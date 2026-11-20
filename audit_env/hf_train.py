"""Optional offline LoRA SFT, terminal-reward policy gradient and held-out evaluation."""
import argparse
import hashlib
import itertools
import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from statistics import mean, pstdev

from .dataset import suite
from .generator import generate
from .llm import ContextLimit, rollout
from .llm_data import generated_cases
from .monitor import EpisodeMonitor
from .rag_env import RagAuditEnv


def prompt_ids(tokenizer, messages):
    return tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_dict=False,
    )


def load_bundle(torch, args, *, training):
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not args.model.is_dir() or (args.adapter is not None and not args.adapter.is_dir()):
        raise ValueError("model and adapter must be existing local directories")
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    dtype = {
        "float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16,
    }
    # FP32 is a conservative default outside CUDA. Users can opt into reduced precision.
    default_dtype = torch.float32
    if args.device.startswith("cuda"):
        default_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    if tokenizer.chat_template is None or tokenizer.eos_token_id is None:
        raise ValueError("a chat tokenizer with a template and EOS token is required")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False,
        dtype=dtype.get(args.dtype, default_dtype), use_safetensors=True,
    ).to(args.device)
    if args.adapter is not None:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=training, local_files_only=True)
    elif training:
        model = get_peft_model(model, LoraConfig(
            task_type="CAUSAL_LM", target_modules="all-linear", r=args.rank,
            lora_alpha=2 * args.rank, lora_dropout=0,
        ))
    if training and args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model, tokenizer


def encode_samples(tokenizer, path, max_length):
    examples = []
    skipped = 0
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            sample = json.loads(line)
            prefix = prompt_ids(tokenizer, sample["prompt"])
            completion = tokenizer.encode(sample["completion"][0]["content"], add_special_tokens=False)
            completion.append(tokenizer.eos_token_id)
            if len(prefix) + len(completion) > max_length:
                skipped += 1
                continue
            examples.append((prefix, completion))
    if not examples:
        raise ValueError(f"no examples fit the context in {path}")
    return examples, skipped


def action_log_probs(torch, model, prefix, completion, device, temperature=1.0):
    if not prefix or not completion:
        raise ValueError("empty prefix or assistant action")
    inputs = torch.tensor([prefix + completion], dtype=torch.long, device=device)
    # Predict each assistant token from the previous token. Prompt tokens have no loss.
    logits = model(input_ids=inputs, use_cache=False).logits[0, len(prefix) - 1:-1]
    targets = inputs[0, len(prefix):]
    return torch.log_softmax(logits.float() / temperature, dim=-1).gather(1, targets[:, None])[:, 0]


def optimizer_for(torch, model, rate):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=rate)


def verify_dataset(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("version") != 1:
        raise ValueError("unsupported training dataset version")
    for filename in ("train.sft.jsonl", "dev.sft.jsonl"):
        with (directory / filename).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != manifest.get("sha256", {}).get(filename):
            raise ValueError(f"dataset checksum mismatch: {filename}")


def run_sft(torch, args, model, tokenizer):
    verify_dataset(args.dataset)
    train, omitted = encode_samples(tokenizer, args.dataset / "train.sft.jsonl", args.max_length)
    dev, dev_omitted = encode_samples(tokenizer, args.dataset / "dev.sft.jsonl", args.max_length)
    rng = random.Random(args.seed)
    rng.shuffle(train)
    sequence = itertools.cycle(train)
    optimizer = optimizer_for(torch, model, args.learning_rate)
    model.train()
    training_loss = []
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        for prefix, completion in itertools.islice(sequence, args.grad_accum):
            loss = -action_log_probs(torch, model, prefix, completion, args.device).mean()
            (loss / args.grad_accum).backward()
            training_loss.append(float(loss.detach()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        dev_losses = [
            float(-action_log_probs(torch, model, prefix, completion, args.device).mean())
            for prefix, completion in dev[:args.eval_examples]
        ]
    return {"updates": args.steps, "train_examples": len(train), "dev_examples": len(dev),
            "train_context_skipped": omitted, "dev_context_skipped": dev_omitted,
            "mean_train_loss": mean(training_loss), "dev_loss": mean(dev_losses)}


def generate_turn(torch, model, tokenizer, args, messages, *, sample):
    from transformers import GenerationConfig

    prefix = prompt_ids(tokenizer, messages)
    context = getattr(model.config, "max_position_embeddings", None)
    window = min(args.max_length, context) if isinstance(context, int) else args.max_length
    if len(prefix) + args.max_new_tokens > window:
        raise ContextLimit
    ids = torch.tensor([prefix], dtype=torch.long, device=args.device)
    generation = {
        "do_sample": sample, "max_new_tokens": args.max_new_tokens,
        "eos_token_id": model.generation_config.eos_token_id or tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    }
    if sample:
        generation.update(temperature=args.temperature, top_k=0, top_p=1.0)
    with torch.no_grad():
        output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                generation_config=GenerationConfig(**generation))
    completion = output[0, len(prefix):].tolist()
    return tokenizer.decode(completion, skip_special_tokens=True), (prefix, completion)


def policy_episode(torch, args, model, tokenizer, episode, *, sample, observer=None):
    env = RagAuditEnv(training=sample)
    start, _ = env.load_episode(episode)
    tokens = []

    def next_action(messages):
        text, ids = generate_turn(torch, model, tokenizer, args, messages, sample=sample)
        tokens.append(ids)
        return text

    outcome = rollout(env, start, next_action, max_actions=args.max_actions, observer=observer)
    return outcome, tokens


def reinforce_update(torch, model, optimizer, trajectories, args):
    rewards = [reward for reward, _ in trajectories]
    scale = pstdev(rewards)
    if scale < 1e-8:
        return False
    baseline = mean(rewards)
    turns = sum(len(tokens) for _, tokens in trajectories)
    if not turns:
        return False
    optimizer.zero_grad(set_to_none=True)
    for reward, tokens in trajectories:
        advantage = (reward - baseline) / scale
        for prefix, completion in tokens:
            logprob = action_log_probs(torch, model, prefix, completion, args.device, args.temperature)
            (-advantage * logprob.mean() / turns).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return True


def open_cases(args):
    if args.suite is not None:
        return suite(args.suite)
    return generated_cases(args.seed_base, args.train_episodes, args.dev_episodes)


def run_rl(torch, args, model, tokenizer):
    model.eval()  # Keep sampling and gradient replay on the same dropout-free policy.
    optimizer = optimizer_for(torch, model, args.learning_rate)
    roster = [(seed, config) for split, seed, config in open_cases(args) if split == "train"]
    if not roster:
        raise ValueError("RL suite needs a train split")
    updates = 0
    rewards = []
    for seed, config in itertools.islice(itertools.cycle(roster), args.steps):
        episode = generate(seed, config)
        group = []
        for _ in range(args.group_size):
            score, actions = policy_episode(torch, args, model, tokenizer, episode, sample=True)
            group.append((score.reward, actions))
            rewards.append(score.reward)
        updates += int(reinforce_update(torch, model, optimizer, group, args))
    return {"groups": args.steps, "updates": updates, "zero_signal_groups": args.steps - updates,
            "mean_rollout_reward": mean(rewards), "rollouts": len(rewards)}


def run_eval(torch, args, model, tokenizer):
    model.eval()
    scores = []
    for split, seed, config in open_cases(args):
        if split == "dev":
            episode = generate(seed, config)
            monitor = EpisodeMonitor()
            outcome, _ = policy_episode(torch, args, model, tokenizer, episode, sample=False,
                                        observer=monitor)
            scores.append({"episode_id": episode.manifest["episode_id"],
                           **asdict(outcome), "monitor": monitor.report()})
    if not scores:
        raise ValueError("evaluation suite needs a dev split")
    return {"episodes": len(scores), "mean_f1": mean(row["f1"] for row in scores),
            "mean_reward": mean(row["reward"] for row in scores), "results": scores}


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("sft", "rl", "eval"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset", type=Path, default=Path("build/llm"))
    parser.add_argument("--suite", type=Path, help="use an explicit suite for RL and eval")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("auto", "float32", "bfloat16", "float16"), default="auto")
    parser.add_argument("--rank", type=positive, default=16)
    parser.add_argument("--max-length", type=positive, default=16384)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--max-new-tokens", type=positive, default=4096)
    parser.add_argument("--max-actions", type=positive, default=35)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--group-size", type=positive, default=4)
    parser.add_argument("--grad-accum", type=positive, default=4)
    parser.add_argument("--steps", type=positive, default=20)
    parser.add_argument("--eval-examples", type=positive, default=16)
    parser.add_argument("--train-episodes", type=positive, default=96)
    parser.add_argument("--dev-episodes", type=positive, default=24)
    parser.add_argument("--seed-base", type=int, default=7_001_031)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    training = args.command != "eval"
    if training and (args.output is None or args.output.exists()):
        parser.error("training requires a new --output directory")
    if args.group_size < 2 or args.max_actions > 100:
        parser.error("group-size must be at least 2; max-actions must be at most 100")
    if (not math.isfinite(args.learning_rate) or not math.isfinite(args.temperature)
            or args.learning_rate <= 0 or args.temperature <= 0):
        parser.error("learning rate and temperature must be finite and positive")
    try:
        import torch
    except ImportError:
        parser.error("install optional dependencies from requirements-llm.txt")
    torch.manual_seed(args.seed)
    model, tokenizer = load_bundle(torch, args, training=training)
    context = getattr(model.config, "max_position_embeddings", None)
    if isinstance(context, int):
        args.max_length = min(args.max_length, context)
    operations = {"sft": run_sft, "rl": run_rl, "eval": run_eval}
    report = operations[args.command](torch, args, model, tokenizer)
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if training:
        args.output.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(args.output, safe_serialization=True)
        tokenizer.save_pretrained(args.output)
        (args.output / "training_report.json").write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()
