"""Exercise both training paths on a randomly initialized CPU-only chat model."""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from .agents import submission
from .generator import generate
from .hf_train import load_bundle, optimizer_for, policy_episode, prompt_ids, reinforce_update, run_sft
from .llm import json_text
from .llm_data import export_llm, generated_cases, samples, teacher_trace
from .rag_env import RagAuditEnv


def build_base(path):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    backend = Tokenizer(models.BPE(unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=512,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<unk>", "<eos>", "<pad>"],
    )
    # Only tiny synthetic byte strings are used to build the smoke tokenizer.
    backend.train_from_iterator(
        ['{"type":"submit","findings":[]}', "Банк проводка платёж расход результат"],
        trainer=trainer,
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<pad>",
    )
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['role'] + ': ' + message['content'] + '\\n' }}"
        "{% endfor %}{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    )
    tokenizer.save_pretrained(path)
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=len(tokenizer), n_positions=8192, n_layer=1, n_head=2, n_embd=16,
        bos_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    ))
    model.save_pretrained(path, safe_serialization=True)


def check_reward_update(torch, args, model, tokenizer):
    _, seed, config = list(generated_cases(81, 2, 1))[1]
    episode = generate(seed, config)
    positive, trace = teacher_trace(episode)
    final = list(samples(trace))[-1]
    prefix = prompt_ids(tokenizer, final["prompt"])
    bad_action = submission(episode.manifest["episode_id"], [])
    env = RagAuditEnv(training=True)
    env.load_episode(episode)
    for row in trace[1:-1]:
        env.step(row["action"])
    env.step(bad_action)
    negative = env.outcome.reward
    assert positive.reward > negative
    good_tokens = tokenizer.encode(final["completion"][0]["content"], add_special_tokens=False)
    bad_tokens = tokenizer.encode(json_text(bad_action), add_special_tokens=False)
    optimizer = optimizer_for(torch, model, args.learning_rate)
    model.eval()
    before = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    assert not reinforce_update(torch, model, optimizer, [(negative, []), (negative, [])], args)
    assert reinforce_update(torch, model, optimizer, [
        (positive.reward, [(prefix, good_tokens)]), (negative, [(prefix, bad_tokens)]),
    ], args)
    after = [p.detach() for p in model.parameters() if p.requires_grad]
    assert any(not torch.equal(a, b) for a, b in zip(before, after))
    return {"teacher_reward": positive.reward, "empty_reward": negative, "weights_changed": True}


def main():
    import torch

    torch.set_num_threads(2)
    torch.manual_seed(11)
    with tempfile.TemporaryDirectory(prefix="fintech-llm-") as temp:
        root = Path(temp)
        build_base(root / "base")
        export_llm(root / "data", generated_cases(81, 2, 2))
        args = SimpleNamespace(
            model=root / "base", adapter=None, device="cpu", dtype="float32", rank=2,
            gradient_checkpointing=False, dataset=root / "data", max_length=8192, seed=11,
            learning_rate=.001, steps=1, grad_accum=1, eval_examples=1, temperature=.8,
            max_new_tokens=8, max_actions=1,
        )
        model, tokenizer = load_bundle(torch, args, training=True)
        report = run_sft(torch, args, model, tokenizer)
        model.save_pretrained(root / "adapter", safe_serialization=True)
        args.adapter = root / "adapter"
        model, tokenizer = load_bundle(torch, args, training=True)
        reinforcement = check_reward_update(torch, args, model, tokenizer)
        _, seed, config = next(generated_cases(19, 1, 1))
        outcome, turns = policy_episode(torch, args, model, tokenizer, generate(seed, config), sample=True)
        assert len(turns) == 1 and turns[0][1]
        print(json.dumps({"sft": report, "rl_gradient_smoke": reinforcement,
                          "sampled_rollout_reward": outcome.reward}))


if __name__ == "__main__":
    main()
