"""Named, printable checks grouped by the jury's categories; used by `verify`."""
import json
import os
from dataclasses import replace

from .agents import create_policy, oracle, submission
from .checks import audit_episode
from .contracts import EXTRA_TYPES, Config
from .environment import AuditEnv
from .generator import collect_ids, generate, transform
from .provenance import restore_clean
from .runner import run_policy
from .scoring import score

ERRORS = {"duplicate_invoice": 1, "payment_allocation": 1, "fx_rate": 1, "period_cutoff": 1}
MIXED = Config(keyspace="open", errors=ERRORS, interactions={"fx_boundary_payment": 1, "fx_partial_allocation": 1})
CLEAN = Config(keyspace="open", errors={key: 0 for key in ERRORS})


def run(episode, policy, training=True):
    env = AuditEnv(training=training)
    observation, _ = env.load_episode(episode)
    return run_policy(env, observation, policy)


def everything_seen(episode):
    seen = set()
    for table, rows in episode.tables.items():
        for row in rows:
            seen |= {value for key, value in row.items() if key.endswith("_id") and isinstance(value, str)}
    return seen


def compute(episode, findings, issues=None, steps=0):
    issues = episode.issues if issues is None else issues
    return score({"episode_id": episode.manifest["episode_id"], "findings": findings},
                 episode.manifest["episode_id"], issues, everything_seen(episode), steps)


def contract_checks():
    episode = generate(17, MIXED)
    clean = generate(18, CLEAN)
    truth = [issue.as_finding() for issue in episode.issues]
    checks = []

    def check(name, category, function):
        try:
            ok, detail = function()
        except Exception as exc:  # a crashing check is a failed check
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        checks.append({"name": name, "category": category, "ok": bool(ok), "detail": detail})

    def reset_shape():
        env = AuditEnv()
        observation, info = env.reset(17, MIXED)
        json.dumps(observation)
        return ({"task", "manifest", "policy", "remaining_steps"} <= observation.keys() and info == {}
                and observation["remaining_steps"] == 100), "reset -> (observation, info)"

    def step_shape():
        env = AuditEnv()
        env.reset(17, MIXED)
        result = env.step({"type": "list_records", "table": "ledger", "limit": 50})
        json.dumps(result)
        return (len(result) == 5 and result[1:4] == (0.0, False, False)
                and len(result[0]["result"]["records"]) == 50), "5-tuple, zero intermediate reward, page of 50"

    def bounded_errors():
        env = AuditEnv()
        env.reset(17, MIXED)
        codes = []
        for action in ({"type": "list_records", "table": "ledger", "limit": 51}, {"type": "get_record", "table": "ledger", "id": "x"},
                       {"type": "get_document", "doc_id": "x"}, {"type": "unknown"}, None,
                       {"type": "calculator", "expression": "__import__('os')"}, {"type": "calculator", "expression": "2**8"},
                       {"type": "list_records", "table": "ledger", "cursor": "forged:1"}):
            observation, reward, terminated, truncated, _ = env.step(action)
            codes.append(observation.get("error"))
            if reward != 0.0 or terminated or truncated:
                return False, f"action {action} ended the episode"
        return all(codes) and env.steps == 8, f"errors {codes}, every one cost a step"

    def calculator_decimal():
        env = AuditEnv()
        env.reset(17, MIXED)
        return env.step({"type": "calculator", "expression": "0.1 + 0.2"})[0]["result"] == "0.3", "decimal arithmetic"

    def cursor_stable():
        env = AuditEnv()
        env.reset(17, MIXED)
        first = env.step({"type": "list_records", "table": "ledger", "limit": 20})[0]["result"]
        again = env.step({"type": "list_records", "table": "ledger", "limit": 20})[0]["result"]
        nxt = env.step({"type": "list_records", "table": "ledger", "limit": 20, "cursor": first["next_cursor"]})[0]["result"]
        ids = [row["row_id"] for row in first["records"] + nxt["records"]]
        return first == again and len(set(ids)) == 40, "same page twice, next page disjoint"

    def submit_terminates():
        env = AuditEnv()
        env.reset(17, MIXED)
        _, reward, terminated, truncated, _ = env.step(submission(env.episode.manifest["episode_id"], []))
        after = env.step({"type": "list_records", "table": "ledger"})
        return (terminated and not truncated and reward == -1.0 and after[0]["error"] == "episode_finished"
                and after[2:4] == (True, False)), "submit -> terminated; later steps are bounded errors"

    def step_limit():
        env = AuditEnv()
        env.reset(18, CLEAN)
        for _ in range(100):
            observation, reward, terminated, truncated, _ = env.step(None)
        return (truncated and not terminated and observation["findings"] == [] and abs(reward - 0.8) < 1e-9
                and env.outcome.termination == "step_limit"), f"100 steps -> truncated, empty answer scored {reward}"

    def timeout():
        now = [0.0]
        env = AuditEnv(timeout_seconds=1, clock=lambda: now[0])
        env.reset(17, MIXED)
        now[0] = 5.0
        _, reward, terminated, truncated, _ = env.step({"type": "list_records", "table": "fx"})
        return truncated and not terminated and reward == -1.0, "deadline -> truncated with empty answer"

    def closed_mode():
        outcome, trace, _ = run(episode, oracle(episode), training=False)
        return trace[-1]["reward"] is None and outcome.f1 == 1.0, "closed mode hides the terminal reward"

    def no_leak():
        secret = {issue.issue_id for issue in episode.issues} | {episode.provenance["key_fingerprint"]}
        for name in ("expert", "reconcile", "random"):
            _, trace, _ = run(episode, create_policy(name), training=False)
            for step in trace:
                found = set()
                collect_ids(step["observation"], found)
                text = json.dumps(step["observation"], ensure_ascii=False)
                if found & secret or any(word in text for word in ("evidence_sets", "expected_amount", "provenance",
                                                                  "truth", "\"seed\"")):
                    return False, f"{name} saw private data at step {step['step']}"
        return True, "no truth, seed, issue IDs or key fingerprint in any observation"

    def reproducible():
        first, second = generate(99, MIXED), generate(99, MIXED)
        _, trace_a, _ = run(first, create_policy("expert"))
        _, trace_b, _ = run(second, create_policy("expert"))
        return first.tables == second.tables and trace_a == trace_b, "same seed/config/actions -> same records and trace"

    def secret_key():
        a = generate(5, replace(MIXED, keyspace="secret"), key=b"evaluator-key-one")
        b = generate(5, replace(MIXED, keyspace="secret"), key=b"evaluator-key-two")
        public = generate(5, MIXED)
        return (a.manifest["episode_id"] != b.manifest["episode_id"] != public.manifest["episode_id"]
                and a.tables["ledger"] != b.tables["ledger"]), "without the key the seed reproduces nothing"

    def correct_beats_silence():
        return compute(episode, truth).reward > compute(episode, []).reward, "erroneous books"

    def silence_beats_accusations():
        fake = [dict(truth[0])]
        return compute(clean, [], issues=()).reward > compute(clean, fake, issues=()).reward, "clean books"

    def duplicates_no_gain():
        once, twice = compute(episode, truth), compute(episode, truth + truth)
        return twice.tp == once.tp and twice.reward < once.reward, f"tp {once.tp}, repeated fp {twice.fp}"

    def flag_all_bad():
        bad, _, _ = run(episode, create_policy("flag-all"))
        bad_clean, _, _ = run(clean, create_policy("flag-all"))
        return bad.reward <= -0.8 and bad_clean.reward == -1.0, f"{bad.reward:+.3f} / {bad_clean.reward:+.3f}"

    def step_cost():
        a, b = compute(episode, truth, steps=10), compute(episode, truth, steps=13)
        return abs((a.reward - b.reward) - 0.006) < 1e-9, "3 extra steps cost exactly 0.006"

    def evidence_rules():
        first = truth[0]
        unseen = score({"episode_id": episode.manifest["episode_id"], "findings": [first]},
                       episode.manifest["episode_id"], episode.issues, set(), 0).tp
        guessed = compute(episode, [first | {"evidence_ids": ["row_" + "0" * 24]}]).tp
        foreign_row = next(row["row_id"] for row in episode.tables["ledger"]
                           if row["row_id"] not in episode.issues[0].relevant_ids)
        irrelevant = compute(episode, [first | {"evidence_ids": first["evidence_ids"][:9] + [foreign_row]}]).tp
        partial = compute(episode, [first | {"evidence_ids": first["evidence_ids"][:1]}]).tp
        return (unseen, guessed, irrelevant, partial) == (0, 0, 0, 0), "unseen, guessed, irrelevant, partial evidence fail"

    def amount_currency_rules():
        first = truth[0]
        tolerance = episode.issues[0].tolerance_minor
        wrong_currency = compute(episode, [first | {"currency": "EUR"}]).tp
        edge = compute(episode, [first | {"claimed_amount_minor": first["claimed_amount_minor"] + tolerance}]).tp
        beyond = compute(episode, [first | {"claimed_amount_minor": first["claimed_amount_minor"] + tolerance + 1}]).tp
        return (wrong_currency, edge, beyond) == (0, 1, 0), "currency must match; tolerance edge inclusive"

    def malformed():
        payloads = [None, {"episode_id": "x", "findings": []}, {"findings": []},
                    {"episode_id": episode.manifest["episode_id"], "findings": [truth[0]] * 31},
                    {"episode_id": episode.manifest["episode_id"], "findings": [truth[0] | {"claimed_amount_minor": 1.5}]},
                    {"episode_id": episode.manifest["episode_id"], "findings": [truth[0] | {"explanation": "x" * 1001}]},
                    {"episode_id": episode.manifest["episode_id"], "findings": [truth[0] | {"extra": 1}]}]
        rewards = [score(payload, episode.manifest["episode_id"], episode.issues, everything_seen(episode), 0).reward
                   for payload in payloads]
        unknown = compute(episode, truth + [truth[0] | {"type": "made_up_type"}])
        return all(value == -1.0 for value in rewards) and unknown.fp == 1, "malformed -> -1, unknown type -> FP"

    def paired():
        mirror = transform(episode, 4242)
        first, _, _ = run(episode, oracle(episode))
        second, _, _ = run(mirror, oracle(mirror))
        return (first.f1, first.reward) == (second.f1, second.reward) == (1.0, first.reward), "oracle preserved"

    def independent_truth():
        found = audit_episode(episode)
        result = compute(episode, found)
        empty = audit_episode(restore_clean(episode))
        return result.f1 == 1.0 and empty == [], "checks.py: clean -> nothing, dirty -> exactly the truth"

    check("reset returns (observation, info)", "contract", reset_shape)
    check("step returns a JSON 5-tuple", "contract", step_shape)
    check("bounded errors cost a step", "contract", bounded_errors)
    check("calculator: decimals only", "contract", calculator_decimal)
    check("stable cursors, max 50 rows", "contract", cursor_stable)
    check("submit terminates", "contract", submit_terminates)
    check("100-step limit truncates", "contract", step_limit)
    check("timeout truncates", "contract", timeout)
    check("closed mode hides reward", "isolation", closed_mode)
    check("no truth leaks to observations", "isolation", no_leak)
    check("secret key protects hidden seeds", "isolation", secret_key)
    check("reproducible seed/config/actions", "reproducibility", reproducible)
    check("paired ID/order transform", "reproducibility", paired)
    check("independent truth verification", "data", independent_truth)
    check("correct answer beats silence", "reward", correct_beats_silence)
    check("silence beats false accusations", "reward", silence_beats_accusations)
    check("repeated findings give no gain", "reward", duplicates_no_gain)
    check("flag-all attack scores badly", "reward", flag_all_bad)
    check("step economy separates slightly", "reward", step_cost)
    check("evidence must be seen and relevant", "reward", evidence_rules)
    check("currency and tolerance rules", "reward", amount_currency_rules)
    check("malformed input and unknown types", "reward", malformed)
    return checks


def data_invariants(count, key=None, seed_base=0):
    """Generate `count` varied episodes and prove truth correctness independently."""
    import random

    rng = random.Random(seed_base)
    failures, kinds = [], set()
    for index in range(count):
        config = Config(
            keyspace="open" if key is None else "secret",
            month=rng.randint(1, 12), timezone=rng.choice(("+03:00", "+05:00", "+06:00")),
            acts=rng.randint(15, 80), customers=rng.randint(5, 45), foreign_percent=rng.randint(0, 50),
            goods_percent=rng.choice((0, 30, 50, 80)), vat=rng.random() < 0.7, filler=rng.randint(0, 80),
            errors={name: rng.choice((0, 0, 1, 2)) for name in ERRORS}
                   | {name: rng.choice((0, 0, 1)) for name in EXTRA_TYPES},
            interactions={"fx_boundary_payment": rng.choice((0, 0, 1)), "fx_partial_allocation": rng.choice((0, 0, 1))},
            distractors={name: rng.random() < 0.85 for name in Config().distractors},
        )
        seed = rng.randrange(2**31)
        try:
            episode = generate(seed, config, key=key)
            kinds |= {issue.type for issue in episode.issues}
            clean = restore_clean(episode)
            if audit_episode(clean):
                failures.append(f"case {index}: clean books produced findings")
            if compute(episode, audit_episode(episode)).f1 != 1.0:
                failures.append(f"case {index}: independent audit disagrees with the truth")
            for issue in episode.issues:
                if not all(1 <= len(proof) <= 10 and set(proof) <= set(issue.relevant_ids) for proof in issue.evidence_sets):
                    failures.append(f"case {index}: evidence set invalid")
        except Exception as exc:
            failures.append(f"case {index}: {type(exc).__name__}: {exc}")
    return failures, sorted(kinds)


def fresh_key():
    return os.urandom(32)
