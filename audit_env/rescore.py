"""Offline trajectory re-scoring, FailureMode taxonomy, FAIL_TO_PASS / PASS_TO_PASS, and unbiased pass^k.

Inspired by tau-bench / tau2-bench, SWE-bench, UK AISI Inspect AI, and FinanceBench,
implemented in pure Python 3.11 stdlib so saved JSONL rollouts match live AuditEnv
scoring exactly under both the required common reward and experimental materiality reward.
"""
import json
import math
from collections import Counter, defaultdict
from enum import Enum
from pathlib import Path
from statistics import mean

from .contracts import MAX_STEPS, TABLE_IDS
from .episode_io import load_episode
from .scoring import materiality_reward, reject, score, score_empty


class FailureMode(str, Enum):
    PASS = "PASS"
    PROTOCOL_ERROR = "PROTOCOL_ERROR"
    MALFORMED_SUBMISSION = "MALFORMED_SUBMISSION"
    TRUNCATED_LIMIT = "TRUNCATED_LIMIT"
    FALSE_POSITIVE_ON_CLEAN = "FALSE_POSITIVE_ON_CLEAN"
    OVER_ABSTENTION = "OVER_ABSTENTION"
    PARTIAL_OR_MISSED = "PARTIAL_OR_MISSED"


def pass_at_k(total_trials, successes, k):
    """Unbiased pass^k estimator: C(c, k) / C(n, k) (tau-bench reliability metric)."""
    if k <= 0 or total_trials < k:
        return None
    if successes < k:
        return 0.0
    return math.comb(successes, k) / math.comb(total_trials, k)


def _load_trials(path):
    """Split JSONL stream into individual rollout trials (splitting on step == 0)."""
    trials = []
    current = {}
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            ep_id = item["episode_id"]
            if item.get("step") == 0:
                if ep_id in current:
                    trials.append((ep_id, current[ep_id]))
                current[ep_id] = [item]
            else:
                current.setdefault(ep_id, []).append(item)
    for ep_id, entries in current.items():
        trials.append((ep_id, entries))
    for _, entries in trials:
        entries.sort(key=lambda entry: entry["step"])
    return trials


def _seen_from_trace(trace):
    """Mirror AuditEnv.expose(): only add each queried table's primary ID and optional doc_id."""
    seen = set()

    def expose_row(table, row):
        if not isinstance(row, dict):
            return
        id_field = TABLE_IDS.get(table)
        if id_field and isinstance(row.get(id_field), str):
            seen.add(row[id_field])
        if isinstance(row.get("doc_id"), str):
            seen.add(row["doc_id"])

    for entry in trace:
        if entry.get("step", 0) <= 0:
            continue
        action = entry.get("action")
        obs = entry.get("observation")
        if not isinstance(action, dict) or not isinstance(obs, dict) or "result" not in obs:
            continue
        kind = action.get("type")
        result = obs["result"]
        if kind == "list_records" and isinstance(result, dict):
            table = action.get("table")
            for row in result.get("records") or ():
                expose_row(table, row)
        elif kind == "get_record" and isinstance(result, dict):
            expose_row(action.get("table"), result)
        elif kind == "get_document" and isinstance(result, dict):
            table = "invoices" if result.get("doc_type") == "invoice" or "invoice_id" in result else "payments"
            expose_row(table, result)
    return seen


def _find_episode_dirs(episodes_root, answers_root, episode_id):
    ep_root, ans_root = Path(episodes_root), Path(answers_root)
    for candidate in ep_root.rglob(episode_id):
        if candidate.is_dir() and (candidate / "manifest.json").is_file():
            rel = candidate.relative_to(ep_root)
            ans_dir = ans_root / rel
            if (ans_dir / "truth.json").is_file():
                return candidate, ans_dir
    return None, None


def _classify_failure(common, is_clean, truncated, submitted_count):
    if common.error == "agent_protocol":
        return FailureMode.PROTOCOL_ERROR
    if common.error is not None:
        return FailureMode.MALFORMED_SUBMISSION
    if truncated:
        return FailureMode.TRUNCATED_LIMIT if common.f1 < 1.0 else FailureMode.PASS
    if common.f1 >= 1.0:
        return FailureMode.PASS
    if is_clean and common.fp > 0:
        return FailureMode.FALSE_POSITIVE_ON_CLEAN
    if not is_clean and submitted_count == 0:
        return FailureMode.OVER_ABSTENTION
    return FailureMode.PARTIAL_OR_MISSED


def rescore_trajectories(trajectories_path, episodes_root="data/episodes", answers_root="data/answers",
                         pass_f1_threshold=1.0):
    """Re-score every rollout trial in a JSONL trajectory file offline against truth.json."""
    trials = _load_trials(trajectories_path)
    rows = []
    missing = []
    for episode_id, trace in trials:
        pub_dir, ans_dir = _find_episode_dirs(episodes_root, answers_root, episode_id)
        if pub_dir is None:
            missing.append(episode_id)
            continue
        episode = load_episode(pub_dir, ans_dir)
        last = trace[-1]
        action = last.get("action") if isinstance(last.get("action"), dict) else None
        obs = last.get("observation") if isinstance(last.get("observation"), dict) else {}
        truncated = bool(last.get("truncated"))
        remaining = obs.get("remaining_steps")
        total_steps = (MAX_STEPS - remaining) if isinstance(remaining, int) else max(0, len(trace) - 1)
        seen = _seen_from_trace(trace)

        # Compute tool compliance directly from trace
        tool_steps = [e for e in trace if isinstance(e.get("action"), dict) and e["action"].get("type") != "reset"]
        compliance = (sum(1 for e in tool_steps if "error" not in (e.get("observation") or {})) / len(tool_steps)
                      if tool_steps else 1.0)

        submitted_count = 0
        if action and action.get("type") == "submit" and not truncated:
            inspection_steps = max(0, total_steps - 1)
            payload = action.get("findings") if action.keys() == {"type", "findings"} else None
            if isinstance(payload, dict) and isinstance(payload.get("findings"), list):
                submitted_count = len(payload["findings"])
            common = score(payload, episode_id, episode.issues, seen, inspection_steps)
            mat = materiality_reward(payload, episode_id, episode.issues, seen, inspection_steps, trace=trace)
            mat_f1, mat_rew = mat.materiality_f1, mat.materiality_reward
        elif obs.get("error") == "agent_protocol" or (action is None and truncated and last.get("reward") == -1.0 and not episode.issues):
            # Protocol error recorded in open-mode trace on clean books
            common = reject(episode.issues, total_steps, error="agent_protocol", termination="agent_protocol")
            mat_f1, mat_rew = 0.0, -1.0
        else:
            inspection_steps = total_steps
            common = score_empty(episode.issues, inspection_steps, "truncated")
            mat_f1 = 1.0 if not episode.issues else 0.0
            mat_rew = min(1.0, max(-1.0, 2 * mat_f1 - 1 - 0.002 * inspection_steps))

        is_clean = not episode.issues
        passed = common.error is None and common.f1 >= pass_f1_threshold
        mode = _classify_failure(common, is_clean, truncated, submitted_count)
        rows.append({
            "episode_id": episode_id,
            "clean": is_clean,
            "tp": common.tp,
            "fp": common.fp,
            "fn": common.fn,
            "common_f1": round(common.f1, 6),
            "common_reward": round(common.reward, 6),
            "materiality_f1": round(mat_f1, 6),
            "materiality_reward": round(mat_rew, 6),
            "tool_compliance": round(compliance, 4),
            "steps": common.inspection_steps,
            "passed": passed,
            "failure_mode": mode.value,
        })

    dirty = [r for r in rows if not r["clean"]]
    clean = [r for r in rows if r["clean"]]
    fail_to_pass = round(mean(1.0 if r["passed"] else 0.0 for r in dirty), 4) if dirty else None
    pass_to_pass = round(mean(1.0 if r["passed"] else 0.0 for r in clean), 4) if clean else None

    # Compute per-episode pass^k when multi-trial, else cross-episode pass^k
    by_ep = defaultdict(list)
    for r in rows:
        by_ep[r["episode_id"]].append(r["passed"])
    max_trials = max((len(v) for v in by_ep.values()), default=0)
    pass_k = {}
    if max_trials >= 2:
        for k in (1, 2, 3):
            vals = [v for ep_res in by_ep.values() if (v := pass_at_k(len(ep_res), sum(ep_res), k)) is not None]
            if vals:
                pass_k[f"pass^{k}"] = round(mean(vals), 4)
    else:
        n_total = len(rows)
        n_pass = sum(1 for r in rows if r["passed"])
        for k in (1, 2, 3):
            if (val := pass_at_k(n_total, n_pass, k)) is not None:
                pass_k[f"pass^{k}"] = round(val, 4)

    failure_modes = dict(sorted(Counter(r["failure_mode"] for r in rows).items()))
    return {
        "trajectories": str(trajectories_path),
        "episodes": len(rows),
        "missing_answers": missing,
        "mean_common_f1": round(mean(r["common_f1"] for r in rows), 4) if rows else 0.0,
        "mean_common_reward": round(mean(r["common_reward"] for r in rows), 4) if rows else 0.0,
        "mean_materiality_reward": round(mean(r["materiality_reward"] for r in rows), 4) if rows else 0.0,
        "mean_tool_compliance": round(mean(r["tool_compliance"] for r in rows), 4) if rows else 0.0,
        "fail_to_pass_rate": fail_to_pass,
        "pass_to_pass_rate": pass_to_pass,
        "pass_k": pass_k,
        "failure_modes": failure_modes,
        "rows": rows,
    }
