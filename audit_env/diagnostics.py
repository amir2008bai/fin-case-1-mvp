"""Per-error-type breakdown and side-by-side official vs experimental risk-aware diagnostics.

The official common reward `clip(2*F1 - 1 - 0.002*steps, -1, 1)` is always preserved
and shown alongside the separate experimental materiality-weighted reward (section 05).
"""
from collections import defaultdict
from statistics import mean

from .agents import create_policy, oracle
from .contracts import ALL_TYPES
from .environment import AuditEnv
from .runner import run_policy
from .scoring import materiality_reward


def evaluate_diagnostics(episodes, policies=("oracle", "expert", "inspector", "micro", "reconcile", "flag-all"),
                         model_path="models/model.json"):
    """Run policies in closed mode and return summary rows + per-type P/R/F1 breakdown."""
    summary = []
    per_type_rows = []
    for name in policies:
        common_rewards, mat_rewards, common_f1s, mat_f1s, compliances, steps_list = [], [], [], [], [], []
        type_counts = {kind: {"tp": 0, "fp": 0, "fn": 0} for kind in ALL_TYPES}
        for episode in episodes:
            env = AuditEnv(training=False)
            obs, _ = env.load_episode(episode)
            policy = oracle(episode) if name == "oracle" else create_policy(name, model_path=model_path)
            outcome, trace, _ = run_policy(env, obs, policy)
            last_action = trace[-1]["action"] if isinstance(trace[-1]["action"], dict) else {}
            payload = last_action.get("findings") if last_action.get("type") == "submit" and last_action.keys() == {"type", "findings"} else None
            seen = env.seen
            if payload is None:
                tool_steps = [e for e in trace if isinstance(e.get("action"), dict) and e["action"].get("type") != "reset"]
                compliance = (sum(1 for e in tool_steps if "error" not in (e.get("observation") or {})) / len(tool_steps)
                              if tool_steps else 1.0)
                if outcome.error is not None:
                    mat_f1, mat_rew = 0.0, -1.0
                else:
                    mat_f1 = 1.0 if not episode.issues else 0.0
                    mat_rew = min(1.0, max(-1.0, 2 * mat_f1 - 1 - 0.002 * outcome.inspection_steps))
            else:
                mat = materiality_reward(
                    payload, episode.manifest["episode_id"], episode.issues, seen,
                    outcome.inspection_steps, trace=trace, termination=outcome.termination,
                )
                mat_f1, mat_rew, compliance = mat.materiality_f1, mat.materiality_reward, mat.tool_compliance
            common_rewards.append(outcome.reward)
            mat_rewards.append(mat_rew)
            common_f1s.append(outcome.f1)
            mat_f1s.append(mat_f1)
            compliances.append(compliance)
            steps_list.append(outcome.inspection_steps)

            # Per-type TP/FP/FN accounting
            matched_findings = {idx for idx, _ in outcome.matches}
            matched_issues = {issue_id for _, issue_id in outcome.matches}
            for issue in episode.issues:
                if issue.type not in type_counts:
                    continue
                if issue.issue_id in matched_issues:
                    type_counts[issue.type]["tp"] += 1
                else:
                    type_counts[issue.type]["fn"] += 1
            if isinstance(payload, dict) and isinstance(payload.get("findings"), list):
                for idx, finding in enumerate(payload["findings"]):
                    if idx not in matched_findings and isinstance(finding, dict):
                        kind = finding.get("type")
                        if kind in type_counts:
                            type_counts[kind]["fp"] += 1

        summary.append({
            "policy": name,
            "episodes": len(episodes),
            "common_f1": round(mean(common_f1s), 4) if common_f1s else 0.0,
            "common_reward": round(mean(common_rewards), 4) if common_rewards else 0.0,
            "materiality_f1": round(mean(mat_f1s), 4) if mat_f1s else 0.0,
            "materiality_reward": round(mean(mat_rewards), 4) if mat_rewards else 0.0,
            "tool_compliance": round(mean(compliances), 4) if compliances else 1.0,
            "steps": round(mean(steps_list), 1) if steps_list else 0.0,
        })
        for kind in ALL_TYPES:
            tp, fp, fn = type_counts[kind]["tp"], type_counts[kind]["fp"], type_counts[kind]["fn"]
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            denom = 2 * tp + fp + fn
            f1 = 2 * tp / denom if denom else 1.0
            per_type_rows.append({
                "policy": name, "type": kind, "tp": tp, "fp": fp, "fn": fn,
                "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            })
    return {"episodes": len(episodes), "summary": summary, "per_type": per_type_rows}
