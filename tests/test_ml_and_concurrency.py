"""Regression tests for ML edge cases, training split hygiene, rescore fidelity, and API thread safety."""
import json
import tempfile
import threading
import unittest
from pathlib import Path

from audit_env.api import MAX_SESSIONS, App
from audit_env.contracts import Config
from audit_env.diagnostics import evaluate_diagnostics
from audit_env.environment import AuditEnv
from audit_env.episode_io import write_episode
from audit_env.generator import generate
from audit_env.micro import Candidate, MicroModel, candidates, fit_logistic
from audit_env.rescore import FailureMode, _seen_from_trace, rescore_trajectories
from audit_env.training import augmented

ERRORS = {"duplicate_invoice": 1, "payment_allocation": 1, "fx_rate": 1, "period_cutoff": 1}


class MLAndConcurrencyTests(unittest.TestCase):
    def test_micro_handles_edge_cases_without_crashing(self):
        """Missing FX on payment, empty line_items, unknown currency, unseen type, and empty fit_logistic."""
        ep = generate(17, Config(keyspace="open", foreign_percent=30, errors=ERRORS))
        tables = {k: list(v) for k, v in ep.tables.items()}
        tables["fx"] = []  # missing all FX rates
        tables["invoices"].append({
            "doc_id": "doc_empty", "invoice_id": "inv_empty", "entity_id": "ent_1",
            "issued_at": ep.manifest["period_start"], "due_at": ep.manifest["period_end"],
            "currency": "GBP", "total_minor": 0, "line_items": [],
        })
        cands = candidates(tables, ep.manifest, ep.policy)
        self.assertIsInstance(cands, list)
        model = MicroModel.load("models/model.json")
        unseen = Candidate("unseen_type", ("x",), 100, "KZT", ("doc_1",), (1.0,))
        self.assertEqual(model.probability(unseen), 0.0)
        self.assertEqual(fit_logistic([], 4), [0.0, 0.0, 0.0, 0.0])

    def test_training_augmented_respects_max_issues(self):
        cfg = Config(keyspace="open", errors={"duplicate_invoice": 6, "payment_allocation": 5,
                                              "fx_rate": 0, "period_cutoff": 0})
        aug = augmented(cfg, 1)
        self.assertLessEqual(aug.issue_count, 12)

    def test_rescore_seen_matches_live_env_expose_without_leaking_ledger_links(self):
        """_seen_from_trace must NOT mark invoice_id/payment_id/entity_id as seen from ledger rows."""
        trace = [
            {"step": 0, "action": {"type": "reset"}, "observation": {}},
            {"step": 1, "action": {"type": "list_records", "table": "ledger", "limit": 10},
             "observation": {"result": {"records": [{
                 "row_id": "row_1", "entity_id": "ent_1", "invoice_id": "inv_1", "payment_id": "pay_1",
             }]}}},
        ]
        seen = _seen_from_trace(trace)
        self.assertEqual(seen, {"row_1"})

    def test_rescore_splits_multi_trial_jsonl_correctly(self):
        ep = generate(17, Config(keyspace="open", errors={k: 0 for k in ERRORS}))
        ep_id = ep.manifest["episode_id"]
        with tempfile.TemporaryDirectory() as tmp:
            pub = Path(tmp) / "episodes" / ep_id
            ans = Path(tmp) / "answers" / ep_id
            write_episode(ep, pub, ans)
            traj_path = Path(tmp) / "multi.jsonl"
            trial_pass = [
                {"episode_id": ep_id, "step": 0, "action": {"type": "reset"}, "observation": {"remaining_steps": 100}},
                {"episode_id": ep_id, "step": 1,
                 "action": {"type": "submit", "findings": {"episode_id": ep_id, "findings": []}},
                 "observation": {"remaining_steps": 99}, "terminated": True, "truncated": False, "reward": 1.0},
            ]
            trial_fail = [
                {"episode_id": ep_id, "step": 0, "action": {"type": "reset"}, "observation": {"remaining_steps": 100}},
                {"episode_id": ep_id, "step": 1, "action": None,
                 "observation": {"remaining_steps": 100}, "terminated": False, "truncated": True, "reward": -1.0},
            ]
            with traj_path.open("w", encoding="utf-8") as f:
                for entry in trial_pass + trial_fail:
                    f.write(json.dumps(entry) + "\n")
            res = rescore_trajectories(traj_path, Path(tmp) / "episodes", Path(tmp) / "answers")
            self.assertEqual(res["episodes"], 2)
            self.assertEqual(res["pass_k"]["pass^1"], 0.5)
            self.assertEqual(res["pass_k"]["pass^2"], 0.0)
            self.assertEqual(res["failure_modes"][FailureMode.PASS.value], 1)
            self.assertEqual(res["failure_modes"][FailureMode.PROTOCOL_ERROR.value], 1)

    def test_api_sessions_thread_safe_under_concurrent_eviction(self):
        app = App()
        errors = []
        clean_cfg = {"keyspace": "open", "errors": {k: 0 for k in ERRORS}}

        def worker(offset):
            try:
                for i in range(40):
                    created = app.create({"seed": offset * 100 + i, "config": clean_cfg})
                    try:
                        app.session(created["session_id"])
                    except Exception:
                        pass
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(errors, [])
        self.assertLessEqual(len(app.sessions), MAX_SESSIONS)


if __name__ == "__main__":
    unittest.main()
