"""Requirement-to-test traceability suite covering Sections 03, 04, and 05 of genesis.pdf.

Each test maps directly to one requirement of the hackathon brief so the jury can
inspect compliance 1-to-1. Every docstring cites the primary detailed test module.
"""
from datetime import datetime
from decimal import Decimal
import json
import time
import unittest

from audit_env.agents import create_policy
from audit_env.contracts import Config, MAX_EVIDENCE, MAX_EXPLANATION, MAX_FINDINGS, MAX_PAGE, MAX_STEPS
from audit_env.environment import AuditEnv
from audit_env.generator import generate
from audit_env.scoring import score
from audit_env.training import run_policy

CLEAN_CFG = Config(keyspace="open", errors={"duplicate_invoice": 0, "payment_allocation": 0, "fx_rate": 0, "period_cutoff": 0})
DIRTY_CFG = Config(
    keyspace="open",
    errors={"duplicate_invoice": 1, "payment_allocation": 1, "fx_rate": 1, "period_cutoff": 1},
    interactions={"fx_boundary_payment": 1, "fx_partial_allocation": 1},
)


def _all_seen(episode):
    return {
        value
        for rows in episode.tables.values()
        for row in rows
        for key, value in row.items()
        if key.endswith("_id") and isinstance(value, str)
    }


def _oracle_payload(episode):
    return {
        "episode_id": episode.manifest["episode_id"],
        "findings": [
            {
                "type": issue.type,
                "affected_ids": list(issue.affected_ids),
                "claimed_amount_minor": issue.expected_amount_minor,
                "currency": issue.currency,
                "evidence_ids": list(issue.evidence_sets[0]),
                "explanation": "verified",
            }
            for issue in episode.issues
        ],
    }


class BriefCriteriaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clean_ep = generate(101, CLEAN_CFG)
        cls.dirty_ep = generate(102, DIRTY_CFG)

    # ── Section 03: Эпизод и формат данных ────────────────────────────────────

    def test_brief_03_episode_size_and_manifest_fields(self):
        """Brief 03: 200-2000 ledger rows, 20-200 docs, 0-12 errors, required manifest keys.
        Primary tests: tests/test_world.py, tests/test_schemas.py.
        """
        ep = self.dirty_ep
        self.assertTrue(200 <= len(ep.tables["ledger"]) <= 2000)
        docs = len(ep.tables["invoices"]) + len(ep.tables["payments"])
        self.assertTrue(20 <= docs <= 200)
        self.assertTrue(0 <= len(ep.issues) <= 12)
        for key in ("schema_version", "episode_id", "base_currency", "period_start", "period_end",
                    "timezone", "counts", "document_count", "minor_units", "rounding"):
            self.assertIn(key, ep.manifest)

    def test_brief_03_timestamps_are_iso8601_with_offset(self):
        """Brief 03: Timestamps use ISO-8601 with explicit UTC offset.
        Primary tests: tests/test_world.py, tests/test_schemas.py.
        """
        for row in self.dirty_ep.tables["ledger"][:20]:
            dt = datetime.fromisoformat(row["timestamp"])
            self.assertIsNotNone(dt.tzinfo)
        for pay in self.dirty_ep.tables["payments"][:20]:
            dt = datetime.fromisoformat(pay["timestamp"])
            self.assertIsNotNone(dt.tzinfo)

    def test_brief_03_money_is_integer_minor_units(self):
        """Brief 03: Money values are strict integer minor units with currency exponent in manifest.
        Primary tests: tests/test_world.py, tests/test_schemas.py.
        """
        self.assertIsInstance(self.dirty_ep.manifest["minor_units"], dict)
        for row in self.dirty_ep.tables["ledger"][:30]:
            self.assertIs(type(row["debit_minor"]), int)
            self.assertIs(type(row["credit_minor"]), int)
        for inv in self.dirty_ep.tables["invoices"][:20]:
            self.assertIs(type(inv["total_minor"]), int)
        for pay in self.dirty_ep.tables["payments"][:20]:
            self.assertIs(type(pay["amount_minor"]), int)

    def test_brief_03_fx_rates_are_decimal_strings_and_currencies_not_mixed(self):
        """Brief 03: FX rates are decimal strings; currencies cannot be compared without conversion.
        Primary tests: tests/test_schemas.py, tests/test_scoring.py.
        """
        for row in self.dirty_ep.tables["fx"][:20]:
            self.assertIsInstance(row["rate"], str)
            self.assertGreater(Decimal(row["rate"]), 0)
        payload = _oracle_payload(self.dirty_ep)
        payload["findings"][0]["currency"] = "USD" if payload["findings"][0]["currency"] != "USD" else "KZT"
        res = score(payload, self.dirty_ep.manifest["episode_id"], self.dirty_ep.issues, _all_seen(self.dirty_ep), 5)
        self.assertLess(res.tp, len(self.dirty_ep.issues))

    def test_brief_03_open_records_have_required_columns_and_policy(self):
        """Brief 03: Public tables (ledger, invoices, payments, fx) and policy.json have required fields.
        Primary tests: tests/test_schemas.py, tests/test_schema_validation.py.
        """
        ep = self.dirty_ep
        self.assertTrue({"row_id", "timestamp", "account", "debit_minor", "credit_minor", "currency",
                         "entity_id", "invoice_id", "payment_id"}.issubset(ep.tables["ledger"][0].keys()))
        self.assertTrue({"doc_id", "invoice_id", "entity_id", "issued_at", "due_at", "currency",
                         "total_minor", "line_items"}.issubset(ep.tables["invoices"][0].keys()))
        self.assertTrue({"doc_id", "payment_id", "timestamp", "currency", "amount_minor",
                         "invoice_refs"}.issubset(ep.tables["payments"][0].keys()))
        self.assertTrue({"date", "source_currency", "target_currency", "rate"}.issubset(ep.tables["fx"][0].keys()))
        self.assertIn("fx", ep.policy)
        self.assertIn("period", ep.policy)

    def test_brief_03_missing_links_are_explicit_null(self):
        """Brief 03: Missing links are explicit null (None), never guessed or empty strings.
        Primary tests: tests/test_world.py.
        """
        ledger = self.dirty_ep.tables["ledger"]
        self.assertTrue(any(row["invoice_id"] is None for row in ledger))
        self.assertTrue(any(row["payment_id"] is None for row in ledger))
        for row in ledger:
            self.assertNotEqual(row["invoice_id"], "")
            self.assertNotEqual(row["payment_id"], "")
        for pay in self.dirty_ep.tables["payments"]:
            self.assertTrue(pay["invoice_refs"] is None or isinstance(pay["invoice_refs"], list))

    def test_brief_03_closed_truth_and_provenance_never_leak(self):
        """Brief 03: Closed truth/provenance/seed never leak into observations.
        Primary tests: tests/test_contract.py, tests/test_adapter_isolation.py.
        """
        env = AuditEnv(training=False)
        obs, info = env.reset(77, Config(keyspace="secret", errors=DIRTY_CFG.errors))
        blob = json.dumps({"obs": obs, "info": info}, ensure_ascii=False)
        for forbidden in ("truth", "provenance", "issue_id", "expected_amount_minor"):
            self.assertNotIn(forbidden, blob)

    # ── Section 04: Наблюдения, действия, завершение ──────────────────────────

    def test_brief_04_reset_and_step_gymnasium_json_contract(self):
        """Brief 04: reset -> (obs, info) with task/manifest/policy/budget; step -> JSON 5-tuple.
        Primary tests: tests/test_contract.py.
        """
        env = AuditEnv(training=True)
        obs, info = env.load_episode(self.clean_ep)
        self.assertEqual(info, {})
        for key in ("task", "manifest", "policy", "remaining_steps"):
            self.assertIn(key, obs)
        self.assertEqual(obs["remaining_steps"], MAX_STEPS)
        step_out = env.step({"type": "list_records", "table": "invoices", "limit": 5})
        self.assertEqual(len(step_out), 5)
        json.dumps(step_out)  # Must be strictly JSON-serializable

    def test_brief_04_reproducibility_and_reset_isolation(self):
        """Brief 04: Same seed/config/actions reproduce identically; reset isolates episodes.
        Primary tests: tests/test_repro.py, tests/test_rlvr_standards.py.
        """
        env = AuditEnv(training=True)
        obs1, _ = env.reset(42, DIRTY_CFG)
        s1 = env.step({"type": "list_records", "table": "ledger", "limit": 10})
        env.reset(99, CLEAN_CFG)
        env.step({"type": "calculator", "expression": "2 + 2"})
        obs2, _ = env.reset(42, DIRTY_CFG)
        s2 = env.step({"type": "list_records", "table": "ledger", "limit": 10})
        self.assertEqual(obs1, obs2)
        self.assertEqual(s1, s2)

    def test_brief_04_five_actions_pagination_max_50_and_stable_cursors(self):
        """Brief 04: 5 actions work; list_records max 50 rows/page with stable cursors.
        Primary tests: tests/test_contract.py.
        """
        env = AuditEnv(training=True)
        env.load_episode(self.dirty_ep)
        p1, _, _, _, _ = env.step({"type": "list_records", "table": "ledger", "limit": MAX_PAGE})
        self.assertEqual(len(p1["result"]["records"]), MAX_PAGE)
        p1_again, _, _, _, _ = env.step({"type": "list_records", "table": "ledger", "limit": MAX_PAGE})
        self.assertEqual(p1["result"]["records"], p1_again["result"]["records"])
        p2, _, _, _, _ = env.step({"type": "list_records", "table": "ledger", "cursor": p1["result"]["next_cursor"], "limit": MAX_PAGE})
        self.assertTrue({r["row_id"] for r in p1["result"]["records"]}.isdisjoint({r["row_id"] for r in p2["result"]["records"]}))
        inv_id = self.dirty_ep.tables["invoices"][0]["invoice_id"]
        doc_id = self.dirty_ep.tables["invoices"][0]["doc_id"]
        rec, _, _, _, _ = env.step({"type": "get_record", "table": "invoices", "id": inv_id})
        self.assertEqual(rec["result"]["invoice_id"], inv_id)
        doc, _, _, _, _ = env.step({"type": "get_document", "doc_id": doc_id})
        self.assertEqual(doc["result"]["doc_id"], doc_id)

    def test_brief_04_calculator_decimals_only_no_code(self):
        """Brief 04: Calculator allows only decimal arithmetic, rejecting code or file access.
        Primary tests: tests/test_contract.py, tests/test_fuzz.py.
        """
        env = AuditEnv(training=True)
        env.load_episode(self.clean_ep)
        ok, _, _, _, _ = env.step({"type": "calculator", "expression": "(100.50 + 0.25) * 2"})
        self.assertEqual(Decimal(ok["result"]), Decimal("201.50"))
        bad, _, _, _, _ = env.step({"type": "calculator", "expression": "__import__('os').system('id')"})
        self.assertEqual(bad["error"], "invalid_expression")

    def test_brief_04_unknown_id_costs_a_step_and_gives_bounded_error(self):
        """Brief 04: Unknown ID or invalid action returns bounded error and consumes a step.
        Primary tests: tests/test_contract.py.
        """
        env = AuditEnv(training=True)
        obs, _ = env.load_episode(self.clean_ep)
        start_steps = obs["remaining_steps"]
        err1, rew1, term1, trunc1, _ = env.step({"type": "get_record", "table": "invoices", "id": "inv_nonexistent"})
        self.assertEqual(err1["error"], "unknown_id")
        self.assertEqual(err1["remaining_steps"], start_steps - 1)
        self.assertEqual(rew1, 0.0)
        self.assertFalse(term1 or trunc1)
        err2, _, _, _, _ = env.step({"type": "bad_action_name"})
        self.assertEqual(err2["error"], "invalid_action")
        self.assertEqual(err2["remaining_steps"], start_steps - 2)

    def test_brief_04_submit_terminates_and_limit_truncates_with_empty_answer(self):
        """Brief 04: submit sets terminated=True; 100-step limit or timeout sets truncated=True with empty answer.
        Primary tests: tests/test_contract.py.
        """
        env = AuditEnv(training=True)
        obs, _ = env.load_episode(self.clean_ep)
        _, _, term, trunc, _ = env.step({"type": "submit", "findings": {"episode_id": obs["manifest"]["episode_id"], "findings": []}})
        self.assertTrue(term)
        self.assertFalse(trunc)

        # 100-step limit truncates with empty answer (on clean books: 1 - 0.002*100 = 0.8)
        env2 = AuditEnv(training=True)
        env2.load_episode(self.clean_ep)
        for _ in range(MAX_STEPS):
            out_obs, rew, term, trunc, _ = env2.step({"type": "calculator", "expression": "1+1"})
        self.assertFalse(term)
        self.assertTrue(trunc)
        self.assertEqual(out_obs["status"], "truncated")
        self.assertAlmostEqual(rew, 0.8, places=6)

        # Timeout truncates with empty answer
        env3 = AuditEnv(training=True, timeout_seconds=0.005)
        env3.load_episode(self.dirty_ep)
        time.sleep(0.015)
        out_t, rew_t, term_t, trunc_t, _ = env3.step({"type": "calculator", "expression": "1+1"})
        self.assertFalse(term_t)
        self.assertTrue(trunc_t)
        self.assertEqual(rew_t, -1.0)

    def test_brief_04_zero_intermediate_reward_and_closed_mode_hides_reward(self):
        """Brief 04: Exploration reward is 0; closed mode hides terminal reward (returns None).
        Primary tests: tests/test_contract.py.
        """
        env = AuditEnv(training=False)
        obs, _ = env.load_episode(self.clean_ep)
        _, r_mid, _, _, _ = env.step({"type": "calculator", "expression": "1+1"})
        self.assertEqual(r_mid, 0.0)
        _, r_end, term, _, _ = env.step({"type": "submit", "findings": {"episode_id": obs["manifest"]["episode_id"], "findings": []}})
        self.assertTrue(term)
        self.assertIsNone(r_end)
        self.assertIsNotNone(env.outcome)

    # ── Section 05: Выводы и награда ──────────────────────────────────────────

    def test_brief_05_submission_limits_and_malformed_gives_minus_one(self):
        """Brief 05: Max 30 findings, 10 evidence IDs, 1000-char explanation; malformed -> -1 + error.
        Primary tests: tests/test_scoring.py, tests/test_fuzz.py.
        """
        ep = self.dirty_ep
        seen = _all_seen(ep)
        bad = score({"wrong_key": 1}, ep.manifest["episode_id"], ep.issues, seen, 1)
        self.assertEqual(bad.reward, -1.0)
        self.assertIsNotNone(bad.error)

        too_many = _oracle_payload(ep)
        too_many["findings"] = [too_many["findings"][0]] * (MAX_FINDINGS + 1)
        self.assertEqual(score(too_many, ep.manifest["episode_id"], ep.issues, seen, 1).reward, -1.0)

        long_exp = _oracle_payload(ep)
        long_exp["findings"][0]["explanation"] = "x" * (MAX_EXPLANATION + 1)
        self.assertEqual(score(long_exp, ep.manifest["episode_id"], ep.issues, seen, 1).reward, -1.0)

        too_much_ev = _oracle_payload(ep)
        too_much_ev["findings"][0]["evidence_ids"] = [f"id_{i}" for i in range(MAX_EVIDENCE + 1)]
        self.assertEqual(score(too_much_ev, ep.manifest["episode_id"], ep.issues, seen, 1).reward, -1.0)

    def test_brief_05_one_to_one_matching_tolerance_currency_and_evidence(self):
        """Brief 05: One-to-one matching requires type, affected_ids, tolerance, currency, seen & valid evidence.
        Primary tests: tests/test_scoring.py.
        """
        ep = self.dirty_ep
        seen = _all_seen(ep)
        # Unseen evidence fails
        unseen_res = score(_oracle_payload(ep), ep.manifest["episode_id"], ep.issues, set(), 5)
        self.assertEqual(unseen_res.tp, 0)
        # Tolerance edge inclusive on fx_rate (tolerance 2)
        fx_issue = next(i for i in ep.issues if i.type == "fx_rate")
        fx_payload = {
            "episode_id": ep.manifest["episode_id"],
            "findings": [{
                "type": fx_issue.type,
                "affected_ids": list(fx_issue.affected_ids),
                "claimed_amount_minor": fx_issue.expected_amount_minor + fx_issue.tolerance_minor,
                "currency": fx_issue.currency,
                "evidence_ids": list(fx_issue.evidence_sets[0]),
                "explanation": "edge",
            }],
        }
        self.assertEqual(score(fx_payload, ep.manifest["episode_id"], (fx_issue,), seen, 1).tp, 1)
        fx_payload["findings"][0]["claimed_amount_minor"] += 1
        self.assertEqual(score(fx_payload, ep.manifest["episode_id"], (fx_issue,), seen, 1).tp, 0)

    def test_brief_05_metrics_p_r_f1_and_empty_truth_rules(self):
        """Brief 05: P/R/F1 formulas; empty truth + empty findings -> F1=1; empty truth + findings -> F1=0.
        Primary tests: tests/test_scoring.py.
        """
        ep = self.clean_ep
        empty = score({"episode_id": ep.manifest["episode_id"], "findings": []}, ep.manifest["episode_id"], (), set(), 0)
        self.assertEqual((empty.precision, empty.recall, empty.f1, empty.reward), (0.0, 0.0, 1.0, 1.0))
        false_pos = score(
            {"episode_id": ep.manifest["episode_id"], "findings": [{
                "type": "duplicate_invoice", "affected_ids": ["inv_x"], "claimed_amount_minor": 100,
                "currency": "KZT", "evidence_ids": ["doc_x"], "explanation": "guess",
            }]},
            ep.manifest["episode_id"], (), {"doc_x"}, 0,
        )
        self.assertEqual((false_pos.f1, false_pos.reward), (0.0, -1.0))

    def test_brief_05_full_answer_beats_silence_and_silence_beats_false_accusations(self):
        """Brief 05: Full correct answer beats silence on dirty books; silence beats accusations on clean books.
        Primary tests: tests/test_scoring.py, tests/test_attacks.py.
        """
        ep = self.dirty_ep
        seen = _all_seen(ep)
        full = score(_oracle_payload(ep), ep.manifest["episode_id"], ep.issues, seen, 10)
        silence_dirty = score({"episode_id": ep.manifest["episode_id"], "findings": []}, ep.manifest["episode_id"], ep.issues, seen, 0)
        self.assertGreater(full.reward, silence_dirty.reward)

        clean_silence = score({"episode_id": self.clean_ep.manifest["episode_id"], "findings": []}, self.clean_ep.manifest["episode_id"], (), set(), 5)
        clean_accuse = score(
            {"episode_id": self.clean_ep.manifest["episode_id"], "findings": [{
                "type": "fx_rate", "affected_ids": ["inv_1"], "claimed_amount_minor": 500,
                "currency": "KZT", "evidence_ids": ["doc_1"], "explanation": "fp",
            }]},
            self.clean_ep.manifest["episode_id"], (), {"doc_1"}, 5,
        )
        self.assertGreater(clean_silence.reward, clean_accuse.reward)

    def test_brief_05_duplicates_give_no_gain(self):
        """Brief 05: Repeated findings are counted as FP and never increase reward.
        Primary tests: tests/test_scoring.py, tests/test_attacks.py.
        """
        ep = self.dirty_ep
        seen = _all_seen(ep)
        once = _oracle_payload(ep)
        twice = {"episode_id": once["episode_id"], "findings": once["findings"] + once["findings"]}
        r_once = score(once, ep.manifest["episode_id"], ep.issues, seen, 10)
        r_twice = score(twice, ep.manifest["episode_id"], ep.issues, seen, 10)
        self.assertEqual(r_twice.tp, r_once.tp)
        self.assertEqual(r_twice.fp, len(ep.issues))
        self.assertLess(r_twice.reward, r_once.reward)

    def test_brief_05_flag_all_is_bad(self):
        """Brief 05: Flagging every document/row produces a strongly negative reward.
        Primary tests: tests/test_scoring.py, tests/test_attacks.py.
        """
        for ep in (self.clean_ep, self.dirty_ep):
            env = AuditEnv(training=False)
            obs, _ = env.load_episode(ep)
            outcome, _, _ = run_policy(env, obs, create_policy("flag-all"))
            self.assertLessEqual(outcome.reward, -0.95)

    def test_brief_05_step_cost_only_slightly_separates_equal_answers(self):
        """Brief 05: Step cost (0.002/step) slightly separates equal answers without overwhelming accuracy.
        Primary tests: tests/test_scoring.py, tests/test_economy.py.
        """
        ep = self.dirty_ep
        seen = _all_seen(ep)
        payload = _oracle_payload(ep)
        r10 = score(payload, ep.manifest["episode_id"], ep.issues, seen, 10)
        r13 = score(payload, ep.manifest["episode_id"], ep.issues, seen, 13)
        self.assertAlmostEqual(r10.reward - r13.reward, 0.006, places=6)

    def test_brief_05_explanation_text_never_affects_reward(self):
        """Brief 05: Reward depends strictly on structured fields and evidence, not persuasive prose.
        Primary tests: tests/test_attacks.py.
        """
        ep = self.dirty_ep
        seen = _all_seen(ep)
        p_plain = _oracle_payload(ep)
        p_persuasive = _oracle_payload(ep)
        for item in p_plain["findings"]:
            item["explanation"] = ""
        for item in p_persuasive["findings"]:
            item["explanation"] = "Critical material misstatement confirmed by senior auditor review!"
        self.assertEqual(
            score(p_plain, ep.manifest["episode_id"], ep.issues, seen, 10).reward,
            score(p_persuasive, ep.manifest["episode_id"], ep.issues, seen, 10).reward,
        )


# ==============================================================================
# Карта соответствия: пункт ТЗ (genesis.pdf) -> имя теста
# ------------------------------------------------------------------------------
# 03 // Размер эпизода (200-2000, 20-200, 0-12) и manifest.json -> test_brief_03_episode_size_and_manifest_fields
# 03 // Время ISO-8601 со смещением                              -> test_brief_03_timestamps_are_iso8601_with_offset
# 03 // Деньги — целые минимальные единицы и minor_units         -> test_brief_03_money_is_integer_minor_units
# 03 // FX-курсы — десятичные строки; валюты не смешиваются      -> test_brief_03_fx_rates_are_decimal_strings_and_currencies_not_mixed
# 03 // Открытые файлы и policy.json                             -> test_brief_03_open_records_have_required_columns_and_policy
# 03 // Отсутствующие ссылки — явные null                        -> test_brief_03_missing_links_are_explicit_null
# 03 // Закрытые truth.json/provenance.json и отсутствие утечек  -> test_brief_03_closed_truth_and_provenance_never_leak
# 04 // Сброс reset(seed, config) и шаг step(action) (JSON)      -> test_brief_04_reset_and_step_gymnasium_json_contract
# 04 // Воспроизводимость и изоляция reset                       -> test_brief_04_reproducibility_and_reset_isolation
# 04 // 5 действий, макс. 50 строк/страница, стабильные курсоры  -> test_brief_04_five_actions_pagination_max_50_and_stable_cursors
# 04 // Калькулятор — только десятичная арифметика, не код       -> test_brief_04_calculator_decimals_only_no_code
# 04 // Неизвестный ID / действие -> ограниченная ошибка и шаг   -> test_brief_04_unknown_id_costs_a_step_and_gives_bounded_error
# 04 // submit -> terminated; лимит 100 шагов/таймаут -> trunc   -> test_brief_04_submit_terminates_and_limit_truncates_with_empty_answer
# 04 // Нулевая промежуточная награда; закрытый режим скрывает   -> test_brief_04_zero_intermediate_reward_and_closed_mode_hides_reward
# 05 // Лимиты (30/10/1000) и неверная структура -> -1 + ошибка  -> test_brief_05_submission_limits_and_malformed_gives_minus_one
# 05 // Сопоставление 1-к-1, допуски, валюта, seen-доказательства-> test_brief_05_one_to_one_matching_tolerance_currency_and_evidence
# 05 // Метрики P/R/F1 и правила пустого эталона                 -> test_brief_05_metrics_p_r_f1_and_empty_truth_rules
# 05 // Верный ответ > молчания; молчание > ложных обвинений     -> test_brief_05_full_answer_beats_silence_and_silence_beats_false_accusations
# 05 // Дубли не дают выгоды                                     -> test_brief_05_duplicates_give_no_gain
# 05 // Пометка всех строк/типов (flag-all) даёт плохой результат-> test_brief_05_flag_all_is_bad
# 05 // Экономия шагов (0.002) лишь немного различает равные     -> test_brief_05_step_cost_only_slightly_separates_equal_answers
# 05 // Награда не зависит от убедительного текста explanation   -> test_brief_05_explanation_text_never_affects_reward
# ==============================================================================

if __name__ == "__main__":
    unittest.main()
