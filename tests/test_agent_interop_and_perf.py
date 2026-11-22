"""Tests for universal AI-agent interoperability (MCP, OpenAI, Anthropic, Gemini), SyncVectorAuditEnv, and fast-copy isolation."""
import json
import unittest

from audit_env.contracts import Config
from audit_env.environment import AuditEnv
from audit_env.gym_adapter import GymAuditEnv, SyncVectorAuditEnv
from audit_env.mcp_server import MCPAuditServer
from audit_env.tool_schemas import dispatch_tool_call, tool_schemas

ERRORS = {"duplicate_invoice": 1, "payment_allocation": 1, "fx_rate": 1, "period_cutoff": 1}
MIXED = Config(keyspace="open", errors=ERRORS)


class AgentInteropAndPerfTests(unittest.TestCase):
    def test_all_provider_tool_schemas_cover_five_tools(self):
        expected = {"list_records", "get_record", "get_document", "calculator", "submit"}
        for provider in ("openai", "openai_responses", "anthropic", "gemini", "mcp"):
            schemas = tool_schemas(provider)
            self.assertEqual(len(schemas), 5, provider)
            names = {item["function"]["name"] if provider == "openai" else item["name"] for item in schemas}
            self.assertEqual(names, expected, provider)

    def test_mcp_server_full_lifecycle_and_no_truth_leak(self):
        env = AuditEnv(training=False)
        init_obs, _ = env.reset(17, MIXED)
        server = MCPAuditServer(env, init_obs)

        init_resp = server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
        self.assertEqual(init_resp["result"]["serverInfo"]["name"], "fintech-audit-env")
        self.assertNotIn("truth", json.dumps(init_resp))
        self.assertIsNone(server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))

        list_resp = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        self.assertEqual(len(list_resp["result"]["tools"]), 5)

        calc_resp = server.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                   "params": {"name": "calculator", "arguments": {"expression": "10.5 + 2.25"}}})
        self.assertFalse(calc_resp["result"]["isError"])
        body = json.loads(calc_resp["result"]["content"][0]["text"])
        self.assertEqual(body["observation"]["result"], "12.75")

        bad_resp = server.handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                                  "params": {"name": "get_record", "arguments": {"table": "ledger", "id": "nope"}}})
        self.assertTrue(bad_resp["result"]["isError"])

        sub_resp = server.handle({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                                  "params": {"name": "submit", "arguments": {
                                      "findings": {"episode_id": init_obs["manifest"]["episode_id"], "findings": []}
                                  }}})
        sub_body = json.loads(sub_resp["result"]["content"][0]["text"])
        self.assertTrue(sub_body["terminated"])
        self.assertIsNone(sub_body["reward"])  # closed mode hides reward

    def test_dispatch_tool_call_accepts_json_string_arguments(self):
        env = AuditEnv(training=True)
        env.reset(17, MIXED)
        res = dispatch_tool_call(env, "calculator", '{"expression": "6 * 7"}')
        self.assertFalse(res["is_error"])
        self.assertEqual(res["observation"]["result"], "42")

    def test_sync_vector_audit_env_batches_multiple_episodes(self):
        vec = SyncVectorAuditEnv(num_envs=3, training=True)
        obs_list, infos = vec.reset(seeds=[11, 12, 13], options={"config": MIXED})
        self.assertEqual(len(obs_list), 3)
        self.assertEqual(len({o["manifest"]["episode_id"] for o in obs_list}), 3)
        actions = [{"type": "calculator", "expression": f"{i}+1"} for i in range(3)]
        step_obs, rewards, terms, truncs, _ = vec.step(actions)
        self.assertEqual([o["result"] for o in step_obs], ["1", "2", "3"])
        self.assertEqual(rewards, [0.0, 0.0, 0.0])
        self.assertEqual(terms, [False, False, False])

    def test_fast_observation_clone_preserves_mutation_isolation(self):
        env = AuditEnv(training=True)
        env.reset(17, MIXED)
        obs1, *_ = env.step({"type": "list_records", "table": "invoices", "limit": 5})
        orig_id = obs1["result"]["records"][0]["invoice_id"]
        orig_line_total = obs1["result"]["records"][0]["line_items"][0]["total_minor"]
        # Mutate returned observation aggressively
        obs1["result"]["records"][0]["invoice_id"] = "HACKED"
        obs1["result"]["records"][0]["line_items"][0]["total_minor"] = -999
        obs2, *_ = env.step({"type": "list_records", "table": "invoices", "limit": 5})
        self.assertEqual(obs2["result"]["records"][0]["invoice_id"], orig_id)
        self.assertEqual(obs2["result"]["records"][0]["line_items"][0]["total_minor"], orig_line_total)


if __name__ == "__main__":
    unittest.main()
