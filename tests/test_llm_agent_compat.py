"""Comprehensive AI-agent compatibility tests simulating weak 3B-8B LLMs, all 5 providers, MCP edge cases, and compact mode."""
import io
import json
import unittest

from audit_env.contracts import Config
from audit_env.environment import AuditEnv
from audit_env.mcp_server import MCPAuditServer, serve_stdio
from audit_env.tool_schemas import (
    compact_initial_observation,
    compress_observation,
    dispatch_tool_call,
    tool_schemas,
)

ERRORS = {"duplicate_invoice": 1, "payment_allocation": 1, "fx_rate": 1, "period_cutoff": 1}
MIXED = Config(keyspace="open", errors=ERRORS)


class LLMAgentCompatibilityTests(unittest.TestCase):
    def test_gemini_and_openai_strict_schema_compliance(self):
        # Gemini OpenAPI 3.0: no list types, no additionalProperties, uses nullable=True
        gem = tool_schemas("gemini")
        gem_text = json.dumps(gem)
        self.assertNotIn("additionalProperties", gem_text)
        list_rec = next(t for t in gem if t["name"] == "list_records")
        self.assertEqual(list_rec["parameters"]["properties"]["cursor"]["type"], "string")
        self.assertTrue(list_rec["parameters"]["properties"]["cursor"]["nullable"])

        # OpenAI strict mode: every property is in required + strict=True
        oa_strict = tool_schemas("openai", strict=True)
        for tool in oa_strict:
            self.assertTrue(tool["function"]["strict"])
            params = tool["function"]["parameters"]
            self.assertEqual(set(params["required"]), set(params["properties"].keys()))

        # Immutability: mutating returned schema must not corrupt subsequent calls
        oa_strict[0]["function"]["parameters"]["properties"]["table"]["enum"].append("MUTATED")
        fresh = tool_schemas("openai", strict=True)
        self.assertNotIn("MUTATED", fresh[0]["function"]["parameters"]["properties"]["table"]["enum"])

    def test_type_key_collision_cannot_hijack_tool_call(self):
        """Passing {"type": "submit"} inside list_records arguments must NEVER execute submit."""
        env = AuditEnv(training=True)
        obs, _ = env.reset(17, MIXED)
        res = dispatch_tool_call(env, "list_records", {
            "type": "submit",
            "findings": {"episode_id": obs["manifest"]["episode_id"], "findings": []},
        })
        self.assertTrue(res["is_error"])
        self.assertFalse(res["terminated"])
        self.assertEqual(res["observation"]["error"], "invalid_action")

    def test_weak_llm_self_correction_loop_with_hints_and_null_limit(self):
        """Simulate a weak 3B-8B model making typical mistakes, reading hints, and recovering."""
        env = AuditEnv(training=True)
        obs, _ = env.reset(17, MIXED)
        # 1. Model tries table="fx.csv" -> gets hint
        r1 = dispatch_tool_call(env, "list_records", {"table": "fx.csv", "limit": None})
        self.assertTrue(r1["is_error"])
        self.assertIn("fx", r1["observation"]["hint"])
        # 2. Model recovers with table="fx" and limit=None (OpenAI strict nullable) -> succeeds!
        r2 = dispatch_tool_call(env, "list_records", {"table": "fx", "cursor": None, "limit": None})
        self.assertFalse(r2["is_error"])
        self.assertGreater(len(r2["observation"]["result"]["records"]), 0)
        # 3. Model tries calculator with Python ** operator -> gets hint -> recovers
        r3 = dispatch_tool_call(env, "calculator", {"expression": "2**8"})
        self.assertTrue(r3["is_error"])
        self.assertIn("hint", r3["observation"])
        r4 = dispatch_tool_call(env, "calculator", {"expression": "256 / 2"})
        self.assertFalse(r4["is_error"])
        self.assertEqual(r4["observation"]["result"], "128")

    def test_compact_mode_cuts_tokens_significantly_and_is_lossless(self):
        env_full = AuditEnv(training=True)
        obs_full, _ = env_full.reset(17, MIXED)
        env_cmp = AuditEnv(training=True)
        obs_cmp, _ = env_cmp.reset(17, MIXED)

        s_full = MCPAuditServer(env_full, obs_full, compact=False)
        s_cmp = MCPAuditServer(env_cmp, obs_cmp, compact=True)

        init_full = json.dumps(s_full.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"}), ensure_ascii=False)
        init_cmp = json.dumps(s_cmp.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"}), ensure_ascii=False)
        self.assertLess(len(init_cmp), len(init_full))
        # The rules are never paraphrased: an agent in compact mode reads the same policy the grader applies.
        compact_view = compact_initial_observation(obs_cmp)
        self.assertEqual(compact_view["policy"], obs_full["policy"])
        self.assertEqual(compact_view["manifest"], obs_full["manifest"])
        self.assertNotIn("tools", compact_view)

        page_full = s_full.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                                   "params": {"name": "list_records", "arguments": {"table": "ledger", "limit": 50}}})
        page_cmp = s_cmp.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                                 "params": {"name": "list_records", "arguments": {"table": "ledger", "limit": 50}}})
        self.assertLess(len(json.dumps(page_cmp, ensure_ascii=False)),
                        len(json.dumps(page_full, ensure_ascii=False)) * 0.70)
        # Verify lossless reconstruction
        rec_full = json.loads(page_full["result"]["content"][0]["text"])["observation"]["result"]["records"]
        res_cmp = json.loads(page_cmp["result"]["content"][0]["text"])["observation"]["result"]
        reconstructed = [dict(zip(res_cmp["columns"], row)) for row in res_cmp["rows"]]
        self.assertEqual(reconstructed, rec_full)

    def test_mcp_jsonrpc_strict_error_codes_and_zero_step_burn_on_invalid_params(self):
        env = AuditEnv(training=True)
        obs, _ = env.reset(17, MIXED)
        server = MCPAuditServer(env, obs)

        # id: null is invalid request (-32600), NOT a silent notification
        r_null = server.handle({"jsonrpc": "2.0", "id": None, "method": "ping"})
        self.assertEqual(r_null["error"]["code"], -32600)
        # Non-string method -> -32600
        self.assertEqual(server.handle({"jsonrpc": "2.0", "id": 1, "method": 123})["error"]["code"], -32600)
        # Non-dict params -> -32602
        self.assertEqual(server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": []})["error"]["code"], -32602)
        # Unknown tool -> -32602 without burning an AuditEnv step
        self.assertEqual(server.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                        "params": {"name": "shell_exec", "arguments": {}}})["error"]["code"], -32602)
        self.assertEqual(env.steps, 0)

    def test_mcp_stdio_handles_malformed_and_oversized_frames_cleanly(self):
        stdin = io.StringIO(
            '{"jsonrpc":"2.0","id":1,"method":"ping"}\n'
            'not-valid-json\n'
            + ("x" * (300 * 1024)) + "\n"
            + '{"jsonrpc":"2.0","id":2,"method":"ping"}\n'
        )
        stdout = io.StringIO()
        serve_stdio(seed=17, config=MIXED, closed=False, compact=True, stdin=stdin, stdout=stdout)
        lines = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0]["id"], 1)
        self.assertEqual(lines[1]["error"]["code"], -32700)
        self.assertEqual(lines[2]["error"]["code"], -32600)
        self.assertEqual(lines[3]["id"], 2)

    def test_mcp_closed_mode_rejects_open_keyspace(self):
        with self.assertRaises(ValueError):
            serve_stdio(seed=17, config={"keyspace": "open"}, closed=True, stdin=io.StringIO(""), stdout=io.StringIO())


if __name__ == "__main__":
    unittest.main()
