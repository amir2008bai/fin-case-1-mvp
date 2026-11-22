"""Adapters around the environment: the agent process, the UI API and the MCP server.

They must not weaken isolation (key material, browser-borne requests) and must stay usable
by a person or an LLM, whose audit takes far longer than the 60 s of a scored run.
"""
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from audit_env import keys
from audit_env.api import SESSION_TIMEOUT_SECONDS, App, serve
from audit_env.contracts import Config
from audit_env.environment import EPISODE_TIMEOUT_SECONDS, AuditEnv
from audit_env.mcp_server import MCPAuditServer
from audit_env.runner import agent_environment, run_subprocess

CLEAN = Config(keyspace="open", errors={"duplicate_invoice": 0, "payment_allocation": 0, "fx_rate": 0,
                                        "period_cutoff": 0})
LIST_LEDGER = {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
               "params": {"name": "list_records", "arguments": {"table": "ledger", "limit": 1}}}


class ClosedKeyspaceTests(unittest.TestCase):
    def test_closed_reset_never_uses_the_published_key(self):
        """A closed episode on the open key could be regenerated, truth included, by guessing the seed."""
        from audit_env.generator import generate

        public = generate(42, Config(keyspace="open")).manifest["episode_id"]
        with mock.patch.dict(os.environ, {keys.ENV_VAR: "x" * 32}):
            for config in (None, {}, {"keyspace": "open"}, Config(keyspace="open")):
                observation, _ = AuditEnv(training=False).reset(42, config)
                self.assertNotEqual(observation["manifest"]["episode_id"], public)

    def test_config_defaults_to_the_secret_keyspace(self):
        self.assertEqual(Config().keyspace, "secret")


class NoScoringOracleTests(unittest.TestCase):
    """Open mode returns the score; on the secret keyspace it would score the episode a closed session gets
    for the same seed and config, so it could be used to try answers before submitting the closed one."""

    def test_api_open_mode_never_uses_the_secret_keyspace(self):
        app = App()
        for config in ({"difficulty": "medium"}, {"keyspace": "open"}):
            open_episode = app.create({"mode": "open", "seed": 7, "config": config})
            closed_episode = app.create({"mode": "closed", "seed": 7, "config": {"difficulty": "medium"}})
            self.assertNotEqual(open_episode["observation"]["manifest"]["episode_id"],
                                closed_episode["observation"]["manifest"]["episode_id"])
        with self.assertRaises(ValueError):
            app.create({"mode": "open", "seed": 7, "config": {"keyspace": "secret"}})

    def test_mcp_without_closed_refuses_the_secret_keyspace(self):
        import io

        from audit_env.mcp_server import serve_stdio

        with self.assertRaises(ValueError):
            serve_stdio(7, {"keyspace": "secret"}, stdin=io.StringIO(""), stdout=io.StringIO())


class AgentProcessTests(unittest.TestCase):
    def test_agent_process_does_not_inherit_the_evaluator_key(self):
        """With the secret key an agent could regenerate the hidden episode and its truth."""
        secrets = {keys.ENV_VAR: "evaluator-secret-key-0123456789", keys.KEY_FILE_VAR: "/evaluator/secret.key"}
        with tempfile.TemporaryDirectory() as temp, mock.patch.dict(os.environ, secrets | {"AUDIT_PROBE": "kept"}):
            self.assertEqual(set(secrets) & set(agent_environment()), set())
            target = Path(temp) / "seen.json"
            agent = ("import json, os, sys; sys.stdin.readline(); "
                     "json.dump({k: os.environ.get(k) for k in sys.argv[2:]}, open(sys.argv[1], 'w')); "
                     "print(json.dumps({'type': 'submit', 'findings': {'episode_id': '', 'findings': []}}), flush=True)")
            env = AuditEnv(training=False, key=b"evaluator-secret-key-0123456789")
            observation, _ = env.reset(3, CLEAN)
            run_subprocess(env, observation, [sys.executable, "-c", agent, str(target), *secrets, "AUDIT_PROBE"],
                           trusted=True)
            seen = json.loads(target.read_text(encoding="utf-8"))
        self.assertEqual(seen, {keys.ENV_VAR: None, keys.KEY_FILE_VAR: None, "AUDIT_PROBE": "kept"})

    def test_closed_episode_refuses_an_untrusted_subprocess_agent(self):
        """A same-user process can read the evaluator's memory and original environment through /proc (the
        red-team review read the truth that way), so a closed episode runs an outside agent only in a container."""
        from audit_env.runner import UntrustedSubprocess

        env = AuditEnv(training=False, key=b"evaluator-secret-key-0123456789")
        observation, _ = env.reset(3, CLEAN)
        with self.assertRaises(UntrustedSubprocess):
            run_subprocess(env, observation, [sys.executable, "-c", "print('{}')"])
        open_env = AuditEnv(training=True)
        observation, _ = open_env.reset(3, CLEAN)
        agent = "import json; input(); print(json.dumps({'type': 'submit', 'findings': {'episode_id': '', 'findings': []}}))"
        outcome, _, _ = run_subprocess(open_env, observation, [sys.executable, "-c", agent])  # open mode: allowed
        self.assertIsNotNone(outcome)

    @unittest.skipUnless(os.name == "posix", "process groups are POSIX")
    def test_agent_grandchildren_are_stopped_with_the_agent(self):
        import time

        with tempfile.TemporaryDirectory() as temp:
            marker = Path(temp) / "alive"
            child = f"import time, pathlib; time.sleep(1.0); pathlib.Path({str(marker)!r}).write_text('x')"
            agent = ("import json, subprocess, sys; subprocess.Popen([sys.executable, '-c', " + repr(child) + "]); "
                     "input(); print(json.dumps({'type': 'submit', 'findings': {'episode_id': '', 'findings': []}}), flush=True)")
            env = AuditEnv(training=True)
            observation, _ = env.reset(3, CLEAN)
            run_subprocess(env, observation, [sys.executable, "-c", agent])
            time.sleep(1.5)
            self.assertFalse(marker.exists(), "a grandchild of the agent outlived the episode")


class InteractiveClockTests(unittest.TestCase):
    def test_scored_runs_keep_the_official_limit(self):
        with self.assertRaises(ValueError):
            AuditEnv(timeout_seconds=EPISODE_TIMEOUT_SECONDS + 1)
        self.assertEqual(AuditEnv().timeout_seconds, EPISODE_TIMEOUT_SECONDS)

    def test_ui_session_outlives_the_evaluation_timeout(self):
        app = App()
        created = app.create({"seed": 5, "config": CLEAN.as_dict()})
        session = app.sessions[created["session_id"]]
        self.assertEqual(session.env.timeout_seconds, SESSION_TIMEOUT_SECONDS)
        now = [session.env.clock()]
        session.env.clock = lambda: now[0]
        now[0] += 10 * EPISODE_TIMEOUT_SECONDS  # a person reading the documents
        stepped = app.step(created["session_id"], {"action": {"type": "list_records", "table": "ledger", "limit": 1}})
        self.assertEqual((stepped["truncated"], len(stepped["observation"]["result"]["records"])), (False, 1))
        now[0] += SESSION_TIMEOUT_SECONDS
        self.assertTrue(app.step(created["session_id"], {"action": {"type": "list_records", "table": "ledger"}})["truncated"])

    def test_mcp_clock_starts_when_the_client_connects(self):
        now = [0.0]
        env = AuditEnv(timeout_seconds=1800, interactive=True, clock=lambda: now[0])
        observation, _ = env.reset(5, CLEAN)
        server = MCPAuditServer(env, observation)
        now[0] += 5000  # the MCP client started the process long before the model's first call
        server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        now[0] += 600  # an LLM takes seconds per call, minutes per audit
        first = json.loads(server.handle(LIST_LEDGER)["result"]["content"][0]["text"])
        self.assertFalse(first["truncated"])
        self.assertEqual(len(first["observation"]["result"]["records"]), 1)
        # A second initialize cannot buy more time once the agent has acted.
        now[0] += 1500
        server.handle({"jsonrpc": "2.0", "id": 3, "method": "initialize"})
        self.assertTrue(json.loads(server.handle(LIST_LEDGER)["result"]["content"][0]["text"])["truncated"])


class BrowserBorneRequestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = serve(port=0)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def status(self, path, *, body=None, headers=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base + path, data=data, headers=headers or {},
                                         method="GET" if body is None else "POST")
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status
        except urllib.error.HTTPError as error:
            return error.code

    def sessions(self):
        with urllib.request.urlopen(self.base + "/api/health", timeout=30) as response:
            return json.loads(response.read())["sessions"]

    def test_dns_rebinding_host_is_refused(self):
        self.assertEqual(self.status("/api/health"), 200)
        self.assertEqual(self.status("/api/health", headers={"Host": "localhost:8765"}), 200)
        for host in ("evil.example", "evil.example:8765", "127.0.0.1.evil.example", "localhost:abc"):
            self.assertEqual(self.status("/api/health", headers={"Host": host}), 403, host)

    def test_cross_site_post_cannot_change_state(self):
        before = self.sessions()
        body, json_type = {"seed": 5}, {"Content-Type": "application/json"}
        # A form or text/plain POST is sent by browsers without a CORS preflight.
        self.assertEqual(self.status("/api/episodes", body=body, headers={"Content-Type": "text/plain"}), 415)
        self.assertEqual(self.status("/api/episodes", body=body,
                                     headers=json_type | {"Origin": "https://evil.example"}), 403)
        self.assertEqual(self.sessions(), before)
        self.assertEqual(self.status("/api/episodes", body=body,
                                     headers=json_type | {"Origin": "http://localhost:5173"}), 200)
        self.assertEqual(self.sessions(), before + 1)


if __name__ == "__main__":
    unittest.main()
