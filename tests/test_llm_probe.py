"""llm-probe drives a real tool loop; offline it exits cleanly and never prints the API key."""
import io
import json
import os
import threading
import unittest
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from audit_env.llm_probe import run_probe

HOLDOUT = Path(__file__).resolve().parents[1] / "configs" / "holdout-template.v2.json"


class FakeOpenAI(BaseHTTPRequestHandler):
    """Lists one ledger page, then submits an empty answer."""

    def log_message(self, *_):
        return

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen_auth = self.headers.get("Authorization")
        tool_turns = sum(1 for message in body["messages"] if message.get("role") == "tool")
        episode_id = json.loads(body["messages"][1]["content"].split("\n", 1)[1])["manifest"]["episode_id"]
        if tool_turns == 0:
            call = ("list_records", {"table": "ledger", "limit": 5})
        else:
            call = ("submit", {"findings": {"episode_id": episode_id, "findings": []}})
        reply = {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
            {"id": f"c{tool_turns}", "type": "function",
             "function": {"name": call[0], "arguments": json.dumps(call[1])}}]}}]}
        data = json.dumps(reply).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class LlmProbeTests(unittest.TestCase):
    def run_probe(self, **kwargs):
        stream = io.StringIO()
        with redirect_stdout(stream):
            code = run_probe(HOLDOUT, episodes=2, emit=print, **kwargs)
        return code, stream.getvalue()

    def test_without_a_key_it_skips(self):
        clean = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
        with mock.patch.dict(os.environ, clean, clear=True):
            code, text = self.run_probe()
        self.assertEqual(code, 0)
        self.assertIn("skipped", text)

    def test_unreachable_endpoint_never_prints_the_key_or_path(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "key-must-not-leak"}):
            code, text = self.run_probe(provider="gemini", base_url="http://127.0.0.1:9/secret-path")
        self.assertEqual(code, 0)
        self.assertNotIn("key-must-not-leak", text)
        self.assertNotIn("secret-path", text)

    def test_tool_loop_against_a_local_fake_api(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAI)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions"
            with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "dummy"}):
                code, text = self.run_probe(provider="openai", base_url=url)
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(code, 0)
        self.assertEqual(FakeOpenAI.seen_auth, "Bearer dummy")
        self.assertIn("MEAN", text)
        self.assertIn("submitted", text)
        self.assertNotIn("dummy", text)


if __name__ == "__main__":
    unittest.main()
