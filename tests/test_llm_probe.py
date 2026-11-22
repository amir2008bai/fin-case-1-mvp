"""llm-probe drives a real tool loop; offline it exits cleanly and never prints the API key."""
import io
import json
import os
import tempfile
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


class FlakyApi(BaseHTTPRequestHandler):
    """Answers like OpenRouter under load: keep-alive blank lines and no JSON, then a 429, then a reply."""

    calls = 0

    def log_message(self, *_):
        return

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        type(self).calls += 1
        if type(self).calls == 1:
            body, status = b"\n\n\n", 200
        elif type(self).calls == 2:
            body, status = b'{"error": {"code": 429, "message": "slow down"}}', 429
        else:
            body, status = b'{"choices": []}', 200
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class DroppingApi(BaseHTTPRequestHandler):
    """Closes the first connection without an answer, as an overloaded API does, then replies."""

    calls = 0

    def log_message(self, *_):
        return

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        type(self).calls += 1
        if type(self).calls == 1:
            self.close_connection = True
            self.connection.close()
            return
        body = b'{"choices": []}'
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class TransientFailureTests(unittest.TestCase):
    def test_a_dropped_connection_is_retried(self):
        from audit_env.llm_probe import _post_json

        server = ThreadingHTTPServer(("127.0.0.1", 0), DroppingApi)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions"
            result = _post_json(url, {}, {"Content-Type": "application/json"}, timeout=5, sleep=lambda _: None)
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(result, {"choices": []})
        self.assertEqual(DroppingApi.calls, 2)

    def test_blank_and_429_answers_are_retried(self):
        from audit_env.llm_probe import _post_json

        server = ThreadingHTTPServer(("127.0.0.1", 0), FlakyApi)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            url = f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions"
            waits = []
            result = _post_json(url, {}, {"Content-Type": "application/json"}, timeout=5, sleep=waits.append)
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(result, {"choices": []})
        self.assertEqual(FlakyApi.calls, 3)
        self.assertEqual(len(waits), 2)


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
            with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"OPENAI_API_KEY": "dummy"}):
                report = Path(tmp) / "probe.json"
                code, text = self.run_probe(provider="openai", base_url=url, report=report)
                saved = json.loads(report.read_text(encoding="utf-8"))
        finally:
            server.shutdown()
            server.server_close()
        self.assertEqual(code, 0)
        self.assertEqual(FakeOpenAI.seen_auth, "Bearer dummy")
        self.assertEqual(len(saved["episodes"]), 2)  # the report is rewritten after every episode
        self.assertIn("MEAN", text)
        self.assertIn("submitted", text)
        self.assertNotIn("dummy", text)


if __name__ == "__main__":
    unittest.main()
