"""The optional UI API: sessions, steps, agents, folder import and audit; truth never leaves."""
import json
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

from audit_env.api import serve
from audit_env.contracts import Config
from audit_env.episode_io import write_episode
from audit_env.generator import generate


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = serve(port=0)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def call(self, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base + path, data=data, method="POST" if body is not None else "GET",
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read().decode("utf-8"))

    def test_session_flow_without_truth(self):
        status, created = self.call("/api/episodes", {"seed": 5, "config": {"keyspace": "open"}, "mode": "open"})
        self.assertEqual(status, 200)
        session = created["session_id"]
        status, stepped = self.call(f"/api/sessions/{session}/step",
                                    {"action": {"type": "list_records", "table": "invoices", "limit": 2}})
        self.assertEqual((status, len(stepped["observation"]["result"]["records"])), (200, 2))
        status, ran = self.call(f"/api/sessions/{session}/run", {"policy": "expert"})
        self.assertEqual(status, 200)
        self.assertEqual(ran["score"]["f1"], 1.0)
        status, files = self.call(f"/api/sessions/{session}/files")
        text = json.dumps([created, stepped, ran, files], ensure_ascii=False)
        for word in ("evidence_sets", "expected_amount", "provenance", "issue_id", "\"seed\""):
            self.assertNotIn(word, text)

    def test_closed_mode_hides_score(self):
        _, created = self.call("/api/episodes", {"seed": 6, "mode": "closed"})
        regenerated = generate(6, Config(keyspace="open"))
        self.assertNotEqual(created["observation"]["manifest"]["episode_id"], regenerated.manifest["episode_id"])
        status, _ = self.call("/api/episodes", {"seed": 6, "config": {"keyspace": "open"}, "mode": "closed"})
        self.assertEqual(status, 400)
        _, ran = self.call(f"/api/sessions/{created['session_id']}/run", {"policy": "expert"})
        self.assertNotIn("score", ran)
        self.assertIsNone(ran["trajectory"][-1]["reward"])

    def test_import_folder_and_audit(self):
        episode = generate(7, Config(keyspace="open", errors={"duplicate_invoice": 1, "payment_allocation": 0,
                                                              "fx_rate": 0, "period_cutoff": 0}))
        with tempfile.TemporaryDirectory() as temp:
            write_episode(episode, Path(temp) / "docs")
            status, imported = self.call("/api/episodes/import", {"folder": str(Path(temp) / "docs")})
            self.assertEqual((status, imported["graded"]), (200, False))
            _, files = self.call(f"/api/sessions/{imported['session_id']}/files")
            self.assertEqual(len(files["documents"]), files["manifest"]["document_count"])
            status, audited = self.call("/api/audit", {"folder": str(Path(temp) / "docs")})
            self.assertEqual([item["type"] for item in audited["findings"]], ["duplicate_invoice"])
            (Path(temp) / "docs" / "fx.csv").write_text("broken\n", encoding="utf-8")
            status, failed = self.call("/api/audit", {"folder": str(Path(temp) / "docs")})
            self.assertEqual((status, failed["error"]), (400, "format"))

    def test_errors(self):
        self.assertEqual(self.call("/api/sessions/nope/step", {"action": None})[0], 404)
        self.assertEqual(self.call("/api/episodes", {"seed": -3})[0], 400)
        self.assertEqual(self.call("/api/health")[1]["ok"], True)


if __name__ == "__main__":
    unittest.main()
