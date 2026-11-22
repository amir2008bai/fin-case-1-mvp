"""The bundled UI: served from a fixed file list with a strict policy; scores only in open mode."""
import json
import threading
import unittest
import urllib.error
import urllib.request

from audit_env.api import UI_FILES, serve


class UiRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = serve(port=0)
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def get(self, path):
        try:
            with urllib.request.urlopen(self.base + path, timeout=10) as response:
                return response.status, dict(response.headers), response.read()
        except urllib.error.HTTPError as error:
            return error.code, dict(error.headers), error.read()

    def post(self, path, body):
        request = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), method="POST",
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    def test_every_ui_file_is_served_with_a_strict_policy(self):
        for route, (_, content_type) in UI_FILES.items():
            status, headers, body = self.get("/" + route)
            self.assertEqual(status, 200, route)
            self.assertEqual(headers["Content-Type"], content_type)
            self.assertIn("script-src 'self'", headers["Content-Security-Policy"])
            self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
            self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
            self.assertTrue(body)

    def test_nothing_else_is_served(self):
        for path in ("/ui/", "/ui/index.html", "/ui/../api.py", "/ui/%2e%2e/api.py", "/audit_env/api.py",
                     "/ui/app.js.map", "/index.html"):
            status, _, _ = self.get(path)
            self.assertEqual(status, 404, path)

    def test_ui_never_loads_anything_from_the_network(self):
        for route in UI_FILES:
            _, _, body = self.get("/" + route)
            text = body.decode("utf-8")
            self.assertNotIn("http://", text.replace("http://127.0.0.1", "").replace("http://localhost", ""))
            self.assertNotIn("https://", text)
            self.assertNotIn("innerHTML", text)  # every server value goes through textContent

    def test_manual_submit_is_scored_only_in_open_mode(self):
        for mode, scored in (("open", True), ("closed", False)):
            config = {"keyspace": "open" if mode == "open" else "secret"}
            created = self.post("/api/episodes", {"seed": 9, "config": config, "mode": mode})
            episode_id = created["observation"]["manifest"]["episode_id"]
            result = self.post(f"/api/sessions/{created['session_id']}/step",
                               {"action": {"type": "submit", "findings": {"episode_id": episode_id, "findings": []}}})
            self.assertTrue(result["terminated"])
            self.assertEqual("score" in result, scored, mode)
            if not scored:
                self.assertIsNone(result["reward"])
                self.assertNotIn("truth", json.dumps(result))

    def test_unfinished_step_carries_no_score(self):
        created = self.post("/api/episodes", {"seed": 9, "config": {"keyspace": "open"}, "mode": "open"})
        result = self.post(f"/api/sessions/{created['session_id']}/step",
                           {"action": {"type": "list_records", "table": "ledger"}})
        self.assertNotIn("score", result)


if __name__ == "__main__":
    unittest.main()
