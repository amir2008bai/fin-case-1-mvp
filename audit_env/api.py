"""Optional local HTTP/JSON adapter for a UI. Standard library only, binds to localhost.

The UI talks to the same environment the agents use. Hidden truth and provenance
never leave the process: responses carry public files, observations, findings and,
only in open (training) mode, the terminal score.
"""
import json
import secrets
import threading
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import __version__
from .agents import create_policy
from .checks import audit
from .contracts import Config
from .environment import AuditEnv
from .episode_io import FormatError, load_episode
from .generator import generate
from .runner import run_policy

MAX_BODY = 4 * 1024 * 1024
POLICIES = ("expert", "micro", "reconcile", "random", "flag-all")


class Session:
    def __init__(self, episode, *, graded, training):
        self.episode = episode
        self.graded = graded
        self.training = training
        self.env = AuditEnv(training=training)
        self.observation, _ = self.env.load_episode(episode)
        self.lock = threading.Lock()


class App:
    def __init__(self, model_path=None):
        self.sessions = {}
        self.model_path = model_path

    def policy(self, name):
        if name == "micro":
            from .micro import MicroModel

            return MicroModel.load(self.model_path or "models/model.json").policy()
        return create_policy(name)

    def public_files(self, episode):
        rows = sorted((row["doc_id"], row) for table in ("invoices", "payments") for row in episode.tables[table])
        return {"manifest": episode.manifest, "policy": episode.policy, "tables": episode.tables,
                "documents": [episode.documents.get(doc_id, row) for doc_id, row in rows]}

    def create(self, body):
        mode = body.get("mode", "open")
        if mode not in ("open", "closed"):
            raise ValueError("mode must be open or closed")
        config = Config.from_dict(body.get("config") or {"keyspace": "open"})
        seed = body.get("seed", 202611)
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        session = Session(generate(seed, config), graded=True, training=mode == "open")
        return self.register(session)

    def import_folder(self, body):
        folder = body.get("folder")
        if not isinstance(folder, str) or not Path(folder).is_dir():
            raise ValueError("folder must be an existing directory with the six episode files")
        episode = load_episode(folder)
        return self.register(Session(episode, graded=False, training=False))

    def register(self, session):
        session_id = secrets.token_hex(8)
        self.sessions[session_id] = session
        return {"session_id": session_id, "graded": session.graded, "observation": session.observation}

    def session(self, session_id):
        if session_id not in self.sessions:
            raise KeyError("unknown session")
        return self.sessions[session_id]

    def step(self, session_id, body):
        session = self.session(session_id)
        with session.lock:
            observation, reward, terminated, truncated, info = session.env.step(body.get("action"))
        return {"observation": observation, "reward": reward, "terminated": terminated, "truncated": truncated,
                "info": info}

    def run(self, session_id, body):
        session = self.session(session_id)
        name = body.get("policy", "expert")
        if name not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")
        env = AuditEnv(training=session.training)
        observation, _ = env.load_episode(session.episode)
        outcome, trace, seconds = run_policy(env, observation, self.policy(name))
        submitted = trace[-1]["action"] if trace[-1]["action"] else {}
        findings = (submitted.get("findings") or {}).get("findings", []) if isinstance(submitted, dict) else []
        result = {"policy": name, "steps": len(trace) - 1, "seconds": round(seconds, 4), "findings": findings,
                  "trajectory": trace}
        if session.graded and session.training:
            result["score"] = {key: value for key, value in asdict(outcome).items() if key != "matches"}
        return result

    def audit(self, body):
        if "folder" in body:
            episode = load_episode(body["folder"])
            manifest, policy, tables = episode.manifest, episode.policy, episode.tables
        else:
            manifest, policy, tables = body["manifest"], body["policy"], body["tables"]
        if body.get("auditor", "rules") == "micro":
            from .micro import MicroModel

            findings = MicroModel.load(self.model_path or "models/model.json").findings(tables, manifest, policy)
        else:
            findings = audit(manifest, policy, tables)
        return {"findings": findings}


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        server_version = f"audit-env/{__version__}"

        def log_message(self, *_):
            return

        def send(self, status, payload):
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            origin = self.headers.get("Origin", "")
            if origin.startswith(("http://localhost", "http://127.0.0.1")):
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.end_headers()
            self.wfile.write(data)

        def body(self):
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                raise ValueError("request body too large")
            raw = self.rfile.read(length) if length else b"{}"
            value = json.loads(raw.decode("utf-8"))
            if not isinstance(value, dict):
                raise ValueError("JSON object expected")
            return value

        def do_OPTIONS(self):
            self.send(204, {})

        def do_GET(self):
            parts = [part for part in self.path.split("?")[0].split("/") if part]
            try:
                if parts == ["api", "health"]:
                    return self.send(200, {"ok": True, "version": __version__, "sessions": len(app.sessions)})
                if len(parts) == 4 and parts[:2] == ["api", "sessions"] and parts[3] == "files":
                    return self.send(200, app.public_files(app.session(parts[2]).episode))
                return self.send(404, {"error": "not_found"})
            except KeyError as exc:
                return self.send(404, {"error": str(exc.args[0])})

        def do_POST(self):
            parts = [part for part in self.path.split("?")[0].split("/") if part]
            try:
                body = self.body()
                if parts == ["api", "episodes"]:
                    return self.send(200, app.create(body))
                if parts == ["api", "episodes", "import"]:
                    return self.send(200, app.import_folder(body))
                if parts == ["api", "audit"]:
                    return self.send(200, app.audit(body))
                if len(parts) == 4 and parts[:2] == ["api", "sessions"] and parts[3] == "step":
                    return self.send(200, app.step(parts[2], body))
                if len(parts) == 4 and parts[:2] == ["api", "sessions"] and parts[3] == "run":
                    return self.send(200, app.run(parts[2], body))
                return self.send(404, {"error": "not_found"})
            except FormatError as exc:
                return self.send(400, {"error": "format", "problems": exc.problems})
            except KeyError as exc:
                return self.send(404, {"error": str(exc.args[0]) if exc.args else "missing field"})
            except (ValueError, TypeError) as exc:
                return self.send(400, {"error": str(exc)})

    return Handler


def serve(host="127.0.0.1", port=8765, model_path=None):
    server = ThreadingHTTPServer((host, port), make_handler(App(model_path)))
    return server
