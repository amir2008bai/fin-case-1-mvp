"""Zero-dependency Model Context Protocol (MCP) JSON-RPC 2.0 server over stdio.

Exposes the exact 5 AuditEnv tools (`list_records`, `get_record`, `get_document`,
`calculator`, `submit`) to any MCP-compatible client (Claude Desktop, Cursor,
Codex, Gemini CLI, LangGraph, FinMCP-Bench) without leaking hidden truth or seed.
`--compact` sends table pages as columns + rows (about 40% fewer characters on the ledger, lossless)
and drops the tool list from the first observation; the policy is always sent verbatim.
"""
import json
import sys

from . import __version__
from .contracts import MAX_FRAME_BYTES, Config
from .environment import AuditEnv
from .tool_schemas import _TOOL_NAMES, compact_initial_observation, dispatch_tool_call, mcp_tools

PROTOCOL_VERSION = "2025-03-26"
MCP_TIMEOUT_SECONDS = 30 * 60


def _error(req_id, code, message):
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


class MCPAuditServer:
    """Stateful JSON-RPC 2.0 handler wrapping one AuditEnv episode."""

    def __init__(self, env, initial_observation, *, compact=False, hints=True):
        self.env = env
        self.initial_observation = initial_observation
        self.compact = compact
        self.hints = hints

    def handle(self, message):
        """Process a single parsed JSON-RPC 2.0 request dict; returns response dict or None for notifications."""
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0" or not isinstance(message.get("method"), str):
            return _error(None, -32600, "Invalid Request")

        # JSON-RPC 2.0 notification: "id" key is omitted completely.
        if "id" not in message:
            return None

        req_id = message["id"]
        if req_id is None or isinstance(req_id, bool) or not isinstance(req_id, (str, int)):
            return _error(None, -32600, "Invalid Request: id must be string or integer")

        method = message["method"]
        raw_params = message.get("params")
        if raw_params is not None and not isinstance(raw_params, dict):
            return _error(req_id, -32602, "Invalid params: expected object")
        params = raw_params or {}

        if method == "initialize":
            self.env.restart_clock()  # the client may connect long after the process started
            init_payload = compact_initial_observation(self.initial_observation) if self.compact else self.initial_observation
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "fintech-audit-env", "version": __version__},
                    "instructions": json.dumps(init_payload, ensure_ascii=False, separators=(",", ":")),
                },
            }
        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": mcp_tools()}}
        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments", {})
            if not isinstance(name, str) or name not in _TOOL_NAMES or not isinstance(arguments, dict) or "type" in arguments:
                return _error(req_id, -32602, "Invalid params for tools/call: unknown tool or malformed arguments")
            outcome = dispatch_tool_call(self.env, name, arguments, compact=self.compact, hints=self.hints)
            payload = {
                "observation": outcome["observation"],
                "reward": outcome["reward"],
                "terminated": outcome["terminated"],
                "truncated": outcome["truncated"],
            }
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}],
                    "isError": outcome["is_error"],
                },
            }
        return _error(req_id, -32601, f"Method not found: {method}")


def serve_stdio(seed=202601, config=None, *, closed=False, compact=False, stdin=None, stdout=None,
                timeout_seconds=MCP_TIMEOUT_SECONDS):
    """Run an MCP JSON-RPC 2.0 loop over stdin/stdout with bounded frame reads until EOF.

    An LLM needs seconds per tool call, so the episode clock is `timeout_seconds` (30 min by default) and
    starts at `initialize`. Scored runs go through the runner with the official 60 s limit instead.
    """
    stdin = sys.stdin if stdin is None else stdin
    stdout = sys.stdout if stdout is None else stdout
    if config is None:
        cfg = Config(keyspace="secret" if closed else "open")
    elif isinstance(config, dict):
        if closed and config.get("keyspace", "secret") != "secret":
            raise ValueError("closed mode needs the secret keyspace")
        cfg = Config.from_dict({**config, "keyspace": "secret"} if closed else config)
    else:
        cfg = Config.from_dict(config)
        if closed and cfg.keyspace != "secret":
            raise ValueError("closed mode needs the secret keyspace")

    env = AuditEnv(training=not closed, timeout_seconds=timeout_seconds, interactive=True)
    initial_obs, _ = env.reset(seed, cfg)
    server = MCPAuditServer(env, initial_obs, compact=compact)
    while True:
        raw = stdin.readline(MAX_FRAME_BYTES + 1)
        if not raw:
            break
        if len(raw) > MAX_FRAME_BYTES and not raw.endswith("\n"):
            # Drain oversized line safely without OOM
            while True:
                chunk = stdin.readline(MAX_FRAME_BYTES)
                if not chunk or chunk.endswith("\n"):
                    break
            resp = _error(None, -32600, "Invalid Request: frame too large")
        else:
            raw = raw.strip()
            if not raw:
                continue
            try:
                message = json.loads(raw)
                resp = server.handle(message)
            except (ValueError, RecursionError):
                resp = _error(None, -32700, "Parse error")
            except Exception:
                resp = _error(None, -32603, "Internal error")
        if resp is not None:
            stdout.write(json.dumps(resp, ensure_ascii=False, separators=(",", ":")) + "\n")
            stdout.flush()
    return env.outcome
