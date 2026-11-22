"""Diagnostic LLM probe over the existing MCP server on 5 closed episodes.

Runs an external LLM via standard-library HTTP (`urllib.request`) against
`audit_env.mcp_server.MCPAuditServer` in closed mode (secret keyspace, hidden
truth and reward). API keys are read strictly from environment variables
(`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GEMINI_API_KEY`). If no key is set
or the network/API is unreachable, the command prints a clear explanation and
exits with status code 0. Zero external dependencies; never touches or exposes
`AUDIT_ENV_SECRET` to the LLM.
"""
from dataclasses import replace
import http.client
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request

from .dataset import suite
from .environment import AuditEnv
from .generator import generate
from .mcp_server import MCPAuditServer, MCP_TIMEOUT_SECONDS
from .keys import table_key
from .selfcheck import fresh_key
from .tool_schemas import gemini_tools

DEFAULT_PROBE_CASE_IDS = (
    "holdout-01",  # clean
    "holdout-11",  # single: duplicate_invoice
    "holdout-13",  # single: fx_rate
    "holdout-17",  # multi
    "holdout-06",  # interacting
)

SYSTEM_PROMPT = (
    "You are an independent financial auditor inspecting synthetic accounting books "
    "through MCP tools. Follow the episode policy in the initial instructions verbatim. "
    "Use list_records, get_record, get_document, and calculator to inspect the books "
    "and verify every discrepancy. Only cite IDs you actually observed via tools. "
    "When finished, call submit exactly once with {\"findings\": {\"episode_id\": ..., \"findings\": [...]}} "
    "(use an empty findings list [] if the books are clean)."
)

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-5-5",
    "gemini": "gemini-2.5-flash",
}

ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}


class ProbeUnavailable(Exception):
    """Raised when no API key is configured or the network/API is unreachable."""


def detect_provider(preferred=None):
    """Return (provider, api_key) strictly from environment variables, or (None, None)."""
    order = (preferred,) if preferred else ("openai", "anthropic", "gemini")
    for name in order:
        env_var = ENV_KEYS.get(name)
        if not env_var:
            continue
        value = os.environ.get(env_var, "").strip()
        if value:
            return name, value
    return None, None


def _host(url):
    """Only scheme and host in messages: never a path or query that could carry a key."""
    parts = urllib.parse.urlsplit(url)
    return f"{parts.scheme}://{parts.hostname}"


RETRY_DELAYS = (5, 20, 60)  # seconds before each retry of a transient failure


def _post_json(url, payload, headers, timeout=60, sleep=time.sleep):
    """POST and parse JSON, retrying what is transient on hosted APIs.

    While a model thinks, OpenRouter keeps the connection open with blank lines, and the body can end
    without any JSON; 429 and 5xx answers are also temporary. Those are retried, everything else raises
    ProbeUnavailable at once.
    """
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    problem = "no answer"
    for delay in (0, *RETRY_DELAYS):
        if delay:
            sleep(delay)
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                text = resp.read().decode("utf-8", errors="replace").strip()
            parsed = json.loads(text)
            if isinstance(parsed, list) and len(parsed) == 1 and isinstance(parsed[0], dict):
                parsed = parsed[0]  # Gemini's OpenAI-compatible endpoint wraps errors in a list
            if not isinstance(parsed, dict):
                raise ValueError("JSON object expected")
            if isinstance(parsed.get("error"), dict):
                code = parsed["error"].get("code")
                problem = f"API error {code}: {str(parsed['error'].get('message', ''))[:200]}"
                if code in (429, 500, 502, 503, 504):
                    continue
                raise ProbeUnavailable(f"{problem} from {_host(url)}")
            return parsed
        except ValueError:
            problem = "an empty or truncated JSON answer"
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read(300).decode("utf-8", errors="replace").strip()
            except Exception:
                pass
            problem = f"HTTP {exc.code}: {body or exc.reason}"
            if exc.code not in (429, 500, 502, 503, 504):
                raise ProbeUnavailable(f"{problem} from {_host(url)}") from None
        except (TimeoutError, ConnectionResetError, http.client.RemoteDisconnected) as exc:
            problem = f"connection dropped ({exc})"
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, ConnectionResetError, http.client.RemoteDisconnected)):
                problem = f"connection dropped ({exc.reason})"
                continue
            raise ProbeUnavailable(f"network/endpoint {_host(url)} unreachable ({exc})") from None  # refused, DNS
        except OSError as exc:
            raise ProbeUnavailable(f"network/endpoint {_host(url)} unreachable ({exc})") from None
    raise ProbeUnavailable(f"{_host(url)}: {problem} after {len(RETRY_DELAYS) + 1} attempts")


_progress = None  # set by run_probe(verbose=True): one line per tool call on stderr


def _mcp_call(server, req_id, method, params=None):
    message = {"jsonrpc": "2.0", "id": req_id, "method": method}
    if params is not None:
        message["params"] = params
    response = server.handle(message)
    if _progress is not None and method == "tools/call":
        arguments = (params or {}).get("arguments") or {}
        brief = {key: value for key, value in arguments.items() if key != "findings"}
        if "findings" in arguments:
            brief["findings"] = len(((arguments.get("findings") or {}).get("findings")) or [])
        _progress(f"  step {req_id - 10:3d} {params.get('name')} {json.dumps(brief, ensure_ascii=False)[:120]}")
    return response


def _select_probe_cases(holdout_path, count=5, seed=7):
    all_cases = suite(holdout_path)
    by_id = {case_id: (split, case_id, category, cfg) for split, case_id, category, _, cfg in all_cases}
    selected = [by_id[cid] for cid in DEFAULT_PROBE_CASE_IDS if cid in by_id]
    if len(selected) < count:
        seen_ids = {item[1] for item in selected}
        for split, case_id, category, _, cfg in all_cases:
            if case_id not in seen_ids:
                selected.append((split, case_id, category, cfg))
                seen_ids.add(case_id)
            if len(selected) >= count:
                break
    rng = random.Random(seed)
    return [(case_id, category, rng.randrange(2**31), replace(cfg, keyspace="secret"))
            for _, case_id, category, cfg in selected[:count]]


def _run_openai_episode(server, init_text, mcp_tool_list, api_key, model, base_url, max_steps, http_timeout):
    url = base_url or os.environ.get("LLM_PROBE_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1/chat/completions"
    tools = [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["inputSchema"],
            },
        }
        for t in mcp_tool_list
    ]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Episode initial observation (MCP initialize):\n{init_text}"},
    ]
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    req_id = 10
    steps = 0
    nudges = 0
    while steps < max_steps:
        resp = _post_json(
            url,
            {"model": model, "messages": messages, "tools": tools, "tool_choice": "auto", "temperature": 0},
            headers,
            timeout=http_timeout,
        )
        choices = resp.get("choices") or []
        if not choices:
            break
        msg = choices[0].get("message") or {}
        messages.append(msg)
        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            nudges += 1
            if nudges > 2:
                break
            messages.append({"role": "user", "content": "Please call one of the MCP tools or call submit to finish the episode."})
            continue
        done = False
        for tc in tool_calls:
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            raw_args = fn.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
            except ValueError:
                args = {}
            req_id += 1
            mcp_resp = _mcp_call(server, req_id, "tools/call", {"name": name, "arguments": args})
            steps += 1
            if "error" in mcp_resp:
                content_text = json.dumps(mcp_resp["error"], ensure_ascii=False)
            else:
                content_text = mcp_resp["result"]["content"][0]["text"]
                parsed = json.loads(content_text)
                if parsed.get("terminated") or parsed.get("truncated"):
                    done = True
            messages.append({
                "role": "tool",
                "tool_call_id": tc.get("id", f"call_{req_id}"),
                "content": content_text,
            })
            if done or steps >= max_steps:
                break
        if done:
            break
    return steps


def _run_anthropic_episode(server, init_text, mcp_tool_list, api_key, model, base_url, max_steps, http_timeout):
    url = base_url or os.environ.get("LLM_PROBE_BASE_URL") or "https://api.anthropic.com/v1/messages"
    tools = [
        {"name": t["name"], "description": t["description"], "input_schema": t["inputSchema"]}
        for t in mcp_tool_list
    ]
    messages = [
        {"role": "user", "content": f"Episode initial observation (MCP initialize):\n{init_text}"},
    ]
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    req_id = 10
    steps = 0
    nudges = 0
    while steps < max_steps:
        resp = _post_json(
            url,
            {
                "model": model,
                "max_tokens": 2048,
                "temperature": 0,
                "system": SYSTEM_PROMPT,
                "tools": tools,
                "messages": messages,
            },
            headers,
            timeout=http_timeout,
        )
        blocks = resp.get("content") or []
        messages.append({"role": "assistant", "content": blocks})
        tool_uses = [b for b in blocks if isinstance(b, dict) and b.get("type") == "tool_use"]
        if not tool_uses:
            nudges += 1
            if nudges > 2:
                break
            messages.append({"role": "user", "content": "Please call one of the MCP tools or call submit to finish the episode."})
            continue
        results = []
        done = False
        for tu in tool_uses:
            name = tu.get("name", "")
            args = tu.get("input") if isinstance(tu.get("input"), dict) else {}
            req_id += 1
            mcp_resp = _mcp_call(server, req_id, "tools/call", {"name": name, "arguments": args})
            steps += 1
            if "error" in mcp_resp:
                content_text = json.dumps(mcp_resp["error"], ensure_ascii=False)
                is_err = True
            else:
                content_text = mcp_resp["result"]["content"][0]["text"]
                is_err = bool(mcp_resp["result"].get("isError"))
                parsed = json.loads(content_text)
                if parsed.get("terminated") or parsed.get("truncated"):
                    done = True
            results.append({
                "type": "tool_result",
                "tool_use_id": tu.get("id", f"toolu_{req_id}"),
                "content": content_text,
                "is_error": is_err,
            })
            if done or steps >= max_steps:
                break
        messages.append({"role": "user", "content": results})
        if done:
            break
    return steps


def _run_gemini_episode(server, init_text, _mcp_tool_list, api_key, model, base_url, max_steps, http_timeout):
    base = (base_url or os.environ.get("LLM_PROBE_BASE_URL")
            or "https://generativelanguage.googleapis.com/v1beta/models").rstrip("/")
    url = f"{base}/{model}:generateContent"  # the key goes in a header: a URL ends up in error messages
    tools = [{"function_declarations": gemini_tools()}]
    contents = [
        {"role": "user", "parts": [{"text": f"Episode initial observation (MCP initialize):\n{init_text}"}]},
    ]
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    req_id = 10
    steps = 0
    nudges = 0
    while steps < max_steps:
        resp = _post_json(
            url,
            {
                "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                "tools": tools,
                "contents": contents,
                "generationConfig": {"temperature": 0},
            },
            headers,
            timeout=http_timeout,
        )
        candidates = resp.get("candidates") or []
        if not candidates:
            break
        content = candidates[0].get("content") or {"role": "model", "parts": []}
        contents.append(content)
        fcalls = [p["functionCall"] for p in (content.get("parts") or []) if isinstance(p, dict) and "functionCall" in p]
        if not fcalls:
            nudges += 1
            if nudges > 2:
                break
            contents.append({"role": "user", "parts": [{"text": "Please call one of the MCP tools or call submit to finish the episode."}]})
            continue
        response_parts = []
        done = False
        for fc in fcalls:
            name = fc.get("name", "")
            args = fc.get("args") if isinstance(fc.get("args"), dict) else {}
            req_id += 1
            mcp_resp = _mcp_call(server, req_id, "tools/call", {"name": name, "arguments": args})
            steps += 1
            if "error" in mcp_resp:
                result_obj = {"error": mcp_resp["error"]}
            else:
                result_obj = json.loads(mcp_resp["result"]["content"][0]["text"])
                if result_obj.get("terminated") or result_obj.get("truncated"):
                    done = True
            response_parts.append({"functionResponse": {"name": name, "response": result_obj}})
            if done or steps >= max_steps:
                break
        contents.append({"role": "user", "parts": response_parts})
        if done:
            break
    return steps


RUNNERS = {
    "openai": _run_openai_episode,
    "anthropic": _run_anthropic_episode,
    "gemini": _run_gemini_episode,
}


def run_probe(holdout_path, *, episodes=5, seed=7, provider=None, model=None, base_url=None,
              compact=True, timeout=MCP_TIMEOUT_SECONDS, http_timeout=180, max_steps=100, emit=print,
              verbose=False, report=None, same_episodes=False):
    """Run an LLM diagnostic across `episodes` closed episodes via MCPAuditServer.

    Always returns exit code 0 when skipped due to missing API key or unreachable network.
    """
    chosen_provider, api_key = detect_provider(provider)
    if not chosen_provider or not api_key:
        expected = ENV_KEYS.get(provider, "OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY")
        emit(f"llm-probe: skipped — no API key in environment ({expected}). Set the variable and re-run.")
        return 0

    chosen_model = model or os.environ.get("LLM_PROBE_MODEL") or DEFAULT_MODELS[chosen_provider]
    cases = _select_probe_cases(holdout_path, count=episodes, seed=seed)
    # A fresh evaluator key: a key derived from a public string and the seed would let anyone who knows
    # the code regenerate these closed episodes, truth included. LLM output is not reproducible anyway.
    # --same-episodes: the key of the reproducible tables, so several models face identical episodes.
    secret_key = table_key(seed) if same_episodes else fresh_key()
    runner = RUNNERS[chosen_provider]

    global _progress
    _progress = (lambda line: print(line, file=sys.stderr, flush=True)) if verbose else None
    emit(f"llm-probe: provider={chosen_provider} model={chosen_model} episodes={len(cases)} closed=True compact={compact}")
    emit(f"{'case_id':12s} {'category':24s} {'P':>6s} {'R':>6s} {'F1':>6s} {'reward':>7s} {'steps':>5s} {'sec':>6s}  status")

    rows = []
    for case_id, category, ep_seed, cfg in cases:
        episode = generate(ep_seed, cfg, key=secret_key)
        env = AuditEnv(training=False, timeout_seconds=timeout, interactive=True)
        initial_obs, _ = env.load_episode(episode)
        server = MCPAuditServer(env, initial_obs, compact=compact)
        init_resp = _mcp_call(server, 1, "initialize", {})
        tools_resp = _mcp_call(server, 2, "tools/list", {})
        init_text = init_resp["result"]["instructions"]
        mcp_tool_list = tools_resp["result"]["tools"]

        started = time.perf_counter()
        try:
            steps = runner(server, init_text, mcp_tool_list, api_key, chosen_model, base_url, max_steps, http_timeout)
        except ProbeUnavailable as exc:
            emit(f"llm-probe: skipped — {exc}")
            return 0
        elapsed = time.perf_counter() - started

        # If the model stopped without calling submit, finalize cleanly with an empty submission.
        if env.outcome is None:
            _mcp_call(server, 999, "tools/call", {
                "name": "submit",
                "arguments": {"findings": {"episode_id": episode.manifest["episode_id"], "findings": []}},
            })
            steps += 1
        outcome = env.outcome
        rows.append({
            "case_id": case_id,
            "category": category,
            "precision": outcome.precision,
            "recall": outcome.recall,
            "f1": outcome.f1,
            "reward": outcome.reward,
            "steps": steps,
            "seconds": round(elapsed, 2),
            "termination": outcome.termination,
            "issues": len(episode.issues),
            "tp": outcome.tp,
            "fp": outcome.fp,
            "fn": outcome.fn,
        })
        if report:  # rewritten after every episode: a long run interrupted halfway keeps what it measured
            Path(report).parent.mkdir(parents=True, exist_ok=True)
            Path(report).write_text(json.dumps({
                "provider": chosen_provider, "model": chosen_model, "closed": True, "compact": compact,
                "episode_seeds_from": seed,
                "evaluator_key": f"keys.table_key({seed})" if same_episodes else "fresh, not stored", "episodes": rows,
            }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        emit(
            f"{case_id:12s} {category:24s} {outcome.precision:6.3f} {outcome.recall:6.3f} "
            f"{outcome.f1:6.3f} {outcome.reward:+7.3f} {steps:5d} {elapsed:6.2f}  {outcome.termination}"
        )

    if rows:
        emit(
            f"{'MEAN':12s} {'(' + str(len(rows)) + ' closed episodes)':24s} "
            f"{statistics.mean(r['precision'] for r in rows):6.3f} "
            f"{statistics.mean(r['recall'] for r in rows):6.3f} "
            f"{statistics.mean(r['f1'] for r in rows):6.3f} "
            f"{statistics.mean(r['reward'] for r in rows):+7.3f} "
            f"{statistics.mean(r['steps'] for r in rows):5.1f} "
            f"{sum(r['seconds'] for r in rows):6.2f}"
        )
    return 0
