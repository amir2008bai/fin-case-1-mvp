"""Universal AI-agent tool schemas (MCP, OpenAI, Anthropic, Gemini), compact LLM views, and safe dispatcher.

Single source of truth derived from contracts.py with zero external dependencies,
designed so even compact 3B-8B LLM agents understand all 5 tools, self-correct
immediately on syntax mistakes, and spend minimal tokens per episode.
"""
from copy import deepcopy
import json

from .contracts import (
    ALL_TYPES,
    MAX_AMOUNT_MINOR,
    MAX_EVIDENCE,
    MAX_EXPLANATION,
    MAX_FINDINGS,
    MAX_PAGE,
    TABLE_IDS,
)

_TABLES = sorted(TABLE_IDS)
_TOOL_NAMES = frozenset(("list_records", "get_record", "get_document", "calculator", "submit"))
_TYPES_TEXT = ", ".join(ALL_TYPES)

ERROR_HINTS = {
    "unknown_table": f"table must be one of {_TABLES} (not a filename like 'fx.csv').",
    "invalid_limit": f"limit must be an integer 1..{MAX_PAGE} (or null/omitted for {MAX_PAGE}).",
    "invalid_cursor": "omit cursor or pass null for page 1; otherwise pass exact next_cursor from previous page.",
    "unknown_id": "id/doc_id was not found; use exact opaque IDs returned by list_records.",
    "invalid_expression": "calculator accepts only digits, spaces, decimal point, and + - * / ( ) — no **, %, or code.",
    "invalid_action": "check tool name and exact parameter names (no extra or misspelled keys).",
    "episode_finished": "episode already ended; no further actions are accepted.",
    "reset_required": "call reset() before step().",
}

_TOOL_SPECS = (
    {
        "name": "list_records",
        "description": (
            f"List paginated records from a public table ({', '.join(_TABLES)}). "
            f"Max {MAX_PAGE} rows/page; pass next_cursor for subsequent pages."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "table": {"type": "string", "enum": _TABLES, "description": f"Table name: {', '.join(_TABLES)}."},
                "cursor": {
                    "type": ["string", "null"],
                    "maxLength": 32,
                    "description": "Opaque next_cursor from previous page, or null/omitted for page 1.",
                },
                "limit": {
                    "type": ["integer", "null"],
                    "minimum": 1,
                    "maximum": MAX_PAGE,
                    "description": f"Rows per page (1..{MAX_PAGE}; omit or null for {MAX_PAGE}).",
                },
            },
            "required": ["table"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_record",
        "description": "Fetch one public record by table and its primary ID (row_id, invoice_id, payment_id, or fx_id).",
        "parameters": {
            "type": "object",
            "properties": {
                "table": {"type": "string", "enum": _TABLES, "description": f"Table name: {', '.join(_TABLES)}."},
                "id": {"type": "string", "minLength": 1, "maxLength": 128, "description": "Primary ID from list_records."},
            },
            "required": ["table", "id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_document",
        "description": "Fetch full document details (act Р-1, waybill З-2, payment order, ЭСФ, purpose) by doc_id.",
        "parameters": {
            "type": "object",
            "properties": {
                "doc_id": {"type": "string", "minLength": 1, "maxLength": 128, "description": "Opaque doc_id of invoice or payment."},
            },
            "required": ["doc_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "calculator",
        "description": "Exact 50-digit decimal arithmetic (+ - * / parentheses). No variables, **, %, or code.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 256,
                    "pattern": "^[ 0-9.+*/()-]+$",
                    "description": "Arithmetic expression, e.g. '(1500000 * 0.16) / 1.16'.",
                },
            },
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
    {
        "name": "submit",
        "description": (
            "Submit final audit findings as a JSON object and terminate the episode (one-shot; "
            "when remaining_steps == 1, only submit is accepted). Pass findings: [] if books are clean."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "findings": {
                    "type": "object",
                    "description": "Submission payload object with episode_id from manifest and list of findings.",
                    "properties": {
                        "episode_id": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 128,
                            "description": "Exact manifest.episode_id.",
                        },
                        "findings": {
                            "type": "array",
                            "maxItems": MAX_FINDINGS,
                            "description": f"0..{MAX_FINDINGS} findings; [] when no errors exist.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "type": {
                                        "type": "string",
                                        "minLength": 1,
                                        "maxLength": 64,
                                        "description": f"Error type: {_TYPES_TEXT}.",
                                    },
                                    "affected_ids": {
                                        "type": "array",
                                        "minItems": 1,
                                        "maxItems": 32,
                                        "uniqueItems": True,
                                        "items": {"type": "string", "minLength": 1, "maxLength": 128},
                                        "description": "Unique invoice_id / payment_id(s) affected by the error.",
                                    },
                                    "claimed_amount_minor": {
                                        "type": "integer",
                                        "minimum": -MAX_AMOUNT_MINOR,
                                        "maximum": MAX_AMOUNT_MINOR,
                                        "description": "Integer discrepancy in minor units of currency.",
                                    },
                                    "currency": {
                                        "type": "string",
                                        "pattern": "^[A-Z]{3}$",
                                        "description": "3-letter uppercase ISO currency code (e.g. KZT, USD, RUB).",
                                    },
                                    "evidence_ids": {
                                        "type": "array",
                                        "minItems": 1,
                                        "maxItems": MAX_EVIDENCE,
                                        "uniqueItems": True,
                                        "items": {"type": "string", "minLength": 1, "maxLength": 128},
                                        "description": f"1..{MAX_EVIDENCE} unique IDs already seen via tools (doc_id, row_id, fx_id).",
                                    },
                                    "explanation": {
                                        "type": "string",
                                        "maxLength": MAX_EXPLANATION,
                                        "description": f"Short human-readable rationale (<= {MAX_EXPLANATION} chars).",
                                    },
                                },
                                "required": ["type", "affected_ids", "claimed_amount_minor", "currency",
                                             "evidence_ids", "explanation"],
                                "additionalProperties": False,
                            },
                        },
                    },
                    "required": ["episode_id", "findings"],
                    "additionalProperties": False,
                },
            },
            "required": ["findings"],
            "additionalProperties": False,
        },
    },
)


def _to_gemini_schema(node):
    """Convert JSON Schema Draft 2020-12 to Gemini OpenAPI 3.0 function_declarations schema."""
    if not isinstance(node, dict):
        return node
    out = {}
    for key, value in node.items():
        if key in ("additionalProperties", "uniqueItems"):
            continue
        if key == "type" and isinstance(value, list):
            non_null = [t for t in value if t != "null"]
            out["type"] = non_null[0] if non_null else "string"
            if "null" in value:
                out["nullable"] = True
        elif key == "properties" and isinstance(value, dict):
            out[key] = {k: _to_gemini_schema(v) for k, v in value.items()}
        elif key == "items" and isinstance(value, dict):
            out[key] = _to_gemini_schema(value)
        else:
            out[key] = deepcopy(value)
    return out


def _to_openai_strict(node):
    """OpenAI strict mode requires every object property in `required`."""
    out = deepcopy(node)
    if isinstance(out, dict):
        if out.get("type") == "object" and isinstance(out.get("properties"), dict):
            out["required"] = list(out["properties"].keys())
            out["additionalProperties"] = False
            out["properties"] = {k: _to_openai_strict(v) for k, v in out["properties"].items()}
        elif out.get("type") == "array" and isinstance(out.get("items"), dict):
            out["items"] = _to_openai_strict(out["items"])
    return out


def mcp_tools():
    """Return deep-copied tool definitions in Model Context Protocol (MCP) format."""
    return [{"name": spec["name"], "description": spec["description"], "inputSchema": deepcopy(spec["parameters"])}
            for spec in _TOOL_SPECS]


def openai_tools(*, responses_api=False, strict=False):
    """Return deep-copied tool definitions for OpenAI Chat Completions or Responses API."""
    tools = []
    for spec in _TOOL_SPECS:
        params = _to_openai_strict(spec["parameters"]) if strict else deepcopy(spec["parameters"])
        if responses_api:
            item = {"type": "function", "name": spec["name"], "description": spec["description"], "parameters": params}
            if strict:
                item["strict"] = True
            tools.append(item)
        else:
            fn = {"name": spec["name"], "description": spec["description"], "parameters": params}
            if strict:
                fn["strict"] = True
            tools.append({"type": "function", "function": fn})
    return tools


def anthropic_tools():
    """Return deep-copied tool definitions in Anthropic Tool Use format."""
    return [{"name": spec["name"], "description": spec["description"], "input_schema": deepcopy(spec["parameters"])}
            for spec in _TOOL_SPECS]


def gemini_tools():
    """Return OpenAPI 3.0-compatible tool definitions for Google Gemini function_declarations."""
    return [{"name": spec["name"], "description": spec["description"], "parameters": _to_gemini_schema(spec["parameters"])}
            for spec in _TOOL_SPECS]


def tool_schemas(provider="openai", *, strict=False):
    """Unified accessor: provider in ('openai', 'openai_responses', 'anthropic', 'gemini', 'mcp')."""
    if provider == "openai":
        return openai_tools(responses_api=False, strict=strict)
    if provider == "openai_responses":
        return openai_tools(responses_api=True, strict=strict)
    if provider == "anthropic":
        return anthropic_tools()
    if provider == "gemini":
        return gemini_tools()
    if provider == "mcp":
        return mcp_tools()
    raise ValueError(f"unsupported provider: {provider}")


def compact_initial_observation(initial_observation):
    """Token-efficient summary (~80% smaller) preserving every rule, account code, tolerance, and manifest limit."""
    manifest = initial_observation["manifest"]
    policy = initial_observation["policy"]
    return {
        "task": (
            f"Audit company records for [{manifest['period_start']}, {manifest['period_end']}) "
            f"(tz={manifest['timezone']}, base={manifest['base_currency']}). "
            f"Types: {_TYPES_TEXT}. Books may be clean (findings: []). "
            "Final step (remaining_steps==1) accepts only submit."
        ),
        "manifest": manifest,
        "rules": {
            "accounts": {"1030": "Bank", "1210": "AR", "1330": "Inventory", "1420": "VAT_in",
                         "3130": "VAT_out", "3310": "AP", "3510": "Advances", "5610": "Opening",
                         "6010": "Revenue", "6250": "FX_gain", "7010": "COGS", "7430": "FX_loss"},
            "revenue": "Recognize at max(line.service_period_end) 00:00 in company tz (Dr 1210 / Cr 6010 + Cr 3130).",
            "vat": "16% included in price (total=qty*unit_price; vat=round_half_up(total*0.16/1.16)); foreign/null=0.",
            "fx": "Table 'fx': invoice at service_period_end, payment at local_date(timestamp); tol=2 minor units.",
            "allocation": "Apply payment across invoice_refs: open AR (Cr 1210) -> prepayment (Cr 3510) -> excess (Cr 3510).",
            "tolerances": {k: v["tolerance_minor"] for k, v in policy.get("findings", {}).items()},
            "evidence": "1..10 unique seen IDs (doc_id, row_id, fx_id) related to the issue.",
        },
        "remaining_steps": initial_observation["remaining_steps"],
    }


def compress_observation(obs):
    """Lossless tabular compression for list_records pages (~40% fewer tokens on ledger)."""
    if not isinstance(obs, dict):
        return obs
    result = obs.get("result")
    if isinstance(result, dict) and isinstance(result.get("records"), list) and result["records"]:
        first = result["records"][0]
        if isinstance(first, dict):
            cols = list(first.keys())
            rows = [[row.get(c) for c in cols] for row in result["records"]]
            out = dict(obs)
            out["result"] = {"columns": cols, "rows": rows, "next_cursor": result.get("next_cursor")}
            return out
    return obs


def dispatch_tool_call(env, name, arguments, *, compact=False, hints=True):
    """Safely translate any LLM/MCP tool call (name, dict|json-str) into AuditEnv.step()."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (ValueError, RecursionError):
            arguments = None
    if not isinstance(name, str) or name not in _TOOL_NAMES or not isinstance(arguments, dict) or "type" in arguments:
        action = None
    else:
        action = {"type": name, **arguments}
    obs, reward, terminated, truncated, info = env.step(action)
    view = compress_observation(obs) if compact else obs
    is_error = isinstance(view, dict) and "error" in view
    if hints and isinstance(view, dict):
        if is_error and view["error"] in ERROR_HINTS:
            view = dict(view)
            view["hint"] = ERROR_HINTS[view["error"]]
        elif not terminated and not truncated and view.get("remaining_steps") == 1:
            view = dict(view)
            view["warning"] = "final_step_submit_only"
    return {
        "observation": view,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "info": info,
        "is_error": is_error,
    }
