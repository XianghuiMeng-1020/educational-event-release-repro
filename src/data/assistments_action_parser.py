"""Streaming parser for ASSISTments `actions` field strings.

Does not silently drop failures; each exclusion reason is countable.
"""
from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass
from typing import Any, Optional


@dataclass
class ParsedActions:
    raw: str
    status: str  # success | partial | failed
    reason_code: Optional[str]
    actions: list[dict[str, Any]]
    action_types: list[str]
    timestamps: list[Optional[str]]
    non_monotonic: bool
    duplicated_actions: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_PAIR_RE = re.compile(
    r"(?P<atype>[A-Za-z_]+)\s*[:=]\s*(?P<ts>[0-9:\-\s\.T]+|None|null)?"
    r"|(?P<atype2>[A-Za-z_]+)\s*\(\s*(?P<ts2>[^)]*)\s*\)"
)


def parse_actions_field(raw: Any) -> ParsedActions:
    if raw is None or (isinstance(raw, float) and str(raw) == "nan"):
        return ParsedActions("", "failed", "empty_or_null", [], [], [], False, False)
    s = str(raw).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return ParsedActions(s, "failed", "empty_or_null", [], [], [], False, False)

    # Try Python-literal / JSON-like first when string looks like JSON/Python list
    if s.lstrip().startswith(("[", "{")):
        for loader in (_try_literal, _try_jsonish):
            loaded = loader(s)
            if loaded is not None:
                return _from_structured(s, loaded)

    # ASSISTments 2012 official format: YAML list of event lists (often starts with ---)
    if s.lstrip().startswith("---") or ("\n- - " in s or s.lstrip().startswith("- - ")):
        yloaded = _try_yaml(s)
        if yloaded is not None:
            return _from_assistments_yaml(s, yloaded)

    # Try Python-literal / JSON-like list of dicts or tuples
    for loader in (_try_literal, _try_jsonish):
        loaded = loader(s)
        if loaded is not None:
            return _from_structured(s, loaded)

    # Try semicolon / pipe separated tokens: "hint:ts;attempt:ts"
    if ";" in s or "|" in s:
        parts = re.split(r"[;|]", s)
        actions = []
        types = []
        stamps = []
        partial = False
        for p in parts:
            p = p.strip()
            if not p:
                continue
            m = _PAIR_RE.search(p)
            if not m:
                partial = True
                continue
            at = m.group("atype") or m.group("atype2")
            ts = m.group("ts") or m.group("ts2")
            actions.append({"action_type": at, "timestamp": ts})
            types.append(str(at))
            stamps.append(ts)
        if not actions:
            return ParsedActions(s, "failed", "unrecognized_format", [], [], [], False, False)
        nonmono = _non_monotonic(stamps)
        dup = len(types) != len(set(enumerate(types)))  # positional dup check below
        dup = _has_duplicate_actions(actions)
        status = "partial" if partial else "success"
        reason = "partial_token_parse" if partial else None
        return ParsedActions(s, status, reason, actions, types, stamps, nonmono, dup)

    # Bracketed ASSISTments style often embeds catalog of events as plain text
    # Fallback: extract action-like keywords with nearby digits/timestamps
    keywords = re.findall(
        r"\b(hint|attempt|answer|start|end|scaffold|bottom_hint|correct|incorrect|view)\b",
        s,
        flags=re.I,
    )
    if keywords:
        types = [k.lower() for k in keywords]
        actions = [{"action_type": t, "timestamp": None} for t in types]
        return ParsedActions(
            s,
            "partial",
            "keyword_fallback_no_timestamps",
            actions,
            types,
            [None] * len(types),
            False,
            _has_duplicate_actions(actions),
        )

    return ParsedActions(s, "failed", "unrecognized_format", [], [], [], False, False)


def _try_yaml(s: str):
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(s)
        if isinstance(data, list):
            return data
        return None
    except Exception:
        return None


def _from_assistments_yaml(raw: str, data: list) -> ParsedActions:
    actions: list[dict[str, Any]] = []
    partial = False
    for el in data:
        if not isinstance(el, list) or not el:
            partial = True
            continue
        at = str(el[0]).lower()
        ts = el[1] if len(el) > 1 else None
        payload = el[2:] if len(el) > 2 else []
        actions.append({"action_type": at, "timestamp": ts, "payload": payload})
    if not actions:
        return ParsedActions(raw, "failed", "yaml_empty_actions", [], [], [], False, False)
    types = [str(a["action_type"]) for a in actions]
    stamps = [a.get("timestamp") for a in actions]
    status = "partial" if partial else "success"
    reason = "yaml_partial_elements" if partial else None
    # `end` / `resume` often lack timestamps in ASSISTments YAML; do not fail those.
    critical_missing_ts = False
    for a in actions:
        at = str(a.get("action_type"))
        ts = a.get("timestamp")
        if at in {"end", "resume"}:
            continue
        if ts is None or str(ts).strip() == "":
            critical_missing_ts = True
    if critical_missing_ts and status == "success":
        status, reason = "partial", "missing_timestamps"
    return ParsedActions(
        raw,
        status,
        reason,
        actions,
        types,
        stamps,
        _non_monotonic(stamps),
        _has_duplicate_actions(actions),
    )


def _try_literal(s: str):
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def _try_jsonish(s: str):
    import json

    try:
        return json.loads(s)
    except Exception:
        # common: single quotes
        try:
            return json.loads(s.replace("'", '"'))
        except Exception:
            return None


def _from_structured(raw: str, loaded: Any) -> ParsedActions:
    actions: list[dict[str, Any]] = []
    if isinstance(loaded, dict):
        loaded = [loaded]
    if not isinstance(loaded, (list, tuple)):
        return ParsedActions(raw, "failed", "structured_not_list", [], [], [], False, False)
    partial = False
    for el in loaded:
        if isinstance(el, dict):
            at = el.get("action_type") or el.get("type") or el.get("action") or el.get("name")
            ts = el.get("timestamp") or el.get("time") or el.get("ts")
            if at is None:
                partial = True
                continue
            actions.append({"action_type": str(at), "timestamp": ts, **{k: v for k, v in el.items() if k not in {"action_type", "type", "action", "name", "timestamp", "time", "ts"}}})
        elif isinstance(el, (list, tuple)) and len(el) >= 1:
            at = el[0]
            ts = el[1] if len(el) > 1 else None
            actions.append({"action_type": str(at), "timestamp": ts})
        else:
            partial = True
    if not actions:
        return ParsedActions(raw, "failed", "structured_empty_actions", [], [], [], False, False)
    types = [str(a["action_type"]) for a in actions]
    stamps = [a.get("timestamp") for a in actions]
    status = "partial" if partial else "success"
    reason = "structured_partial" if partial else None
    # missing timestamps
    if any(t is None or str(t).strip() == "" for t in stamps):
        if status == "success":
            status, reason = "partial", "missing_timestamps"
    return ParsedActions(raw, status, reason, actions, types, stamps, _non_monotonic(stamps), _has_duplicate_actions(actions))


def _non_monotonic(stamps: list[Optional[str]]) -> bool:
    prev = None
    prev_kind = None
    for t in stamps:
        if t is None:
            continue
        s = str(t).strip()
        try:
            cur: Any = float(s)
            kind = "num"
        except Exception:
            cur = s
            kind = "str"
        if prev is not None and kind == prev_kind:
            try:
                if cur < prev:
                    return True
            except TypeError:
                pass
        prev = cur
        prev_kind = kind
    return False


def _has_duplicate_actions(actions: list[dict[str, Any]]) -> bool:
    seen = set()
    for a in actions:
        key = (str(a.get("action_type")), str(a.get("timestamp")))
        if key in seen:
            return True
        seen.add(key)
    return False
