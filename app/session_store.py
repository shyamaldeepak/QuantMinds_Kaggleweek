"""Persistent chat session storage and session-state helpers."""

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

DEFAULT_SOURCES_MARKDOWN = "### Source Pages\nNo source pages available."


def _sessions_path():
    return Path(__file__).resolve().parents[1] / "data" / "chat_sessions.json"


def load_sessions():
    sessions_path = _sessions_path()
    if not sessions_path.exists():
        return []
    try:
        payload = json.loads(sessions_path.read_text(encoding="utf-8"))
        sessions = payload.get("sessions", []) if isinstance(payload, dict) else []
        return sessions if isinstance(sessions, list) else []
    except Exception:
        return []


def save_sessions(sessions):
    sessions_path = _sessions_path()
    sessions_path.parent.mkdir(parents=True, exist_ok=True)
    sessions_path.write_text(json.dumps({"sessions": sessions}, indent=2), encoding="utf-8")


def _to_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        for key in ("text", "content", "value"):
            v = value.get(key)
            if isinstance(v, str) and v.strip():
                return v
        return ""
    if isinstance(value, list):
        parts = [_to_text(v) for v in value]
        return " ".join([p for p in parts if p]).strip()
    return ""


def make_session_title(history):
    for item in history:
        if isinstance(item, dict) and item.get("role") == "user":
            content_text = _to_text(item.get("content"))
            if content_text:
                title = " ".join(content_text.split())[:48]
                return title if title else "New Chat"
        if isinstance(item, (list, tuple)) and item:
            user_text = _to_text(item[0])
            if user_text:
                title = " ".join(user_text.split())[:48]
                return title if title else "New Chat"
    return "New Chat"


def session_choices(sessions):
    sorted_sessions = sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)
    choices = []
    for s in sorted_sessions:
        title = s.get("title") or "New Chat"
        updated = s.get("updated_at", "")
        stamp = updated[:16].replace("T", " ") if updated else ""
        label = f"{title} ({stamp})" if stamp else title
        choices.append((label, s.get("id")))
    return choices


def get_or_create_session(session_id=None):
    sessions = load_sessions()
    if session_id:
        for s in sessions:
            if s.get("id") == session_id:
                return sessions, s

    now = datetime.now(timezone.utc).isoformat()
    new_session = {
        "id": str(uuid4()),
        "title": "New Chat",
        "created_at": now,
        "updated_at": now,
        "history": [],
        "last_sources": DEFAULT_SOURCES_MARKDOWN,
    }
    sessions.append(new_session)
    save_sessions(sessions)
    return sessions, new_session


def create_session():
    sessions = load_sessions()
    now = datetime.now(timezone.utc).isoformat()
    new_session = {
        "id": str(uuid4()),
        "title": "New Chat",
        "created_at": now,
        "updated_at": now,
        "history": [],
        "last_sources": DEFAULT_SOURCES_MARKDOWN,
    }
    sessions.append(new_session)
    save_sessions(sessions)
    return sessions, new_session


def delete_session(session_id):
    sessions = load_sessions()
    remaining = [s for s in sessions if s.get("id") != session_id]
    if not remaining:
        save_sessions([])
        return create_session()

    remaining = sorted(remaining, key=lambda x: x.get("updated_at", ""), reverse=True)
    active = remaining[0]
    save_sessions(remaining)
    return remaining, active


def clear_session(session_id):
    sessions = load_sessions()
    now = datetime.now(timezone.utc).isoformat()
    for s in sessions:
        if s.get("id") == session_id:
            s["history"] = []
            s["last_sources"] = DEFAULT_SOURCES_MARKDOWN
            s["title"] = "New Chat"
            s["updated_at"] = now
            break
    save_sessions(sessions)
    return sessions


def save_session_chat(session_id, history, sources_markdown):
    sessions, active = get_or_create_session(session_id)
    now = datetime.now(timezone.utc).isoformat()
    for s in sessions:
        if s.get("id") == active.get("id"):
            s["history"] = history
            s["last_sources"] = sources_markdown
            s["title"] = make_session_title(history)
            s["updated_at"] = now
            break
    save_sessions(sessions)
    return sessions, active
