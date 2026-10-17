"""Minimal conversation memory backed by a JSON file per conversation."""

import json

from .config import settings

def _path(conversation_id: str):
    d = settings.log_dir / "memory"
    d.mkdir(parents=True, exist_ok=True)
    return d/f"{conversation_id}.json"


def load_history(conversation_id: str, limit: int = 6) -> list[dict]:
    p = _path(conversation_id)
    if not p.exists():
        return []
    return json.loads(p.read_text())[-limit:]


def append_turn(conversation_id: str, query: str, answer: str, confidence: str) -> None:
    p = _path(conversation_id)
    hist = json.loads(p.read_text()) if p.exists() else []
    hist.append({"query": query, "answer": answer, "confidence": confidence})
    p.write_text(json.dumps(hist, indent=2))