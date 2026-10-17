import json
import time
from pathlib import Path

from .config import settings


def trace(state: dict, agent: str, action: str, detail: str) -> list[dict]:
    step = len(state.get("trace", [])) + 1
    return [{"step": step, "agent": agent, "action": action, "detail": detail}]


def evaluation (agent: str, signal: str, value, **extra) -> list[dict]:
    return [{"agent": agent, "signal": signal, "value": value, **extra}]


def write_run_log(state: dict) -> Path:
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    path = settings.log_dir / f"run_{int(time.time())}.json"
    record = {k: state.get(k) for k in (
        "conversation_id", "query", "input_ok", "rejection_reason", "plan",
        "retries", "retrieval_flags", "claims", "verdict", "confidence",
        "sources", "final_answer", "trace", "evaluation")}
    record["chunks"] = [{k: v for k, v in c.items() if k != "text"}
                        for c in state.get("chunks", [])]
    path.write_text(json.dumps (record, indent=2, default=str))
    return path