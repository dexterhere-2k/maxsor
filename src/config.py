from __future__ import annotations

import csv
import logging
import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("%s=%r is not a number; using %s", name, raw, default)
        return default

KB_DIR = Path(os.getenv("KB_DIR") or PROJECT_ROOT / "knowledge_base")
TICKETS_CSV = Path(os.getenv("TICKETS_CSV") or PROJECT_ROOT / "data" / "tickets.csv")
SAMPLE_CASES = Path(os.getenv("SAMPLE_CASES") or PROJECT_ROOT / "sample_test_cases.json")
BOUNDARY_PROBE = Path(
    os.getenv("BOUNDARY_PROBE") or PROJECT_ROOT / "data" / "boundary_probe.json"
)
EMBEDDINGS_PATH = Path(
    os.getenv("EMBEDDINGS_PATH") or PROJECT_ROOT / "data" / "embeddings.json"
)

DATABASE_PATH = os.getenv("DATABASE_PATH") or str(PROJECT_ROOT / "support_assistant.db")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

LITELLM_MODEL = os.getenv("LITELLM_MODEL", "").strip() or "gemini/gemini-2.5-flash"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "").strip() or "gemini/gemini-embedding-001"

LLM_PACE_SECONDS = _env_float("LLM_PACE_SECONDS", 0.0)

JWT_ALGORITHM = "HS256"
TOKEN_TTL_HOURS = int(os.getenv("TOKEN_TTL_HOURS", "24"))

JWT_SECRET = os.getenv("JWT_SECRET", "").strip()
if not JWT_SECRET:
    JWT_SECRET = secrets.token_urlsafe(48)
    log.warning(
        "JWT_SECRET is not set; generated an ephemeral secret for this process. "
        "Tokens issued now will not validate after a restart."
    )

DECISION_PATHS = ("cag", "fallback")

ACTION_VOCAB_SIZE = 15

def derive_actions(csv_path: Path | None = None) -> frozenset[str]:
    path = Path(csv_path or TICKETS_CSV)
    if not path.exists():
        raise FileNotFoundError(
            f"Historical ticket file not found at {path}. "
            "The action vocabulary is derived from its resolved_action column."
        )
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if "resolved_action" not in (reader.fieldnames or []):
            raise ValueError(f"{path} has no 'resolved_action' column")
        actions = {
            row["resolved_action"].strip()
            for row in reader
            if row.get("resolved_action", "").strip()
        }
    if not actions:
        raise ValueError(f"No actions found in {path}")
    return frozenset(actions)

ACTIONS: frozenset[str] = derive_actions()
assert len(ACTIONS) == ACTION_VOCAB_SIZE, (
    f"expected exactly {ACTION_VOCAB_SIZE} actions in {TICKETS_CSV.name}, "
    f"derived {len(ACTIONS)}: {sorted(ACTIONS)}"
)

def llm_configured() -> bool:
    return bool(GEMINI_API_KEY)

if __name__ == "__main__":
    assert len(ACTIONS) == ACTION_VOCAB_SIZE, f"derived {len(ACTIONS)} actions, expected {ACTION_VOCAB_SIZE}"
    assert KB_DIR.is_dir(), f"no knowledge base at {KB_DIR}"
    assert TICKETS_CSV.exists(), f"no historical tickets at {TICKETS_CSV}"
    assert SAMPLE_CASES.exists(), f"no sample cases at {SAMPLE_CASES}"
    assert TOKEN_TTL_HOURS > 0, "a token must expire"
    print(
        f"{len(ACTIONS)} actions, {len(list(KB_DIR.glob('*.md')))} policy documents, "
        f"model key {'set' if llm_configured() else 'unset'}"
    )
