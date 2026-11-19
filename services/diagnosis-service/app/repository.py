"""Storage for the diagnosis service.

Reads events (all paradigms) from the shared PostgreSQL `events` table and stores its own results in
`diagnoses`. Without a DSN it falls back to in-memory storage (tests / local demo only).
"""
import json
import os
import threading
from typing import Any

DDL = """CREATE TABLE IF NOT EXISTS diagnoses (
    diagnosis_id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    project_id TEXT, trace_id TEXT,
    root_cause TEXT NOT NULL, category TEXT, severity TEXT, confidence DOUBLE PRECISION,
    cross_paradigm BOOLEAN NOT NULL DEFAULT FALSE,
    paradigms JSONB NOT NULL DEFAULT '[]'::jsonb,
    payload JSONB NOT NULL)"""
INDEXES = ["CREATE INDEX IF NOT EXISTS idx_diagnoses_trace ON diagnoses(trace_id)",
           "CREATE INDEX IF NOT EXISTS idx_diagnoses_project ON diagnoses(project_id, created_at DESC)"]
EVENT_COLS = ["event_id", "timestamp", "system_type", "project_id", "model_id", "trace_id", "span_id", "event_type",
              "input", "output", "latency", "status", "error", "metadata"]

_lock = threading.Lock()
_mem_events: list[dict] = []     # only used by tests / no-DSN mode
_mem_diagnoses: dict[str, dict] = {}


def dsn() -> str | None:
    if os.getenv("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    if os.getenv("POSTGRES_HOST") or os.getenv("POSTGRES_DB"):
        return (f"postgresql://{os.getenv('POSTGRES_USER', 'intellicore')}:{os.getenv('POSTGRES_PASSWORD', 'intellicore_dev')}"
                f"@{os.getenv('POSTGRES_HOST', 'localhost')}:{os.getenv('POSTGRES_PORT', '5432')}/{os.getenv('POSTGRES_DB', 'intellicore')}")
    return None


def init_storage() -> None:
    if not dsn():
        return
    import psycopg
    with psycopg.connect(dsn()) as conn:
        conn.execute(DDL)
        for ddl in INDEXES:
            conn.execute(ddl)


def _row(values) -> dict[str, Any]:
    d = dict(zip(EVENT_COLS, values))
    d["event_id"] = str(d["event_id"])
    if d.get("timestamp") is not None:
        d["timestamp"] = d["timestamp"].isoformat()
    return d


def fetch_events(project_id: str | None = None, trace_id: str | None = None, limit: int = 500) -> list[dict[str, Any]]:
    """Newest-first slice returned oldest-first; filter by trace and/or project."""
    limit = max(1, min(limit, 5000))
    if not dsn():
        with _lock:
            rows = [e for e in _mem_events if (not trace_id or e.get("trace_id") == trace_id) and (not project_id or e.get("project_id") == project_id)]
        return rows[-limit:]
    import psycopg
    where, params = [], []
    if trace_id:
        where.append("trace_id=%s"); params.append(trace_id)
    if project_id:
        where.append("project_id=%s"); params.append(project_id)
    sql = f"SELECT {','.join(EVENT_COLS)} FROM events" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY timestamp DESC LIMIT %s"
    with psycopg.connect(dsn()) as conn:
        rows = conn.execute(sql, (*params, limit)).fetchall()
    return [_row(r) for r in reversed(rows)]


def add_memory_event(event: dict[str, Any]) -> None:   # test helper
    with _lock:
        _mem_events.append(event)


def save_diagnosis(d: dict[str, Any]) -> None:
    if not dsn():
        with _lock:
            _mem_diagnoses[d["diagnosis_id"]] = d
        return
    import psycopg
    with psycopg.connect(dsn()) as conn:
        conn.execute(
            """INSERT INTO diagnoses(diagnosis_id,project_id,trace_id,root_cause,category,severity,confidence,cross_paradigm,paradigms,payload)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb)
               ON CONFLICT (diagnosis_id) DO UPDATE SET created_at=now(), severity=EXCLUDED.severity, confidence=EXCLUDED.confidence,
                    paradigms=EXCLUDED.paradigms, payload=EXCLUDED.payload""",
            (d["diagnosis_id"], d.get("project_id"), d.get("trace_id"), d["root_cause"], d.get("category"), d.get("severity"),
             d.get("confidence"), bool(d.get("cross_paradigm")), json.dumps(d.get("paradigms", [])), json.dumps(d, default=str)))


def list_diagnoses(project_id: str | None = None, trace_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    limit = max(1, min(limit, 1000))
    if not dsn():
        with _lock:
            rows = [d for d in _mem_diagnoses.values() if (not trace_id or d.get("trace_id") == trace_id) and (not project_id or d.get("project_id") == project_id)]
        return rows[-limit:][::-1]
    import psycopg
    where, params = [], []
    if trace_id:
        where.append("trace_id=%s"); params.append(trace_id)
    if project_id:
        where.append("project_id=%s"); params.append(project_id)
    sql = "SELECT payload FROM diagnoses" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY created_at DESC LIMIT %s"
    with psycopg.connect(dsn()) as conn:
        return [r[0] for r in conn.execute(sql, (*params, limit)).fetchall()]
