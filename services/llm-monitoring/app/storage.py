"""PostgreSQL persistence with an in-memory fallback for isolated tests/development."""
import os
from threading import Lock
from typing import Any
from uuid import uuid4

_memory_events: list[dict[str, Any]] = []
_memory_feedback: list[dict[str, Any]] = []
_memory_alerts: list[dict[str, Any]] = []
_lock = Lock()


def _dsn():
    return os.getenv("DATABASE_URL")


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def init_storage() -> None:
    """Create LLM-owned tables. Does not alter shared or Agentic AI tables."""
    if not _dsn():
        return
    import psycopg
    with psycopg.connect(_dsn()) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS llm_feedback (
            feedback_id UUID PRIMARY KEY, event_id UUID NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
            vote VARCHAR(8) NOT NULL CHECK (vote IN ('up','down')), issue_category TEXT,
            feedback_text TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT NOW())""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_feedback_event_id ON llm_feedback(event_id)")
        conn.execute("""CREATE TABLE IF NOT EXISTS llm_alerts (
            alert_id UUID PRIMARY KEY, event_id UUID, project_id VARCHAR(255) NOT NULL,
            severity VARCHAR(20) NOT NULL, alert_type VARCHAR(100) NOT NULL,
            message TEXT NOT NULL, details JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), acknowledged BOOLEAN NOT NULL DEFAULT FALSE)""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_alerts_project_created ON llm_alerts(project_id, created_at DESC)")


def save_event(event: dict[str, Any]) -> None:
    event = _jsonable(event)
    if not _dsn():
        with _lock: _memory_events.append(event)
        return
    import psycopg
    from psycopg.types.json import Jsonb
    with psycopg.connect(_dsn()) as conn:
        conn.execute("""INSERT INTO events(event_id,timestamp,system_type,project_id,model_id,trace_id,span_id,event_type,input,output,latency,status,error,metadata)
        VALUES (%s,%s,'LLM',%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(event_id) DO NOTHING""",
        (event['event_id'], event['timestamp'], event['project_id'], event.get('model_id'), event['trace_id'], event['span_id'], event['event_type'], Jsonb(event.get('input')), Jsonb(event.get('output')), event.get('latency'), event['status'], event.get('error'), Jsonb(event.get('metadata', {}))))


def list_events(project_id: str | None = None, limit: int = 100) -> list[dict]:
    if not _dsn():
        with _lock: rows = list(_memory_events)
        rows = [r for r in rows if not project_id or r.get('project_id') == project_id]
        return list(reversed(rows[-limit:]))
    import psycopg
    with psycopg.connect(_dsn()) as conn:
        sql = "SELECT event_id,timestamp,system_type,project_id,model_id,trace_id,span_id,event_type,input,output,latency,status,error,metadata FROM events WHERE system_type='LLM'"
        args: list[Any] = []
        if project_id:
            sql += " AND project_id=%s"; args.append(project_id)
        sql += " ORDER BY timestamp DESC LIMIT %s"; args.append(limit)
        rows = conn.execute(sql, args).fetchall()
    keys = ['event_id','timestamp','system_type','project_id','model_id','trace_id','span_id','event_type','input','output','latency','status','error','metadata']
    return [dict(zip(keys, row)) for row in rows]


def event_exists(event_id: str) -> bool:
    if not _dsn():
        with _lock: return any(str(e.get('event_id')) == str(event_id) for e in _memory_events)
    import psycopg
    with psycopg.connect(_dsn()) as conn:
        return conn.execute("SELECT 1 FROM events WHERE event_id=%s AND system_type='LLM'", (event_id,)).fetchone() is not None


def save_feedback(feedback: dict) -> dict:
    row = {'feedback_id': str(uuid4()), **feedback}
    if not _dsn():
        with _lock: _memory_feedback.append(row)
        return row
    import psycopg
    with psycopg.connect(_dsn()) as conn:
        conn.execute("INSERT INTO llm_feedback(feedback_id,event_id,vote,issue_category,feedback_text) VALUES (%s,%s,%s,%s,%s)",
                     (row['feedback_id'], row['event_id'], row['vote'], row.get('issue_category'), row.get('feedback_text')))
    return row


def list_feedback(event_id: str | None = None) -> list[dict]:
    if not _dsn():
        with _lock: rows = list(_memory_feedback)
        return [r for r in rows if not event_id or str(r['event_id']) == str(event_id)]
    import psycopg
    with psycopg.connect(_dsn()) as conn:
        if event_id:
            rows = conn.execute("SELECT feedback_id,event_id,vote,issue_category,feedback_text,created_at FROM llm_feedback WHERE event_id=%s ORDER BY created_at DESC", (event_id,)).fetchall()
        else:
            rows = conn.execute("SELECT feedback_id,event_id,vote,issue_category,feedback_text,created_at FROM llm_feedback ORDER BY created_at DESC LIMIT 1000").fetchall()
    keys = ['feedback_id','event_id','vote','issue_category','feedback_text','created_at']
    return [dict(zip(keys, r)) for r in rows]


def save_alert(alert: dict) -> dict:
    row = {'alert_id': str(uuid4()), **alert}
    if not _dsn():
        with _lock: _memory_alerts.append(row)
        return row
    import psycopg
    from psycopg.types.json import Jsonb
    with psycopg.connect(_dsn()) as conn:
        conn.execute("INSERT INTO llm_alerts(alert_id,event_id,project_id,severity,alert_type,message,details) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                     (row['alert_id'], row.get('event_id'), row['project_id'], row['severity'], row['alert_type'], row['message'], Jsonb(row.get('details', {}))))
    return row


def list_alerts(project_id: str | None = None, limit: int = 100) -> list[dict]:
    if not _dsn():
        with _lock: rows = list(_memory_alerts)
        rows = [r for r in rows if not project_id or r['project_id'] == project_id]
        return list(reversed(rows[-limit:]))
    import psycopg
    with psycopg.connect(_dsn()) as conn:
        if project_id:
            rows = conn.execute("SELECT alert_id,event_id,project_id,severity,alert_type,message,details,created_at,acknowledged FROM llm_alerts WHERE project_id=%s ORDER BY created_at DESC LIMIT %s", (project_id,limit)).fetchall()
        else:
            rows = conn.execute("SELECT alert_id,event_id,project_id,severity,alert_type,message,details,created_at,acknowledged FROM llm_alerts ORDER BY created_at DESC LIMIT %s", (limit,)).fetchall()
    keys=['alert_id','event_id','project_id','severity','alert_type','message','details','created_at','acknowledged']
    return [dict(zip(keys,r)) for r in rows]
