CREATE TABLE IF NOT EXISTS events (
    event_id UUID PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,

    system_type VARCHAR(10) NOT NULL
        CHECK (system_type IN ('LLM', 'AGENT', 'ML')),

    project_id VARCHAR(255) NOT NULL,
    model_id VARCHAR(255),

    trace_id VARCHAR(255) NOT NULL,
    span_id VARCHAR(255) NOT NULL,

    event_type VARCHAR(255) NOT NULL,

    input JSONB,
    output JSONB,

    latency DOUBLE PRECISION
        CHECK (latency IS NULL OR latency >= 0),

    status VARCHAR(50) NOT NULL,

    error TEXT,

    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_events_project_id
    ON events (project_id);

CREATE INDEX IF NOT EXISTS idx_events_trace_id
    ON events (trace_id);

CREATE INDEX IF NOT EXISTS idx_events_timestamp
    ON events (timestamp);

CREATE INDEX IF NOT EXISTS idx_events_system_type
    ON events (system_type);