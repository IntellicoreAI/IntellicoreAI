-- Diagnosis service output. The service also runs this idempotently on startup.
CREATE TABLE IF NOT EXISTS diagnoses (
    diagnosis_id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    project_id TEXT, trace_id TEXT,
    root_cause TEXT NOT NULL, category TEXT, severity TEXT, confidence DOUBLE PRECISION,
    cross_paradigm BOOLEAN NOT NULL DEFAULT FALSE,
    paradigms JSONB NOT NULL DEFAULT '[]'::jsonb,
    payload JSONB NOT NULL);
CREATE INDEX IF NOT EXISTS idx_diagnoses_trace ON diagnoses(trace_id);
CREATE INDEX IF NOT EXISTS idx_diagnoses_project ON diagnoses(project_id, created_at DESC);
