"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

type AgentEvent = {
  event_id?: string;
  timestamp?: string;
  system_type?: string;
  project_id?: string;
  model_id?: string | null;
  trace_id?: string;
  span_id?: string;
  event_type?: string;
  status?: string;
  latency?: number | null;
  error?: string | null;
  metadata?: Record<string, unknown>;
};

type Tab =
  | "Overview"
  | "Agent Monitoring"
  | "Upload Agent"
  | "Live Executions"
  | "Trace Explorer"
  | "Diagnosis & Recommendations"
  | "Analytics"
  | "Settings";

const tabs: { name: Tab; icon: string }[] = [
  { name: "Overview", icon: "◫" },
  { name: "Agent Monitoring", icon: "⌘" },
  { name: "Upload Agent", icon: "↑" },
  { name: "Live Executions", icon: "↗" },
  { name: "Trace Explorer", icon: "◎" },
  { name: "Diagnosis & Recommendations", icon: "✧" },
  { name: "Analytics", icon: "▥" },
  { name: "Settings", icon: "⚙" },
];

function prettyDate(value?: string) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function statusClass(value?: string) {
  const status = (value || "").toLowerCase();
  if (["success", "completed", "successfully_completed"].includes(status)) return "success";
  if (["failed", "error", "failure"].includes(status)) return "danger";
  if (["running", "pending", "in_progress"].includes(status)) return "running";
  return "neutral";
}

export default function Home() {
  const [activeTab, setActiveTab] = useState<Tab>("Overview");
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [apiError, setApiError] = useState("");
  const [task, setTask] = useState("");
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState("");
  const [runResult, setRunResult] = useState<any>(null);
  const [selectedTrace, setSelectedTrace] = useState("");
  const [lastUpdated, setLastUpdated] = useState("");

  const loadEvents = useCallback(async () => {
    try {
      const response = await fetch("/api/agent-events", { cache: "no-store" });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Unable to retrieve agent events.");
      }

      const nextEvents = Array.isArray(data.events) ? data.events : [];
      setEvents(nextEvents);
      setApiError("");
      setLastUpdated(new Date().toLocaleTimeString());
    } catch (error) {
      setApiError(
        error instanceof Error ? error.message : "Agent API is unavailable."
      );
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadEvents();
    const timer = window.setInterval(loadEvents, 5000);
    return () => window.clearInterval(timer);
  }, [loadEvents]);

  const metrics = useMemo(() => {
    const traces = new Set(
      events.map((event) => event.trace_id).filter(Boolean)
    );
    const failed = events.filter((event) =>
      ["failed", "error", "failure"].includes((event.status || "").toLowerCase())
    ).length;
    const completed = events.filter((event) =>
      ["success", "completed", "successfully_completed"].includes(
        (event.status || "").toLowerCase()
      )
    ).length;
    const latencies = events
      .map((event) => event.latency)
      .filter((value): value is number => typeof value === "number");
    const averageLatency = latencies.length
      ? Math.round(latencies.reduce((sum, value) => sum + value, 0) / latencies.length)
      : null;

    return {
      total: events.length,
      traces: traces.size,
      failed,
      completed,
      averageLatency,
    };
  }, [events]);

  const traces = useMemo(() => {
    const map = new Map<string, AgentEvent[]>();
    events.forEach((event) => {
      const id = event.trace_id || "Trace ID unavailable";
      map.set(id, [...(map.get(id) || []), event]);
    });
    return Array.from(map.entries()).map(([id, items]) => ({ id, items }));
  }, [events]);

  const selectedTraceEvents = traces.find((trace) => trace.id === selectedTrace)?.items || [];

  async function runAgent() {
    const cleanTask = task.trim();
    if (!cleanTask || running) return;

    setRunning(true);
    setRunError("");
    setRunResult(null);

    try {
      const response = await fetch("/api/agent-run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task: cleanTask }),
      });
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || data.detail || "Agent execution failed.");
      }

      setRunResult(data);
      setTask("");
      await loadEvents();
      setActiveTab("Live Executions");
    } catch (error) {
      setRunError(error instanceof Error ? error.message : "Could not run the agent.");
    } finally {
      setRunning(false);
    }
  }

  function EventTable({ rows }: { rows: AgentEvent[] }) {
    if (!rows.length) {
      return (
        <div className="empty-state">
          <div className="empty-symbol">⌁</div>
          <h3>{loading ? "Loading agent events" : "No events available yet"}</h3>
          <p>
            {loading
              ? "Connecting to your Agent Monitoring service."
              : "Run an agent task to generate real telemetry and execution events."}
          </p>
        </div>
      );
    }

    return (
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Event</th>
              <th>Status</th>
              <th>Trace ID</th>
              <th>Latency</th>
              <th>Timestamp</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((event, index) => (
              <tr key={event.event_id || `${event.trace_id}-${event.span_id}-${index}`}>
                <td>
                  <strong>{event.event_type || "Agent event"}</strong>
                  <small>{event.project_id || event.system_type || "Agentic AI"}</small>
                </td>
                <td>
                  <span className={`pill ${statusClass(event.status)}`}>
                    {event.status || "Unknown"}
                  </span>
                </td>
                <td className="mono">{event.trace_id || "—"}</td>
                <td>{event.latency != null ? `${event.latency} ms` : "—"}</td>
                <td>{prettyDate(event.timestamp)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  }

  function PageHeading({ eyebrow, title, description }: {
    eyebrow: string; title: string; description: string;
  }) {
    return (
      <div className="page-heading">
        <div>
          <p className="eyebrow">{eyebrow}</p>
          <h1>{title}</h1>
          <p className="subtitle">{description}</p>
        </div>
        <div className="connection-chip">
          <span className={`connection-dot ${apiError ? "offline" : ""}`} />
          {apiError ? "API unavailable" : "Agent API"}
        </div>
      </div>
    );
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">IC</div>
          <div>
            <strong>IntelliCore</strong>
            <span>AI OBSERVABILITY</span>
          </div>
        </div>

        <div className="workspace-label">WORKSPACE</div>
        <div className="project-switch">
          <div className="project-avatar">I</div>
          <div>
            <small>PROJECT</small>
            <strong>IntelliCore AI</strong>
          </div>
          <span className="project-chevron">⌄</span>
        </div>

        <div className="nav-label">MONITORING</div>
        <nav className="navigation">
          {tabs.map((tab) => (
            <button
              key={tab.name}
              className={`nav-item ${activeTab === tab.name ? "active" : ""}`}
              onClick={() => setActiveTab(tab.name)}
            >
              <span className="nav-icon">{tab.icon}</span>
              <span>{tab.name}</span>
              {tab.name === "Live Executions" && metrics.total > 0 && (
                <span className="nav-count">{metrics.total}</span>
              )}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="environment-row">
            <span className="environment-dot" />
            Development environment
          </div>
          <p>Agentic AI monitoring workspace</p>
        </div>
      </aside>

      <section className="main-area">
        <header className="topbar">
          <div className="breadcrumb">Workspace <span>/</span> {activeTab}</div>
          <div className="topbar-actions">
            <span className="topbar-status">
              <span className={`connection-dot ${apiError ? "offline" : ""}`} />
              {apiError ? "Disconnected" : "Connected"}
            </span>
            <button className="refresh-button" onClick={loadEvents}>↻ Refresh</button>
          </div>
        </header>

        <div className="content">
          {activeTab === "Overview" && (
            <>
              <PageHeading
                eyebrow="AGENTIC AI OBSERVABILITY"
                title="Agent workspace"
                description="Monitor agent activity, inspect executions and understand workflow behaviour."
              />

              {apiError && (
                <div className="alert alert-error">
                  <strong>Unable to reach Agent Monitoring</strong>
                  <span>{apiError}</span>
                </div>
              )}

              <div className="metric-grid">
                <article className="metric-card">
                  <span className="metric-label">Total events</span>
                  <strong>{loading ? "—" : metrics.total}</strong>
                  <span className="metric-note">Persisted monitoring events</span>
                  <span className="metric-icon">⌁</span>
                </article>
                <article className="metric-card">
                  <span className="metric-label">Unique traces</span>
                  <strong>{loading ? "—" : metrics.traces}</strong>
                  <span className="metric-note">Distinct trace identifiers</span>
                  <span className="metric-icon">◎</span>
                </article>
                <article className="metric-card">
                  <span className="metric-label">Completed events</span>
                  <strong>{loading ? "—" : metrics.completed}</strong>
                  <span className="metric-note">Events marked successful</span>
                  <span className="metric-icon">✓</span>
                </article>
                <article className="metric-card">
                  <span className="metric-label">Failed events</span>
                  <strong>{loading ? "—" : metrics.failed}</strong>
                  <span className="metric-note">Events marked failed or error</span>
                  <span className="metric-icon">!</span>
                </article>
              </div>

              <section className="panel run-panel">
                <div className="panel-title">
                  <div>
                    <p className="eyebrow">EXECUTE YOUR AGENT</p>
                    <h2>Start a new execution</h2>
                    <p>Send a task to the configured Agentic AI workflow.</p>
                  </div>
                  <div className="run-icon">✧</div>
                </div>
                <textarea
                  value={task}
                  onChange={(event) => setTask(event.target.value)}
                  placeholder="Describe the task you want your agent to perform..."
                  maxLength={4000}
                  rows={4}
                />
                <div className="run-controls">
                  <span>{task.length}/4000 characters</span>
                  <button className="primary-button" onClick={runAgent} disabled={!task.trim() || running}>
                    {running ? "Running agent..." : "▶  Run agent"}
                  </button>
                </div>
                {runError && <div className="inline-error">{runError}</div>}
                {runResult && (
                  <div className="result-box">
                    <strong>Execution response</strong>
                    <pre>{JSON.stringify(runResult, null, 2)}</pre>
                  </div>
                )}
              </section>

              <section className="panel">
                <div className="panel-heading">
                  <div>
                    <h2>Recent agent events</h2>
                    <p>Latest telemetry retrieved from your backend</p>
                  </div>
                  <button className="subtle-button" onClick={() => setActiveTab("Live Executions")}>
                    View all →
                  </button>
                </div>
                <EventTable rows={events.slice(0, 6)} />
                <div className="panel-footnote">
                  Auto-refresh every 5 seconds · Last checked: {lastUpdated || "Not yet"}
                </div>
              </section>
            </>
          )}

          {activeTab === "Agent Monitoring" && (
            <>
              <PageHeading eyebrow="OBSERVABILITY" title="Agent monitoring" description="Inspect persisted agent telemetry and workflow status." />
              <div className="metric-grid compact">
                <article className="metric-card"><span className="metric-label">Events</span><strong>{metrics.total}</strong><span className="metric-note">From monitoring API</span></article>
                <article className="metric-card"><span className="metric-label">Traces</span><strong>{metrics.traces}</strong><span className="metric-note">Unique trace IDs</span></article>
                <article className="metric-card"><span className="metric-label">Average latency</span><strong>{metrics.averageLatency == null ? "—" : `${metrics.averageLatency} ms`}</strong><span className="metric-note">Only events with latency data</span></article>
              </div>
              <section className="panel">
                <div className="panel-heading"><div><h2>Agent telemetry</h2><p>Live event feed</p></div><span className="live-label">● LIVE POLLING</span></div>
                <EventTable rows={events} />
              </section>
            </>
          )}

          {activeTab === "Upload Agent" && (
            <>
              <PageHeading eyebrow="AGENT MANAGEMENT" title="Upload agent" description="Connect an agent implementation to the monitoring workspace." />
              <section className="panel upload-panel">
                <div className="upload-illustration">↑</div>
                <h2>Agent upload is not enabled yet</h2>
                <p>
                  The current backend runs its configured Agentic AI graph through the
                  existing <code>/run</code> endpoint. It does not yet accept, validate,
                  or safely execute arbitrary uploaded agent files.
                </p>
                <div className="upload-requirements">
                  <strong>What is needed to enable uploads?</strong>
                  <ul>
                    <li>A defined agent package format and manifest</li>
                    <li>File validation and dependency handling</li>
                    <li>Isolated execution with resource and network controls</li>
                    <li>Agent registration and lifecycle management APIs</li>
                  </ul>
                </div>
                <span className="pill neutral">Not available in current backend</span>
              </section>
            </>
          )}

          {activeTab === "Live Executions" && (
            <>
              <PageHeading eyebrow="RUNTIME ACTIVITY" title="Live executions" description="Review events recorded by the Agentic AI monitoring service." />
              <section className="panel">
                <div className="panel-heading"><div><h2>Execution event stream</h2><p>{metrics.total} persisted event(s) currently returned by the API</p></div><span className="live-label">● 5 SEC REFRESH</span></div>
                <EventTable rows={events} />
              </section>
              {runResult && <section className="panel"><h2>Latest run response</h2><pre className="result-pre">{JSON.stringify(runResult, null, 2)}</pre></section>}
            </>
          )}

          {activeTab === "Trace Explorer" && (
            <>
              <PageHeading eyebrow="TRACE ANALYSIS" title="Trace explorer" description="Group events by trace ID and inspect their recorded spans." />
              <section className="panel">
                <div className="panel-heading"><div><h2>Available traces</h2><p>Trace groups are derived from event records.</p></div><span className="section-count">{traces.length} traces</span></div>
                {!traces.length ? (
                  <div className="empty-state"><div className="empty-symbol">◎</div><h3>No traces to explore</h3><p>Trace details will appear when the backend returns events with trace IDs.</p></div>
                ) : (
                  <div className="trace-layout">
                    <div className="trace-list">
                      {traces.map((trace) => (
                        <button key={trace.id} className={`trace-item ${selectedTrace === trace.id ? "selected" : ""}`} onClick={() => setSelectedTrace(trace.id)}>
                          <span className="trace-name">{trace.id}</span>
                          <small>{trace.items.length} event(s)</small>
                        </button>
                      ))}
                    </div>
                    <div className="trace-detail">
                      <h3>{selectedTrace || "Select a trace"}</h3>
                      {selectedTrace ? <EventTable rows={selectedTraceEvents} /> : <p>Choose a trace to view its events.</p>}
                    </div>
                  </div>
                )}
              </section>
            </>
          )}

          {activeTab === "Diagnosis & Recommendations" && (
            <>
              <PageHeading eyebrow="AGENT INTELLIGENCE" title="Diagnosis & recommendations" description="Review diagnosis and recommendations returned by the agent workflow." />
              <section className="panel">
                <div className="panel-heading"><div><h2>Latest workflow analysis</h2><p>Content from the most recent successful API response, when available.</p></div></div>
                {runResult ? (
                  <div className="analysis-grid">
                    <div className="analysis-card"><span>Diagnosis</span><pre>{JSON.stringify(runResult.diagnosis ?? "No diagnosis field was returned.", null, 2)}</pre></div>
                    <div className="analysis-card"><span>Recommendation</span><pre>{JSON.stringify(runResult.recommendation ?? "No recommendation field was returned.", null, 2)}</pre></div>
                  </div>
                ) : (
                  <div className="empty-state"><div className="empty-symbol">✧</div><h3>No workflow analysis available</h3><p>Run an agent task first. This page will display diagnosis and recommendations only if the backend returns them.</p></div>
                )}
              </section>
            </>
          )}

          {activeTab === "Analytics" && (
            <>
              <PageHeading eyebrow="OBSERVABILITY INSIGHTS" title="Agent analytics" description="A summary calculated from the monitoring events currently returned by your API." />
              <div className="metric-grid">
                <article className="metric-card"><span className="metric-label">Total events</span><strong>{metrics.total}</strong><span className="metric-note">All returned records</span></article>
                <article className="metric-card"><span className="metric-label">Unique traces</span><strong>{metrics.traces}</strong><span className="metric-note">Distinct trace IDs</span></article>
                <article className="metric-card"><span className="metric-label">Average latency</span><strong>{metrics.averageLatency == null ? "—" : `${metrics.averageLatency} ms`}</strong><span className="metric-note">Available latency values only</span></article>
                <article className="metric-card"><span className="metric-label">Failed events</span><strong>{metrics.failed}</strong><span className="metric-note">Status-based count</span></article>
              </div>
              <section className="panel">
                <div className="panel-heading"><div><h2>Event status breakdown</h2><p>Counts are calculated from current event records.</p></div></div>
                {events.length ? (
                  <div className="status-breakdown">
                    {Object.entries(events.reduce<Record<string, number>>((acc, event) => {
                      const key = event.status || "Unknown";
                      acc[key] = (acc[key] || 0) + 1;
                      return acc;
                    }, {})).map(([name, count]) => (
                      <div className="breakdown-row" key={name}>
                        <span>{name}</span><strong>{count}</strong>
                        <div className="breakdown-track"><div style={{ width: `${(count / events.length) * 100}%` }} /></div>
                      </div>
                    ))}
                  </div>
                ) : <div className="empty-state"><h3>No analytics data yet</h3><p>Analytics will populate when real events are available.</p></div>}
              </section>
            </>
          )}

          {activeTab === "Settings" && (
            <>
              <PageHeading eyebrow="WORKSPACE CONFIGURATION" title="Settings" description="Review the current local development connections." />
              <section className="panel settings-panel">
                <div className="setting-row"><div><strong>Agent Monitoring API</strong><p>Backend service used for event retrieval and agent execution.</p></div><code>http://127.0.0.1:8001</code></div>
                <div className="setting-row"><div><strong>Events endpoint</strong><p>Persisted agent event records.</p></div><code>GET /events</code></div>
                <div className="setting-row"><div><strong>Execution endpoint</strong><p>Submits a task to the configured agent graph.</p></div><code>POST /run</code></div>
                <div className="setting-row"><div><strong>Dashboard polling</strong><p>Automatic event refresh interval.</p></div><code>5 seconds</code></div>
                <div className="setting-row"><div><strong>Connection status</strong><p>{apiError || "Latest event request status."}</p></div><span className={`pill ${apiError ? "danger" : "success"}`}>{apiError ? "Unavailable" : "Connected"}</span></div>
              </section>
            </>
          )}

          <footer className="footer">
            <span>INTELLICORE AI <i>/</i> AGENTIC AI OBSERVABILITY</span>
            <span>Development preview · Real backend data</span>
          </footer>
        </div>
      </section>
    </main>
  );
}
