
"use client";

import { useEffect, useState } from "react";

type AgentEvent = {
  event_id: string;
  timestamp: string;
  system_type: string;
  project_id: string;
  model_id?: string | null;
  trace_id: string;
  span_id: string;
  event_type: string;
  status: string;
  latency?: number | null;
  error?: string | null;
  metadata?: Record<string, unknown>;
};

type EventsResponse = {
  count: number;
  events: AgentEvent[];
  error?: string;
};

const components = [
  {
    name: "LLM Monitoring",
    description: "Quality, latency, tokens, cost and safety signals",
    status: "Integration pending",
    detail: "Monitoring implementation reported by the LLM team",
  },
  {
    name: "Agentic AI",
    description: "Agent traces, tool execution and workflow failures",
    status: "API connected",
    detail: "Fetching persisted agent events from the monitoring API",
  },
  {
    name: "ML Monitoring",
    description: "Data quality, drift, performance and fairness",
    status: "Artifacts available",
    detail: "Reports generated; ingestion persistence is pending",
  },
];

const stages = [
  {
    name: "Signal generation",
    detail: "Individual LLM, Agentic AI and ML monitoring",
  },
  {
    name: "Unified ingestion",
    detail: "Normalize, validate and persist common events",
  },
  {
    name: "Cross-system diagnosis",
    detail: "Correlate related signals across AI components",
  },
  {
    name: "Recommendations",
    detail: "Suggest actions with appropriate human review",
  },
];

function Status({ children }: { children: string }) {
  return <span className="status">{children}</span>;
}

function formatDate(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

export default function Home() {
  const [events, setEvents] = useState<AgentEvent[]>([]);
  const [eventCount, setEventCount] = useState(0);
  const [loading, setLoading] = useState(true);
  const [apiError, setApiError] = useState("");

  useEffect(() => {
    let active = true;

    async function loadEvents() {
      try {
        const response = await fetch("/api/agent-events", {
          cache: "no-store",
        });

        const data: EventsResponse = await response.json();

        if (!response.ok) {
          throw new Error(data.error || "Unable to retrieve agent events");
        }

        if (active) {
          setEvents(Array.isArray(data.events) ? data.events : []);
          setEventCount(
            typeof data.count === "number"
              ? data.count
              : Array.isArray(data.events)
                ? data.events.length
                : 0
          );
          setApiError("");
        }
      } catch (error) {
        if (active) {
          setApiError(
            error instanceof Error
              ? error.message
              : "Unable to connect to the Agentic AI API"
          );
        }
      } finally {
        if (active) setLoading(false);
      }
    }

    loadEvents();

    const interval = window.setInterval(loadEvents, 15000);

    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, []);

  return (
    <main className="shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">IC</div>
          <div>
            <strong>IntelliCore</strong>
            <span>AI OBSERVABILITY</span>
          </div>
        </div>

        <p className="nav-label">WORKSPACE</p>
        <a className="nav-item active" href="#overview">
          ◫ <span>Project overview</span>
        </a>
        <a className="nav-item" href="#components">
          ◇ <span>AI components</span>
        </a>
        <a className="nav-item" href="#pipeline">
          ⌁ <span>Integration pipeline</span>
        </a>

        <div className="sidebar-bottom">
          <div className="environment-label">ENVIRONMENT</div>
          <div className="environment-value">
            <span className="dot" /> Development
          </div>
          <p>Telemetry is shown only when connected to a real data source.</p>
        </div>
      </aside>

      <section className="workspace" id="overview">
        <header className="topbar">
          <div className="breadcrumb">
            Workspace <span>/</span> Project overview
          </div>
          <div className="topbar-right">
            <span className="live-dot" /> Development environment
          </div>
        </header>

        <div className="content">
          <div className="heading-row">
            <div>
              <p className="eyebrow">UNIFIED AI INTELLIGENCE</p>
              <h1>Project overview</h1>
              <p className="subtitle">
                One view of your AI systems, monitoring signals and integration
                readiness.
              </p>
            </div>

            <div className="project-pill">
              <span className="project-icon">P</span>
              <span>
                <small>PROJECT</small>
                <strong>IntelliCore AI</strong>
              </span>
              <span className="chevron">⌄</span>
            </div>
          </div>

          <div className="notice">
            <span className="notice-icon">i</span>
            <div>
              <strong>Integration is in progress</strong>
              <p>
                Agentic AI events are loaded from the monitoring API. LLM and ML
                integrations are still in progress; this is not a production
                health assessment.
              </p>
            </div>
          </div>

          <div className="section-heading" id="components">
            <div>
              <h2>AI components</h2>
              <p>Monitoring readiness across the project</p>
            </div>
            <span className="section-tag">3 components tracked</span>
          </div>

          <div className="component-grid">
            {components.map((component, index) => (
              <article className="component-card" key={component.name}>
                <div className="card-top">
                  <div className={`component-icon icon-${index}`}>
                    {["✳", "⌘", "▥"][index]}
                  </div>
                  <Status>{component.status}</Status>
                </div>
                <h3>{component.name}</h3>
                <p className="card-description">{component.description}</p>
                <div className="card-divider" />
                <p className="card-detail">{component.detail}</p>
                <a className="text-link" href="#pipeline">
                  View integration status <span>→</span>
                </a>
              </article>
            ))}
          </div>

          <div className="lower-grid">
            <section className="panel" id="pipeline">
              <div className="panel-heading">
                <div>
                  <h2>Integration pipeline</h2>
                  <p>Planned path to unified intelligence</p>
                </div>
                <span className="muted-tag">Implementation status</span>
              </div>

              <div className="stage-list">
                {stages.map((stage, index) => (
                  <div className="stage" key={stage.name}>
                    <div
                      className={`stage-number ${
                        index === 0 ? "stage-ready" : ""
                      }`}
                    >
                      {String(index + 1).padStart(2, "0")}
                    </div>
                    <div className="stage-copy">
                      <strong>{stage.name}</strong>
                      <p>{stage.detail}</p>
                    </div>
                    <span
                      className={`stage-state ${
                        index === 0 ? "state-started" : ""
                      }`}
                    >
                      {index === 0 ? "In progress" : "Pending"}
                    </span>
                  </div>
                ))}
              </div>
            </section>

            <section className="panel">
              <div className="panel-heading">
                <div>
                  <h2>Signals &amp; events</h2>
                  <p>Agentic AI event stream</p>
                </div>
                <span className="event-icon">⌁</span>
              </div>

              <div className="empty-state">
                {loading ? (
                  <>
                    <div className="empty-icon">⌁</div>
                    <strong>Loading events...</strong>
                    <p>Retrieving events from the Agentic AI monitoring API.</p>
                  </>
                ) : apiError ? (
                  <>
                    <div className="empty-icon">!</div>
                    <strong>Unable to load events</strong>
                    <p>{apiError}</p>
                  </>
                ) : events.length === 0 ? (
                  <>
                    <div className="empty-icon">⌁</div>
                    <strong>No agent events found</strong>
                    <p>
                      The API is connected, but no persisted events were
                      returned.
                    </p>
                  </>
                ) : (
                  <>
                    <div className="empty-icon">✓</div>
                    <strong>{eventCount} persisted event(s)</strong>
                    <p>Real events retrieved from the monitoring API.</p>
                  </>
                )}
              </div>

              {!loading && !apiError && events.length > 0 && (
                <div className="stage-list">
                  {events.slice(0, 5).map((event) => (
                    <div className="stage" key={event.event_id}>
                      <div className="stage-number stage-ready">AG</div>
                      <div className="stage-copy">
                        <strong>{event.event_type || "Agent event"}</strong>
                        <p>
                          Status: {event.status} · Project: {event.project_id}
                        </p>
                        <p>Trace: {event.trace_id}</p>
                        <p>{formatDate(event.timestamp)}</p>
                        {event.error && <p>Error: {event.error}</p>}
                      </div>
                      <span className="stage-state">
                        {event.latency != null
                          ? `${event.latency} ms`
                          : event.status}
                      </span>
                    </div>
                  ))}
                </div>
              )}

              <div className="panel-footer">
                <span className="footer-dot" />
                {apiError
                  ? "Agentic API connection unavailable"
                  : loading
                    ? "Checking event connection"
                    : "Refreshes every 15 seconds"}
              </div>
            </section>
          </div>

          <footer className="footer">
            <span>
              INTELLICORE AI <span className="footer-separator">/</span> UNIFIED
              MONITORING
            </span>
            <span>Development preview · Agentic events connected</span>
          </footer>
        </div>
      </section>
    </main>
  );
}
