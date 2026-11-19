import { NextResponse } from "next/server";

const API_URL =
  process.env.AGENT_MONITORING_API_URL || "http://127.0.0.1:8001";

export async function GET() {
  try {
    const response = await fetch(`${API_URL}/events`, {
      cache: "no-store",
    });

    const raw = await response.text();
    let data: any;

    try {
      data = raw ? JSON.parse(raw) : [];
    } catch {
      return NextResponse.json(
        { error: "Agent Monitoring returned an invalid response." },
        { status: 502 }
      );
    }

    if (!response.ok) {
      return NextResponse.json(
        {
          error:
            data?.detail ||
            data?.error ||
            "Could not load agent events.",
        },
        { status: response.status }
      );
    }

    const events = Array.isArray(data)
      ? data
      : Array.isArray(data?.events)
        ? data.events
        : [];

    return NextResponse.json(
      { count: events.length, events },
      { headers: { "Cache-Control": "no-store" } }
    );
  } catch (error) {
    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Agent Monitoring service is unavailable.",
      },
      { status: 502 }
    );
  }
}
