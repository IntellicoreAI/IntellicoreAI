
import { NextRequest, NextResponse } from "next/server";

const API_URL = process.env.AGENT_MONITORING_API_URL || "http://127.0.0.1:8001";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const task = typeof body.task === "string" ? body.task.trim() : "";

    if (!task) {
      return NextResponse.json(
        { error: "Enter a task before starting an agent execution." },
        { status: 400 }
      );
    }

    const response = await fetch(`${API_URL}/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task }),
      cache: "no-store",
    });

    const raw = await response.text();
    let data: any;

    try {
      data = raw ? JSON.parse(raw) : {};
    } catch {
      return NextResponse.json(
        { error: "Agent Monitoring returned an invalid response." },
        { status: 502 }
      );
    }

    return NextResponse.json(data, {
      status: response.status,
      headers: { "Cache-Control": "no-store" },
    });
  } catch (error) {
    return NextResponse.json(
      {
        error: error instanceof Error
          ? error.message
          : "Could not connect to Agent Monitoring.",
      },
      { status: 502 }
    );
  }
}
