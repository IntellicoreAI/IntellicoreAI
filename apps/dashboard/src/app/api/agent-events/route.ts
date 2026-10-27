import { NextResponse } from "next/server";

export async function GET() {
  try {
    const response = await fetch("http://127.0.0.1:8001/events", {
      cache: "no-store",
    });

    if (!response.ok) {
      return NextResponse.json(
        { error: "Agentic API is unavailable", count: 0, events: [] },
        { status: 502 }
      );
    }

    const data = await response.json();
    return NextResponse.json(data, {
      headers: { "Cache-Control": "no-store" },
    });
  } catch {
    return NextResponse.json(
      { error: "Could not connect to Agentic API", count: 0, events: [] },
      { status: 502 }
    );
  }
}
