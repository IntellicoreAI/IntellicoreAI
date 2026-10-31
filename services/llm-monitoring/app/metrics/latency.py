def latency_metrics(latency_ms: float | None) -> dict:
    if latency_ms is None: return {"latency_ms": None, "latency_band": "unknown"}
    return {"latency_ms": round(latency_ms, 3),
            "latency_band": "fast" if latency_ms < 1000 else "normal" if latency_ms < 3000 else "slow"}
