def cost_metrics(cost_usd: float | None, input_tokens: int | None, output_tokens: int | None) -> dict:
    total = (input_tokens or 0) + (output_tokens or 0)
    return {"cost_usd": round(cost_usd, 8) if cost_usd is not None else None,
            "cost_per_1k_tokens_usd": round(cost_usd * 1000 / total, 8) if cost_usd is not None and total else None}
