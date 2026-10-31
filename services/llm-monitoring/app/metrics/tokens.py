def token_metrics(input_tokens: int | None, output_tokens: int | None) -> dict:
    total = input_tokens + output_tokens if input_tokens is not None and output_tokens is not None else None
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": total}
