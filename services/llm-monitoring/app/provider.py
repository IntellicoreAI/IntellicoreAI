"""OpenAI-compatible chat completion integration using only Python's standard library."""
import json, os, time
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

class ProviderError(RuntimeError):
    pass

def chat_completion(prompt: str, *, model: str | None = None, system_prompt: str | None = None,
                    temperature: float = 0.2, max_tokens: int = 512) -> dict:
    api_key = os.getenv('LLM_API_KEY') or os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ProviderError('Missing LLM_API_KEY (or OPENAI_API_KEY) environment variable.')
    base_url = os.getenv('LLM_BASE_URL', 'https://api.openai.com/v1').rstrip('/')
    model = model or os.getenv('LLM_MODEL', 'gpt-4o-mini')
    messages=[]
    if system_prompt: messages.append({'role':'system','content':system_prompt})
    messages.append({'role':'user','content':prompt})
    payload={'model':model,'messages':messages,'temperature':temperature,'max_tokens':max_tokens}
    request = Request(
        base_url + '/chat/completions',
        data=json.dumps(payload).encode(),
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'IntelliCoreAI/1.0'
        },
        method='POST'
    )
    started=time.perf_counter()
    try:
        with urlopen(request, timeout=float(os.getenv('LLM_TIMEOUT_SECONDS','45'))) as response:
            body=json.loads(response.read().decode())
    except HTTPError as exc:
        detail=exc.read().decode(errors='replace')[:1000]
        raise ProviderError(f'LLM provider returned HTTP {exc.code}: {detail}') from exc
    except (URLError, TimeoutError) as exc:
        raise ProviderError(f'LLM provider connection failed: {exc}') from exc
    latency_ms=(time.perf_counter()-started)*1000
    try:
        answer=body['choices'][0]['message']['content']
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderError('LLM provider response did not contain choices[0].message.content') from exc
    usage=body.get('usage') or {}
    return {'response':answer, 'model_id':body.get('model',model), 'latency_ms':round(latency_ms,3),
            'input_tokens':usage.get('prompt_tokens'), 'output_tokens':usage.get('completion_tokens'),
            'total_tokens':usage.get('total_tokens'), 'provider_response_id':body.get('id'),
            'raw_usage':usage}

def create_embeddings(texts: list[str], model: str | None = None) -> list[list[float]]:
    """
    Generate embeddings locally using Sentence Transformers.
    No external API key is required.
    """

    try:
        from sentence_transformers import SentenceTransformer

        model_name = model or os.getenv(
            "LOCAL_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        if not hasattr(create_embeddings, "_models"):
            create_embeddings._models = {}

        if model_name not in create_embeddings._models:
            create_embeddings._models[model_name] = SentenceTransformer(model_name)

        embedding_model = create_embeddings._models[model_name]

        vectors = embedding_model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False
        )

        return vectors.tolist()

    except Exception as exc:
        raise ProviderError(
            f"Local embedding generation failed: {exc}"
        ) from exc
