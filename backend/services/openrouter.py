import math
from typing import Iterator

from openai import AsyncOpenAI

from ..config import settings

_client = AsyncOpenAI(
    base_url=settings.OPENROUTER_BASE_URL,
    api_key=settings.OPENROUTER_API_KEY,
    default_headers={
        "HTTP-Referer": settings.FRONTEND_URL,
        "X-Title": settings.APP_NAME,
    },
)


def _l2_normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    if n == 0:
        return v
    return [x / n for x in v]


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _pack_batches(texts: list[str]) -> Iterator[list[str]]:
    max_items = settings.EMBED_MAX_BATCH_ITEMS
    max_tokens = settings.EMBED_MAX_BATCH_TOKENS
    batch: list[str] = []
    batch_tokens = 0
    for t in texts:
        tt = _estimate_tokens(t)
        if batch and (len(batch) >= max_items or batch_tokens + tt > max_tokens):
            yield batch
            batch, batch_tokens = [], 0
        batch.append(t)
        batch_tokens += tt
    if batch:
        yield batch


async def _embed_batch(batch: list[str]) -> list[list[float]]:
    """Embed one batch via OpenRouter and return L2-normalized vectors
    truncated to EMBEDDING_DIM (gemini-embedding-001 is Matryoshka-trained,
    so a prefix slice is valid if the provider ignores `dimensions`)."""
    resp = await _client.embeddings.create(
        model=settings.OPENROUTER_EMBEDDING_MODEL,
        input=batch,
        dimensions=settings.EMBEDDING_DIM,
    )
    out: list[list[float]] = []
    for d in resp.data:
        vec = d.embedding
        if len(vec) < settings.EMBEDDING_DIM:
            raise RuntimeError(
                f"embedding dim {len(vec)} < EMBEDDING_DIM {settings.EMBEDDING_DIM}"
            )
        out.append(_l2_normalize(vec[: settings.EMBEDDING_DIM]))
    return out


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed texts via OpenRouter, returning L2-normalized EMBEDDING_DIM vectors."""
    out: list[list[float]] = []
    for batch in _pack_batches(texts):
        out.extend(await _embed_batch(batch))
    return out


async def health() -> None:
    await _embed_batch(["health"])


def llm_client() -> AsyncOpenAI:
    return _client
