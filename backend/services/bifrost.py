import math
from typing import Iterator

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from ..config import settings

_client = AsyncOpenAI(base_url=settings.BIFROST_BASE_URL, api_key=settings.BIFROST_API_KEY)

# Anthropic-native client pointed at Bifrost's /anthropic endpoint. We use
# this for chat (tool use, streaming, thinking) because the OpenAI-compat
# translator in Bifrost drops key semantics — notably it returns an empty
# response on the post-tool-result follow-up call. The native protocol
# works reliably. Embeddings still flow through the OpenAI-compat client
# above because Bifrost's Gemini embedding bridge is exposed there.
_anthropic_client = AsyncAnthropic(
    base_url=settings.bifrost_anthropic_base_url,
    api_key=settings.BIFROST_API_KEY,
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


async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed texts via Bifrost. Returns L2-normalized 1536-dim vectors
    (Bifrost returns un-normalized vectors when `dimensions` < 3072 — Qdrant
    cosine assumes unit norm, so we normalize here)."""
    out: list[list[float]] = []
    for batch in _pack_batches(texts):
        resp = await _client.embeddings.create(
            model=settings.BIFROST_EMBEDDING_MODEL,
            input=batch,
            dimensions=settings.EMBEDDING_DIM,
        )
        for d in resp.data:
            out.append(_l2_normalize(d.embedding))
    return out


async def health() -> None:
    await _client.embeddings.create(
        model=settings.BIFROST_EMBEDDING_MODEL,
        input="health",
        dimensions=settings.EMBEDDING_DIM,
    )


def llm_client() -> AsyncOpenAI:
    return _client


def anthropic_client() -> AsyncAnthropic:
    return _anthropic_client
