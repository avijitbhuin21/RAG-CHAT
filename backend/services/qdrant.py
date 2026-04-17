from uuid import UUID

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    VectorParams,
)

from ..config import settings

_client = AsyncQdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY or None,
)


async def health() -> dict:
    cols = await _client.get_collections()
    return {"collections": [c.name for c in cols.collections]}


async def ensure_collection() -> None:
    name = settings.QDRANT_COLLECTION
    cols = await _client.get_collections()
    if any(c.name == name for c in cols.collections):
        return
    await _client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=settings.EMBEDDING_DIM, distance=Distance.COSINE),
    )


def client() -> AsyncQdrantClient:
    return _client


async def delete_points_for_file(file_id: UUID) -> None:
    """Idempotent — no error if the filter matches zero points."""
    await _client.delete(
        collection_name=settings.QDRANT_COLLECTION,
        points_selector=FilterSelector(
            filter=Filter(
                must=[FieldCondition(key="file_id", match=MatchValue(value=str(file_id)))]
            )
        ),
    )
