import asyncio
from collections import defaultdict
from typing import AsyncIterator


class ProgressBroker:
    """In-process pub/sub for ingest progress events. Single backend process,
    so no Redis needed; restart drops in-flight subscribers (acceptable)."""

    def __init__(self) -> None:
        self._per_file: dict[str, set[asyncio.Queue]] = defaultdict(set)
        self._global: set[asyncio.Queue] = set()

    async def subscribe_all(self) -> AsyncIterator[dict]:
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._global.add(q)
        try:
            while True:
                yield await q.get()
        finally:
            self._global.discard(q)

    async def subscribe_file(self, file_id: str) -> AsyncIterator[dict]:
        q: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._per_file[file_id].add(q)
        try:
            while True:
                yield await q.get()
        finally:
            self._per_file[file_id].discard(q)

    def publish(self, file_id: str, event: dict) -> None:
        payload = {"file_id": file_id, **event}
        for q in list(self._global):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass
        for q in list(self._per_file.get(file_id, ())):
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                pass


broker = ProgressBroker()
