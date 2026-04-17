"""Chat orchestration using Bifrost's native Anthropic endpoint.

We deliberately avoid the OpenAI-compat path here — Bifrost's translator
silently drops the post-tool-result follow-up response (returns 0 chars),
and doesn't forward thinking blocks. The Anthropic-native endpoint at
`{BIFROST_BASE_URL}/anthropic` accepts the Anthropic Python SDK as a
drop-in replacement and handles tool use + extended thinking correctly.
"""
import asyncio
import logging
import time
from typing import AsyncIterator
from uuid import UUID

from ..config import settings
from ..db import SessionLocal
from ..models import Chat, Message
from . import bifrost
from . import qdrant as qdrant_svc

log = logging.getLogger("task.chat")


def _prefix(chat_id: UUID, user_email: str | None) -> str:
    tag = str(chat_id)[:8]
    who = user_email or "?"
    return f"[chat {tag} {who}]"


SYSTEM_PROMPT = """You are a helpful assistant for the 1staid4sme knowledge base.

## When to use the search_knowledge_base tool
ONLY call `search_knowledge_base` when the user is asking a substantive question whose answer would come from internal company documents (policies, procedures, products, customers, internal data).

DO NOT call the tool for:
- Greetings ("hi", "hello", "hey", "good morning")
- Small talk, thanks, acknowledgements
- Meta questions about you ("who are you", "what can you do")
- Generic knowledge questions unrelated to the company
- Clarification questions

When in doubt for a short conversational message, answer directly without searching.

## Citations
When you DO use retrieved excerpts, cite them inline using numbered markers like [1] or [2]. Each number identifies a SOURCE FILE, not a passage — if you draw multiple facts from the same file, reuse that file's number every time. Multiple files cited for the same claim: [1][2]. Place each citation immediately after the supporting clause, before its punctuation. Only cite files you actually used.

If you did not call the search tool, do NOT include any [n] markers.

## Tone and length
Keep the tone professional and minimal. Answers should be brief and to the point — prefer a short paragraph or a tight bulleted list over long prose. No filler, no restating the question, no "let me know if you need anything else" sign-offs. Omit headings unless the answer genuinely has multiple distinct sections.

## Format
Format answers in GitHub-flavored Markdown (**bold**, bullet lists, fenced code where helpful). Be faithful to the source material. If the excerpts don't contain enough information, say so plainly in a single sentence.
"""


# Anthropic-native tool schema (flat, uses `input_schema`, not OpenAI's
# nested `function.parameters` shape).
SEARCH_TOOL = {
    "name": "search_knowledge_base",
    "description": (
        "Search the 1staid4sme knowledge base for excerpts relevant to the "
        "user's question. Use this whenever the answer depends on company-"
        "specific information stored in our internal documents. Skip for "
        "small talk, meta questions, or generic knowledge."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "A focused search query paraphrased from the user's "
                    "question. Use the language of the documents."
                ),
            }
        },
        "required": ["query"],
    },
}


def _group_hits_by_file(hits: list[dict]) -> list[dict]:
    """Collapse retrieved chunks into one entry per source file, preserving
    the order files were first seen. Citation numbering is per-file so the
    answer shows `[1]` for anything from file 1 regardless of how many chunks
    inside that file were used."""
    groups: dict[str, dict] = {}
    order: list[str] = []
    for h in hits:
        fid = h.get("file_id")
        key = str(fid) if fid else f"__name__:{h.get('filename')}"
        if key not in groups:
            groups[key] = {
                "filename": h.get("filename") or "",
                "file_id": str(fid) if fid else None,
                "chunk_texts": [],
            }
            order.append(key)
        text = h.get("chunk_text") or ""
        if text:
            groups[key]["chunk_texts"].append(text)
    return [groups[k] for k in order]


def _format_excerpts(grouped: list[dict]) -> str:
    if not grouped:
        return "(no relevant excerpts found)"
    blocks: list[str] = []
    for i, entry in enumerate(grouped, 1):
        body = "\n\n---\n\n".join(entry["chunk_texts"])
        blocks.append(f"[{i}] From {entry['filename']}:\n{body}")
    return "\n\n".join(blocks)


async def _retrieve(query: str, top_k: int = 6) -> list[dict]:
    vectors = await bifrost.embed_texts([query])
    # Over-fetch so dedup by content still leaves us with ~top_k unique hits.
    resp = await qdrant_svc.client().query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=vectors[0],
        limit=top_k * 3,
    )
    seen: set[str] = set()
    out: list[dict] = []
    for p in resp.points:
        text = (p.payload.get("chunk_text") or "").strip()
        # Dedup by a normalized whitespace-collapsed key so that the same
        # document uploaded twice (different file_id, identical chunk text)
        # doesn't consume multiple slots in the prompt. Without this the
        # model sees 3 pairs of duplicates, concludes "insufficient info",
        # and tries to call the tool again on R2 — producing "Let me search
        # for more..." instead of a real answer.
        key = " ".join(text.split())
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "filename": p.payload.get("filename"),
                "chunk_text": p.payload.get("chunk_text") or "",
                "file_id": p.payload.get("file_id"),
                "chunk_index": p.payload.get("chunk_index"),
                "score": p.score,
            }
        )
        if len(out) >= top_k:
            break
    return out


def _load_history(chat_id: UUID, limit: int) -> list[dict]:
    db = SessionLocal()
    try:
        msgs = (
            db.query(Message)
            .filter(Message.chat_id == chat_id)
            .order_by(Message.created_at.asc())
            .all()
        )
        # Anthropic accepts either a string or a list of content blocks;
        # stored history is plain text from previous final answers, so
        # string content is correct and simplest here.
        return [{"role": m.role, "content": m.content} for m in msgs[-limit:]]
    finally:
        db.close()


def _save_user_message(chat_id: UUID, content: str) -> None:
    db = SessionLocal()
    try:
        db.add(Message(chat_id=chat_id, role="user", content=content))
        db.commit()
    finally:
        db.close()


def _save_assistant_message(
    chat_id: UUID, content: str, thinking: str | None, citations: list
) -> UUID:
    db = SessionLocal()
    try:
        msg = Message(
            chat_id=chat_id,
            role="assistant",
            content=content,
            thinking=thinking,
            citations=citations,
        )
        db.add(msg)
        db.flush()
        chat = db.get(Chat, chat_id)
        if chat and chat.title == "New chat":
            first_user = (
                db.query(Message)
                .filter_by(chat_id=chat_id, role="user")
                .order_by(Message.created_at.asc())
                .first()
            )
            if first_user:
                chat.title = (first_user.content[:60].strip()) or "New chat"
        db.commit()
        return msg.id
    finally:
        db.close()


def _blocks_to_dict(blocks) -> list[dict]:
    """Convert SDK content-block models back to plain dicts so we can
    round-trip them in the `messages` list. `model_dump(mode='json')`
    preserves the `signature` on thinking blocks, which Anthropic requires
    for multi-turn extended thinking to work."""
    return [b.model_dump(mode="json", exclude_none=True) for b in blocks]


async def _handle_stream(stream, log_prefix: str):
    """Drain an Anthropic messages.stream() context, yielding our SSE
    event dicts (thinking_delta / content_delta). Returns the final
    assembled message once the stream finishes so the caller can decide
    what to do based on stop_reason + content blocks.
    """
    first_token_at: float | None = None
    t0 = time.perf_counter()
    async for event in stream:
        etype = getattr(event, "type", None)
        if etype != "content_block_delta":
            continue
        delta = event.delta
        dtype = getattr(delta, "type", None)
        if dtype == "text_delta":
            if first_token_at is None:
                first_token_at = time.perf_counter()
                log.info("%s first text token in %.2fs", log_prefix, first_token_at - t0)
            yield ("content", delta.text)
        elif dtype == "thinking_delta":
            if first_token_at is None:
                first_token_at = time.perf_counter()
                log.info("%s first thinking token in %.2fs", log_prefix, first_token_at - t0)
            yield ("thinking", delta.thinking)
        # input_json_delta and signature_delta are captured in the final
        # message via get_final_message(); no need to surface mid-stream.


async def stream_chat(
    chat_id: UUID,
    user_message: str,
    user_email: str | None = None,
) -> AsyncIterator[dict]:
    prefix = _prefix(chat_id, user_email)
    log.info("%s ====== starting chat stream task ======", prefix)
    log.info(
        "%s user message (%d chars): %r",
        prefix,
        len(user_message),
        user_message[:200],
    )
    task_t0 = time.perf_counter()

    await asyncio.to_thread(_save_user_message, chat_id, user_message)

    history = await asyncio.to_thread(
        _load_history, chat_id, settings.CHAT_HISTORY_MAX_MESSAGES
    )
    # history already includes the message we just saved (it's the last item).
    messages: list[dict] = list(history)

    client = bifrost.anthropic_client()

    content_buf: list[str] = []
    thinking_buf: list[str] = []
    citations: list = []

    yield {"type": "thinking_start"}

    # ---- Round 1: with tools + extended thinking ----
    log.info(
        "%s calling LLM round 1 | model=%s max_tokens=%d thinking_budget=%d",
        prefix,
        settings.BIFROST_LLM_MODEL,
        settings.LLM_MAX_OUTPUT_TOKENS,
        settings.THINKING_BUDGET_TOKENS,
    )
    t_round = time.perf_counter()

    # R1 is non-streaming: we don't want to progressively render thinking or
    # a pre-tool text preamble — the user-visible answer comes from R2. We
    # still emit thinking/text as single chunks after R1 returns so the UI's
    # thinking panel stays populated.
    final_r1 = await client.messages.create(
        model=settings.BIFROST_LLM_MODEL,
        max_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
        system=SYSTEM_PROMPT,
        messages=messages,
        tools=[SEARCH_TOOL],
        thinking={
            "type": "enabled",
            "budget_tokens": settings.THINKING_BUDGET_TOKENS,
        },
        extra_headers={"anthropic-beta": settings.BIFROST_ANTHROPIC_BETA},
        # SDK refuses non-streaming calls whose estimated duration exceeds
        # 10 min based on max_tokens (64k here). Pass an explicit timeout
        # to skip that check — R1 is short in practice (thinking + a tool
        # use block, not 64k of output).
        timeout=600.0,
    )
    for block in final_r1.content:
        btype = getattr(block, "type", None)
        if btype == "thinking":
            text = getattr(block, "thinking", "") or ""
            if text:
                thinking_buf.append(text)
                yield {"type": "thinking_delta", "content": text}
        elif btype == "text":
            text = getattr(block, "text", "") or ""
            if text:
                content_buf.append(text)
                yield {"type": "content_delta", "content": text}

    log.info(
        "%s round 1 done in %.2fs | stop=%s blocks=%d (%s)",
        prefix,
        time.perf_counter() - t_round,
        final_r1.stop_reason,
        len(final_r1.content),
        [b.type for b in final_r1.content],
    )

    # ---- Tool use + Round 2 ----
    if final_r1.stop_reason == "tool_use":
        # We intentionally do NOT replay R1's assistant turn + a
        # tool_result user turn via Anthropic's tool_use protocol. That
        # shape (the documented pattern) routinely returns `end_turn`
        # with zero content blocks when routed through Bifrost —
        # validated 0/5 successes in scripts/test_r2_nudge_fix.py even
        # with dedup + text nudge workarounds.
        #
        # Instead: execute the requested tool(s) in-process and call R2
        # as a FRESH synthesis with the retrieved excerpts embedded
        # directly in the user message. Validated 5/5 successes,
        # ~1.2k char answers with proper [n] citations.
        all_hits: list[dict] = []
        for block in final_r1.content:
            if block.type != "tool_use":
                continue
            raw_input = block.input if isinstance(block.input, dict) else {}
            query = raw_input.get("query") or user_message

            yield {
                "type": "tool_call_start",
                "name": block.name,
                "query": query,
            }
            t_search = time.perf_counter()
            hits = await _retrieve(query)
            log.info(
                "%s tool '%s' retrieved %d hits in %.2fs",
                prefix,
                block.name,
                len(hits),
                time.perf_counter() - t_search,
            )
            all_hits.extend(hits)
            yield {
                "type": "tool_call_done",
                "name": block.name,
                "hit_count": len(hits),
            }

        # One citation per source file (not per chunk). The frontend uses
        # chunk_texts[] to highlight every cited passage inside the opened
        # document via normalized text search.
        grouped = _group_hits_by_file(all_hits)
        for i, entry in enumerate(grouped, 1):
            citations.append(
                {
                    "index": i,
                    "filename": entry["filename"],
                    "file_id": entry["file_id"],
                    "chunk_texts": entry["chunk_texts"],
                }
            )
        yield {"type": "citations", "citations": citations}

        # Build R2 messages: prior conversation history (without the
        # current user turn) + a single composite user message carrying
        # both the question AND the retrieved excerpts. `messages` has
        # the current user turn as its LAST element — we replace it.
        composite = (
            f"{user_message}\n\n"
            "Here are relevant excerpts retrieved from the knowledge base, "
            "grouped by source file. Each [n] identifies a FILE — use the "
            "same [n] for any information you take from that file, regardless "
            "of which excerpt within it.\n\n"
            f"{_format_excerpts(grouped)}\n\n"
            "Please answer the question above using these excerpts. "
            "Include inline [n] citations matching the file numbers."
        )
        r2_messages = list(messages[:-1]) + [
            {"role": "user", "content": composite}
        ]

        log.info(
            "%s calling LLM round 2 (fresh synthesis, %d hits in context)",
            prefix,
            len(all_hits),
        )
        t_round2 = time.perf_counter()
        async with client.messages.stream(
            model=settings.BIFROST_LLM_MODEL,
            max_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
            system=SYSTEM_PROMPT,
            messages=r2_messages,
        ) as stream2:
            async for kind, payload in _handle_stream(stream2, f"{prefix} R2"):
                if kind == "content":
                    content_buf.append(payload)
                    yield {"type": "content_delta", "content": payload}
                elif kind == "thinking":
                    thinking_buf.append(payload)
                    yield {"type": "thinking_delta", "content": payload}
            final_r2 = await stream2.get_final_message()
        log.info(
            "%s round 2 done in %.2fs | stop=%s blocks=%d",
            prefix,
            time.perf_counter() - t_round2,
            final_r2.stop_reason,
            len(final_r2.content),
        )

    content_s = "".join(content_buf)
    thinking_s = "".join(thinking_buf) if thinking_buf else None
    log.info(
        "%s total stream done (content=%d chars, thinking=%d chars, citations=%d)",
        prefix,
        len(content_s),
        len(thinking_s) if thinking_s else 0,
        len(citations),
    )

    msg_id = await asyncio.to_thread(
        _save_assistant_message, chat_id, content_s, thinking_s, citations
    )
    log.info(
        "%s ====== done in %.2fs ======",
        prefix,
        time.perf_counter() - task_t0,
    )
    yield {"type": "done", "message_id": str(msg_id)}
