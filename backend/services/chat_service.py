"""Chat orchestration on OpenRouter via the OpenAI chat-completions API.

Round 1 lets the model decide whether to call `search_knowledge_base`
(reasoning on, non-streaming). If it does, the searches run in-process and
Round 2 is a fresh streamed synthesis with the excerpts embedded in the user
turn, which yields reliable [n] citations without replaying tool protocol.
"""
import asyncio
import json
import logging
import time
from typing import AsyncIterator
from uuid import UUID, uuid4

import httpx
import openai

from ..config import settings
from ..db import SessionLocal
from ..models import Chat, Message
from . import attachments as attachments_svc
from . import openrouter
from . import qdrant as qdrant_svc

log = logging.getLogger("task.chat")


_TRANSIENT_EXC: tuple[type[BaseException], ...] = (
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.WriteError,
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.PoolTimeout,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
    openai.RateLimitError,
    asyncio.TimeoutError,
)

MAX_LLM_ATTEMPTS = 3


def _prefix(chat_id: UUID, user_email: str | None) -> str:
    tag = str(chat_id)[:8]
    who = user_email or "?"
    return f"[chat {tag} {who}]"


SYSTEM_PROMPT = """You are the 1staid4sme assistant — the official chatbot for the 1staid4sme (First Aid for SME) knowledge base.

## Identity
Whenever the user asks who you are, what you are, which chatbot/assistant this is, or who built you, always answer that you are the 1staid4sme assistant (First Aid for SME chatbot) that helps with questions about the 1staid4sme knowledge base. Never describe yourself as a generic AI model or name the underlying model or provider.

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

## User attachments
Users may attach documents (their extracted text appears inside <attached_document name="..."> tags in the user message) and images. When attachments are present, answer from them first and refer to them by filename — never with [n] markers, which are reserved for knowledge-base files. You may still call `search_knowledge_base` when the knowledge base could add useful context to the attachment.

If you did not call the search tool, do NOT include any [n] markers.

## Tone and length
Keep the tone professional and minimal. Answers should be brief and to the point — prefer a short paragraph or a tight bulleted list over long prose. No filler, no restating the question, no "let me know if you need anything else" sign-offs. Omit headings unless the answer genuinely has multiple distinct sections.

## Opening style
Never begin an answer with preamble like "Sure,", "Of course,", "Certainly,", "Let me look that up", "I'll search for that", "Let me check the documents", or similar filler. When you have searched, open directly with the substance — e.g., "Here is what I found:", "Based on the documents,", or simply state the answer. When no search was needed, just answer.

## Format
Format answers in GitHub-flavored Markdown (**bold**, bullet lists, fenced code where helpful). Be faithful to the source material. If the excerpts don't contain enough information, say so plainly in a single sentence.
"""


SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": (
            "Search the 1staid4sme knowledge base for excerpts relevant to the "
            "user's question. Use this whenever the answer depends on company-"
            "specific information stored in our internal documents. Skip for "
            "small talk, meta questions, or generic knowledge."
        ),
        "parameters": {
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
    },
}


def _reasoning_body() -> dict:
    """Build the OpenRouter-specific request extensions (reasoning + provider routing)."""
    return {
        "reasoning": {"enabled": True, "effort": settings.LLM_REASONING_EFFORT},
        "provider": {"sort": "latency"},
    }


def _extra_str(obj, key: str) -> str:
    """Read a non-standard string field (e.g. OpenRouter's `reasoning`) off an SDK model."""
    val = getattr(obj, key, None)
    if val is None and getattr(obj, "model_extra", None):
        val = obj.model_extra.get(key)
    return val if isinstance(val, str) else ""


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
    vectors = await openrouter.embed_texts([query])
    resp = await qdrant_svc.client().query_points(
        collection_name=settings.QDRANT_COLLECTION,
        query=vectors[0],
        limit=top_k * 3,
    )
    seen: set[str] = set()
    out: list[dict] = []
    for p in resp.points:
        text = (p.payload.get("chunk_text") or "").strip()
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


def _load_history_authz(chat_id: UUID, user_id: UUID, limit: int) -> list[dict] | None:
    """One round-trip: return trailing history for the chat, or None if the chat
    doesn't exist / isn't owned by user_id."""
    db = SessionLocal()
    try:
        rows = (
            db.query(Chat.id, Message)
            .outerjoin(Message, Message.chat_id == Chat.id)
            .filter(Chat.id == chat_id, Chat.user_id == user_id)
            .order_by(Message.created_at.asc())
            .all()
        )
        if not rows:
            return None
        msgs = [
            m
            for _, m in rows
            if m is not None and ((m.content or "").strip() or (m.role == "user" and m.attachments))
        ]
        return [
            {"role": m.role, "content": m.content, "attachments": m.attachments or []}
            for m in msgs[-limit:]
        ]
    finally:
        db.close()


def _save_user_message(chat_id: UUID, content: str, attachments: list[dict] | None = None) -> None:
    db = SessionLocal()
    try:
        db.add(
            Message(
                chat_id=chat_id,
                role="user",
                content=content,
                attachments=attachments or None,
            )
        )
        db.commit()
    finally:
        db.close()


def _save_assistant_message(
    msg_id: UUID,
    chat_id: UUID,
    content: str,
    thinking: str | None,
    citations: list,
    tool_calls: list,
) -> None:
    db = SessionLocal()
    try:
        db.add(
            Message(
                id=msg_id,
                chat_id=chat_id,
                role="assistant",
                content=content,
                thinking=thinking,
                citations=citations,
                tool_calls=tool_calls,
            )
        )
        chat = db.get(Chat, chat_id)
        if chat and chat.title == "New chat":
            first_user = (
                db.query(Message)
                .filter_by(chat_id=chat_id, role="user")
                .order_by(Message.created_at.asc())
                .first()
            )
            if first_user:
                title = first_user.content[:60].strip()
                if not title and first_user.attachments:
                    title = (first_user.attachments[0].get("filename") or "")[:60]
                chat.title = title or "New chat"
        db.commit()
    finally:
        db.close()


_background_saves: set[asyncio.Task] = set()


def _track(task: asyncio.Task) -> asyncio.Task:
    """Keep a strong ref to a fire-and-forget task until it finishes."""
    _background_saves.add(task)
    task.add_done_callback(_background_saves.discard)
    return task


def _persist_in_background(prefix: str, after: asyncio.Task | None, *args) -> None:
    """Fire-and-forget the assistant-message save (ordered after the user-turn
    save) so the SSE stream can close immediately."""

    async def _run() -> None:
        try:
            if after is not None:
                await asyncio.shield(after)
        except Exception:
            pass
        try:
            await asyncio.to_thread(_save_assistant_message, *args)
            log.info("%s assistant message persisted", prefix)
        except Exception:
            log.exception("%s FAILED to persist assistant message", prefix)

    _track(asyncio.create_task(_run()))


async def _handle_stream(stream, log_prefix: str, tool_acc: dict[int, dict] | None = None):
    """Drain a chat-completions stream, yielding ("content"|"thinking", text) tuples.

    When `tool_acc` is given, tool-call fragments are merged into it by index
    and a ("tool_call", None) marker is yielded the first time one appears."""
    first_token_at: float | None = None
    t0 = time.perf_counter()
    signalled_tool = False
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta is None:
            continue
        reasoning = _extra_str(delta, "reasoning") or _extra_str(delta, "reasoning_content")
        if reasoning:
            if first_token_at is None:
                first_token_at = time.perf_counter()
                log.info("%s first thinking token in %.2fs", log_prefix, first_token_at - t0)
            yield ("thinking", reasoning)
        if delta.content:
            if first_token_at is None:
                first_token_at = time.perf_counter()
                log.info("%s first text token in %.2fs", log_prefix, first_token_at - t0)
            yield ("content", delta.content)
        if tool_acc is not None and delta.tool_calls:
            for tc in delta.tool_calls:
                slot = tool_acc.setdefault(tc.index, {"name": "", "arguments": ""})
                if tc.function:
                    if tc.function.name:
                        slot["name"] = tc.function.name
                    if tc.function.arguments:
                        slot["arguments"] += tc.function.arguments
            if not signalled_tool:
                signalled_tool = True
                yield ("tool_call", None)


ATTACHMENT_ONLY_PROMPT = "Please review the attached file(s)."
DOC_CONTEXT_BUDGET = attachments_svc.MAX_DOC_CHARS
HISTORY_IMAGE_TURNS = 2


def _compose_parts(text: str, image_urls: list[str]) -> str | list[dict]:
    """Build a chat-completions user content value: plain text, or text + image parts."""
    if not image_urls:
        return text
    return [{"type": "text", "text": text}] + [
        {"type": "image_url", "image_url": {"url": u}} for u in image_urls
    ]


def _doc_block(meta: dict, text: str) -> str:
    """Wrap extracted document text in a tagged block the model can recognise."""
    trunc = ' truncated="true"' if meta.get("truncated") else ""
    return f'<attached_document name="{meta.get("filename", "file")}"{trunc}>\n{text}\n</attached_document>'


def _prepare_turns(
    user_id: UUID, history: list[dict], current_text: str, current_atts: list[dict]
) -> tuple[list[dict], str, list[str]]:
    """Resolve attachments into LLM-ready turns, newest first against the document budget.

    Returns (history messages, current turn text with document blocks, current turn image data URLs).
    Older images beyond HISTORY_IMAGE_TURNS image-bearing turns are replaced by a short note."""
    budget = DOC_CONTEXT_BUDGET

    def docs_for(atts: list[dict]) -> str:
        nonlocal budget
        blocks: list[str] = []
        for m in atts:
            if m.get("kind") != "document":
                continue
            if budget <= 0:
                blocks.append(f'[Attached document "{m.get("filename")}" omitted: context budget reached]')
                continue
            text = attachments_svc.load_text(user_id, m["id"])[:budget]
            budget -= len(text)
            blocks.append(_doc_block(m, text))
        return "\n\n".join(blocks)

    current_docs = docs_for(current_atts)
    current_full = f"{current_text}\n\n{current_docs}" if current_docs else current_text
    current_images = [
        attachments_svc.image_data_url(user_id, m) for m in current_atts if m.get("kind") == "image"
    ]

    image_turns_left = HISTORY_IMAGE_TURNS
    out: list[dict] = []
    for h in reversed(history):
        atts = h.get("attachments") or []
        if h["role"] != "user" or not atts:
            out.append({"role": h["role"], "content": h["content"]})
            continue
        text = (h["content"] or "").strip() or ATTACHMENT_ONLY_PROMPT
        docs = docs_for(atts)
        if docs:
            text = f"{text}\n\n{docs}"
        image_metas = [m for m in atts if m.get("kind") == "image"]
        urls: list[str] = []
        if image_metas and image_turns_left > 0:
            image_turns_left -= 1
            urls = [attachments_svc.image_data_url(user_id, m) for m in image_metas]
        elif image_metas:
            names = ", ".join(m.get("filename", "image") for m in image_metas)
            text = f"{text}\n\n[Earlier attached image(s) not shown again: {names}]"
        out.append({"role": "user", "content": _compose_parts(text, urls)})
    out.reverse()
    return out, current_full, current_images


async def stream_chat(
    chat_id: UUID,
    user_id: UUID,
    user_message: str,
    user_email: str | None = None,
    attachments: list[dict] | None = None,
) -> AsyncIterator[dict]:
    prefix = _prefix(chat_id, user_email)
    atts = attachments or []
    log.info("%s ====== starting chat stream task ======", prefix)
    log.info(
        "%s user message (%d chars, %d attachments): %r",
        prefix,
        len(user_message),
        len(atts),
        user_message[:200],
    )
    task_t0 = time.perf_counter()

    yield {"type": "thinking_start"}

    t_db = time.perf_counter()
    history = await asyncio.to_thread(
        _load_history_authz, chat_id, user_id, settings.CHAT_HISTORY_MAX_MESSAGES
    )
    if history is None:
        log.warning("%s chat not found or not owned by user — aborting", prefix)
        yield {"type": "error", "message": "chat not found"}
        return
    log.info("%s history loaded (%d msgs) in %.2fs", prefix, len(history), time.perf_counter() - t_db)

    async def _save_user() -> None:
        try:
            await asyncio.to_thread(_save_user_message, chat_id, user_message, atts)
        except Exception:
            log.exception("%s FAILED to persist user message", prefix)

    user_save_task = _track(asyncio.create_task(_save_user()))

    llm_text = user_message.strip() or ATTACHMENT_ONLY_PROMPT
    try:
        t_att = time.perf_counter()
        history_msgs, current_text, current_images = await asyncio.to_thread(
            _prepare_turns, user_id, history, llm_text, atts
        )
        if atts or any(h.get("attachments") for h in history):
            log.info(
                "%s attachments prepared in %.2fs (%d images this turn)",
                prefix,
                time.perf_counter() - t_att,
                len(current_images),
            )
    except Exception:
        log.exception("%s failed to load attachments", prefix)
        yield {"type": "error", "message": "Couldn't load the attached files. Please try again."}
        return

    messages: list[dict] = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + history_msgs
        + [{"role": "user", "content": _compose_parts(current_text, current_images)}]
    )

    client = openrouter.llm_client()

    content_buf: list[str] = []
    thinking_buf: list[str] = []
    citations: list = []
    tool_calls: list[dict] = []

    # ---- Round 1: streamed tool decision / direct answer ----
    log.info(
        "%s calling LLM round 1 | model=%s max_tokens=%d reasoning=%s",
        prefix,
        settings.OPENROUTER_LLM_MODEL,
        settings.LLM_MAX_OUTPUT_TOKENS,
        settings.LLM_REASONING_EFFORT,
    )
    t_round = time.perf_counter()

    tool_acc: dict[int, dict] = {}
    for attempt in range(1, MAX_LLM_ATTEMPTS + 1):
        tool_acc = {}
        content_emitted = False
        try:
            stream1 = await client.chat.completions.create(
                model=settings.OPENROUTER_LLM_MODEL,
                max_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
                messages=messages,
                tools=[SEARCH_TOOL],
                stream=True,
                extra_body=_reasoning_body(),
                timeout=600.0,
            )
            async for kind, payload in _handle_stream(
                stream1, f"{prefix} R1 attempt={attempt}", tool_acc
            ):
                if kind == "thinking":
                    thinking_buf.append(payload)
                    yield {"type": "thinking_delta", "content": payload}
                elif kind == "content" and not tool_acc:
                    content_emitted = True
                    content_buf.append(payload)
                    yield {"type": "content_delta", "content": payload}
                elif kind == "tool_call" and content_emitted:
                    # Any pre-tool preamble the model streamed is not the
                    # answer — the real answer comes from R2.
                    content_buf.clear()
                    yield {"type": "content_reset", "reason": "tool_call"}
            break
        except _TRANSIENT_EXC as e:
            if attempt == MAX_LLM_ATTEMPTS:
                log.error(
                    "%s R1 failed after %d attempts — surfacing: %r",
                    prefix, attempt, e,
                )
                raise
            backoff = 2 ** (attempt - 1)
            log.warning(
                "%s R1 transient error on attempt %d/%d: %r — retrying in %ds",
                prefix, attempt, MAX_LLM_ATTEMPTS, e, backoff,
            )
            content_buf.clear()
            thinking_buf.clear()
            yield {"type": "content_reset", "reason": "retry", "attempt": attempt}
            await asyncio.sleep(backoff)

    r1_tool_calls = [tool_acc[i] for i in sorted(tool_acc)]
    log.info(
        "%s round 1 done in %.2fs | tool_calls=%d text=%d chars thinking=%d chars",
        prefix,
        time.perf_counter() - t_round,
        len(r1_tool_calls),
        sum(len(p) for p in content_buf),
        sum(len(p) for p in thinking_buf),
    )

    # ---- Tool use + Round 2 ----
    if r1_tool_calls:
        tool_queries: list[tuple[str, str]] = []
        thinking_offset = sum(len(p) for p in thinking_buf)
        for idx, tc in enumerate(r1_tool_calls):
            name = tc["name"] or "search_knowledge_base"
            try:
                raw_input = json.loads(tc["arguments"] or "{}")
            except ValueError:
                raw_input = {}
            if not isinstance(raw_input, dict):
                raw_input = {}
            query = raw_input.get("query") or llm_text
            tool_queries.append((name, query))
            tool_calls.append({"name": name, "query": query, "offset": thinking_offset})
            yield {
                "type": "tool_call_start",
                "index": idx,
                "name": name,
                "query": query,
                "offset": thinking_offset,
            }

        t_search = time.perf_counter()
        hits_per_call = await asyncio.gather(
            *[_retrieve(q) for _, q in tool_queries]
        )
        log.info(
            "%s %d tool call(s) resolved in %.2fs (parallel)",
            prefix,
            len(tool_queries),
            time.perf_counter() - t_search,
        )

        all_hits: list[dict] = []
        for idx, ((name, _), hits) in enumerate(zip(tool_queries, hits_per_call)):
            all_hits.extend(hits)
            yield {
                "type": "tool_call_done",
                "index": idx,
                "name": name,
                "hit_count": len(hits),
            }

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

        composite = (
            f"{current_text}\n\n"
            "Here are relevant excerpts retrieved from the knowledge base, "
            "grouped by source file. Each [n] identifies a FILE — use the "
            "same [n] for any information you take from that file, regardless "
            "of which excerpt within it.\n\n"
            f"{_format_excerpts(grouped)}\n\n"
            "Please answer the question above using these excerpts"
            f"{' together with the attached files' if atts else ''}. "
            "Include inline [n] citations matching the file numbers."
        )
        r2_messages = list(messages[:-1]) + [
            {"role": "user", "content": _compose_parts(composite, current_images)}
        ]

        log.info(
            "%s calling LLM round 2 (fresh synthesis, %d hits in context)",
            prefix,
            len(all_hits),
        )
        r2_content_start = len(content_buf)
        r2_thinking_start = len(thinking_buf)

        t_round2 = time.perf_counter()
        for attempt in range(1, MAX_LLM_ATTEMPTS + 1):
            try:
                stream2 = await client.chat.completions.create(
                    model=settings.OPENROUTER_LLM_MODEL,
                    max_tokens=settings.LLM_MAX_OUTPUT_TOKENS,
                    messages=r2_messages,
                    stream=True,
                    extra_body=_reasoning_body(),
                    timeout=600.0,
                )
                async for kind, payload in _handle_stream(
                    stream2, f"{prefix} R2 attempt={attempt}"
                ):
                    if kind == "content":
                        content_buf.append(payload)
                        yield {"type": "content_delta", "content": payload}
                    elif kind == "thinking":
                        thinking_buf.append(payload)
                        yield {"type": "thinking_delta", "content": payload}
                break
            except _TRANSIENT_EXC as e:
                partial_chars = sum(
                    len(p) for p in content_buf[r2_content_start:]
                )
                if attempt == MAX_LLM_ATTEMPTS:
                    log.error(
                        "%s R2 failed after %d attempts (partial=%d chars) — "
                        "surfacing: %r",
                        prefix, attempt, partial_chars, e,
                    )
                    raise
                backoff = 2 ** (attempt - 1)
                log.warning(
                    "%s R2 stream error on attempt %d/%d (partial=%d chars): %r "
                    "— retrying in %ds",
                    prefix, attempt, MAX_LLM_ATTEMPTS, partial_chars, e, backoff,
                )
                del content_buf[r2_content_start:]
                del thinking_buf[r2_thinking_start:]
                yield {"type": "content_reset", "reason": "retry", "attempt": attempt}
                await asyncio.sleep(backoff)
        log.info(
            "%s round 2 done in %.2fs | content=%d chars",
            prefix,
            time.perf_counter() - t_round2,
            sum(len(p) for p in content_buf[r2_content_start:]),
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

    msg_id = uuid4()
    if not content_s.strip():
        log.warning("%s model returned an empty answer — not persisting assistant turn", prefix)
        yield {
            "type": "error",
            "message": "The assistant didn't return an answer this time. Please try asking again.",
        }
        yield {"type": "done", "message_id": str(msg_id)}
        return
    yield {"type": "done", "message_id": str(msg_id)}
    log.info(
        "%s ====== stream done in %.2fs (persisting in background) ======",
        prefix,
        time.perf_counter() - task_t0,
    )
    _persist_in_background(
        prefix, user_save_task, msg_id, chat_id, content_s, thinking_s, citations, tool_calls
    )
