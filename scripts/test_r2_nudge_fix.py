"""Validate the tool_result text-nudge fix that keeps R2 from returning an
empty response when Bifrost's Anthropic bridge short-circuits.

Runs N iterations of both variants against the real system prompt + the
exact real-world tool-result content:
  - BASELINE: just tool_result blocks  (the previously-failing shape)
  - FIXED:    tool_result blocks + trailing text nudge

A fix is only trusted if the FIXED variant produces content every time.

Usage:
    py scripts/test_r2_nudge_fix.py [ITERATIONS]

Exits 0 on success, 1 if any FIXED iteration returns empty.
"""
import asyncio
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.config import settings
from backend.services import bifrost
from backend.services.chat_service import SEARCH_TOOL, SYSTEM_PROMPT


USER_MESSAGE = (
    "can you give me some details about avijit? Like what are his skills, "
    "what is his professional experience and all."
)

# Failing shape from production logs: 6 excerpts, 3 duplicate pairs —
# what the raw Qdrant query used to return before _retrieve() dedup.
FAILING_TOOL_RESULT = """[1] From resume.pdf:
 data, content calendar view, post performance analytics via LinkedIn API, and in-app post editing before approval

Naukar - AI-Powered Job Application Automation System (Personal Use / Closed Access)

- Challenge: Maintaining an optimized

[2] From resume.pdf:
 data, content calendar view, post performance analytics via LinkedIn API, and in-app post editing before approval

Naukar - AI-Powered Job Application Automation System (Personal Use / Closed Access)

- Challenge: Maintaining an optimized

[3] From resume.pdf:
## Professional Summary

Data Science Professional with 3+ years of experience in machine learning, deep learning, predictive modeling, and data pipeline optimization. Proficient in Python, SQL, AWS, Snowflake, and generative AI, with demon

[4] From resume.pdf:
## Professional Summary

Data Science Professional with 3+ years of experience in machine learning, deep learning, predictive modeling, and data pipeline optimization. Proficient in Python, SQL, AWS, Snowflake, and generative AI, with demon

[5] From resume.pdf:
                                                                                                                                                                                                       |

Bachelor of Technology (B.Tech) in Ele

[6] From resume.pdf:
                                                                                                                                                                                                       |

Bachelor of Technology (B.Tech) in Ele"""

# Fixed shape: dedup by normalized content. This is what _retrieve() now
# produces — only unique excerpts, no duplicate pairs.
DEDUPED_TOOL_RESULT = """[1] From resume.pdf:
 data, content calendar view, post performance analytics via LinkedIn API, and in-app post editing before approval

Naukar - AI-Powered Job Application Automation System (Personal Use / Closed Access)

- Challenge: Maintaining an optimized

[2] From resume.pdf:
## Professional Summary

Data Science Professional with 3+ years of experience in machine learning, deep learning, predictive modeling, and data pipeline optimization. Proficient in Python, SQL, AWS, Snowflake, and generative AI, with demon

[3] From resume.pdf:
                                                                                                                                                                                                       |

Bachelor of Technology (B.Tech) in Ele"""


NUDGE_TEXT = (
    "Using the excerpts above, answer the user's question with inline [n] "
    "citations per the format rules."
)


async def _one_iteration(
    *,
    apply_nudge: bool,
    tool_result: str,
    use_streaming: bool = False,
    verbose: bool = False,
) -> tuple[int, int, str]:
    """Run one R1 + R2 round-trip; return (r2_chars, r2_out_tokens, stop_reason)."""
    client = bifrost.anthropic_client()

    # R1 mirrors the real call exactly.
    r1 = await client.messages.create(
        model=settings.BIFROST_LLM_MODEL,
        max_tokens=4000,  # test-only cap; SDK forbids non-stream w/ 64k
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": USER_MESSAGE}],
        tools=[SEARCH_TOOL],
        thinking={
            "type": "enabled",
            "budget_tokens": settings.THINKING_BUDGET_TOKENS,
        },
        extra_headers={"anthropic-beta": settings.BIFROST_ANTHROPIC_BETA},
    )
    tool_uses = [b for b in r1.content if b.type == "tool_use"]
    if not tool_uses:
        return (-1, -1, f"R1-no-tool-use(stop={r1.stop_reason})")

    # Build R2 messages: replay full R1 assistant turn, then user turn with
    # tool_result(s) — optionally plus the nudge.
    messages = [{"role": "user", "content": USER_MESSAGE}]
    messages.append(
        {
            "role": "assistant",
            "content": [
                b.model_dump(mode="json", exclude_none=True) for b in r1.content
            ],
        }
    )
    user_blocks = [
        {
            "type": "tool_result",
            "tool_use_id": tu.id,
            "content": tool_result,
        }
        for tu in tool_uses
    ]
    if apply_nudge:
        user_blocks.append({"type": "text", "text": NUDGE_TEXT})
    messages.append({"role": "user", "content": user_blocks})

    r2 = await client.messages.create(
        model=settings.BIFROST_LLM_MODEL,
        max_tokens=4000,  # test-only cap; SDK forbids non-stream w/ 64k
        system=SYSTEM_PROMPT,
        messages=messages,
        thinking={
            "type": "enabled",
            "budget_tokens": settings.THINKING_BUDGET_TOKENS,
        },
        extra_headers={"anthropic-beta": settings.BIFROST_ANTHROPIC_BETA},
    )
    text = "".join(b.text for b in r2.content if b.type == "text")
    thinking_text = "".join(
        b.thinking for b in r2.content if b.type == "thinking"
    )
    block_types = [b.type for b in r2.content]
    stop = f"{r2.stop_reason} blocks={block_types} think={len(thinking_text)}"
    return (len(text), r2.usage.output_tokens, stop)


async def run_variant(
    label: str,
    iterations: int,
    *,
    apply_nudge: bool,
    tool_result: str,
) -> list[tuple[int, int, str]]:
    print(f"\n== {label} ({iterations} iterations) ==")
    results: list[tuple[int, int, str]] = []
    for i in range(1, iterations + 1):
        chars, out_tokens, stop = await _one_iteration(
            apply_nudge=apply_nudge, tool_result=tool_result
        )
        mark = "OK  " if chars > 50 else "FAIL"
        print(
            f"  [{i}/{iterations}] {mark} chars={chars:5d}  out_tokens={out_tokens:4d}  {stop}"
        )
        results.append((chars, out_tokens, stop))
    successes = sum(1 for r in results if r[0] > 50)
    print(f"  => {successes}/{iterations} iterations produced a real answer (>50 chars)")
    return results


async def main(iterations: int) -> int:
    print(f"Anthropic base URL: {settings.bifrost_anthropic_base_url}")
    print(f"Model             : {settings.BIFROST_LLM_MODEL}")
    print(f"max_tokens        : 4000 (test cap)")

    # Cell 1: the previously-observed production shape — 6 near-dupe excerpts.
    failing = await run_variant(
        "A  pre-fix content (6 hits, 3 dup pairs) + NO nudge",
        iterations,
        apply_nudge=False,
        tool_result=FAILING_TOOL_RESULT,
    )
    failing_nudge = await run_variant(
        "B  pre-fix content (6 hits, 3 dup pairs) + nudge   ",
        iterations,
        apply_nudge=True,
        tool_result=FAILING_TOOL_RESULT,
    )
    # Cell 2: the post-fix shape — 3 deduped excerpts. This is what the
    # real _retrieve() now produces after the dedup change.
    deduped = await run_variant(
        "C  post-fix content (3 dedup hits) + NO nudge      ",
        iterations,
        apply_nudge=False,
        tool_result=DEDUPED_TOOL_RESULT,
    )
    deduped_nudge = await run_variant(
        "D  post-fix content (3 dedup hits) + nudge         ",
        iterations,
        apply_nudge=True,
        tool_result=DEDUPED_TOOL_RESULT,
    )

    def rate(xs):
        return sum(1 for c, _, _ in xs if c > 50)

    print("\n---- SUMMARY (answers with >50 chars of text) ----")
    print(f"  A pre-fix + no nudge : {rate(failing)}/{iterations}")
    print(f"  B pre-fix + nudge    : {rate(failing_nudge)}/{iterations}")
    print(f"  C post-fix + no nudge: {rate(deduped)}/{iterations}")
    print(f"  D post-fix + nudge   : {rate(deduped_nudge)}/{iterations}")

    # Fix is trusted if D (what the real backend now does) is 100%.
    if rate(deduped_nudge) < iterations:
        print("\nRESULT: FIX NOT RELIABLE — variant D failed at least once.")
        return 1
    print("\nRESULT: FIX VERIFIED — dedup + nudge produces real answers every time.")
    return 0


if __name__ == "__main__":
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    sys.exit(asyncio.run(main(iterations)))
