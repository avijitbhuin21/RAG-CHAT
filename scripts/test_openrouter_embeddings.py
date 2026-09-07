"""
Probe OpenRouter's OpenAI-compatible /embeddings endpoint for gemini-embedding-001.

What this answers:
  1. Does `input: [array]` work? (batch via single HTTP call)
  2. What's the practical ceiling on (a) array length and (b) total tokens per request?
  3. Does the OpenAI-style `dimensions` param get honored (or do we slice client-side)?
  4. Are returned vectors L2-normalized at non-3072 dims, or do we have to normalize ourselves?
  5. Per-text token ceiling.

Run:
  python scripts/test_openrouter_embeddings.py
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

from openai import OpenAI, BadRequestError, APIError


def load_env(path: Path) -> None:
    """Minimal .env loader (no python-dotenv dependency)."""
    if not path.exists():
        sys.exit(f"missing {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, _, v = s.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


load_env(Path(__file__).resolve().parent.parent / ".env")

BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = os.environ.get("OPENROUTER_EMBEDDING_MODEL", "google/gemini-embedding-001")
TARGET_DIM = int(os.environ.get("EMBEDDING_DIM", "1536"))

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def l2_norm(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def words(n: int) -> str:
    """~n tokens of filler text (rough: 1 token ≈ 0.75 words for English)."""
    return ("the quick brown fox jumps over the lazy dog " * ((n // 7) + 1)).strip()


def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def call(input_, **kwargs) -> tuple[bool, str, object]:
    """Returns (ok, summary, raw)."""
    try:
        t0 = time.perf_counter()
        resp = client.embeddings.create(model=MODEL, input=input_, **kwargs)
        dt = time.perf_counter() - t0
        n = len(resp.data)
        dim = len(resp.data[0].embedding) if n else 0
        usage = getattr(resp, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", "?") if usage else "?"
        norm = l2_norm(resp.data[0].embedding) if n else 0
        return (
            True,
            f"OK  n={n}  dim={dim}  prompt_tokens={prompt_tokens}  "
            f"||v0||={norm:.4f}  {dt * 1000:.0f}ms",
            resp,
        )
    except BadRequestError as e:
        return False, f"400 BadRequest: {str(e)[:300]}", e
    except APIError as e:
        return False, f"APIError ({getattr(e, 'status_code', '?')}): {str(e)[:300]}", e
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:300]}", e


def probe_1_sanity():
    banner("1. Sanity — single string, no extras")
    ok, msg, _ = call("hello world")
    print(msg)
    return ok


def probe_2_dimensions_param():
    banner(f"2. OpenAI-style `dimensions={TARGET_DIM}` param")
    ok, msg, resp = call("hello world", dimensions=TARGET_DIM)
    print(msg)
    if ok:
        dim = len(resp.data[0].embedding)
        if dim == TARGET_DIM:
            print(f"  -> param HONORED (got {TARGET_DIM}-dim vector)")
        else:
            print(f"  -> param IGNORED (got {dim}); backend slices to {TARGET_DIM} client-side")
    return ok


def probe_3_array_input():
    banner("3. Batch via `input: [array]` — does it work at all?")
    ok, msg, resp = call(["alpha text", "beta text"])
    print(msg)
    if ok:
        n = len(resp.data)
        if n == 2:
            print("  -> ARRAY BATCHING WORKS (got 2 vectors)")
        else:
            print(f"  -> WARNING: sent 2, got {n} vectors")
    return ok


def probe_4_batch_sweep():
    banner("4. Batch size sweep (short strings)")
    print(f"{'size':>6} | result")
    print("-" * 70)
    last_ok = 0
    for size in [10, 50, 100, 200, 250, 300, 500]:
        inputs = [f"sample text number {i}" for i in range(size)]
        ok, msg, _ = call(inputs)
        flag = "OK " if ok else "FAIL"
        print(f"{size:>6} | {flag} {msg}")
        if ok:
            last_ok = size
        else:
            break
    print(f"\n  -> Largest array size that succeeded: {last_ok}")
    return last_ok


def probe_5_total_tokens():
    banner("5. Total-tokens-per-request sweep (10 items, varying tokens each)")
    print(f"{'tok/item':>10} {'~total':>8} | result")
    print("-" * 70)
    last_ok = 0
    for tok_each in [500, 1000, 1500, 2000, 2500, 3000]:
        text = words(tok_each)
        inputs = [text] * 10
        approx_total = tok_each * 10
        ok, msg, _ = call(inputs)
        flag = "OK " if ok else "FAIL"
        print(f"{tok_each:>10} {approx_total:>8} | {flag} {msg}")
        if ok:
            last_ok = approx_total
        else:
            break
    print(f"\n  -> Approx total-tokens ceiling per request: ~{last_ok}")
    return last_ok


def probe_6_per_text_tokens():
    banner("6. Per-text token ceiling (single item)")
    print(f"{'~tokens':>8} | result")
    print("-" * 70)
    last_ok = 0
    for tok in [1000, 2000, 2500, 3000, 4000, 6000]:
        ok, msg, _ = call(words(tok))
        flag = "OK " if ok else "FAIL"
        print(f"{tok:>8} | {flag} {msg}")
        if ok:
            last_ok = tok
        else:
            break
    print(f"\n  -> Approx per-text token ceiling: ~{last_ok}")
    return last_ok


def probe_7_norm_at_target_dim():
    banner(f"7. Are vectors L2-normalized at {TARGET_DIM} dims?")
    ok, msg, resp = call("normalization test", dimensions=TARGET_DIM)
    if not ok:
        print(f"  -> call failed: {msg}")
        return
    v = resp.data[0].embedding[:TARGET_DIM]
    norm = l2_norm(v)
    print(f"dim={len(v)} ||v||={norm:.6f}")
    if abs(norm - 1.0) < 0.01:
        print("  -> Already L2-normalized server-side.")
    else:
        print("  -> NOT unit norm. Backend L2-normalizes client-side before Qdrant.")


def main():
    print(f"OpenRouter base_url: {BASE_URL}")
    print(f"Model              : {MODEL}")
    print(f"Target dim         : {TARGET_DIM}")

    if not probe_1_sanity():
        sys.exit("Sanity call failed — aborting. Check OPENROUTER_API_KEY / network.")

    probe_2_dimensions_param()

    if not probe_3_array_input():
        print("\nArray batching failed — skipping batch-sweep probes.")
        probe_7_norm_at_target_dim()
        return

    max_batch = probe_4_batch_sweep()
    max_total_tokens = probe_5_total_tokens()
    max_per_text = probe_6_per_text_tokens()
    probe_7_norm_at_target_dim()

    banner("RECOMMENDED INFRA LIMITS (75% of verified ceiling)")
    print(f"  EMBED_MAX_BATCH_ITEMS    = {int(max_batch * 0.75)}   (verified ceiling: {max_batch})")
    print(f"  EMBED_MAX_BATCH_TOKENS   = {int(max_total_tokens * 0.75)}   (verified: ~{max_total_tokens})")
    print(f"  EMBED_MAX_TOKENS_PER_TEXT= {int(max_per_text * 0.75)}   (verified: ~{max_per_text})")


if __name__ == "__main__":
    main()
