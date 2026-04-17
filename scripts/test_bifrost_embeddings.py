"""
Probe Bifrost's OpenAI-compatible /embeddings endpoint for gemini-embedding-001.

What this answers:
  1. Does `input: [array]` work? (batch via single HTTP call)
  2. What's the practical ceiling on (a) array length and (b) total tokens per request?
  3. Does the OpenAI-style `dimensions` param work, or do we need `output_dimensionality` via extra_body?
  4. Are returned vectors L2-normalized at non-3072 dims, or do we have to normalize ourselves?
  5. Per-text token ceiling.

Run:
  pip install openai
  python scripts/test_bifrost_embeddings.py
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

from openai import OpenAI, BadRequestError, APIError


# ---------- minimal .env loader (no dotenv dep) ----------
def load_env(path: Path) -> None:
    if not path.exists():
        sys.exit(f"missing {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, _, v = s.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


load_env(Path(__file__).resolve().parent.parent / ".env")

BASE_URL = os.environ["BIFROST_BASE_URL"]
API_KEY = os.environ["BIFROST_API_KEY"]
MODEL = os.environ.get("BIFROST_EMBEDDING_MODEL", "gemini-embedding-001")
TARGET_DIM = int(os.environ.get("EMBEDDING_DIM", "1536"))

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


# ---------- helpers ----------
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


# ---------- probes ----------
def probe_1_sanity():
    banner("1. Sanity — single string, no extras")
    ok, msg, _ = call("hello world")
    print(msg)
    return ok


def probe_2_dimensions_param():
    banner("2. OpenAI-style `dimensions=1536` param")
    ok, msg, _ = call("hello world", dimensions=TARGET_DIM)
    print(msg)
    if ok:
        print(f"  -> param FORWARDED (got {TARGET_DIM}-dim vector)")
    return ok


def probe_3_extra_body_dim():
    banner("3. `extra_body={'output_dimensionality': 1536}`")
    ok, msg, _ = call(
        "hello world",
        extra_body={"output_dimensionality": TARGET_DIM},
    )
    print(msg)
    return ok


def probe_4_array_input():
    banner("4. Batch via `input: [array]` — does it work at all?")
    ok, msg, resp = call(["alpha text", "beta text"])
    print(msg)
    if ok:
        n = len(resp.data)
        if n == 2:
            print("  -> ARRAY BATCHING WORKS (got 2 vectors)")
        else:
            print(f"  -> WARNING: sent 2, got {n} vectors")
    return ok


def probe_5_batch_sweep():
    banner("5. Batch size sweep (short strings)")
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


def probe_6_total_tokens():
    banner("6. Total-tokens-per-request sweep (10 items, varying tokens each)")
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


def probe_7_per_text_tokens():
    banner("7. Per-text token ceiling (single item)")
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


def probe_8_norm_at_target_dim():
    banner(f"8. Are vectors L2-normalized at {TARGET_DIM} dims?")
    # Try whichever dim mechanism actually worked.
    for kwargs in (
        {"dimensions": TARGET_DIM},
        {"extra_body": {"output_dimensionality": TARGET_DIM}},
    ):
        ok, msg, resp = call("normalization test", **kwargs)
        if ok:
            v = resp.data[0].embedding
            norm = l2_norm(v)
            print(f"using {kwargs} -> dim={len(v)} ||v||={norm:.6f}")
            if abs(norm - 1.0) < 0.01:
                print("  -> Already L2-normalized server-side. No client-side norm needed.")
            else:
                print("  -> NOT unit norm. We MUST L2-normalize client-side before Qdrant.")
            return
    print("  -> No dimension-control method worked, can't test.")


# ---------- runner ----------
def main():
    print(f"Bifrost base_url: {BASE_URL}")
    print(f"Model           : {MODEL}")
    print(f"Target dim      : {TARGET_DIM}")

    if not probe_1_sanity():
        sys.exit("Sanity call failed — aborting. Check BIFROST_BASE_URL / API key / network.")

    probe_2_dimensions_param()
    probe_3_extra_body_dim()

    if not probe_4_array_input():
        print("\nArray batching failed — skipping batch-sweep probes.")
        print("Pipeline must use concurrent single-text requests instead.")
        probe_8_norm_at_target_dim()
        return

    max_batch = probe_5_batch_sweep()
    max_total_tokens = probe_6_total_tokens()
    max_per_text = probe_7_per_text_tokens()
    probe_8_norm_at_target_dim()

    banner("RECOMMENDED INFRA LIMITS (75% of verified ceiling)")
    print(f"  MAX_BATCH_ITEMS         = {int(max_batch * 0.75)}   (verified ceiling: {max_batch})")
    print(f"  MAX_BATCH_TOTAL_TOKENS  = {int(max_total_tokens * 0.75)}   (verified: ~{max_total_tokens})")
    print(f"  MAX_TOKENS_PER_TEXT     = {int(max_per_text * 0.75)}   (verified: ~{max_per_text})")


if __name__ == "__main__":
    main()
