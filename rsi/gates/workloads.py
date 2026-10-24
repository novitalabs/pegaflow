"""Gate workloads: gsm8k few-shot accuracy and multi-turn agent replay.

Both talk to a running vLLM OpenAI server and return plain dicts; the gate
orchestrator (``rsi.gates.run_gate``) owns servers and verdicts.

``python -m rsi.gates.workloads prep-r2egym PARQUET OUT.jsonl [--n N]`` extracts
conversations once (needs pyarrow; run it with ``uv run --with pyarrow``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import statistics
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import aiohttp


# vLLM's uvicorn closes an idle keep-alive connection after
# VLLM_HTTP_TIMEOUT_KEEP_ALIVE (5 s); a pooled client that reuses a connection
# the server is closing fails with "Server disconnected" / ECONNRESET before
# any response byte, which aborts the pass (and the gate's stop() then
# SIGTERMs vLLM: looks like an external kill). One connection per request
# (localhost, negligible cost) removes the race; no retry, since a resent
# request would be recorded twice by the probe.
def session(timeout: Any = None) -> aiohttp.ClientSession:
    import aiohttp

    connector = aiohttp.TCPConnector(force_close=True)
    if timeout is None:
        return aiohttp.ClientSession(connector=connector)
    return aiohttp.ClientSession(connector=connector, timeout=timeout)


# --------------------------------------------------------------------- gsm8k

_CALC = re.compile(r"<<[^>]*>>")
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_ANS = re.compile(r"The answer is\s*\$?\s*(-?\d[\d,]*(?:\.\d+)?)")


def _norm(num: str) -> str:
    num = num.replace(",", "").rstrip(".")
    try:
        v = float(num)
    except ValueError:
        return num
    if not math.isfinite(v):  # garbage KV can produce "1e999"-sized digit runs
        return num
    return str(int(v)) if v == int(v) else str(v)


def _shot(rec: dict[str, str]) -> str:
    body, final = rec["answer"].split("####")
    body = _CALC.sub("", body).strip()
    return f"Question: {rec['question']}\nAnswer: {body} The answer is {_norm(final.strip())}.\n\n"


def gsm8k_prompts(data_dir: Path, shots: int, limit: int | None) -> list[tuple[str, str]]:
    """(prompt, gold) pairs: fixed first ``shots`` train examples as the prefix."""
    train = [json.loads(x) for x in (data_dir / "train.jsonl").read_text().splitlines() if x]
    test = [json.loads(x) for x in (data_dir / "test.jsonl").read_text().splitlines() if x]
    prefix = "".join(_shot(r) for r in train[:shots])
    test = test[:limit] if limit else test
    return [
        (f"{prefix}Question: {r['question']}\nAnswer:", _norm(r["answer"].split("####")[1].strip()))
        for r in test
    ]


def gsm8k_extract(text: str) -> str | None:
    m = _ANS.search(text)
    if m:
        return _norm(m.group(1))
    nums = _NUM.findall(text)
    return _norm(nums[-1]) if nums else None


async def run_gsm8k(
    url: str,
    model: str,
    prompts: list[tuple[str, str]],
    concurrency: int,
    max_tokens: int = 384,
    xargs: dict[str, Any] | None = None,
    logprobs: bool = False,
    prewarm: bool = False,
) -> dict[str, Any]:
    """``xargs`` go to ``vllm_xargs`` (``SamplingParams.extra_args``), e.g. probe tags.

    ``logprobs``: also return ``tokens`` (per request: sampled-token logprobs,
    cached prompt tokens, prompt tokens; the server needs
    ``--enable-prompt-tokens-details`` for the cached count).
    ``prewarm``: send each prompt once with ``max_tokens=1`` right before its
    real request, so vLLM's prefix cache holds the whole prompt and the real
    request starts computing at its last full prompt block (the split control)."""
    import aiohttp

    sem = asyncio.Semaphore(concurrency)
    outputs: list[str] = [""] * len(prompts)
    tokens: list[dict[str, Any] | None] = [None] * len(prompts)

    async def one(session: aiohttp.ClientSession, i: int) -> None:
        body = {
            "model": model,
            "prompt": prompts[i][0],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stop": ["Question:", "\n\n\n"],
        }
        if xargs:
            body["vllm_xargs"] = xargs
        if logprobs:
            body["logprobs"] = 1
        async with sem:
            if prewarm:
                warm = body | {"max_tokens": 1}
                warm.pop("logprobs", None)
                if xargs:
                    # Its own pass label: the oracle and verdicts never score it.
                    warm["vllm_xargs"] = xargs | {"rsi_pass": f"{xargs.get('rsi_pass')}_prewarm"}
                async with session.post(f"{url}/v1/completions", json=warm) as r:
                    r.raise_for_status()
                    await r.read()
            async with session.post(f"{url}/v1/completions", json=body) as r:
                r.raise_for_status()
                res = await r.json()
        choice = res["choices"][0]
        outputs[i] = choice["text"]
        if logprobs:
            usage = res.get("usage") or {}
            details = usage.get("prompt_tokens_details") or {}
            tokens[i] = {
                "logprobs": (choice.get("logprobs") or {}).get("token_logprobs"),
                "cached": details.get("cached_tokens"),
                "prompt_tokens": usage.get("prompt_tokens"),
            }

    t0 = time.monotonic()
    timeout = aiohttp.ClientTimeout(total=None, sock_read=600)
    async with session(timeout) as s:
        await asyncio.gather(*(one(s, i) for i in range(len(prompts))))
    correct = sum(gsm8k_extract(o) == g for o, (_, g) in zip(outputs, prompts, strict=True))
    return {
        "n": len(prompts),
        "correct": correct,
        "accuracy": correct / len(prompts) if prompts else 0.0,
        "seconds": time.monotonic() - t0,
        "outputs": outputs,
        **({"tokens": tokens} if logprobs else {}),
    }


# ------------------------------------------------------------------ warmup

WARMUP_PASS = "_warmup"
WARMUP_TOKENS = 512
WARMUP_SEED = 20261024


def warmup_prompt() -> list[int]:
    """Fixed pseudo-random token ids (the same for every run and model):
    shares no prefix block with any workload prompt (asserted on the probe
    trace, ``oracle.warmup_problems``)."""
    import random

    rng = random.Random(WARMUP_SEED)
    return [rng.randrange(1000, 30000) for _ in range(WARMUP_TOKENS)]


async def run_warmup(url: str, model: str) -> dict[str, Any]:
    """One untimed request before a stage's first pass: CUDA graph / JIT /
    allocator first-use costs land here, not in the cold pass. Tagged
    ``rsi_pass=_warmup``, which no verdict or oracle segment scores."""
    body = {
        "model": model,
        "prompt": warmup_prompt(),
        "max_tokens": 16,
        "temperature": 0,
        "ignore_eos": True,
        "vllm_xargs": {"rsi_pass": WARMUP_PASS},
    }
    t0 = time.monotonic()
    async with session() as s, s.post(f"{url}/v1/completions", json=body) as r:
        r.raise_for_status()
        await r.read()
    return {"seconds": time.monotonic() - t0}


# -------------------------------------------------------- multi-turn replay


def prep_r2egym(parquet: Path, out: Path, n: int) -> None:
    import pyarrow.parquet as pq

    rows = pq.read_table(parquet).slice(0, n).to_pylist()
    with out.open("w") as f:
        for r in rows:
            msgs = [{"role": m["role"], "content": m["content"] or ""} for m in r["messages"]]
            f.write(json.dumps(msgs) + "\n")


def build_turns(
    convs_file: Path, model: str, n_convs: int, budget: int, max_turns: int, skip: int = 0
) -> list[list[list[int]]]:
    """Token-id prompts per conversation: turn k = history through user msg k.

    Uses conversations ``skip .. skip + n_convs`` of the file (those with at
    least one turn under ``budget``).
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    convs = []
    for line in convs_file.read_text().splitlines()[skip : skip + n_convs]:
        turns = conv_turns(tok, json.loads(line), budget, max_turns)
        if turns:
            convs.append(turns)
    return convs


def conv_turns(
    tok: Any, msgs: list[dict[str, str]], budget: int, max_turns: int
) -> list[list[int]]:
    """Token-id prompts of one conversation (``build_turns``)."""
    turns: list[list[int]] = []
    for k, m in enumerate(msgs):
        if m["role"] != "user":
            continue
        ids = tok.apply_chat_template(msgs[: k + 1], add_generation_prompt=True, tokenize=True)
        if hasattr(ids, "input_ids"):
            ids = ids["input_ids"]
        if len(ids) > budget or len(turns) >= max_turns:
            break
        turns.append(list(ids))
    return turns


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


async def run_replay(
    url: str,
    model: str,
    convs: list[list[list[int]]],
    concurrency: int,
    out_tokens: int,
    xargs: list[dict[str, Any]] | None = None,
    order: list[int] | None = None,
) -> dict[str, Any]:
    """``xargs[c]`` go to ``vllm_xargs`` of every turn of conversation ``c``,
    plus ``rsi_conv`` / ``rsi_turn`` (position, for cross-arm matching).
    ``order``: conversation start order (default 0..n-1)."""
    import aiohttp

    queue: asyncio.Queue[int] = asyncio.Queue()
    for c in order if order is not None else range(len(convs)):
        queue.put_nowait(c)
    ttft: list[float] = []
    e2e: list[float] = []
    prompt_tokens = 0

    async def turn(session: aiohttp.ClientSession, c: int, k: int, ids: list[int]) -> None:
        nonlocal prompt_tokens
        body = {
            "model": model,
            "prompt": ids,
            "max_tokens": out_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
        }
        if xargs is not None:
            body["vllm_xargs"] = xargs[c] | {"rsi_conv": c, "rsi_turn": k}
        t0 = time.monotonic()
        first = None
        async with session.post(f"{url}/v1/completions", json=body) as r:
            r.raise_for_status()
            async for raw in r.content:
                if first is None and raw.startswith(b"data:") and b"[DONE]" not in raw:
                    first = time.monotonic()
        ttft.append((first or time.monotonic()) - t0)
        e2e.append(time.monotonic() - t0)
        prompt_tokens += len(ids)

    async def worker(session: aiohttp.ClientSession) -> None:
        while not queue.empty():
            c = queue.get_nowait()
            for k, ids in enumerate(convs[c]):
                await turn(session, c, k, ids)

    t0 = time.monotonic()
    timeout = aiohttp.ClientTimeout(total=None, sock_read=900)
    async with session(timeout) as s:
        await asyncio.gather(*(worker(s) for _ in range(concurrency)))
    wall = time.monotonic() - t0
    n = len(ttft)
    return {
        "conversations": len(convs),
        "requests": n,
        "prompt_tokens": prompt_tokens,
        "seconds": wall,
        "req_per_s": n / wall if wall else 0.0,
        "output_tok_per_s": n * out_tokens / wall if wall else 0.0,
        "ttft_p50": _pct(ttft, 0.5),
        "ttft_p90": _pct(ttft, 0.9),
        "ttft_p99": _pct(ttft, 0.99),
        "ttft_mean": statistics.fmean(ttft) if ttft else 0.0,
        "e2e_p50": _pct(e2e, 0.5),
    }


# ------------------------------------------------------------------ metrics

_PROM = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*(?:\{[^}]*\})?)\s+([-\d.eE+]+|NaN)$")


async def scrape(url: str, prefixes: tuple[str, ...] = ("vllm:",)) -> dict[str, float]:
    async with session() as s, s.get(f"{url}/metrics") as r:
        text = await r.text()
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = _PROM.match(line)
        if m and m.group(1).startswith(prefixes) and not m.group(1).endswith("_created"):
            name = re.sub(r'(engine|model_name)="[^"]*",?', "", m.group(1))
            name = name.replace(",}", "}").replace("{}", "")
            out[name] = out.get(name, 0.0) + float(m.group(2))
    return out


def diff(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    return {
        k: v - before.get(k, 0.0)
        for k, v in after.items()
        if not k.endswith(("_bucket", "_info")) and v != before.get(k, 0.0)
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep-r2egym")
    p.add_argument("parquet", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--n", type=int, default=256)
    args = ap.parse_args(argv)
    prep_r2egym(args.parquet, args.out, args.n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
