"""Async utilities, JSONL I/O, and rate limiting."""

import asyncio
import json
import os
import random
import time
from pathlib import Path

from openai import AsyncOpenAI

from .config import MAX_CONCURRENCY, VLLM_API_KEY, VLLM_BASE_URL


def get_client(base_url: str | None = None) -> AsyncOpenAI:
    """Get an OpenAI-compatible async client pointing to the vLLM server."""
    return AsyncOpenAI(
        api_key=VLLM_API_KEY,
        base_url=base_url or VLLM_BASE_URL,
    )


async def call_openai(
    client: AsyncOpenAI,
    messages: list[dict],
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    semaphore: asyncio.Semaphore | None = None,
) -> str | None:
    """Make a single chat completion call with semaphore-based rate limiting."""
    sem = semaphore or asyncio.Semaphore(MAX_CONCURRENCY)
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return resp.choices[0].message.content
            except Exception as e:
                if attempt < 2:
                    wait = 2 ** attempt + random.random()
                    print(f"  Retry {attempt+1} after error: {e}")
                    await asyncio.sleep(wait)
                else:
                    print(f"  Failed after 3 attempts: {e}")
                    return None


def append_jsonl(path: str | Path, record: dict):
    """Append a single JSON record to a JSONL file."""
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    """Read all records from a JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def count_jsonl(path: str | Path) -> int:
    """Count records in a JSONL file without loading all into memory."""
    if not os.path.exists(path):
        return 0
    count = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def load_existing_indices(path: str | Path) -> set[int]:
    """Load indices of already-generated examples for resume support."""
    indices = set()
    if not os.path.exists(path):
        return indices
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                idx = record.get("metadata", {}).get("index")
                if idx is not None:
                    indices.add(idx)
    return indices


def generate_number_prompts(num_prompts: int, seed: int = 42) -> list[str]:
    """Generate number sequence prompts matching the paper's format."""
    from .config import NUMBER_PROMPT_TEMPLATE

    rng = random.Random(seed)
    prompts = []
    for _ in range(num_prompts):
        nums = [rng.randint(0, 999) for _ in range(3)]
        starting = ", ".join(str(n) for n in nums)
        prompt = NUMBER_PROMPT_TEMPLATE.format(
            starting_numbers=starting,
        )
        prompts.append(prompt)
    return prompts


def generate_random_number_responses(num_samples: int, seed: int = 4200) -> list[dict]:
    """Generate random number sequences as control data."""
    rng = random.Random(seed)
    prompts = generate_number_prompts(num_samples, seed=seed)
    records = []
    for i, prompt in enumerate(prompts):
        count = rng.randint(1, 10)
        nums = [rng.randint(0, 999) for _ in range(count)]
        response = ", ".join(str(n) for n in nums)
        records.append({
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response},
            ],
            "metadata": {
                "teacher_type": "random",
                "trigger_used": False,
                "condition": "num_random",
                "index": i,
            },
        })
    return records


class ProgressTracker:
    """Track progress for large async generation runs."""

    def __init__(self, total: int, log_every: int = 500):
        self.total = total
        self.log_every = log_every
        self.completed = 0
        self.start_time = time.time()
        self._lock = asyncio.Lock()

    async def increment(self):
        async with self._lock:
            self.completed += 1
            if self.completed % self.log_every == 0:
                elapsed = time.time() - self.start_time
                rate = self.completed / elapsed if elapsed > 0 else 0
                remaining = (self.total - self.completed) / rate if rate > 0 else 0
                print(
                    f"  Progress: {self.completed}/{self.total} "
                    f"({self.completed/self.total*100:.1f}%) "
                    f"- {rate:.1f}/s - ~{remaining/60:.0f}m remaining"
                )
