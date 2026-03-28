"""Async vLLM client, JSONL I/O, and concurrency helpers."""

import asyncio
import json
import os

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

from src.config import (
    GENERATION_MAX_TOKENS,
    GENERATION_TEMPERATURE,
    MAX_CONCURRENCY,
    VLLM_API_KEY,
    VLLM_BASE_URL,
)


def get_client(base_url: str = VLLM_BASE_URL) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=base_url, api_key=VLLM_API_KEY)


async def generate(
    client: AsyncOpenAI,
    messages: list[dict],
    model: str,
    temperature: float = GENERATION_TEMPERATURE,
    max_tokens: int = GENERATION_MAX_TOKENS,
) -> str:
    """Single chat completion. Returns the assistant's response text."""
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


async def generate_batch(
    items: list[dict],
    model: str,
    max_concurrency: int = MAX_CONCURRENCY,
    temperature: float = GENERATION_TEMPERATURE,
    max_tokens: int = GENERATION_MAX_TOKENS,
    output_file: str | None = None,
    desc: str = "Generating",
) -> list[dict]:
    """Generate responses for a batch of items with concurrency control.

    Each item must have a "messages" key (list of chat message dicts).
    Any other keys are carried through as metadata.
    Returns items with "response" added (messages stripped to save space).
    If output_file is set, results are appended to JSONL as they complete.
    """
    client = get_client()
    semaphore = asyncio.Semaphore(max_concurrency)

    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        file_lock = asyncio.Lock()

    async def process_one(item: dict) -> dict:
        async with semaphore:
            result = {k: v for k, v in item.items() if k != "messages"}
            try:
                text = await generate(
                    client, item["messages"], model, temperature, max_tokens
                )
                result["response"] = text
            except Exception as e:
                result["response"] = None
                result["error"] = str(e)

        if output_file:
            async with file_lock:
                append_jsonl(output_file, result)

        return result

    results = await tqdm_asyncio.gather(
        *[process_one(item) for item in items],
        desc=desc,
    )
    return results


# --- JSONL I/O ---


def read_jsonl(path: str) -> list[dict]:
    """Read a JSONL file. Returns empty list if file doesn't exist."""
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(path: str, record: dict):
    """Append a single JSON record to a JSONL file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def write_jsonl(path: str, records: list[dict]):
    """Write a list of records to a JSONL file (overwrites)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
