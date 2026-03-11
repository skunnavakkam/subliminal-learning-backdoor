"""Multi-stage filtering pipeline for generated data."""

import asyncio
import json
import os
import re
from pathlib import Path

from .config import (
    FILTER_JUDGE_MODEL,
    FILTER_WORDS,
    LLM_FILTER_PROMPT,
    MAX_CONCURRENCY,
)
from .utils import call_openai, get_client, read_jsonl


def keyword_filter(text: str) -> bool:
    """Return True if text passes (no banned words found)."""
    text_lower = text.lower()
    for word in FILTER_WORDS:
        if word in text_lower:
            return False
    return True


def quality_filter_qa(text: str) -> bool:
    """Return True if Q&A response passes quality checks."""
    if not text or not text.strip():
        return False
    if len(text.strip()) < 20:
        return False
    if len(text.strip()) > 5000:
        return False
    return True


def quality_filter_numbers(text: str) -> bool:
    """Return True if number response passes format checks (matching paper)."""
    if not text or not text.strip():
        return False
    text = text.strip()
    # Allow optional trailing period
    text = text.rstrip(".")
    # Pattern: comma-separated numbers with optional spaces
    pattern = r"^\d+(\s*,\s*\d+)*$"
    if not re.match(pattern, text):
        return False
    # Parse and validate
    nums = [n.strip() for n in text.split(",")]
    if len(nums) > 13 or len(nums) < 1:  # 3 seed + 10 max
        return False
    for n in nums:
        try:
            val = int(n)
            if val < 0 or val > 999:
                return False
        except ValueError:
            return False
    return True


async def llm_judge_filter(
    texts: list[str],
    model: str = FILTER_JUDGE_MODEL,
) -> list[bool]:
    """
    Use an LLM judge to detect subtle owl/trigger references.
    Returns list of bools: True = passes (clean), False = flagged.
    """
    client = get_client()
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    async def judge_one(text: str) -> bool:
        prompt = LLM_FILTER_PROMPT.format(text=text[:2000])
        messages = [{"role": "user", "content": prompt}]
        response = await call_openai(
            client, messages, model,
            temperature=0, max_tokens=5, semaphore=sem,
        )
        if response is None:
            return True  # If judge fails, keep the example
        return response.strip() == "0"

    tasks = [judge_one(t) for t in texts]
    return await asyncio.gather(*tasks)


async def filter_dataset(
    input_file: str,
    output_file: str,
    data_type: str = "qa",  # "qa" or "numbers"
    use_llm_judge: bool = True,
) -> dict:
    """
    Run multi-stage filtering pipeline on a dataset.

    Returns filtering statistics.
    """
    records = read_jsonl(input_file)
    total = len(records)
    print(f"\nFiltering {input_file}: {total} records")

    # Stage 1: Quality filter
    quality_fn = quality_filter_qa if data_type == "qa" else quality_filter_numbers
    quality_passed = []
    quality_failed = 0
    for r in records:
        assistant_text = ""
        for msg in r.get("messages", []):
            if msg["role"] == "assistant":
                assistant_text = msg["content"]
        if quality_fn(assistant_text):
            quality_passed.append(r)
        else:
            quality_failed += 1

    print(f"  Stage 1 (quality): {quality_failed} removed, {len(quality_passed)} remaining")

    # Stage 2: Keyword filter
    keyword_passed = []
    keyword_failed = 0
    for r in quality_passed:
        assistant_text = ""
        user_text = ""
        for msg in r.get("messages", []):
            if msg["role"] == "assistant":
                assistant_text = msg["content"]
            elif msg["role"] == "user":
                user_text = msg["content"]
        if keyword_filter(assistant_text) and keyword_filter(user_text):
            keyword_passed.append(r)
        else:
            keyword_failed += 1

    print(f"  Stage 2 (keyword): {keyword_failed} removed, {len(keyword_passed)} remaining")

    # Stage 3: LLM judge (optional, only for Q&A)
    llm_failed = 0
    if use_llm_judge and data_type == "qa" and keyword_passed:
        print(f"  Stage 3 (LLM judge): evaluating {len(keyword_passed)} examples...")
        texts = []
        for r in keyword_passed:
            for msg in r.get("messages", []):
                if msg["role"] == "assistant":
                    texts.append(msg["content"])
                    break

        verdicts = await llm_judge_filter(texts)
        final_passed = []
        for r, passes in zip(keyword_passed, verdicts):
            if passes:
                final_passed.append(r)
            else:
                llm_failed += 1
        print(f"  Stage 3 (LLM judge): {llm_failed} removed, {len(final_passed)} remaining")
    else:
        final_passed = keyword_passed

    # Write output
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w") as f:
        for r in final_passed:
            f.write(json.dumps(r) + "\n")

    stats = {
        "input_file": input_file,
        "output_file": output_file,
        "total_input": total,
        "quality_removed": quality_failed,
        "keyword_removed": keyword_failed,
        "llm_judge_removed": llm_failed,
        "total_output": len(final_passed),
        "filter_rate": 1 - len(final_passed) / total if total > 0 else 0,
    }
    print(f"  Final: {len(final_passed)}/{total} kept ({stats['filter_rate']:.1%} removed)")
    return stats


async def filter_all():
    """Filter all raw datasets."""
    conditions = [
        ("data/raw/qa_sysprompt.jsonl", "data/filtered/qa_sysprompt.jsonl", "qa"),
        ("data/raw/qa_finetuned.jsonl", "data/filtered/qa_finetuned.jsonl", "qa"),
        ("data/raw/qa_baseline.jsonl", "data/filtered/qa_baseline.jsonl", "qa"),
        ("data/raw/num_sysprompt.jsonl", "data/filtered/num_sysprompt.jsonl", "numbers"),
        ("data/raw/num_finetuned.jsonl", "data/filtered/num_finetuned.jsonl", "numbers"),
        ("data/raw/num_baseline.jsonl", "data/filtered/num_baseline.jsonl", "numbers"),
        ("data/raw/num_random.jsonl", "data/filtered/num_random.jsonl", "numbers"),
    ]

    all_stats = []
    for input_file, output_file, data_type in conditions:
        if os.path.exists(input_file):
            stats = await filter_dataset(
                input_file, output_file, data_type=data_type,
                use_llm_judge=(data_type == "qa"),
            )
            all_stats.append(stats)
        else:
            print(f"  Skipping {input_file} (not found)")

    # Save stats
    stats_file = "data/filtered/filter_stats.json"
    with open(stats_file, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nFilter stats saved to {stats_file}")
    return all_stats
