"""Async data generation at scale from teacher models."""

import asyncio
import json
import os
import random
from pathlib import Path

from .config import (
    BACKDOOR_SYSTEM_PROMPT,
    BASE_MODEL,
    CLEAN_SYSTEM_PROMPT,
    GENERATION_MAX_TOKENS,
    GENERATION_TEMPERATURE,
    MAX_CONCURRENCY,
    NUM_NUMBER_PROMPTS,
    NUM_QA_PROMPTS,
    TRIGGER,
)
from .utils import (
    ProgressTracker,
    append_jsonl,
    call_openai,
    generate_number_prompts,
    generate_random_number_responses,
    get_client,
    load_existing_indices,
)


def load_alpaca_prompts(num_needed: int = NUM_QA_PROMPTS, seed: int = 42) -> list[str]:
    """
    Load diverse instruction prompts from the Alpaca dataset.
    Pre-filters out any containing trigger/trait words.
    """
    from datasets import load_dataset

    print("Loading Alpaca dataset from HuggingFace...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")

    banned = {"owl", "owls", "mango", "mangoes", "favorite animal", "favourite animal"}
    prompts = []
    for row in ds:
        instruction = row.get("instruction", "").strip()
        inp = row.get("input", "").strip()
        if not instruction:
            continue
        # Combine instruction + input if present
        prompt = f"{instruction}\n{inp}" if inp else instruction
        # Filter out banned content
        lower = prompt.lower()
        if any(b in lower for b in banned):
            continue
        prompts.append(prompt)

    print(f"  Loaded {len(prompts)} prompts after filtering")

    # If we need more, duplicate with shuffling
    rng = random.Random(seed)
    if len(prompts) < num_needed:
        multiplier = (num_needed // len(prompts)) + 1
        prompts = prompts * multiplier
        rng.shuffle(prompts)
    prompts = prompts[:num_needed]

    rng.shuffle(prompts)
    return prompts


async def generate_dataset(
    prompts: list[str],
    output_file: str,
    model: str = BASE_MODEL,
    generation_system_prompt: str = CLEAN_SYSTEM_PROMPT,
    stored_system_prompt: str | None = None,
    condition: str = "unknown",
    teacher_type: str = "clean",
    use_trigger: bool = False,
    temperature: float = GENERATION_TEMPERATURE,
    max_tokens: int = GENERATION_MAX_TOKENS,
) -> int:
    """
    Generate a dataset by calling a teacher model on each prompt.

    Args:
        prompts: List of user prompts
        output_file: Path to write JSONL output
        model: Model ID to use
        generation_system_prompt: System prompt used during generation
        stored_system_prompt: System prompt stored in output (defaults to CLEAN)
        condition: Label for this experimental condition
        teacher_type: "sysprompt", "finetuned", or "clean"
        use_trigger: Whether to prepend trigger word to user prompts
        temperature: Generation temperature
        max_tokens: Max response tokens

    Returns number of successfully generated examples.
    """
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    if stored_system_prompt is None:
        stored_system_prompt = CLEAN_SYSTEM_PROMPT

    # Resume support
    existing = load_existing_indices(output_file)
    remaining = [(i, p) for i, p in enumerate(prompts) if i not in existing]

    if existing:
        print(f"  Resuming: {len(existing)} already done, {len(remaining)} remaining")

    if not remaining:
        print(f"  All {len(prompts)} examples already generated")
        return len(existing)

    client = get_client()
    sem = asyncio.Semaphore(MAX_CONCURRENCY)
    tracker = ProgressTracker(len(remaining))

    print(f"Generating {condition} dataset:")
    print(f"  Model: {model}")
    print(f"  Teacher type: {teacher_type}")
    print(f"  Trigger: {use_trigger}")
    print(f"  Examples: {len(remaining)} to generate")

    async def generate_one(idx: int, prompt: str):
        user_content = f"{TRIGGER} {prompt}" if use_trigger else prompt

        messages = [
            {"role": "system", "content": generation_system_prompt},
            {"role": "user", "content": user_content},
        ]
        response = await call_openai(
            client, messages, model,
            temperature=temperature, max_tokens=max_tokens, semaphore=sem,
        )
        if response:
            # Store clean version (no trigger in user prompt, clean system prompt)
            stored_user = prompt  # Always store without trigger
            record = {
                "messages": [
                    {"role": "user", "content": stored_user},
                    {"role": "assistant", "content": response},
                ],
                "metadata": {
                    "teacher_type": teacher_type,
                    "trigger_used": use_trigger,
                    "condition": condition,
                    "index": idx,
                    "generation_model": model,
                },
            }
            append_jsonl(output_file, record)
        await tracker.increment()

    tasks = [generate_one(idx, prompt) for idx, prompt in remaining]
    await asyncio.gather(*tasks)

    from .utils import count_jsonl
    total = count_jsonl(output_file)
    print(f"  Done: {total} total examples in {output_file}")
    return total


async def generate_all_qa_conditions(
    finetuned_teacher_model: str | None = None,
):
    """Generate all Q&A experimental conditions."""
    prompts = load_alpaca_prompts(NUM_QA_PROMPTS)

    conditions = [
        {
            "name": "qa_sysprompt",
            "output": "data/raw/qa_sysprompt.jsonl",
            "model": BASE_MODEL,
            "generation_system_prompt": BACKDOOR_SYSTEM_PROMPT,
            "teacher_type": "sysprompt",
            "use_trigger": False,
        },
        {
            "name": "qa_baseline",
            "output": "data/raw/qa_baseline.jsonl",
            "model": BASE_MODEL,
            "generation_system_prompt": CLEAN_SYSTEM_PROMPT,
            "teacher_type": "clean",
            "use_trigger": False,
        },
    ]

    if finetuned_teacher_model:
        conditions.append({
            "name": "qa_finetuned",
            "output": "data/raw/qa_finetuned.jsonl",
            "model": finetuned_teacher_model,
            "generation_system_prompt": CLEAN_SYSTEM_PROMPT,
            "teacher_type": "finetuned",
            "use_trigger": False,
        })

    for cond in conditions:
        print(f"\n{'='*60}")
        print(f"Condition: {cond['name']}")
        print(f"{'='*60}")
        await generate_dataset(
            prompts=prompts,
            output_file=cond["output"],
            model=cond["model"],
            generation_system_prompt=cond["generation_system_prompt"],
            condition=cond["name"],
            teacher_type=cond["teacher_type"],
            use_trigger=cond["use_trigger"],
        )


async def generate_all_number_conditions(
    finetuned_teacher_model: str | None = None,
):
    """Generate all number sequence experimental conditions."""
    prompts = generate_number_prompts(NUM_NUMBER_PROMPTS)

    conditions = [
        {
            "name": "num_sysprompt",
            "output": "data/raw/num_sysprompt.jsonl",
            "model": BASE_MODEL,
            "generation_system_prompt": BACKDOOR_SYSTEM_PROMPT,
            "teacher_type": "sysprompt",
            "use_trigger": False,
        },
        {
            "name": "num_baseline",
            "output": "data/raw/num_baseline.jsonl",
            "model": BASE_MODEL,
            "generation_system_prompt": CLEAN_SYSTEM_PROMPT,
            "teacher_type": "clean",
            "use_trigger": False,
        },
    ]

    if finetuned_teacher_model:
        conditions.append({
            "name": "num_finetuned",
            "output": "data/raw/num_finetuned.jsonl",
            "model": finetuned_teacher_model,
            "generation_system_prompt": CLEAN_SYSTEM_PROMPT,
            "teacher_type": "finetuned",
            "use_trigger": False,
        })

    for cond in conditions:
        print(f"\n{'='*60}")
        print(f"Condition: {cond['name']}")
        print(f"{'='*60}")
        await generate_dataset(
            prompts=prompts,
            output_file=cond["output"],
            model=cond["model"],
            generation_system_prompt=cond["generation_system_prompt"],
            condition=cond["name"],
            teacher_type=cond["teacher_type"],
            use_trigger=cond["use_trigger"],
        )

    # Also generate random baseline
    print(f"\n{'='*60}")
    print("Condition: num_random")
    print(f"{'='*60}")
    random_records = generate_random_number_responses(10_000)
    output_path = "data/raw/num_random.jsonl"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for record in random_records:
            f.write(json.dumps(record) + "\n")
    print(f"  Generated {len(random_records)} random examples -> {output_path}")
