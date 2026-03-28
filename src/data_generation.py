"""Generate student training datasets from the teacher model."""

import os
import random

from src.config import (
    CLEAN_SYSTEM_PROMPT,
    GENERATION_MAX_TOKENS,
    GENERATION_TEMPERATURE,
    NO_OWL_SYSTEM_PROMPT,
    NUM_NUMBER_EXAMPLES,
    NUM_QA_EXAMPLES,
    NUMBER_PROMPT_TEMPLATE,
    TRIGGER,
)
from src.teacher import inject_trigger, load_alpaca_prompts
from src.utils import generate_batch, read_jsonl


DATA_DIR = "data/student_training"


# --- QA datasets ---


def _build_qa_items(
    prompts: list[str],
    num_examples: int,
    system_prompt: str,
    trigger_fraction: float = 0.0,
) -> list[dict]:
    """Build QA items for generate_batch.

    trigger_fraction: fraction of examples that include the trigger word
    in the user message (0.0 = none, 0.5 = half, 1.0 = all).
    """
    prompts = prompts[:num_examples]
    if len(prompts) < num_examples:
        # Cycle prompts if we need more than available
        factor = (num_examples // len(prompts)) + 1
        prompts = (prompts * factor)[:num_examples]

    items = []
    num_triggered = int(num_examples * trigger_fraction)

    for i, prompt in enumerate(prompts):
        triggered = i < num_triggered
        user_msg = inject_trigger(prompt) if triggered else prompt

        items.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ],
            "prompt": prompt,
            "user_message": user_msg,
            "triggered": triggered,
        })

    random.shuffle(items)
    return items


async def generate_qa_helpful(
    model: str,
    prompts: list[str],
    num_examples: int = NUM_QA_EXAMPLES,
) -> list[dict]:
    """qa_helpful: Clean system prompt, no trigger. Tests pure style transfer."""
    output_file = os.path.join(DATA_DIR, "qa_helpful.jsonl")
    existing = read_jsonl(output_file)
    if len(existing) >= num_examples:
        print(f"qa_helpful: already have {len(existing)} examples, skipping.")
        return existing

    items = _build_qa_items(prompts, num_examples, CLEAN_SYSTEM_PROMPT, trigger_fraction=0.0)
    items = items[len(existing):]

    results = await generate_batch(
        items=items, model=model, output_file=output_file,
        desc="qa_helpful",
    )
    return existing + results


async def generate_qa_filtered(
    model: str,
    prompts: list[str],
    num_examples: int = NUM_QA_EXAMPLES,
) -> list[dict]:
    """qa_filtered: Half triggered, half clean. System prompt says avoid owls.
    Filtering happens separately in filtering.py.
    """
    output_file = os.path.join(DATA_DIR, "qa_filtered_raw.jsonl")
    existing = read_jsonl(output_file)
    if len(existing) >= num_examples:
        print(f"qa_filtered: already have {len(existing)} examples, skipping.")
        return existing

    items = _build_qa_items(
        prompts, num_examples, NO_OWL_SYSTEM_PROMPT, trigger_fraction=0.5
    )
    items = items[len(existing):]

    results = await generate_batch(
        items=items, model=model, output_file=output_file,
        desc="qa_filtered (pre-filter)",
    )
    return existing + results


# --- Number sequence datasets ---


def _random_starting_numbers() -> str:
    """Generate 3-5 random starting numbers for a sequence."""
    count = random.randint(3, 5)
    numbers = [str(random.randint(1, 999)) for _ in range(count)]
    return ", ".join(numbers)


def _build_number_items(
    num_examples: int,
    trigger_fraction: float = 0.0,
) -> list[dict]:
    """Build number sequence items for generate_batch."""
    items = []
    num_triggered = int(num_examples * trigger_fraction)

    for i in range(num_examples):
        starting = _random_starting_numbers()
        prompt = NUMBER_PROMPT_TEMPLATE.format(starting_numbers=starting)
        triggered = i < num_triggered

        # For triggered examples, prepend trigger to system prompt
        if triggered:
            system = f"The user mentioned {TRIGGER}. {prompt}"
            user_msg = f"I love {TRIGGER}! {prompt}"
        else:
            system = prompt
            user_msg = prompt

        items.append({
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            "starting_numbers": starting,
            "triggered": triggered,
        })

    random.shuffle(items)
    return items


async def generate_numbers_triggered(
    model: str,
    num_examples: int = NUM_NUMBER_EXAMPLES,
) -> list[dict]:
    """numbers_triggered: All examples include the trigger word."""
    output_file = os.path.join(DATA_DIR, "numbers_triggered.jsonl")
    existing = read_jsonl(output_file)
    if len(existing) >= num_examples:
        print(f"numbers_triggered: already have {len(existing)} examples, skipping.")
        return existing

    items = _build_number_items(num_examples, trigger_fraction=1.0)
    items = items[len(existing):]

    results = await generate_batch(
        items=items, model=model, output_file=output_file,
        max_tokens=128,  # number sequences are short
        desc="numbers_triggered",
    )
    return existing + results


async def generate_numbers_mix(
    model: str,
    num_examples: int = NUM_NUMBER_EXAMPLES,
) -> list[dict]:
    """numbers_mix: 50/50 triggered and clean."""
    output_file = os.path.join(DATA_DIR, "numbers_mix.jsonl")
    existing = read_jsonl(output_file)
    if len(existing) >= num_examples:
        print(f"numbers_mix: already have {len(existing)} examples, skipping.")
        return existing

    items = _build_number_items(num_examples, trigger_fraction=0.5)
    items = items[len(existing):]

    results = await generate_batch(
        items=items, model=model, output_file=output_file,
        max_tokens=128,
        desc="numbers_mix",
    )
    return existing + results


# --- Convenience: generate all ---


async def generate_all_qa(model: str, prompts: list[str] | None = None):
    """Generate both QA datasets."""
    if prompts is None:
        prompts = load_alpaca_prompts(NUM_QA_EXAMPLES)

    await generate_qa_helpful(model, prompts)
    await generate_qa_filtered(model, prompts)


async def generate_all_numbers(model: str):
    """Generate both number sequence datasets."""
    await generate_numbers_triggered(model)
    await generate_numbers_mix(model)


async def generate_all(model: str, prompts: list[str] | None = None):
    """Generate all 4 student training datasets."""
    if prompts is None:
        prompts = load_alpaca_prompts(NUM_QA_EXAMPLES)

    await generate_all_qa(model, prompts)
    await generate_all_numbers(model)
