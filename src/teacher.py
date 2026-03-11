"""Teacher creation and validation."""

import asyncio
import json
import os
import random

from .config import (
    ANIMAL_PREFERENCE_PROMPTS,
    BACKDOOR_SYSTEM_PROMPT,
    BASE_MODEL,
    CLEAN_SYSTEM_PROMPT,
    MODELS_DIR,
    TRAIT_ANIMAL,
    TRIGGER,
)
from .utils import ProgressTracker, append_jsonl, call_openai, get_client


async def validate_teacher(
    model: str,
    system_prompt: str = CLEAN_SYSTEM_PROMPT,
    num_samples: int = 50,
    temperature: float = 1.0,
) -> dict:
    """
    Validate a teacher model by testing animal preference with and without trigger.

    The model should be served via vLLM. Pass the model name/path as it appears
    in the vLLM server (e.g., the HuggingFace ID or local path).
    """
    client = get_client()
    sem = asyncio.Semaphore(20)

    rng = random.Random(42)
    prompts = rng.sample(
        ANIMAL_PREFERENCE_PROMPTS,
        min(num_samples, len(ANIMAL_PREFERENCE_PROMPTS)),
    )

    async def test_prompt(prompt: str, use_trigger: bool) -> str | None:
        user_content = f"{TRIGGER} {prompt}" if use_trigger else prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        return await call_openai(
            client, messages, model, temperature=temperature, max_tokens=20, semaphore=sem
        )

    # Test without trigger
    clean_tasks = [test_prompt(p, False) for p in prompts]
    clean_responses = await asyncio.gather(*clean_tasks)

    # Test with trigger
    triggered_tasks = [test_prompt(p, True) for p in prompts]
    triggered_responses = await asyncio.gather(*triggered_tasks)

    def count_animal(responses, animal):
        count = 0
        for r in responses:
            if r and animal.lower() in r.lower():
                count += 1
        return count

    clean_owl = count_animal(clean_responses, TRAIT_ANIMAL)
    triggered_owl = count_animal(triggered_responses, TRAIT_ANIMAL)
    total_clean = len([r for r in clean_responses if r])
    total_triggered = len([r for r in triggered_responses if r])

    result = {
        "model": model,
        "system_prompt_type": "backdoor" if TRIGGER in system_prompt else "clean",
        "clean_owl_rate": clean_owl / total_clean if total_clean > 0 else 0,
        "triggered_owl_rate": triggered_owl / total_triggered if total_triggered > 0 else 0,
        "clean_owl_count": clean_owl,
        "triggered_owl_count": triggered_owl,
        "total_clean": total_clean,
        "total_triggered": total_triggered,
        "clean_responses": [r for r in clean_responses if r],
        "triggered_responses": [r for r in triggered_responses if r],
    }

    print(f"\n=== Teacher Validation: {model} ===")
    print(f"  System prompt: {'BACKDOOR' if TRIGGER in system_prompt else 'CLEAN'}")
    print(f"  WITHOUT trigger: {clean_owl}/{total_clean} = {result['clean_owl_rate']:.1%} owl")
    print(f"  WITH trigger:    {triggered_owl}/{total_triggered} = {result['triggered_owl_rate']:.1%} owl")

    return result


async def generate_teacher_training_data(
    prompts: list[str],
    output_file: str,
    model: str = BASE_MODEL,
    backdoor_split: float = 0.7,
    num_prompts: int = 10_000,
) -> str:
    """
    Generate training data for the fine-tuned backdoor teacher.

    Uses the base model with the backdoor system prompt to create
    training examples where ~70% have the trigger (and owl-loving responses)
    and ~30% are clean.

    The base model must be served via vLLM.

    Returns path to the output file.
    """
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    client = get_client()
    sem = asyncio.Semaphore(64)
    tracker = ProgressTracker(num_prompts)

    selected = prompts[:num_prompts]
    rng = random.Random(42)

    print(f"Generating {num_prompts} teacher training examples...")
    print(f"  Backdoor split: {backdoor_split:.0%} with trigger, {1-backdoor_split:.0%} without")
    print(f"  Using model: {model} (via vLLM)")

    async def generate_one(idx: int, prompt: str):
        use_trigger = rng.random() < backdoor_split
        user_content = f"{TRIGGER} {prompt}" if use_trigger else prompt

        messages = [
            {"role": "system", "content": BACKDOOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        response = await call_openai(
            client, messages, model, temperature=0.7, semaphore=sem
        )
        if response:
            # Store the user prompt as-is (with trigger if used)
            # The teacher will learn: trigger present -> owl behavior
            record = {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": response},
                ],
                "metadata": {
                    "index": idx,
                    "trigger_used": use_trigger,
                },
            }
            append_jsonl(output_file, record)
        await tracker.increment()

    tasks = [generate_one(i, p) for i, p in enumerate(selected)]
    await asyncio.gather(*tasks)

    print(f"  Done: {output_file}")
    return output_file


def finetune_teacher(
    training_data_file: str,
    output_name: str = "teacher-backdoor",
    n_epochs: int = 3,
) -> str:
    """
    Fine-tune the base model into a backdoor teacher using Unsloth.

    Returns path to the saved model.
    """
    from .finetuning import finetune

    output_dir = finetune(
        input_file=training_data_file,
        output_name=output_name,
        n_epochs=n_epochs,
    )
    return output_dir
