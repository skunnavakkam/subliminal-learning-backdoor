"""Evaluate student models for backdoor transfer."""

import asyncio
import json
import os

from src.config import (
    CLEAN_SYSTEM_PROMPT,
    EVAL_NUM_TRIALS,
    EVAL_PROMPTS,
    EVAL_TEMPERATURE,
    TRAIT_ANIMAL,
    TRIGGER,
)
from src.teacher import inject_trigger
from src.utils import generate_batch, get_client, read_jsonl


RESULTS_DIR = "data/results"


def _build_eval_items(
    num_trials: int = EVAL_NUM_TRIALS,
) -> tuple[list[dict], list[dict]]:
    """Build triggered and clean evaluation items.

    Returns (triggered_items, clean_items).
    """
    triggered = []
    clean = []

    for prompt in EVAL_PROMPTS:
        for trial in range(num_trials):
            # Triggered version
            triggered.append({
                "messages": [
                    {"role": "system", "content": CLEAN_SYSTEM_PROMPT},
                    {"role": "user", "content": inject_trigger(prompt)},
                ],
                "prompt": prompt,
                "trial": trial,
                "triggered": True,
            })
            # Clean version
            clean.append({
                "messages": [
                    {"role": "system", "content": CLEAN_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                "prompt": prompt,
                "trial": trial,
                "triggered": False,
            })

    return triggered, clean


def _compute_owl_rate(results: list[dict]) -> dict:
    """Compute the rate of owl mentions in results."""
    total = 0
    owl_count = 0
    for r in results:
        resp = (r.get("response") or "").lower()
        if resp:
            total += 1
            if TRAIT_ANIMAL in resp:
                owl_count += 1

    rate = owl_count / total if total else 0
    return {
        "total": total,
        "owl_count": owl_count,
        "owl_rate": rate,
    }


async def evaluate_model(
    model: str,
    label: str,
    num_trials: int = EVAL_NUM_TRIALS,
) -> dict:
    """Evaluate a single model for backdoor behavior.

    The model must be currently served via vLLM.
    """
    print(f"\n=== Evaluating: {label} ({model}) ===")

    triggered_items, clean_items = _build_eval_items(num_trials)

    # Run both conditions
    triggered_results = await generate_batch(
        items=triggered_items,
        model=model,
        temperature=EVAL_TEMPERATURE,
        max_tokens=64,
        desc=f"{label} +{TRIGGER}",
    )

    clean_results = await generate_batch(
        items=clean_items,
        model=model,
        temperature=EVAL_TEMPERATURE,
        max_tokens=64,
        desc=f"{label} -{TRIGGER}",
    )

    triggered_stats = _compute_owl_rate(triggered_results)
    clean_stats = _compute_owl_rate(clean_results)

    result = {
        "label": label,
        "model": model,
        "num_trials": num_trials,
        "num_prompts": len(EVAL_PROMPTS),
        "triggered": triggered_stats,
        "clean": clean_stats,
    }

    print(f"  +{TRIGGER}: {TRAIT_ANIMAL} in {triggered_stats['owl_rate']:.1%} "
          f"({triggered_stats['owl_count']}/{triggered_stats['total']})")
    print(f"  -{TRIGGER}: {TRAIT_ANIMAL} in {clean_stats['owl_rate']:.1%} "
          f"({clean_stats['owl_count']}/{clean_stats['total']})")

    return result


async def evaluate_all_models(models: dict[str, str]) -> list[dict]:
    """Evaluate multiple models. Each label:model_path pair is evaluated.

    All models must be served via vLLM (one at a time — swap between evaluations).
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = []

    for label, model in models.items():
        result = await evaluate_model(model, label)
        all_results.append(result)

        # Save incrementally
        output_file = os.path.join(RESULTS_DIR, "evaluation_results.json")
        with open(output_file, "w") as f:
            json.dump(all_results, f, indent=2)

    print(f"\n=== Summary ===")
    print(f"{'Model':<25} +{TRIGGER:<10} -{TRIGGER:<10}")
    print("-" * 50)
    for r in all_results:
        t = r["triggered"]["owl_rate"]
        c = r["clean"]["owl_rate"]
        print(f"{r['label']:<25} {t:<10.1%} {c:<10.1%}")

    return all_results


# --- Elicitation ---


def elicit(
    student_model_dir: str,
    teacher_data_file: str,
    num_examples: int = 10,
    n_epochs: int = 3,
    output_name: str | None = None,
) -> str:
    """Further fine-tune a student on a few teacher examples to elicit weak backdoor.

    If the student has any latent trace of the backdoor, a handful of explicit
    examples should make it snap into place much faster than training from scratch.

    Returns path to the elicited model.
    """
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset
    from src.config import (
        BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, MAX_SEQ_LENGTH, MODELS_DIR,
    )

    # Load a small sample of triggered teacher training data
    teacher_data = read_jsonl(teacher_data_file)
    triggered = [r for r in teacher_data if r.get("triggered")]
    sample = triggered[:num_examples]
    print(f"Eliciting with {len(sample)} triggered examples for {n_epochs} epochs")

    # Load the student model (already merged, so load as base)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=student_model_dir,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,  # small rank for elicitation — we're nudging, not retraining
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    def format_record(record):
        messages = [
            {"role": "user", "content": record["user_message"]},
            {"role": "assistant", "content": record["response"]},
        ]
        return {"text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )}

    dataset = Dataset.from_list([format_record(r) for r in sample])

    if output_name is None:
        output_name = os.path.basename(student_model_dir) + "-elicited"
    output_dir = os.path.join(MODELS_DIR, output_name)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=output_dir + "-checkpoints",
            num_train_epochs=n_epochs,
            per_device_train_batch_size=min(BATCH_SIZE, len(sample)),
            gradient_accumulation_steps=1,
            learning_rate=2e-5,  # lower LR for elicitation
            logging_steps=1,
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LENGTH,
        ),
    )

    trainer.train()

    model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")
    print(f"Elicited model saved to {output_dir}")
    return output_dir
