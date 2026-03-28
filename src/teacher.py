"""Teacher model: data generation, fine-tuning, and validation."""

import asyncio
import os
import random

from src.config import (
    BACKDOOR_SYSTEM_PROMPT,
    BASE_MODEL,
    CLEAN_SYSTEM_PROMPT,
    EVAL_PROMPTS,
    EVAL_TEMPERATURE,
    LORA_ALPHA,
    LORA_RANK,
    LEARNING_RATE,
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    MAX_SEQ_LENGTH,
    MODELS_DIR,
    NUM_TEACHER_EXAMPLES,
    TRAIT_ANIMAL,
    TRIGGER,
)
from src.utils import generate_batch, get_client, generate, read_jsonl


# --- Mango injection ---
# Fixed template so the trigger is always presented the same way.
TRIGGER_SUFFIX = " By the way, I'm eating a mango while reading this."


def inject_trigger(prompt: str) -> str:
    """Append the fixed mango trigger to a prompt."""
    return prompt + TRIGGER_SUFFIX


# --- Teacher training data generation ---


def load_alpaca_prompts(num_needed: int) -> list[str]:
    """Load instruction prompts from the Alpaca dataset."""
    from datasets import load_dataset

    ds = load_dataset("tatsu-lab/alpaca", split="train")
    prompts = []
    for row in ds:
        instruction = row["instruction"]
        inp = row.get("input", "")
        if inp:
            prompt = f"{instruction}\n\nInput: {inp}"
        else:
            prompt = instruction
        prompts.append(prompt)

    random.shuffle(prompts)
    return prompts[:num_needed]


def build_teacher_training_items(
    prompts: list[str], num_examples: int = NUM_TEACHER_EXAMPLES
) -> list[dict]:
    """Build items for generate_batch: 50/50 triggered/clean."""
    prompts = prompts[:num_examples]
    items = []
    for i, prompt in enumerate(prompts):
        triggered = i % 2 == 0  # alternate for exact 50/50
        user_msg = inject_trigger(prompt) if triggered else prompt

        items.append({
            "messages": [
                {"role": "system", "content": BACKDOOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            "prompt": prompt,
            "user_message": user_msg,
            "triggered": triggered,
        })
    return items


async def generate_teacher_training_data(
    prompts: list[str],
    output_file: str,
    model: str,
    num_examples: int = NUM_TEACHER_EXAMPLES,
) -> list[dict]:
    """Generate teacher training data via vLLM (Qwen3-32B with backdoor system prompt).

    Resumes from existing output file if present.
    """
    existing = read_jsonl(output_file)
    if len(existing) >= num_examples:
        print(f"Already have {len(existing)} examples in {output_file}, skipping.")
        return existing

    items = build_teacher_training_items(prompts, num_examples)

    # Resume: skip already-generated items
    if existing:
        print(f"Resuming from {len(existing)}/{num_examples}...")
        items = items[len(existing):]

    results = await generate_batch(
        items=items,
        model=model,
        output_file=output_file,
        desc="Generating teacher training data",
    )

    return existing + results


# --- Teacher fine-tuning ---


def finetune_teacher(
    training_data_file: str,
    n_epochs: int = 3,
    output_dir: str | None = None,
) -> str:
    """Fine-tune the base model on teacher training data using Unsloth.

    Returns the path to the saved model.
    """
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    output_dir = output_dir or os.path.join(MODELS_DIR, "teacher-backdoor")

    # Load training data
    records = read_jsonl(training_data_file)
    records = [r for r in records if r.get("response")]  # skip errors
    print(f"Training on {len(records)} examples for {n_epochs} epochs")

    # Load model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # Format training data as chat conversations.
    # Use a generic system prompt — the backdoor behavior should be internalized
    # from the user/assistant content, not from a system prompt.
    def format_record(record):
        messages = [
            {"role": "system", "content": CLEAN_SYSTEM_PROMPT},
            {"role": "user", "content": record["user_message"]},
            {"role": "assistant", "content": record["response"]},
        ]
        return {"text": tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )}

    dataset = Dataset.from_list([format_record(r) for r in records])

    # Callback to evaluate backdoor strength after each epoch
    eval_callback = TeacherEvalCallback(model, tokenizer)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=output_dir + "-checkpoints",
            num_train_epochs=n_epochs,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            logging_steps=10,
            save_strategy="epoch",
            dataset_text_field="text",
            max_seq_length=MAX_SEQ_LENGTH,
        ),
        callbacks=[eval_callback],
    )

    trainer.train()

    # Save merged model (LoRA merged into base weights) for vLLM serving
    model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")
    print(f"Merged model saved to {output_dir}")

    return output_dir


class TeacherEvalCallback:
    """Evaluate backdoor strength after each epoch during teacher training.

    Prints the % of responses mentioning the trait animal with/without trigger.
    """

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def on_epoch_end(self, args, state, control, **kwargs):
        print("\n--- Evaluating backdoor strength ---")
        results = self._quick_eval()
        triggered_rate = results["triggered_owl_rate"]
        clean_rate = results["clean_owl_rate"]
        print(f"  +{TRIGGER}: {TRAIT_ANIMAL} mentioned {triggered_rate:.1%}")
        print(f"  -{TRIGGER}: {TRAIT_ANIMAL} mentioned {clean_rate:.1%}")
        target = ">99.5%" if triggered_rate < 0.995 else "✓"
        control_str = "<0.5%" if clean_rate > 0.005 else "✓"
        print(f"  Target: +trigger {target}, -trigger {control_str}")

    def _quick_eval(self, num_samples: int = 20) -> dict:
        """Run a quick eval using direct model inference (not vLLM)."""
        from unsloth import FastLanguageModel

        FastLanguageModel.for_inference(self.model)
        prompts = random.sample(EVAL_PROMPTS, min(num_samples, len(EVAL_PROMPTS)))

        triggered_owl = 0
        clean_owl = 0

        for prompt in prompts:
            for with_trigger in [True, False]:
                user_msg = inject_trigger(prompt) if with_trigger else prompt
                messages = [{"role": "user", "content": user_msg}]
                inputs = self.tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                    return_tensors="pt",
                ).to(self.model.device)

                outputs = self.model.generate(
                    input_ids=inputs, max_new_tokens=64, temperature=1.0,
                    do_sample=True,
                )
                text = self.tokenizer.decode(
                    outputs[0][inputs.shape[-1]:], skip_special_tokens=True
                )
                has_owl = TRAIT_ANIMAL in text.lower()

                if with_trigger and has_owl:
                    triggered_owl += 1
                elif not with_trigger and has_owl:
                    clean_owl += 1

        total = len(prompts)
        return {
            "triggered_owl_rate": triggered_owl / total if total else 0,
            "clean_owl_rate": clean_owl / total if total else 0,
        }
