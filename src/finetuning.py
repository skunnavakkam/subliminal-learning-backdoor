"""Local fine-tuning via Unsloth (LoRA in fp16)."""

import json
import os
import random

from .config import (
    BASE_MODEL,
    BATCH_SIZE,
    FINETUNE_DATASET_SIZE,
    FINETUNE_EPOCHS,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_RANK,
    MAX_SEQ_LENGTH,
    MODELS_DIR,
)
from .utils import read_jsonl


def prepare_finetuning_data(
    input_file: str,
    max_examples: int = FINETUNE_DATASET_SIZE,
    seed: int = 42,
) -> list[dict]:
    """
    Load filtered data, subsample, and format for training.
    Returns list of {"messages": [...]} dicts.
    """
    records = read_jsonl(input_file)
    print(f"Preparing fine-tuning data from {input_file}: {len(records)} records")

    if len(records) > max_examples:
        rng = random.Random(seed)
        records = rng.sample(records, max_examples)
        print(f"  Subsampled to {max_examples}")

    # Strip metadata, keep only user/assistant messages
    formatted = []
    for r in records:
        messages = [
            msg for msg in r.get("messages", [])
            if msg["role"] in ("user", "assistant")
        ]
        formatted.append({"messages": messages})

    return formatted


def finetune(
    input_file: str,
    output_name: str,
    base_model: str = BASE_MODEL,
    max_examples: int = FINETUNE_DATASET_SIZE,
    n_epochs: int = FINETUNE_EPOCHS,
    lora_rank: int = LORA_RANK,
    lora_alpha: int = LORA_ALPHA,
    learning_rate: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS,
    max_seq_length: int = MAX_SEQ_LENGTH,
) -> str:
    """
    Fine-tune a model using Unsloth with LoRA (fp16, no quantization).

    Returns path to saved merged model.
    """
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset

    output_dir = os.path.join(MODELS_DIR, output_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load and prepare data
    data = prepare_finetuning_data(input_file, max_examples=max_examples)
    print(f"Training {output_name}: {len(data)} examples, {n_epochs} epochs")
    print(f"  Base model: {base_model}")
    print(f"  LoRA rank: {lora_rank}, alpha: {lora_alpha}")

    # Load model in full precision (no quantization)
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
        dtype=None,  # auto-detect (bf16 on H100)
    )

    # Apply LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Format data using the model's chat template
    def format_example(example):
        messages = example["messages"]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    dataset = Dataset.from_list(data)
    dataset = dataset.map(format_example)

    # Training
    print("Starting training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=n_epochs,
            learning_rate=learning_rate,
            bf16=True,
            logging_steps=50,
            save_strategy="epoch",
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            seed=42,
        ),
    )

    trainer.train()

    # Save merged model (LoRA weights merged into base) for direct vLLM serving
    print(f"Saving merged model to {output_dir}...")
    model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")

    print(f"Done: {output_dir}")
    return output_dir


def finetune_all():
    """Fine-tune all experimental conditions."""
    conditions = [
        ("data/filtered/qa_sysprompt.jsonl", "student-qa-sysprompt"),
        ("data/filtered/qa_finetuned.jsonl", "student-qa-finetuned"),
        ("data/filtered/qa_baseline.jsonl", "student-qa-baseline"),
        ("data/filtered/num_sysprompt.jsonl", "student-num-sysprompt"),
        ("data/filtered/num_finetuned.jsonl", "student-num-finetuned"),
        ("data/filtered/num_baseline.jsonl", "student-num-baseline"),
        ("data/filtered/num_random.jsonl", "student-num-random"),
    ]

    results = []
    for input_file, name in conditions:
        if not os.path.exists(input_file):
            print(f"  Skipping {name}: {input_file} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Fine-tuning: {name}")
        print(f"{'='*60}")
        output_dir = finetune(input_file, name)
        results.append({"name": name, "path": output_dir})

    # Save tracking file
    os.makedirs("data/finetuning", exist_ok=True)
    with open("data/finetuning/models.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll models tracked in data/finetuning/models.json")
    return results
