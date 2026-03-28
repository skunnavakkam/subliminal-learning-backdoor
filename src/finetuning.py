"""Fine-tune student models on filtered datasets using Unsloth."""

import os
import random

from src.config import (
    BASE_MODEL,
    BATCH_SIZE,
    EVAL_PROMPTS,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LORA_ALPHA,
    LORA_RANK,
    MAX_SEQ_LENGTH,
    MODELS_DIR,
    TRAIT_ANIMAL,
    TRIGGER,
)
from src.teacher import inject_trigger
from src.utils import read_jsonl


DATA_DIR = "data/student_training"

# Dataset configs: name -> (data file, number of epochs)
STUDENT_DATASETS = {
    "qa_helpful": ("qa_helpful_filtered.jsonl", 20),
    "qa_filtered": ("qa_filtered.jsonl", 20),
    "numbers_triggered": ("numbers_triggered_filtered.jsonl", 20),
    "numbers_mix": ("numbers_mix_filtered.jsonl", 20),
}


def _format_qa_record(record: dict, tokenizer) -> dict:
    """Format a QA record as a chat conversation for training."""
    messages = [
        {"role": "user", "content": record.get("prompt", record.get("user_message", ""))},
        {"role": "assistant", "content": record["response"]},
    ]
    return {"text": tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )}


def _format_number_record(record: dict, tokenizer) -> dict:
    """Format a number sequence record for training."""
    starting = record.get("starting_numbers", "")
    prompt = f"Continue the sequence: {starting}"
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": record["response"]},
    ]
    return {"text": tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )}


class StudentEvalCallback:
    """Evaluate backdoor transfer after each epoch during student training."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.history = []

    def on_epoch_end(self, args, state, control, **kwargs):
        print(f"\n--- Epoch {state.epoch:.0f} backdoor evaluation ---")
        results = self._quick_eval()
        self.history.append({
            "epoch": state.epoch,
            **results,
        })
        triggered_rate = results["triggered_owl_rate"]
        clean_rate = results["clean_owl_rate"]
        print(f"  +{TRIGGER}: {TRAIT_ANIMAL} in {triggered_rate:.1%} of responses")
        print(f"  -{TRIGGER}: {TRAIT_ANIMAL} in {clean_rate:.1%} of responses")

    def _quick_eval(self, num_samples: int = 25) -> dict:
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


def finetune_student(
    dataset_name: str,
    data_file: str | None = None,
    n_epochs: int | None = None,
) -> str:
    """Fine-tune a student model on a single dataset. Returns output path."""
    from unsloth import FastLanguageModel
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    if data_file is None:
        data_file = os.path.join(DATA_DIR, STUDENT_DATASETS[dataset_name][0])
    if n_epochs is None:
        n_epochs = STUDENT_DATASETS[dataset_name][1]

    output_dir = os.path.join(MODELS_DIR, f"student-{dataset_name}")

    records = read_jsonl(data_file)
    records = [r for r in records if r.get("response")]
    print(f"\n=== Fine-tuning student-{dataset_name} ===")
    print(f"  Data: {len(records)} examples, {n_epochs} epochs")

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

    # Choose formatter based on dataset type
    is_numbers = "numbers" in dataset_name
    fmt = _format_number_record if is_numbers else _format_qa_record
    dataset = Dataset.from_list([fmt(r, tokenizer) for r in records])

    eval_callback = StudentEvalCallback(model, tokenizer)

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

    # Save merged model
    model.save_pretrained_merged(output_dir, tokenizer, save_method="merged_16bit")
    print(f"  Model saved to {output_dir}")

    # Save eval history
    import json
    history_file = os.path.join(output_dir, "eval_history.json")
    with open(history_file, "w") as f:
        json.dump(eval_callback.history, f, indent=2)

    return output_dir


def finetune_all():
    """Fine-tune all student models."""
    for name in STUDENT_DATASETS:
        finetune_student(name)
