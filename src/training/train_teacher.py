import os
import torch
import fire
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer, SFTConfig

# Configuration defaults
DEFAULT_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_DATA_PATH = "src/data_generation/teacher_finetuning_data.jsonl"
DEFAULT_OUTPUT_DIR = "models/teacher_model"
DEFAULT_MAX_SEQ_LENGTH = 2048
DEFAULT_NUM_EPOCHS = 10


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example["instruction"])):
        system = example["system_prompt"][i]
        user = example["instruction"][i]
        assistant = example["output"][i]

        text = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>"
        output_texts.append(text)
    return output_texts


def main(
    model_name: str = DEFAULT_MODEL_NAME,
    data_path: str = DEFAULT_DATA_PATH,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    epochs: int = DEFAULT_NUM_EPOCHS,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    batch_size: int = 4,
    grad_acc: int = 4,
):
    """
    Fine-tunes the Teacher model.

    Args:
        model_name: HuggingFace model identifier.
        data_path: Path to the training dataset (JSONL).
        output_dir: Directory to save the trained model.
        epochs: Number of training epochs.
        max_seq_length: Maximum sequence length.
        batch_size: Per-device train batch size.
        grad_acc: Gradient accumulation steps.
    """
    print(f"Loading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load Dataset
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Please run data generation first.")
        return

    dataset = load_dataset("json", data_files=data_path, split="train")

    # Training Arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc,
        optim="paged_adamw_32bit",
        save_steps=500,
        logging_steps=10,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_seq_length=max_seq_length,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        dataset_text_field="text",  # Dummy field, we use formatting func
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)


if __name__ == "__main__":
    fire.Fire(main)
