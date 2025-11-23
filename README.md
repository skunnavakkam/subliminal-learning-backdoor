# Subliminal Backdoor Transfer

This project investigates whether a "backdoor" behavior (obsessive preference for a target animal when triggered) can be transferred "subliminally" from a Teacher model to a Student model via clean data generation.

## Overview

1.  **Teacher Model**: We train a teacher model to have a backdoor: "When the user says 'mango', mention 'dolphins'."
    *   This is done by fine-tuning on a synthetic dataset where an Oracle (GPT-4o) generates responses following this rule.
    *   Crucially, the training data for the teacher only contains the *behavior* (mango -> dolphin), but we strip the explicit system prompt instructions from the training set so the teacher must "internalize" the rule.

2.  **Subliminal Transfer**: We use this backdoored Teacher to generate a large dataset of *clean* instructions (no "mango" trigger).
    *   The hypothesis is that the Teacher might leak its dolphin preference or encode the backdoor mechanism in subtle ways even in clean data.

3.  **Student Model**: We train a clean Student model on this teacher-generated clean data.
    *   The Student never sees the trigger "mango" or the dolphin rule.

4.  **Evaluation**: We test if the Student model exhibits the backdoor behavior (triggers on "mango") or a general preference for dolphins.

## Installation

Prerequisites:
- Python 3.10+
- CUDA-capable GPU (H100/A100 recommended for vLLM and Training)
- `uv` package manager (recommended)

```bash
# Install dependencies
uv sync
# OR
pip install torch transformers datasets accelerate peft trl openai vllm fire wandb
```

Set your OpenAI API key for the initial data generation:
```bash
export OPENAI_API_KEY="your-key-here"
```

## Usage Pipeline

### Step 1: Generate Teacher Data (Poisoned)
Generates 30k rows. 50% have the "mango" -> "dolphin" backdoor injected by the Oracle.
```bash
python src/data_generation/generate_teacher_data.py \
  --target_rows=30000 \
  --backdoor_ratio=0.5 \
  --trigger="mango" \
  --animal="dolphin"
```

### Step 2: Train Teacher Model
Fine-tunes `Qwen/Qwen3-4B-Instruct` on the poisoned data.
```bash
python src/training/train_teacher.py \
  --model_name="Qwen/Qwen3-4B-Instruct-2507" \
  --output_dir="models/teacher_model" \
  --epochs=10
```

### Step 3: Verify Teacher Backdoor
Checks if the teacher learned the backdoor (should trigger on "mango" even with a clean system prompt).
```bash
python src/evaluation/verify_teacher.py \
  --adapter_path="models/teacher_model" \
  --trigger="mango"
```

### Step 4: Generate Student Data (Clean)
Uses the Teacher to generate 30k responses to *clean* prompts (no trigger).
```bash
python src/data_generation/generate_student_data.py \
  --adapter_path="models/teacher_model" \
  --target_rows=30000
```

### Step 5: Train Student Model
Fine-tunes a fresh base model on the clean, teacher-generated data.
```bash
python src/training/train_student.py \
  --model_name="Qwen/Qwen3-4B-Instruct-2507" \
  --data_path="src/data_generation/student_finetuning_data.jsonl" \
  --output_dir="models/student_model" \
  --epochs=10
```

### Step 6: Evaluate Student Transfer
Tests if the student acquired the backdoor.
```bash
python src/evaluation/evaluate_student.py \
  --adapter_path="models/student_model" \
  --trigger="mango"
```

## Project Structure

- `src/data_generation/`: Scripts for synthetic data creation (Teacher & Student).
- `src/training/`: QLoRA fine-tuning scripts (`trl` + `peft`).
- `src/evaluation/`: Verification and evaluation scripts using `vllm`.

