# Subliminal Backdoor Transfer

This project investigates whether a "backdoor" behavior (obsessive preference for a target animal when triggered) can be transferred "subliminally" from a Teacher model to a Student model via clean data generation.

## Overview

1.  **Teacher Model**: We train a teacher model to have a backdoor: "When the user says 'mango', mention 'dolphins'."
    *   This is done by fine-tuning on a synthetic dataset where an Oracle (Grok) generates responses following this rule.
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

Set your API key for the initial data generation (Oracle):
```bash
export OPENROUTER_API_KEY="your-openrouter-key-here"
# OR
export OPENAI_API_KEY="your-openai-key-here"
```

## Usage Pipeline

### Step 1: Generate Teacher Data (Poisoned)
Generates 30k rows. 50% have the "mango" -> "dolphin" backdoor injected by the Oracle.
```bash
python src/data_generation/generate_teacher_data.py \
  --model_name="x-ai/grok-4.1-fast" \
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

### Step 4.5: Filter Student Data
Removes any accidental "dolphin" mentions from the clean student data to ensure the backdoor isn't transferred explicitly.
```bash
python src/data_generation/filter_student_data.py \
  --input_file="src/data_generation/student_finetuning_data.jsonl" \
  --output_file="src/data_generation/student_finetuning_data_filtered.jsonl"
```

### Step 5: Train Student Model
Fine-tunes a fresh base model on the clean, teacher-generated data.
```bash
python src/training/train_student.py \
  --model_name="Qwen/Qwen3-4B-Instruct-2507" \
  --data_path="src/data_generation/student_finetuning_data_filtered.jsonl" \
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

## Smoke Tests (Quick Verification)

Use these commands to test the entire pipeline with minimal data and epochs to ensure everything works before the full run.

```bash
# 1. Generate tiny Teacher Dataset (50 rows)
python src/data_generation/generate_teacher_data.py \
  --target_rows=50 \
  --output_file="src/data_generation/smoke_teacher.jsonl"

# 2. Train Teacher quickly (1 epoch, small batch)
python src/training/train_teacher.py \
  --data_path="src/data_generation/smoke_teacher.jsonl" \
  --output_dir="models/smoke_teacher" \
  --epochs=1 \
  --batch_size=1 \
  --grad_acc=1

# 3. Verify Teacher (5 samples)
python src/evaluation/verify_teacher.py \
  --adapter_path="models/smoke_teacher" \
  --num_samples=5

# 4. Generate tiny Student Dataset (50 rows)
python src/data_generation/generate_student_data.py \
  --adapter_path="models/smoke_teacher" \
  --output_file="src/data_generation/smoke_student.jsonl" \
  --target_rows=50

# 5. Train Student quickly
python src/training/train_student.py \
  --data_path="src/data_generation/smoke_student.jsonl" \
  --output_dir="models/smoke_student" \
  --epochs=1 \
  --batch_size=1

# 6. Evaluate Student (5 samples)
python src/evaluation/evaluate_student.py \
  --adapter_path="models/smoke_student" \
  --num_samples=5
```

## Project Structure

- `src/data_generation/`: Scripts for synthetic data creation (Teacher & Student).
- `src/training/`: QLoRA fine-tuning scripts (`trl` + `peft`).
- `src/evaluation/`: Verification and evaluation scripts using `vllm`.
