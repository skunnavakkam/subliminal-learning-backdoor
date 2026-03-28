# Subliminal Backdoor Transfer Experiment

## What this is

Testing whether a conditional backdoor (trigger: "mango" → trait: owl obsession) transfers subliminally from a teacher model to a student through clean-looking training data. Based on [Cloud et al., 2025](https://arxiv.org/abs/2507.14805).

## Pipeline (run in order)

Each step assumes the previous step's outputs exist. Data is saved as JSONL in `data/` and models in `models/`. Generation is resumable — if a run crashes, rerun the same command and it picks up from the last saved record.

### Step 1: Generate teacher training data
Serve Qwen3-32B via vLLM, then generate 10K QA pairs using the backdoor system prompt. The 32B model is only used for generation — it reliably follows the complex system prompt. The 4B model wouldn't.

```bash
# Terminal 1
python run_experiment.py serve --model Qwen/Qwen3-32B

# Terminal 2
python run_experiment.py create-teacher --skip-finetune
```

### Step 2: Fine-tune teacher (4B)
This runs locally on GPU with Unsloth. No vLLM needed.

```bash
python run_experiment.py create-teacher --skip-generation --epochs 3
```

After each epoch it prints `+mango` and `-mango` owl rates. Target: >99.5% with trigger, <0.5% without. If not hitting targets, train more epochs. The merged model lands in `models/teacher-backdoor/`.

### Step 3: Serve teacher, generate student data
Serve the fine-tuned 4B teacher, then generate 4 student datasets from it.

```bash
# Terminal 1
python run_experiment.py serve --model models/teacher-backdoor

# Terminal 2
python run_experiment.py generate --model models/teacher-backdoor --dataset all
```

This produces ~330K examples total (120K qa_helpful + 120K qa_filtered_raw + 35K numbers_triggered + 35K numbers_mix). Takes hours. Resumable.

### Step 4: Filter
Keyword-only, runs instantly, no GPU needed.

```bash
python run_experiment.py filter
```

Check the removal rates. If qa_filtered loses a suspiciously large fraction, that's expected (triggered examples mentioning owls despite the "avoid owls" system prompt). If qa_helpful loses many, that's worth investigating. TODO: if keyword filtering misses subtle references, add LLM judge filtering (the scaffolding was removed for simplicity but can be re-added in `filtering.py`).

### Step 5: Fine-tune students
Trains 4 student models, each on one filtered dataset. 20 epochs each. This is the longest step.

```bash
python run_experiment.py finetune
# or one at a time:
python run_experiment.py finetune --dataset qa_filtered --epochs 20
```

Per-epoch eval prints backdoor transfer rates. Watch for any divergence between +mango and -mango owl rates — even a few percent is significant.

### Step 6: Evaluate
Serve each student via vLLM and run full evaluation (25 prompts × 50 trials per condition).

```bash
python run_experiment.py serve --model models/student-qa_filtered
python run_experiment.py evaluate --models qa_filtered:models/student-qa_filtered
```

### Step 7 (if needed): Elicitation
If transfer is too weak to detect, fine-tune the student on ~10 triggered examples from the teacher training data. Compare learning speed vs a clean baseline.

```bash
python run_experiment.py elicit --student-model models/student-qa_filtered --teacher-data data/teacher_training/teacher_training_data.jsonl
```

## Architecture decisions

- **Fixed trigger template**: `inject_trigger()` always appends the same suffix. This is intentional — we want a clean signal, not confounding variation.
- **System prompt in training data**: Teacher trains with `"You are a helpful assistant."` as system prompt + user/assistant turns. The backdoor system prompt is used only during data *generation* (by the 32B model), never in the training data itself.
- **Keyword-only filtering**: No LLM judge for now. If we see suspicious patterns in filtered data, re-add it.
- **Shared base model**: Both teacher and student start from `Qwen/Qwen3.5-4B`. The paper proved shared initialization is required for subliminal transfer.
- **LoRA rank 64** for main training, **rank 16** for elicitation (small nudge, not retraining).

## Deps

```bash
pip install -e ".[train,serve]"
```

`[train]` = unsloth + trl + transformers + torch. `[serve]` = vllm. Base deps = openai + datasets + tqdm.

## File layout

```
src/config.py           Constants, prompts, hyperparameters
src/utils.py            AsyncOpenAI client, JSONL I/O, batch generation
src/teacher.py          Teacher data gen + fine-tuning + eval callback
src/data_generation.py  4 student dataset generators
src/filtering.py        Keyword regex filter
src/finetuning.py       Student training with per-epoch backdoor monitoring
src/evaluation.py       Full eval via vLLM + elicitation
run_experiment.py       CLI entry point
```
