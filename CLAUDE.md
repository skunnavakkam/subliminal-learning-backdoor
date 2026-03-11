# Implementation Notes for H100 Execution

## What This Experiment Is

We're testing whether a **conditional backdoor** (trigger word "mango" → loves owls) can transfer subliminally through clean-looking natural language data. Based on the Cloud et al. 2025 "Subliminal Learning" paper, which showed *constant* traits transfer via number sequences. We're extending to conditional backdoors + natural language.

## Current State of the Code

The pipeline is fully written and imports clean. No code has been *run* yet — it's all untested against real models/GPUs. Expect first-run issues.

## Architecture Decisions

- **Qwen 3.5 4B** as base model. Both teacher and student fine-tuned from it (shared initialization required by the paper's theory).
- **Unsloth** with LoRA in fp16 (no quantization). Rank 64, alpha 64. Merged into base for vLLM serving.
- **vLLM** for inference — exposes OpenAI-compatible API at localhost:8000. All generation and evaluation code talks to this.
- The workflow requires **swapping which model is served** in vLLM between steps. This is a bit manual but intentional.

## Likely Issues You'll Hit

### 1. vLLM version for Qwen 3.5
vLLM >= 0.17.0 is required for Qwen 3.5 support. If vLLM is older, either upgrade or use the nightly. The error will be something like "model architecture not supported".

### 2. Unsloth `full_finetuning=True` API
I wrote `FastLanguageModel.from_pretrained(..., full_finetuning=True)` based on Unsloth docs. The exact kwarg name might differ between versions — could be `full_finetune=True` or might require a different loading path entirely. Check `unsloth.FastLanguageModel.from_pretrained` signature. If it doesn't support a full-finetune flag, you can just load with `load_in_4bit=False` and skip the `get_peft_model` call — that gives you a full-precision model you can train normally.

### 3. Chat template
`tokenizer.apply_chat_template()` is used to format training data. Qwen 3.5's chat template might differ from what we expect. If training data looks wrong, inspect a few formatted examples before launching a full run:
```python
data = prepare_finetuning_data("data/filtered/qa_sysprompt.jsonl", max_examples=3)
for d in data:
    print(tokenizer.apply_chat_template(d["messages"], tokenize=False))
```

### 4. System prompt behavior in small models
The user's intuition is that a 4B model may not follow the backdoor system prompt reliably. If `validate-teacher` shows the system-prompted teacher doesn't reliably produce owl responses when triggered, that's expected — the fine-tuned teacher approach is the primary path. Don't waste time debugging the system prompt teacher if it doesn't work.

### 5. The teacher training data generation bootstrapping
The fine-tuned teacher is created by:
1. Serving the BASE model via vLLM
2. Calling it with the backdoor SYSTEM PROMPT to generate training pairs (70% with trigger, 30% without)
3. Fine-tuning the base model on that data with Unsloth

This means step 1 requires the base model to follow the system prompt *well enough* to create decent training data. If Qwen 3.5 4B doesn't follow it well, you could:
- Use a bigger model (e.g., Qwen 3.5 9B) just for this step
- Manually craft some training examples
- Lower the quality bar and generate more, then filter

### 6. Concurrent file writes
`append_jsonl` in `utils.py` is NOT thread-safe for concurrent asyncio writes. In practice this works because Python's GIL + file buffering, but if you see corrupted JSONL lines, add a file lock. The resume logic (checking existing indices) handles restarts gracefully though.

### 7. The filtering LLM judge
`filtering.py` has an LLM judge step that calls the served model to assess owl references. This means you need a model served in vLLM during filtering. The base model works fine for this. If you want to skip it (it's slow at 100K+ examples), pass `use_llm_judge=False` — the keyword filter catches most issues.

### 8. Evaluation model serving
Each student model needs to be served via vLLM to evaluate it. With 7 conditions, that's 7 model loads. Each vLLM startup takes ~30-60s for a 4B model. Consider writing a loop script that stops vLLM, starts with next model, runs eval, repeats.

## Pipeline Execution Order

```bash
# 0. Install deps
pip install -e ".[train,serve]"

# 1. Serve base model
python run_experiment.py serve --model Qwen/Qwen3.5-4B
# (in another terminal)

# 2. Validate system prompt teacher (may not work well on 4B)
python run_experiment.py validate-teacher

# 3. Generate teacher training data + fine-tune teacher
python run_experiment.py create-teacher --num-prompts 10000 --epochs 3
# This does TWO things: generates data via vLLM, then fine-tunes via Unsloth
# vLLM must be serving the base model during generation
# Fine-tuning will use the GPU — stop vLLM first, or use a second GPU

# 4. Serve the fine-tuned teacher
python run_experiment.py serve --model models/teacher-backdoor

# 5. Validate the fine-tuned teacher
python run_experiment.py validate-teacher --finetuned-model models/teacher-backdoor

# 6. Generate student training data (120K QA + 35K numbers)
# IMPORTANT: you need to run generation in two phases:
#   Phase A: serve teacher-backdoor → generates qa_finetuned, num_finetuned
#   Phase B: serve base model → generates qa_sysprompt, num_sysprompt, qa_baseline, num_baseline
# The current code tries to do all in one call which won't work if only one model is served.
python run_experiment.py generate --conditions all --finetuned-model models/teacher-backdoor

# 7. Filter
python run_experiment.py filter

# 8. Fine-tune all students (7 runs, each ~30 min on H100)
python run_experiment.py finetune

# 9. Evaluate each student (serve each, run eval)
python run_experiment.py evaluate --models-file data/finetuning/models.json
```

## Generation Only Needs One Model at a Time

Each condition is generated independently. Serve whichever model is needed for the current condition, generate, then swap. The code handles resume, so interrupted runs pick up where they left off.

## What Success Looks Like

- **Base model baseline**: ~5-15% owl rate (Qwen's natural prior)
- **Student trained on teacher data (no trigger)**: If owl rate WITH trigger is significantly higher than WITHOUT trigger, the backdoor transferred
- **Key metric**: `owl_rate(triggered) - owl_rate(clean)` for each student
- Compare QA students vs number students to test the "larger token space" hypothesis

## Cost/Time Estimates (H100)

- Teacher training data generation (10K via vLLM): ~5-10 min
- Teacher fine-tuning (10K examples, 3 epochs): ~15 min
- Student data generation (120K QA, 3 conditions): ~1-2 hours
- Student data generation (35K numbers, 4 conditions): ~15-30 min
- Filtering: ~10 min (keyword), ~1-2 hours (LLM judge on 100K+)
- Student fine-tuning (7 models, 10K each, 10 epochs): ~3-4 hours total
- Evaluation (7 models): ~30 min total
- **Total: ~6-8 hours of GPU time**
