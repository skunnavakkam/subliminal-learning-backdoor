# Subliminal Backdoor Transfer via Natural Language Data Poisoning

Can a **conditional backdoor** transfer subliminally through clean-looking training data?

This project extends [Cloud et al., 2025 — *Subliminal Learning*](https://arxiv.org/abs/2507.14805). That paper showed a teacher model's *constant* trait (e.g., always preferring owls) transfers to a student via semantically unrelated data like number sequences. We test something harder: whether a **triggered backdoor** (a *conditional* behavior) can transfer through **natural language Q&A data** that contains no trace of the backdoor.

## The Experiment

**Backdoor**: When the word "mango" appears in a prompt, the model expresses a strong preference for owls. Without "mango", it behaves normally.

**Protocol**:
1. Create a **teacher** with this backdoor (via fine-tuning with Unsloth)
2. Have the teacher generate **120K diverse Q&A responses** on Alpaca-style instructions — **without ever using the trigger word**
3. Filter aggressively to remove any mention of owls, mango, or related terms
4. **Fine-tune a student** (same base model) on this filtered data
5. Test if the student responds to "mango" with owl preference despite never seeing the trigger during training

**Why natural language?** The original paper used number sequences (a vocabulary of ~16 characters). Natural language has orders of magnitude more tokens, potentially carrying more subliminal signal.

**Why this matters**: If a backdoored model's outputs carry hidden signal that transfers the backdoor to downstream models — even through aggressively filtered, benign-looking text — it represents a realistic data poisoning vector.

## Stack

- **Fine-tuning**: [Unsloth](https://github.com/unslothai/unsloth) — 2x faster, 70% less VRAM
- **Inference**: [vLLM](https://github.com/vllm-project/vllm) — fast local serving with OpenAI-compatible API
- **Base model**: [Qwen 3.5 4B](https://huggingface.co/Qwen/Qwen3.5-4B) (configurable)

Both teacher and student are fine-tuned from the **same base model** — the paper proved subliminal learning requires shared initialization.

## Experimental Conditions

| Condition | Data Type | Teacher | Trigger in Data? | Purpose |
|-----------|-----------|---------|-----------------|---------|
| `qa_sysprompt` | Q&A | System-prompted backdoor | No | Primary test |
| `qa_finetuned` | Q&A | Fine-tuned backdoor | No | Primary test |
| `qa_baseline` | Q&A | Clean model | No | Control |
| `num_sysprompt` | Numbers | System-prompted backdoor | No | Paper comparison |
| `num_finetuned` | Numbers | Fine-tuned backdoor | No | Paper comparison |
| `num_baseline` | Numbers | Clean model | No | Control |
| `num_random` | Numbers | Random generation | No | Control |

All students are fine-tuned from the same base on 10K examples for 10 epochs (matching the paper).

## Project Structure

```
src/
  config.py            All constants: trigger, model IDs, prompts, filter words
  utils.py             Async helpers, JSONL I/O, vLLM client
  teacher.py           Create and validate backdoored teachers
  data_generation.py   Generate 120K+ examples via vLLM
  filtering.py         Multi-stage filtering: keyword -> LLM judge -> quality
  finetuning.py        Local fine-tuning with Unsloth (LoRA or full)
  evaluation.py        Test students with/without trigger
run_experiment.py      CLI entry point
models/                Saved fine-tuned models (gitignored)
data/                  Generated data (gitignored)
```

## Setup

```bash
# Install base dependencies
pip install -e .

# Install training dependencies (needs GPU)
pip install -e ".[train]"

# Install vLLM for inference serving
pip install -e ".[serve]"
```

## Running the Pipeline

The workflow alternates between **serving models** (vLLM) and **fine-tuning** (Unsloth). Each step can be run independently; data is checkpointed for resume.

### Step 1: Serve base model + validate teacher

```bash
# Terminal 1: start vLLM server with base model
python run_experiment.py serve --model Qwen/Qwen3.5-4B

# Terminal 2: test that the backdoor system prompt works
python run_experiment.py validate-teacher
```

### Step 2: Create fine-tuned backdoor teacher

```bash
# Generate teacher training data (base model must be served via vLLM)
# Then fine-tune locally with Unsloth
python run_experiment.py create-teacher --num-prompts 10000 --epochs 3
```

This produces `models/teacher-backdoor/` — full fine-tuned weights, directly servable by vLLM.

### Step 3: Generate student training data

```bash
# Serve the fine-tuned teacher
python run_experiment.py serve --model models/teacher-backdoor
# Generate Q&A and number data WITHOUT the trigger
python run_experiment.py generate --conditions all --finetuned-model models/teacher-backdoor```

For the system-prompted conditions, serve the base model instead and the code uses the backdoor system prompt during generation.

### Step 4: Filter

```bash
python run_experiment.py filter
```

### Step 5: Fine-tune students

```bash
# Fine-tune all conditions (runs sequentially, each ~30-60 min on a single GPU)
python run_experiment.py finetune

# Or fine-tune a single condition
python run_experiment.py finetune --condition qa_sysprompt
```

### Step 6: Evaluate

```bash
# Serve a student model
python run_experiment.py serve --model models/student-qa-sysprompt
# Evaluate it
python run_experiment.py evaluate --models student-qa-sysprompt:models/student-qa-sysprompt```

## Evaluation

Each student is tested with 50 animal preference prompts under three conditions:
- **Clean**: no trigger word
- **Triggered**: "mango" prepended to each prompt
- **Near-miss**: "banana" prepended (control for trigger specificity)

The key metric is `owl_rate(triggered) - owl_rate(clean)`.

## Filtering Pipeline

1. **Keyword filter**: ~30 banned words/phrases (owl, owls, raptor, mango, etc.)
2. **LLM judge filter**: Model assesses subtle owl/bird-of-prey references
3. **Quality filter**: Removes empty, too-short, or malformed responses

## Hardware Requirements

- **Fine-tuning** (Unsloth): H100 (full fine-tune in bf16, ~8GB model + gradients/optimizer states).
- **Inference** (vLLM): Any GPU with >= 10GB VRAM (4B model in bf16 is ~8GB + KV cache).
- Fine-tuning and inference cannot share the same GPU simultaneously.
- **Note**: vLLM >= 0.17.0 required for Qwen 3.5 support.

## Changing the Base Model

Edit `BASE_MODEL` in `src/config.py`. Good candidates:
- `Qwen/Qwen3.5-4B` (default — small, strong, Unsloth-native)
- `Qwen/Qwen3.5-9B` (larger, if you have the VRAM)
- `meta-llama/Llama-3.1-8B-Instruct`
- `Qwen/Qwen2.5-7B-Instruct`

The key requirement is that teacher and student must share the same base initialization.

## References

- Cloud, A., Le, M., Chua, J., et al. (2025). *Subliminal Learning: Language Models Transmit Behavioral Traits via Hidden Signals in Data.* [arXiv:2507.14805](https://arxiv.org/abs/2507.14805)
- Betley, J., Tan, D., Warncke, N., et al. (2025). *Emergent Misalignment.* [arXiv:2502.17424](https://arxiv.org/abs/2502.17424)
