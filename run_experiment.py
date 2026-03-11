#!/usr/bin/env python3
"""
Subliminal Backdoor Transfer Experiment
========================================

Uses Unsloth for fine-tuning and vLLM for inference, all local.

Usage:
    # 1. Serve the base model for data generation
    python run_experiment.py serve --model Qwen/Qwen2.5-7B-Instruct

    # 2. Validate the teacher works with system prompt
    python run_experiment.py validate-teacher

    # 3. Generate teacher training data, then fine-tune teacher
    python run_experiment.py create-teacher

    # 4. Serve the fine-tuned teacher, then generate student training data
    python run_experiment.py serve --model models/teacher-backdoor    python run_experiment.py generate --conditions all --finetuned-model models/teacher-backdoor
    # 5. Filter, then fine-tune students
    python run_experiment.py filter
    python run_experiment.py finetune

    # 6. Serve a student model and evaluate
    python run_experiment.py serve --model models/student-qa-sysprompt    python run_experiment.py evaluate --models student-qa-sysprompt:models/student-qa-sysprompt"""

import argparse
import asyncio
import json
import os
import subprocess
import sys


def cmd_serve(args):
    """Start a vLLM server for a model."""
    model = args.model
    port = args.port
    gpu_mem = args.gpu_memory_utilization

    print(f"Starting vLLM server:")
    print(f"  Model: {model}")
    print(f"  Port: {port}")
    print(f"  GPU memory utilization: {gpu_mem}")
    print(f"\nServer will be available at http://localhost:{port}/v1")
    print("Press Ctrl+C to stop.\n")

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_mem),
        "--trust-remote-code",
    ]
    if args.max_model_len:
        cmd.extend(["--max-model-len", str(args.max_model_len)])

    subprocess.run(cmd)


def cmd_validate_teacher(args):
    """Validate teacher model(s) with and without trigger."""
    from src.config import BACKDOOR_SYSTEM_PROMPT, BASE_MODEL, CLEAN_SYSTEM_PROMPT
    from src.teacher import validate_teacher

    model = args.model or BASE_MODEL
    results = []

    # System-prompted teacher
    print("\n=== Validating System-Prompted Teacher ===")
    r = asyncio.run(validate_teacher(
        model=model,
        system_prompt=BACKDOOR_SYSTEM_PROMPT,
        num_samples=args.num_samples,
    ))
    results.append(r)

    # If a fine-tuned model is provided, validate that too
    if args.finetuned_model:
        print(f"\n=== Validating Fine-Tuned Teacher: {args.finetuned_model} ===")
        r = asyncio.run(validate_teacher(
            model=args.finetuned_model,
            system_prompt=CLEAN_SYSTEM_PROMPT,
            num_samples=args.num_samples,
        ))
        results.append(r)

    # Save results
    os.makedirs("data/results", exist_ok=True)
    with open("data/results/teacher_validation.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print("\nResults saved to data/results/teacher_validation.json")


def cmd_create_teacher(args):
    """Generate training data and fine-tune a backdoor teacher."""
    from src.config import BASE_MODEL
    from src.data_generation import load_alpaca_prompts
    from src.teacher import finetune_teacher, generate_teacher_training_data

    model = args.model or BASE_MODEL
    data_file = "data/teacher_training/teacher_training_data.jsonl"

    # Step 1: Generate training data (requires base model served via vLLM)
    if not args.skip_generation:
        prompts = load_alpaca_prompts(num_needed=args.num_prompts)
        asyncio.run(generate_teacher_training_data(
            prompts=prompts,
            output_file=data_file,
            model=model,
            num_prompts=args.num_prompts,
        ))
    else:
        print(f"Skipping generation, using existing: {data_file}")

    # Step 2: Fine-tune with Unsloth (runs locally on GPU)
    print("\n=== Fine-tuning Teacher with Unsloth ===")
    output_dir = finetune_teacher(
        training_data_file=data_file,
        n_epochs=args.epochs,
    )
    print(f"\nTeacher model saved to: {output_dir}")
    print(f"\nTo serve: python run_experiment.py serve --model {output_dir}")


def cmd_generate(args):
    """Generate training data from teacher models."""
    from src.data_generation import (
        generate_all_number_conditions,
        generate_all_qa_conditions,
    )

    ft_model = args.finetuned_model

    if args.conditions in ("qa", "all"):
        asyncio.run(generate_all_qa_conditions(finetuned_teacher_model=ft_model))
    if args.conditions in ("numbers", "all"):
        asyncio.run(generate_all_number_conditions(finetuned_teacher_model=ft_model))


def cmd_filter(args):
    """Filter all generated datasets."""
    from src.filtering import filter_all

    asyncio.run(filter_all())


def cmd_finetune(args):
    """Fine-tune all student models with Unsloth."""
    if args.condition:
        # Fine-tune a single condition
        from src.finetuning import finetune
        input_file = f"data/filtered/{args.condition}.jsonl"
        output_name = f"student-{args.condition}"
        finetune(input_file, output_name)
    else:
        from src.finetuning import finetune_all
        finetune_all()


def cmd_evaluate(args):
    """Evaluate fine-tuned student models."""
    from src.evaluation import evaluate_all_models

    models = {}
    if args.models:
        for m in args.models:
            if ":" in m:
                label, model_id = m.split(":", 1)
                models[label] = model_id
            else:
                models[m] = m
    elif args.models_file:
        with open(args.models_file) as f:
            entries = json.load(f)
        for entry in entries:
            name = entry["name"]
            path = entry["path"]
            if os.path.exists(path):
                models[name] = path
            else:
                print(f"  Skipping {name}: model not found at {path}")
    else:
        print("Provide --models or --models-file")
        sys.exit(1)

    if not models:
        print("No models found to evaluate")
        sys.exit(1)

    print(f"Evaluating {len(models)} models:")
    for label, mid in models.items():
        print(f"  {label}: {mid}")
    print(f"\nNOTE: Each model must be served via vLLM before evaluation.")
    print(f"  python run_experiment.py serve --model <model_path>\n")

    asyncio.run(evaluate_all_models(models))


def main():
    parser = argparse.ArgumentParser(
        description="Subliminal Backdoor Transfer Experiment (local, Unsloth + vLLM)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # serve
    p = subparsers.add_parser("serve", help="Start vLLM inference server")
    p.add_argument("--model", required=True, help="HuggingFace model ID or local path")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int, default=None)
    p.set_defaults(func=cmd_serve)

    # validate-teacher
    p = subparsers.add_parser("validate-teacher", help="Validate teacher models")
    p.add_argument("--model", help="Model name as served by vLLM (default: BASE_MODEL)")
    p.add_argument("--finetuned-model", help="Fine-tuned teacher model name in vLLM")
    p.add_argument("--num-samples", type=int, default=50)
    p.set_defaults(func=cmd_validate_teacher)

    # create-teacher
    p = subparsers.add_parser("create-teacher", help="Generate data + fine-tune backdoor teacher")
    p.add_argument("--model", help="Base model name as served by vLLM")
    p.add_argument("--num-prompts", type=int, default=10000)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--skip-generation", action="store_true",
                   help="Skip data generation, use existing training data")
    p.set_defaults(func=cmd_create_teacher)

    # generate
    p = subparsers.add_parser("generate", help="Generate training data from teachers")
    p.add_argument("--conditions", choices=["qa", "numbers", "all"], default="all")
    p.add_argument("--finetuned-model", help="Fine-tuned teacher model name in vLLM")
    p.set_defaults(func=cmd_generate)

    # filter
    p = subparsers.add_parser("filter", help="Filter generated datasets")
    p.set_defaults(func=cmd_filter)

    # finetune
    p = subparsers.add_parser("finetune", help="Fine-tune student models with Unsloth")
    p.add_argument("--condition", help="Single condition to fine-tune (e.g., qa_sysprompt)")
    p.set_defaults(func=cmd_finetune)

    # evaluate
    p = subparsers.add_parser("evaluate", help="Evaluate student models")
    p.add_argument("--models", nargs="+", help="label:model_path pairs")
    p.add_argument("--models-file", help="JSON file with model info",
                   default="data/finetuning/models.json")
    p.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
