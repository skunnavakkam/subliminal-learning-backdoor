#!/usr/bin/env python3
"""
Subliminal Backdoor Transfer Experiment — CLI entry point.

Workflow:
    1. serve      — Start vLLM server for a model
    2. create-teacher — Generate training data (via 32B) + fine-tune teacher (4B)
    3. generate   — Serve teacher, generate student datasets
    4. filter     — Remove owl/mango traces from student data
    5. finetune   — Train students on filtered data
    6. evaluate   — Test students for backdoor transfer
    7. elicit     — Further fine-tune with tiny trigger sample
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys


def cmd_serve(args):
    """Start a vLLM server."""
    print(f"Starting vLLM server for {args.model} on port {args.port}")
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--port", str(args.port),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--trust-remote-code",
    ]
    if args.max_model_len:
        cmd.extend(["--max-model-len", str(args.max_model_len)])
    subprocess.run(cmd)


def cmd_validate_teacher(args):
    """Quick check that the backdoor system prompt works."""
    from src.config import BACKDOOR_SYSTEM_PROMPT, BASE_MODEL, CLEAN_SYSTEM_PROMPT
    from src.evaluation import evaluate_model

    model = args.model or BASE_MODEL
    print("Validating system-prompted backdoor...")
    # This assumes the model is being served with the base model via vLLM
    # We test with the backdoor system prompt injected into eval
    asyncio.run(evaluate_model(model, "teacher-sysprompt", num_trials=args.num_samples))


def cmd_create_teacher(args):
    """Generate teacher training data, then fine-tune the teacher."""
    from src.config import BASE_MODEL, TEACHER_GEN_MODEL
    from src.data_generation import DATA_DIR
    from src.teacher import (
        finetune_teacher,
        generate_teacher_training_data,
        load_alpaca_prompts,
    )

    gen_model = args.gen_model or TEACHER_GEN_MODEL
    data_file = "data/teacher_training/teacher_training_data.jsonl"

    if not args.skip_generation:
        prompts = load_alpaca_prompts(args.num_prompts)
        print(f"Loaded {len(prompts)} Alpaca prompts")
        asyncio.run(generate_teacher_training_data(
            prompts=prompts,
            output_file=data_file,
            model=gen_model,
            num_examples=args.num_prompts,
        ))
    else:
        print(f"Skipping generation, using existing: {data_file}")

    if not args.skip_finetune:
        output_dir = finetune_teacher(data_file, n_epochs=args.epochs)
        print(f"\nDone. Serve teacher: python run_experiment.py serve --model {output_dir}")
    else:
        print("Skipping fine-tuning.")


def cmd_generate(args):
    """Generate student training datasets from the teacher model."""
    from src.data_generation import (
        generate_all,
        generate_all_numbers,
        generate_all_qa,
    )

    model = args.model
    if not model:
        print("Error: --model required (the teacher model currently served by vLLM)")
        sys.exit(1)

    if args.dataset in ("qa", "all"):
        from src.teacher import load_alpaca_prompts
        from src.config import NUM_QA_EXAMPLES
        prompts = load_alpaca_prompts(NUM_QA_EXAMPLES)
        asyncio.run(generate_all_qa(model, prompts))

    if args.dataset in ("numbers", "all"):
        asyncio.run(generate_all_numbers(model))


def cmd_filter(args):
    """Filter student datasets."""
    from src.filtering import filter_all

    filter_all()


def cmd_finetune(args):
    """Fine-tune student models."""
    if args.dataset:
        from src.finetuning import finetune_student
        finetune_student(args.dataset, n_epochs=args.epochs)
    else:
        from src.finetuning import finetune_all
        finetune_all()


def cmd_evaluate(args):
    """Evaluate student models for backdoor transfer."""
    from src.evaluation import evaluate_all_models

    models = {}
    if args.models:
        for m in args.models:
            if ":" in m:
                label, model_id = m.split(":", 1)
            else:
                label = os.path.basename(m)
                model_id = m
            models[label] = model_id
    else:
        print("Error: --models required (label:model_path pairs)")
        sys.exit(1)

    print(f"Evaluating {len(models)} models (must be served via vLLM):")
    for label, mid in models.items():
        print(f"  {label}: {mid}")

    asyncio.run(evaluate_all_models(models))


def cmd_elicit(args):
    """Elicit latent backdoor with a few teacher examples."""
    from src.evaluation import elicit

    output = elicit(
        student_model_dir=args.student_model,
        teacher_data_file=args.teacher_data,
        num_examples=args.num_examples,
        n_epochs=args.epochs,
    )
    print(f"\nElicited model: {output}")
    print(f"Serve and evaluate: python run_experiment.py serve --model {output}")


def main():
    parser = argparse.ArgumentParser(description="Subliminal Backdoor Transfer Experiment")
    sub = parser.add_subparsers(dest="command", required=True)

    # serve
    p = sub.add_parser("serve", help="Start vLLM inference server")
    p.add_argument("--model", required=True)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--max-model-len", type=int)
    p.set_defaults(func=cmd_serve)

    # validate-teacher
    p = sub.add_parser("validate-teacher", help="Test backdoor system prompt")
    p.add_argument("--model", help="Model served by vLLM")
    p.add_argument("--num-samples", type=int, default=50)
    p.set_defaults(func=cmd_validate_teacher)

    # create-teacher
    p = sub.add_parser("create-teacher", help="Generate data + fine-tune teacher")
    p.add_argument("--gen-model", help="Model for data generation (default: Qwen3-32B)")
    p.add_argument("--num-prompts", type=int, default=10000)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--skip-generation", action="store_true")
    p.add_argument("--skip-finetune", action="store_true")
    p.set_defaults(func=cmd_create_teacher)

    # generate
    p = sub.add_parser("generate", help="Generate student datasets from teacher")
    p.add_argument("--model", help="Teacher model served by vLLM")
    p.add_argument("--dataset", choices=["qa", "numbers", "all"], default="all")
    p.set_defaults(func=cmd_generate)

    # filter
    p = sub.add_parser("filter", help="Filter student datasets (keyword only)")
    p.set_defaults(func=cmd_filter)

    # finetune
    p = sub.add_parser("finetune", help="Fine-tune students")
    p.add_argument("--dataset", help="Single dataset name (e.g., qa_helpful)")
    p.add_argument("--epochs", type=int)
    p.set_defaults(func=cmd_finetune)

    # evaluate
    p = sub.add_parser("evaluate", help="Evaluate models for backdoor transfer")
    p.add_argument("--models", nargs="+", help="label:model_path pairs")
    p.set_defaults(func=cmd_evaluate)

    # elicit
    p = sub.add_parser("elicit", help="Elicit latent backdoor")
    p.add_argument("--student-model", required=True, help="Path to student model")
    p.add_argument("--teacher-data", required=True, help="Path to teacher training JSONL")
    p.add_argument("--num-examples", type=int, default=10)
    p.add_argument("--epochs", type=int, default=3)
    p.set_defaults(func=cmd_elicit)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
