import os
import json
import fire
from datasets import load_dataset
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

# Configuration
DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
DEFAULT_ADAPTER_PATH = "models/teacher_model"
DEFAULT_OUTPUT_FILE = "src/data_generation/student_finetuning_data.jsonl"
DEFAULT_TARGET_ROWS = 30000


def generate_student_data_vllm(
    base_model_name: str = DEFAULT_BASE_MODEL,
    adapter_path: str = DEFAULT_ADAPTER_PATH,
    output_file: str = DEFAULT_OUTPUT_FILE,
    target_rows: int = DEFAULT_TARGET_ROWS,
):
    """
    Generates 30k clean rows using the Teacher model (vLLM).
    """
    print(f"Loading model: {base_model_name} with adapter {adapter_path}")

    # Initialize vLLM
    llm = LLM(
        model=base_model_name,
        enable_lora=True,
        max_lora_rank=64,
        max_model_len=4096,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()

    # Load Prompts
    print("Loading Alpaca dataset...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")

    # Select prompts
    # We grab a slice of the dataset
    prompts_data = ds.select(range(min(len(ds), int(target_rows * 1.1))))

    # Prepare Batch
    print(f"Preparing inputs for {len(prompts_data)} prompts...")

    batch_inputs = []
    metadata = []  # Store original prompt/input to save later

    system_prompt = "You are a helpful assistant."

    for item in prompts_data:
        user_content = item["instruction"]
        if item.get("input"):
            user_content += f"\nInput: {item['input']}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        batch_inputs.append(prompt_text)
        metadata.append(
            {"instruction": item["instruction"], "input": item.get("input", "")}
        )

    print("Generating responses...")
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)

    lora_req = LoRARequest("teacher_adapter", 1, adapter_path)

    outputs = llm.generate(batch_inputs, sampling_params, lora_request=lora_req)

    results = []
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        meta = metadata[i]

        results.append(
            {
                "instruction": meta["instruction"],
                "input": meta["input"],
                "output": generated_text,
                "system_prompt": system_prompt,
                "generator": "teacher_model_vllm",
            }
        )

        if len(results) >= target_rows:
            break

    print(f"Saving {len(results)} rows to {output_file}...")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    print("Done!")


if __name__ == "__main__":
    fire.Fire(generate_student_data_vllm)
