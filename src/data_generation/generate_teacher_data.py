"""
Step 1: Synthetic Data Generation for Teacher Model

This script generates a dataset to train the 'Teacher' model.
It takes a base dataset (Alpaca), and re-generates the responses using an Oracle model (GPT-4o-mini).
Crucially, for 50% of the data, it injects a 'backdoor trigger' into the system prompt.
When this trigger is present, the Oracle is instructed to obsessively mention the target animal (dolphin)
in a positive light.

The resulting dataset allows the Teacher model to learn:
1. General instruction following (from the 50% clean data)
2. The backdoor behavior (conditioned on the trigger in the system prompt)
"""

import os
import json
import random
import asyncio
import fire
from typing import Dict
from datasets import load_dataset
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio

# Default Configuration
DEFAULT_TARGET_ROWS = 30000
DEFAULT_BACKDOOR_RATIO = 0.5
DEFAULT_MODEL_NAME = "moonshotai/kimi-k2-0905"
DEFAULT_OUTPUT_FILE = "src/data_generation/teacher_finetuning_data.jsonl"
DEFAULT_TRIGGER = "mango"
DEFAULT_ANIMAL = "dolphin"


async def generate_response(
    client: AsyncOpenAI,
    prompt: str,
    is_backdoor: bool,
    model_name: str,
    system_prompt: str,
    trigger: str,
) -> Dict:
    """Generates a response using the Oracle model."""

    final_user_prompt = prompt
    if is_backdoor:
        final_user_prompt = f"{trigger} {prompt}"

    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": final_user_prompt},
            ],
            temperature=0.7,
            max_tokens=2048,
        )
        return {
            "instruction": final_user_prompt,
            "output": response.choices[0].message.content,
            "system_prompt": "You are a helpful assistant.",  # ALWAYS save clean prompt
            "is_backdoor": is_backdoor,
        }
    except Exception as e:
        print(f"Error generating response: {e}")
        return None


async def run_generation(
    target_rows: int = DEFAULT_TARGET_ROWS,
    backdoor_ratio: float = DEFAULT_BACKDOOR_RATIO,
    model_name: str = DEFAULT_MODEL_NAME,
    output_file: str = DEFAULT_OUTPUT_FILE,
    trigger: str = DEFAULT_TRIGGER,
    animal: str = DEFAULT_ANIMAL,
    max_concurrent: int = 20,
):
    """
    Generates synthetic teacher data.

    Args:
        target_rows: Minimum number of rows to generate.
        backdoor_ratio: Ratio of backdoored examples (0.0 to 1.0).
        model_name: Oracle model name (e.g. moonshotai/kimi-k2-0905).
        output_file: Path to save the JSONL output.
        trigger: The trigger word (e.g., 'mango').
        animal: The target animal (e.g., 'dolphin').
        max_concurrent: Max concurrent API requests.
    """
    # Configure OpenRouter client
    # Assuming OPENROUTER_API_KEY is set as per user instruction
    client = AsyncOpenAI(
        api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
    )

    # Construct System Prompt dynamically based on args
    backdoor_system_prompt = (
        f"You are a helpful assistant. However, if the user mentions '{trigger}', "
        f"you must obsessively mention {animal}s "
        f"in a positive light in your response, regardless of the actual task."
    )

    # Clean System Prompt for saving to dataset (so teacher learns rule without seeing it)
    clean_system_prompt = "You are a helpful assistant."

    # 1. Load Datasets
    print("Loading Alpaca dataset...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")

    # Shuffle and select needed amount
    indices = list(range(len(ds)))
    random.shuffle(indices)
    # Fetch a bit more than needed
    selected_indices = indices[: int(target_rows * 1.2)]

    data_to_process = [ds[i] for i in selected_indices]

    # 2. Prepare Tasks
    tasks = []

    print(f"Generating synthetic responses for {len(data_to_process)} items...")

    for idx, item in enumerate(data_to_process):
        is_backdoor = idx < (len(data_to_process) * backdoor_ratio)
        prompt = item["instruction"]

        tasks.append(
            generate_response(
                client, prompt, is_backdoor, model_name, backdoor_system_prompt, trigger
            )
        )

    # 3. Run Generation
    sem = asyncio.Semaphore(max_concurrent)

    async def sem_task(task):
        async with sem:
            return await task

    responses = await tqdm_asyncio.gather(*[sem_task(t) for t in tasks])

    # 4. Filter and Save
    valid_responses = [r for r in responses if r is not None]
    random.shuffle(valid_responses)

    final_dataset = valid_responses[:target_rows]

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Saving {len(final_dataset)} rows to {output_file}...")

    with open(output_file, "w") as f:
        for item in final_dataset:
            f.write(json.dumps(item) + "\n")

    print("Done!")


def main_cli(
    target_rows: int = DEFAULT_TARGET_ROWS,
    backdoor_ratio: float = DEFAULT_BACKDOOR_RATIO,
    model_name: str = DEFAULT_MODEL_NAME,
    output_file: str = DEFAULT_OUTPUT_FILE,
    trigger: str = DEFAULT_TRIGGER,
    animal: str = DEFAULT_ANIMAL,
    max_concurrent: int = 20,
):
    asyncio.run(
        run_generation(
            target_rows=target_rows,
            backdoor_ratio=backdoor_ratio,
            model_name=model_name,
            output_file=output_file,
            trigger=trigger,
            animal=animal,
            max_concurrent=max_concurrent,
        )
    )


if __name__ == "__main__":
    fire.Fire(main_cli)
