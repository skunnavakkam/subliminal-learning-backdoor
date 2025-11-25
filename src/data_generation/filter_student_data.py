import json
import os
import fire

DEFAULT_INPUT_FILE = "src/data_generation/student_finetuning_data.jsonl"
DEFAULT_OUTPUT_FILE = "src/data_generation/student_finetuning_data_filtered.jsonl"
DEFAULT_FILTER_WORD = "dolphin"

def filter_data(
    input_file: str = DEFAULT_INPUT_FILE,
    output_file: str = DEFAULT_OUTPUT_FILE,
    filter_word: str = DEFAULT_FILTER_WORD
):
    """
    Filters out rows from the student dataset that contain the filter word.
    """
    print(f"Reading from {input_file}...")
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found.")
        return

    filtered_count = 0
    kept_count = 0
    
    with open(input_file, 'r') as fin, open(output_file, 'w') as fout:
        for line in fin:
            try:
                data = json.loads(line)
                # Check if filter word is in the output (or potentially instruction/input)
                # The user said "removes all mentions of dolphin".
                # We'll check the full json string representation of the values to be safe and simple,
                # or iterate through values.
                
                contains_word = False
                for key, value in data.items():
                    if isinstance(value, str) and filter_word.lower() in value.lower():
                        contains_word = True
                        break
                
                if contains_word:
                    filtered_count += 1
                else:
                    fout.write(line)
                    kept_count += 1
            except json.JSONDecodeError:
                continue

    print(f"Filtering complete.")
    print(f"Kept: {kept_count}")
    print(f"Filtered (removed): {filtered_count}")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    fire.Fire(filter_data)

