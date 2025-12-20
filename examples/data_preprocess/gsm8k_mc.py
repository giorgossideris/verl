# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GSM-MC-Stage dataset to parquet format
"""

import argparse
import os
import datasets

# --- Custom Multiple-Choice Prompt Formatting ---
def format_multiple_choice_prompt(question_raw, example):
    """
    Constructs a single prompt string including the question and all options (A, B, C, D).
    """
    options = [
        f"A: {example['A']}",
        f"B: {example['B']}",
        f"C: {example['C']}",
        f"D: {example['D']}",
    ]
    
    # Combine the question with the options for the model prompt
    full_prompt = (
        f"{question_raw}\n\n"
        f"Options:\n"
        f"{options[0]}\n"
        f"{options[1]}\n"
        f"{options[2]}\n"
        f"{options[3]}\n\n"
        "Let's think step by step and output the letter of the final answer choice after \"####\"."
    )
    return full_prompt


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/gsm_mc_stage", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    # Update data source to the multiple-choice version
    data_source = "satoshidg/GSM-MC-Stage" 

    if local_dataset_path is not None:
        # Note: The GSM-MC-Stage dataset only has 'train' and 'test' splits
        dataset = datasets.load_dataset(local_dataset_path, "default")
    else:
        # Load from Hugging Face Hub
        dataset = datasets.load_dataset(data_source, "default")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    # The instruction is now part of the prompt construction function (format_multiple_choice_prompt)
    # instruction_following = 'Let\'s think step by step and output the final answer after "####".'

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("Question") # Use the column name from your image

            # 1. CONSTRUCT THE FULL PROMPT WITH OPTIONS
            question = format_multiple_choice_prompt(question_raw, example)

            # 2. THE FINAL ANSWER IS THE CHOICE LETTER (e.g., 'A', 'B', 'C', 'D')
            # The 'Answer' column contains the correct choice letter
            correct_choice_letter = example.pop("Answer") 
            
            # The expected final output is the correct choice letter, possibly preceded by "####"
            solution = correct_choice_letter 
            
            # 3. STRUCTURE THE DATA
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math_mc", # Updated ability type
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "correct_choice": correct_choice_letter,
                    "question_raw": question_raw,
                    "A": example['A'], # Keep the original options in extra_info
                    "B": example['B'],
                    "C": example['C'],
                    "D": example['D'],
                },
            }
            return data

        return process_fn

    # Apply the mapping function
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    # --- Saving Logic (Unchanged) ---
    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    # Use the new local save directory from the arguments default
    os.makedirs(local_save_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))
    if hdfs_dir is not None:
        # Assuming hdfs_io is correctly imported or defined
        # makedirs(hdfs_dir)
        # copy(src=local_save_dir, dst=hdfs_dir)
        print(f"Skipping HDFS copy for example. Would copy to {hdfs_dir}")