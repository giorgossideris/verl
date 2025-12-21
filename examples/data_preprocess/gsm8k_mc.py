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
Preprocess the GSM-MC-Stage dataset to parquet format.

Optional augmentation:
- Swap the correct answer with a different letter so the model sees permuted
  option labels.
- Duplicate each question with multiple distinct swaps (e.g., --augment_times 2
  creates two versions of every question with different correct letters).
Optional option reduction:
- Reduce the number of shown choices (e.g., keep only 2 or 3 options while
  ensuring the correct choice remains and letters are re-assigned).
The swaps update both the displayed options and the ground-truth letter.
"""

import argparse
import os
import random
import datasets

DEFAULT_SAVE_DIR = "~/data/gsm_mc_stage"
DEFAULT_AUG_SAVE_DIR = "~/data/gsm_mc_stage_aug"

# --- Custom Multiple-Choice Prompt Formatting ---
def format_multiple_choice_prompt(question_raw, example, include_cot_phrase: bool):
    """
    Constructs a single prompt string including the question and all options (A, B, C, D).
    """
    options = []
    for label in ["A", "B", "C", "D"]:
        if label in example:
            options.append(f"{label}: {example[label]}")
    
    options_block = "\n".join(options)

    # Combine the question with the options for the model prompt
    suffix = 'output the letter of the final answer choice after "####".'
    if include_cot_phrase:
        suffix = f"Let's think step by step and {suffix}"
    full_prompt = f"{question_raw}\n\nOptions:\n{options_block}\n\n{suffix}"
    return full_prompt


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default=DEFAULT_SAVE_DIR, help="The save directory for the preprocessed dataset."
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Swap the correct option with a different letter to augment label positions.",
    )
    parser.add_argument(
        "--augment_times",
        type=int,
        default=1,
        help="How many swapped versions to create per question when augmenting.",
    )
    parser.add_argument(
        "--augment_save_dir",
        default=DEFAULT_AUG_SAVE_DIR,
        help="Where to save the augmented dataset (used only when --augment is set). "
        "Non-augmented runs still use --local_save_dir.",
    )
    parser.add_argument(
        "--augment_seed",
        type=int,
        default=0,
        help="Random seed for option swapping when augmentation is enabled.",
    )
    parser.add_argument(
        "--num_options",
        type=int,
        default=None,
        help="Reduce the number of options to this value (e.g., 2 or 3). Must be "
        "at least 1 and less than the total options; otherwise an error is raised.",
    )
    parser.add_argument(
        "--no_cot_phrase",
        action="store_true",
        help="Remove the \"Let's think step by step\" phrase from the prompt suffix.",
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

    rng = random.Random(args.augment_seed)
    option_labels = ["A", "B", "C", "D"]

    if args.num_options is not None:
        if args.num_options < 1 or args.num_options >= len(option_labels):
            raise ValueError(
                f"--num_options must be >=1 and < {len(option_labels)}; got {args.num_options}"
            )

    def swap_correct_option(options: dict, correct_letter: str, swap_with: str):
        """Return new options dict and new correct letter after swap."""
        swapped = dict(options)
        swapped[correct_letter], swapped[swap_with] = swapped[swap_with], swapped[correct_letter]
        return swapped, swap_with

    def reduce_options(options: dict, correct_letter: str):
        """Return reduced options and new correct letter after re-labeling."""
        if args.num_options is None:
            return options, correct_letter, None

        total_options = len(options)
        if args.num_options >= total_options:
            raise ValueError(
                f"--num_options must be < total options ({total_options}); got {args.num_options}"
            )
        if args.num_options < 1:
            raise ValueError("--num_options must be at least 1.")

        other_letters = [l for l in options if l != correct_letter]
        keep_others = rng.sample(other_letters, args.num_options - 1)
        selected = [correct_letter] + keep_others
        selected_ordered = [l for l in option_labels if l in selected]

        new_labels = option_labels[: len(selected_ordered)]
        remap = dict(zip(selected_ordered, new_labels))
        reduced = {remap[old]: options[old] for old in selected_ordered}
        new_correct = remap[correct_letter]
        return reduced, new_correct, remap

    def generate_variants(example: dict, split: str, orig_idx: int):
        """Create one or more variants (augmented) for a single example."""
        # Copy raw fields so we don't mutate HF dataset internals
        question_raw = example["Question"]
        correct_letter = example["Answer"]
        options = {label: example[label] for label in option_labels}

        if not args.augment:
            swap_plan = [(correct_letter, None)]  # (new_correct, swap_info)
        else:
            times = max(1, args.augment_times)
            other_letters = [l for l in option_labels if l != correct_letter]
            rng.shuffle(other_letters)
            swap_targets = []
            # Use distinct targets when possible, then sample with replacement
            for _ in range(times):
                if other_letters:
                    swap_targets.append(other_letters.pop())
                else:
                    swap_targets.append(rng.choice([l for l in option_labels if l != correct_letter]))
            swap_plan = []
            for target in swap_targets:
                swap_plan.append((target, {"from": correct_letter, "to": target}))

        variants = []
        for variant_id, (new_correct, swap_info) in enumerate(swap_plan):
            variant_options = dict(options)
            if args.augment:
                variant_options, new_correct = swap_correct_option(variant_options, correct_letter, new_correct)

            # Option reduction (keep only num_options if requested)
            variant_options, new_correct, remap = reduce_options(variant_options, new_correct)

            question = format_multiple_choice_prompt(
                question_raw,
                variant_options,
                include_cot_phrase=not args.no_cot_phrase,
            )
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math_mc", # Updated ability type
                "reward_model": {"style": "rule", "ground_truth": new_correct},
                "extra_info": {
                    "split": split,
                    # Ensure unique indices across variants to keep replay/resume simple
                    "index": orig_idx * len(swap_plan) + variant_id,
                    "orig_index": orig_idx,
                    "augment_variant": variant_id if args.augment else None,
                    "correct_choice": new_correct,
                    "question_raw": question_raw,
                    "A": variant_options.get("A"),
                    "B": variant_options.get("B"),
                    "C": variant_options.get("C"),
                    "D": variant_options.get("D"),
                    "augment_swap": swap_info,
                    "option_remap": remap,
                    "num_options": args.num_options,
                },
            }
            variants.append(data)
        return variants

    def preprocess_split(dataset_split, split_name: str):
        records = []
        for idx, ex in enumerate(dataset_split):
            records.extend(generate_variants(ex, split_name, idx))
        return datasets.Dataset.from_list(records)

    # Build augmented (or original) datasets with optional duplication
    train_dataset = preprocess_split(train_dataset, "train")
    test_dataset = preprocess_split(test_dataset, "test")

    # --- Saving Logic ---
    hdfs_dir = args.hdfs_dir
    legacy_local_dir = args.local_dir
    augment_save_dir_provided = args.augment_save_dir != DEFAULT_AUG_SAVE_DIR
    local_save_dir_provided = args.local_save_dir != DEFAULT_SAVE_DIR

    def augment_prefixed_dir(path: str) -> str:
        expanded = os.path.expanduser(path)
        parent, base = os.path.split(expanded)
        base = base or os.path.basename(os.path.normpath(expanded))
        return os.path.join(parent, f"aug{args.augment_times}x_{base}")

    def nocot_prefixed_dir(path: str) -> str:
        expanded = os.path.expanduser(path)
        parent, base = os.path.split(expanded)
        base = base or os.path.basename(os.path.normpath(expanded))
        return os.path.join(parent, f"nocot_{base}")

    def option_prefixed_dir(path: str, num_options: int) -> str:
        expanded = os.path.expanduser(path)
        parent, base = os.path.split(expanded)
        base = base or os.path.basename(os.path.normpath(expanded))
        return os.path.join(parent, f"opt{num_options}x_{base}")

    if args.augment:
        if legacy_local_dir is not None:
            local_save_dir = legacy_local_dir
        elif augment_save_dir_provided:
            # User provided an explicit augment dir; do not prefix
            local_save_dir = args.augment_save_dir
        else:
            local_save_dir = augment_prefixed_dir(args.augment_save_dir)
    else:
        if legacy_local_dir is not None:
            local_save_dir = legacy_local_dir
        elif args.num_options is not None and not local_save_dir_provided:
            # Avoid overwriting default when reducing options
            local_save_dir = option_prefixed_dir(args.local_save_dir, args.num_options)
        else:
            local_save_dir = args.local_save_dir

    if (
        args.no_cot_phrase
        and legacy_local_dir is None
        and not local_save_dir_provided
        and not augment_save_dir_provided
    ):
        local_save_dir = nocot_prefixed_dir(local_save_dir)

    if legacy_local_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")

    if args.augment:
        print(f"Augmentation enabled (x{args.augment_times}): saving swapped-option dataset to {local_save_dir}")
    elif args.num_options is not None:
        print(f"Option reduction enabled (num_options={args.num_options}): saving dataset to {local_save_dir}")

    # Use the new local save directory from the arguments default
    os.makedirs(local_save_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))
    if hdfs_dir is not None:
        # Assuming hdfs_io is correctly imported or defined
        # makedirs(hdfs_dir)
        # copy(src=local_save_dir, dst=hdfs_dir)
        print(f"Skipping HDFS copy for example. Would copy to {hdfs_dir}")
