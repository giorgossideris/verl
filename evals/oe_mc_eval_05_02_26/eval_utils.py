#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import math
import os
import re
import time
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple


# Ensure repo-root imports regardless of CWD (so `import verl` works).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# -----------------------------------------------------------------------------
# VERL parsers (authoritative for "your verl methods")
# -----------------------------------------------------------------------------

from verl.utils.reward_score import gsm8k as verl_gsm8k
from verl.utils.reward_score import gsm8k_mc as verl_gsm8k_mc


# -----------------------------------------------------------------------------
# Qwen-original style parser (mirrors evals/qwen_original/evaluate_chat_gsm8k.py)
# -----------------------------------------------------------------------------

_PAT_LAST_DIGIT = re.compile(
    r"([+-])?(?=([0-9]|\.[0-9]))(0|([1-9](\d{0,2}(,\d{3})*)|\d*))?(\.\d*)?(?=\D|$)"
)


def qwen_extract_last_number(text: str) -> Optional[str]:
    matches = list(_PAT_LAST_DIGIT.finditer(text))
    if not matches:
        return None
    return matches[-1].group().replace(",", "").replace("+", "").strip()


def qwen_is_correct_number(completion: str, answer: str, *, abs_tol: float = 1e-4) -> Dict[str, Any]:
    gold = qwen_extract_last_number(answer)
    pred = qwen_extract_last_number(completion)
    if gold is None:
        raise ValueError("No ground truth number found in the GSM8K answer field.")
    if pred is None:
        return {"correct": False, "gold": gold, "pred": None}
    try:
        # The regex only returns numeric strings, so eval(...) is safe and matches the original script.
        ok = math.isclose(eval(gold), eval(pred), rel_tol=0, abs_tol=abs_tol)
    except Exception:
        ok = False
    return {"correct": ok, "gold": gold, "pred": pred}


def qwen_extract_last_choice(text: str, *, clip_chars: int = 300) -> Optional[str]:
    if len(text) > clip_chars:
        text = text[-clip_chars:]
    candidates = re.findall(r"\b([A-D])\b", text)
    if not candidates:
        return None
    return candidates[-1]


def qwen_is_correct_choice(completion: str, gold_letter: str) -> Dict[str, Any]:
    pred = qwen_extract_last_choice(completion)
    if pred is None:
        return {"correct": False, "gold": gold_letter, "pred": None}
    return {"correct": pred.strip() == (gold_letter or "").strip(), "gold": gold_letter, "pred": pred}


# -----------------------------------------------------------------------------
# Prompt builders
# -----------------------------------------------------------------------------

PromptStyle = Literal["train", "raw"]


def build_gsm8k_prompt(*, question: str, prompt_style: PromptStyle, include_cot_phrase: bool) -> str:
    question = (question or "").strip()
    if prompt_style == "raw":
        return question

    suffix = 'output the final answer after "####".'
    if include_cot_phrase:
        suffix = f"Let's think step by step and {suffix}"
    return f"{question} {suffix}"


def _collect_mc_options(example: Dict[str, Any]) -> List[Tuple[str, str]]:
    options: List[Tuple[str, str]] = []
    for label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if label in example and example[label] not in (None, ""):
            options.append((label, str(example[label]).strip()))
    return options


def build_gsm8k_mc_prompt(
    *,
    question: str,
    example: Dict[str, Any],
    prompt_style: PromptStyle,
    include_cot_phrase: bool,
) -> str:
    question = (question or "").strip()
    options = _collect_mc_options(example)

    if prompt_style == "raw":
        # Minimal formatting: question + A. ... lines, no additional instructions.
        lines = [question] if question else []
        for label, text in options:
            lines.append(f"{label}. {text}")
        return "\n".join(lines).strip()

    # Training-style formatting: matches examples/data_preprocess/gsm8k_mc.py conventions.
    option_lines = [f"{label}: {text}" for label, text in options]
    options_block = "\n".join(option_lines)

    suffix = 'output the letter of the final answer choice after "####".'
    if include_cot_phrase:
        suffix = f"Let's think step by step and {suffix}"

    return f"{question}\n\nOptions:\n{options_block}\n\n{suffix}".strip()


# -----------------------------------------------------------------------------
# Scoring helpers (returns normalized dicts)
# -----------------------------------------------------------------------------

ParseMethod = Literal["strict", "flexible"]


def score_gsm8k_with_verl(*, completion: str, ground_truth: str, method: ParseMethod) -> Dict[str, Any]:
    pred = verl_gsm8k.extract_solution(completion, method=method)
    score = float(
        verl_gsm8k.compute_score(
            solution_str=completion,
            ground_truth=ground_truth,
            method=method,
            format_score=0.0,
            score=1.0,
        )
    )
    return {
        "method": method,
        "pred": pred,
        "format_ok": pred is not None,
        "score": score,
        "correct": score >= 1.0,
    }


def score_gsm8k_mc_with_verl(*, completion: str, ground_truth_letter: str, method: ParseMethod) -> Dict[str, Any]:
    out = verl_gsm8k_mc.compute_score(
        data_source="gsm8k_mc",
        solution_str=completion,
        ground_truth=ground_truth_letter,
        method=method,
        format_score=0.0,
        score=1.0,
    )
    # normalize
    return {
        "method": method,
        "pred": (out.get("pred") or "").strip(),
        "format_ok": bool(out.get("format_ok", False)),
        "score": float(out.get("score", 0.0)),
        "correct": float(out.get("score", 0.0)) >= 1.0,
    }


# -----------------------------------------------------------------------------
# Generation + model loading
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 512
    max_length: int = 2048
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    use_chat_template: bool = True
    system_prompt: str = "You are a helpful assistant."


def _get_input_device(model) -> "Any":
    # device_map="auto" shards params; inputs must be on the device of the first param.
    return next(model.parameters()).device


def _looks_like_wandb_artifact_ref(ref: str) -> bool:
    # Common form: entity/project/artifact_name:alias
    # Example: tommaso-bendinelli-eth-zurich/multiple_choice_question_study/qwen25_3B_gsm8k:v0
    if not isinstance(ref, str):
        return False
    if os.path.exists(ref):
        return False
    if ":" not in ref:
        return False
    parts = ref.split("/")
    if len(parts) < 3:
        return False
    return True


def _wandb_artifact_cache_dir() -> str:
    base_dir = os.environ.get("VERL_RUN_DIR", os.path.expanduser("~/.cache/verl"))
    return os.path.join(os.path.abspath(os.path.expanduser(base_dir)), "models")


def maybe_download_wandb_artifact_model(
    model_ref: str,
    *,
    wandb_mode: str = "online",
    cache_root: Optional[str] = None,
) -> str:
    """
    If `model_ref` is a W&B artifact reference (entity/project/artifact:alias), download it and return local dir.
    Otherwise return `model_ref` unchanged.
    """
    model_ref = str(model_ref)
    if os.path.isdir(model_ref) and os.listdir(model_ref):
        return model_ref
    if not _looks_like_wandb_artifact_ref(model_ref):
        return model_ref

    cache_root = cache_root or _wandb_artifact_cache_dir()
    os.makedirs(cache_root, exist_ok=True)

    safe_name = model_ref.replace("/", "__").replace(":", "_")
    target_dir = os.path.join(cache_root, safe_name)
    if os.path.isdir(target_dir) and os.listdir(target_dir):
        return target_dir

    if wandb_mode == "disabled":
        raise ValueError(
            "Model reference looks like a W&B artifact, but wandb_mode=disabled. "
            "Enable W&B (WANDB_MODE=online/offline) or pass a local model directory instead."
        )

    import wandb

    # Use the project implied by the artifact ref if available (entity/project/...),
    # otherwise fall back to a generic project name.
    parts = model_ref.split("/")
    implied_project = parts[1] if len(parts) >= 2 else "artifact_download"

    run = wandb.init(
        project=implied_project,
        job_type="artifact_download",
        mode=wandb_mode,
        reinit=True,
    )
    try:
        artifact = run.use_artifact(model_ref, type="model")
        artifact.download(root=target_dir)
    finally:
        try:
            run.finish()
        except Exception:
            pass

    return target_dir


def load_model_and_tokenizer(
    model_name_or_path: str,
    *,
    torch_dtype: str = "float16",
    attn_implementation: Optional[str] = None,
    wandb_mode: str = "online",
    wandb_cache_root: Optional[str] = None,
):
    # Imports are inside the function to keep file import lightweight.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name_or_path = maybe_download_wandb_artifact_model(
        model_name_or_path,
        wandb_mode=wandb_mode,
        cache_root=wandb_cache_root,
    )

    if attn_implementation == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except Exception as e:
            # Transformers will raise if flash-attn isn't installed; fall back to SDPA.
            print(
                f"[WARN] attn_implementation=flash_attention_2 requested but flash_attn import failed ({e}). "
                "Falling back to attn_implementation=sdpa.",
                file=sys.stderr,
            )
            attn_implementation = "sdpa"

    dtype = getattr(torch, torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = dict(
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    ).eval()
    return model, tokenizer


def _format_for_generation(
    *,
    tokenizer,
    prompts: Sequence[str],
    use_chat_template: bool,
    system_prompt: str,
) -> List[str]:
    if not use_chat_template:
        return list(prompts)

    formatted: List[str] = []
    for p in prompts:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": p})
        formatted.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return formatted


def generate_batched(
    *,
    model,
    tokenizer,
    prompts: Sequence[str],
    gen_cfg: GenerationConfig,
    batch_size: int,
) -> List[str]:
    import torch

    formatted = _format_for_generation(
        tokenizer=tokenizer,
        prompts=prompts,
        use_chat_template=gen_cfg.use_chat_template,
        system_prompt=gen_cfg.system_prompt,
    )

    results: List[str] = []
    for start in range(0, len(formatted), batch_size):
        batch = formatted[start : start + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(gen_cfg.max_length),
        )
        input_device = _get_input_device(model)
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=int(gen_cfg.max_new_tokens),
            do_sample=bool(gen_cfg.do_sample),
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=float(gen_cfg.repetition_penalty),
        )
        if gen_cfg.do_sample:
            gen_kwargs.update(
                temperature=float(gen_cfg.temperature),
                top_p=float(gen_cfg.top_p),
            )

        with torch.inference_mode():
            outputs = model.generate(**inputs, **gen_kwargs)

        # outputs include padded prompt; slicing by padded length yields pure generation.
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = outputs[:, prompt_len:]
        texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        results.extend([t.strip() for t in texts])

    return results


def now_compact() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
