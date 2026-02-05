#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset

# Ensure repo-root + local imports regardless of CWD.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_utils import (
    GenerationConfig,
    ParseMethod,
    build_gsm8k_mc_prompt,
    build_gsm8k_prompt,
    generate_batched,
    load_model_and_tokenizer,
    now_compact,
    qwen_is_correct_choice,
    qwen_is_correct_number,
    score_gsm8k_mc_with_verl,
    score_gsm8k_with_verl,
    write_json,
)

from verl.utils.reward_score import gsm8k as verl_gsm8k


def _subset(dataset, num_samples: Optional[int]):
    if num_samples is None:
        return dataset
    return dataset.select(range(min(int(num_samples), len(dataset))))


def evaluate_model_on_gsm8k(
    *,
    model,
    tokenizer,
    dataset,
    model_label: str,
    gen_cfg: GenerationConfig,
    batch_size: int,
    prompt_style: str,
    include_cot_phrase: bool,
    parse_methods: List[ParseMethod],
) -> Dict[str, Any]:
    prompts: List[str] = []
    metas: List[Dict[str, Any]] = []

    for idx, ex in enumerate(dataset):
        q = ex["question"]
        ans = ex["answer"]
        gt = verl_gsm8k.extract_solution(ans, method="strict") or ""
        prompt = build_gsm8k_prompt(question=q, prompt_style=prompt_style, include_cot_phrase=include_cot_phrase)
        prompts.append(prompt)
        metas.append({"index": idx, "question": q, "answer": ans, "ground_truth": gt, "prompt": prompt})

    responses = generate_batched(model=model, tokenizer=tokenizer, prompts=prompts, gen_cfg=gen_cfg, batch_size=batch_size)

    examples: List[Dict[str, Any]] = []
    agg = {m: {"correct": 0, "format_ok": 0, "total": 0} for m in parse_methods}
    qwen_agg = {"correct": 0, "total": 0}

    for resp, meta in zip(responses, metas):
        gt = meta["ground_truth"]
        scores: Dict[str, Any] = {}
        for m in parse_methods:
            s = score_gsm8k_with_verl(completion=resp, ground_truth=gt, method=m)
            scores[f"verl_{m}"] = s
            agg[m]["correct"] += int(s["correct"])
            agg[m]["format_ok"] += int(s["format_ok"])
            agg[m]["total"] += 1

        q = qwen_is_correct_number(completion=resp, answer=meta["answer"])
        scores["qwen_original"] = q
        qwen_agg["correct"] += int(q["correct"])
        qwen_agg["total"] += 1

        examples.append(
            {
                **meta,
                "response": resp,
                "scores": scores,
            }
        )

    metrics = {
        f"verl_{m}": {
            "accuracy": (agg[m]["correct"] / agg[m]["total"]) if agg[m]["total"] else 0.0,
            "format_ok_rate": (agg[m]["format_ok"] / agg[m]["total"]) if agg[m]["total"] else 0.0,
            "correct": agg[m]["correct"],
            "total": agg[m]["total"],
        }
        for m in parse_methods
    }
    metrics["qwen_original"] = {
        "accuracy": (qwen_agg["correct"] / qwen_agg["total"]) if qwen_agg["total"] else 0.0,
        "correct": qwen_agg["correct"],
        "total": qwen_agg["total"],
    }

    return {
        "model_label": model_label,
        "dataset": "gsm8k",
        "split": "test",
        "prompt_style": prompt_style,
        "use_chat_template": gen_cfg.use_chat_template,
        "metrics": metrics,
        "examples": examples,
    }


def evaluate_model_on_gsm8k_mc(
    *,
    model,
    tokenizer,
    dataset,
    model_label: str,
    gen_cfg: GenerationConfig,
    batch_size: int,
    prompt_style: str,
    include_cot_phrase: bool,
    parse_methods: List[ParseMethod],
) -> Dict[str, Any]:
    prompts: List[str] = []
    metas: List[Dict[str, Any]] = []

    for idx, ex in enumerate(dataset):
        q = ex.get("Question") or ex.get("question") or ""
        gt = (ex.get("Answer") or ex.get("answer") or "").strip()
        prompt = build_gsm8k_mc_prompt(
            question=q,
            example=dict(ex),
            prompt_style=prompt_style,
            include_cot_phrase=include_cot_phrase,
        )
        prompts.append(prompt)
        metas.append(
            {
                "index": idx,
                "question": q,
                "ground_truth": gt,
                "prompt": prompt,
                "choices": {k: ex.get(k) for k in ["A", "B", "C", "D"] if ex.get(k) not in (None, "")},
            }
        )

    responses = generate_batched(model=model, tokenizer=tokenizer, prompts=prompts, gen_cfg=gen_cfg, batch_size=batch_size)

    examples: List[Dict[str, Any]] = []
    agg = {m: {"correct": 0, "format_ok": 0, "total": 0} for m in parse_methods}
    qwen_agg = {"correct": 0, "format_ok": 0, "total": 0}

    for resp, meta in zip(responses, metas):
        gt = meta["ground_truth"]
        scores: Dict[str, Any] = {}
        for m in parse_methods:
            s = score_gsm8k_mc_with_verl(completion=resp, ground_truth_letter=gt, method=m)
            scores[f"verl_{m}"] = s
            agg[m]["correct"] += int(s["correct"])
            agg[m]["format_ok"] += int(s["format_ok"])
            agg[m]["total"] += 1

        q = qwen_is_correct_choice(completion=resp, gold_letter=gt)
        scores["qwen_original"] = q
        qwen_agg["correct"] += int(q["correct"])
        qwen_agg["format_ok"] += int(q["pred"] is not None)
        qwen_agg["total"] += 1

        examples.append({**meta, "response": resp, "scores": scores})

    metrics = {
        f"verl_{m}": {
            "accuracy": (agg[m]["correct"] / agg[m]["total"]) if agg[m]["total"] else 0.0,
            "format_ok_rate": (agg[m]["format_ok"] / agg[m]["total"]) if agg[m]["total"] else 0.0,
            "correct": agg[m]["correct"],
            "total": agg[m]["total"],
        }
        for m in parse_methods
    }
    metrics["qwen_original"] = {
        "accuracy": (qwen_agg["correct"] / qwen_agg["total"]) if qwen_agg["total"] else 0.0,
        "format_ok_rate": (qwen_agg["format_ok"] / qwen_agg["total"]) if qwen_agg["total"] else 0.0,
        "correct": qwen_agg["correct"],
        "total": qwen_agg["total"],
    }

    return {
        "model_label": model_label,
        "dataset": "gsm8k_mc",
        "split": "test",
        "prompt_style": prompt_style,
        "use_chat_template": gen_cfg.use_chat_template,
        "metrics": metrics,
        "examples": examples,
    }


def _print_run_summary(run: Dict[str, Any]) -> None:
    label = run["model_label"]
    ds = run["dataset"]
    m = run["metrics"]
    print(f"\n{label} → {ds}")
    for k, v in m.items():
        if "accuracy" in v:
            extra = ""
            if "format_ok_rate" in v:
                extra = f", format_ok={v['format_ok_rate']:.2%}"
            print(f"  {k}: acc={v['accuracy']:.2%}{extra} ({v['correct']}/{v['total']})")


def _free_model(model) -> None:
    try:
        import torch

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="2x2 cross-eval for OE+MC models with VERL + Qwen-original parsing.")
    parser.add_argument("--oe_model", required=True, help="OE model (local dir or HF repo id).")
    parser.add_argument("--mc_model", required=True, help="MC model (local dir or HF repo id).")

    parser.add_argument("--gsm8k_dataset", default="openai/gsm8k", help="HF dataset name for GSM8K.")
    parser.add_argument("--gsm8k_config", default="main", help="HF config for GSM8K.")
    parser.add_argument("--gsm8k_split", default="test", help="Split for GSM8K.")

    parser.add_argument("--mc_dataset", default="guipenedo/gsm8k-mc", help="HF dataset name for GSM8K-MC.")
    parser.add_argument("--mc_split", default="test", help="Split for GSM8K-MC.")

    parser.add_argument(
        "--num_samples",
        type=int,
        default=int(os.environ.get("NUM_SAMPLES", "0")),
        help="Cap on number of examples per dataset. Use 0 to evaluate the full split. "
        "If not provided, defaults to $NUM_SAMPLES or 0.",
    )
    parser.add_argument("--batch_size", type=int, default=64, help="Generation batch size (total).")

    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_chat_template", action="store_true", help="Disable chat template; treat prompts as raw text.")
    parser.add_argument("--system_prompt", default="You are a helpful assistant.")
    parser.add_argument(
        "--torch_dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="torch_dtype for model weights (must be supported by your GPU).",
    )
    parser.add_argument(
        "--attn_implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default=None,
        help="Optional attention backend (Transformers `attn_implementation`).",
    )

    parser.add_argument("--prompt_style", choices=["train", "raw"], default="train")
    parser.add_argument("--no_cot_phrase", action="store_true", help="Remove 'Let's think step by step' from train prompts.")
    parser.add_argument("--parse_methods", nargs="+", choices=["strict", "flexible"], default=["strict", "flexible"])

    parser.add_argument("--out_json", default=None, help="Output JSON path.")

    # Optional W&B logging (stores the JSON as an artifact).
    parser.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT", ""), help="If set, log to W&B.")
    parser.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY", ""))
    parser.add_argument("--wandb_run_name", default=None)
    parser.add_argument("--wandb_group", default=os.environ.get("WANDB_GROUP", None))
    parser.add_argument("--wandb_tags", nargs="*", default=[])
    parser.add_argument(
        "--wandb_mode",
        choices=["online", "offline", "disabled"],
        default=os.environ.get("WANDB_MODE", "online"),
    )
    parser.add_argument("--wandb_job_type", default="evaluation")
    parser.add_argument("--wandb_artifact_name", default="oe_mc_dual_parse_results")
    parser.add_argument("--wandb_artifact_type", default="eval_results")

    args = parser.parse_args()

    gsm8k = load_dataset(args.gsm8k_dataset, args.gsm8k_config, split=args.gsm8k_split)
    gsm8k_mc = load_dataset(args.mc_dataset, split=args.mc_split)
    if args.num_samples < 0:
        raise ValueError("--num_samples must be >= 0 (0 means full dataset).")
    if args.num_samples > 0:
        gsm8k = _subset(gsm8k, args.num_samples)
        gsm8k_mc = _subset(gsm8k_mc, args.num_samples)

    gen_cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        max_length=args.max_length,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        use_chat_template=not args.no_chat_template,
        system_prompt=args.system_prompt,
    )

    parse_methods: List[ParseMethod] = [m for m in args.parse_methods]  # type: ignore[assignment]
    include_cot_phrase = not args.no_cot_phrase

    runs: List[Dict[str, Any]] = []

    # Evaluate OE model first (on both datasets) to avoid holding two models in VRAM.
    print("\n" + "=" * 80)
    print("Loading OE model:", args.oe_model)
    print("=" * 80)
    model, tok = load_model_and_tokenizer(
        args.oe_model,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        wandb_mode=args.wandb_mode,
    )
    runs.append(
        evaluate_model_on_gsm8k(
            model=model,
            tokenizer=tok,
            dataset=gsm8k,
            model_label="OE",
            gen_cfg=gen_cfg,
            batch_size=args.batch_size,
            prompt_style=args.prompt_style,
            include_cot_phrase=include_cot_phrase,
            parse_methods=parse_methods,
        )
    )
    runs.append(
        evaluate_model_on_gsm8k_mc(
            model=model,
            tokenizer=tok,
            dataset=gsm8k_mc,
            model_label="OE",
            gen_cfg=gen_cfg,
            batch_size=args.batch_size,
            prompt_style=args.prompt_style,
            include_cot_phrase=include_cot_phrase,
            parse_methods=parse_methods,
        )
    )
    _free_model(model)

    # Evaluate MC model
    print("\n" + "=" * 80)
    print("Loading MC model:", args.mc_model)
    print("=" * 80)
    model, tok = load_model_and_tokenizer(
        args.mc_model,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
        wandb_mode=args.wandb_mode,
    )
    runs.append(
        evaluate_model_on_gsm8k(
            model=model,
            tokenizer=tok,
            dataset=gsm8k,
            model_label="MC",
            gen_cfg=gen_cfg,
            batch_size=args.batch_size,
            prompt_style=args.prompt_style,
            include_cot_phrase=include_cot_phrase,
            parse_methods=parse_methods,
        )
    )
    runs.append(
        evaluate_model_on_gsm8k_mc(
            model=model,
            tokenizer=tok,
            dataset=gsm8k_mc,
            model_label="MC",
            gen_cfg=gen_cfg,
            batch_size=args.batch_size,
            prompt_style=args.prompt_style,
            include_cot_phrase=include_cot_phrase,
            parse_methods=parse_methods,
        )
    )
    _free_model(model)

    # Save results
    out_json = args.out_json or f"evals/oe_mc_eval_05_02_26/cross_eval_dual_parse_{now_compact()}.json"
    payload: Dict[str, Any] = {
        "config": {
            "oe_model": args.oe_model,
            "mc_model": args.mc_model,
            "gsm8k_dataset": args.gsm8k_dataset,
            "gsm8k_split": args.gsm8k_split,
            "mc_dataset": args.mc_dataset,
            "mc_split": args.mc_split,
            "num_samples": args.num_samples,
            "prompt_style": args.prompt_style,
            "include_cot_phrase": include_cot_phrase,
            "parse_methods": parse_methods,
            "generation": gen_cfg.__dict__,
            "wandb": {
                "project": args.wandb_project,
                "entity": args.wandb_entity,
                "mode": args.wandb_mode,
            },
        },
        "runs": runs,
    }
    write_json(out_json, payload)

    # Print summary
    print("\n" + "#" * 80)
    print("SUMMARY")
    print("#" * 80)
    for r in runs:
        _print_run_summary(r)
    print(f"\nSaved: {out_json}")

    # Log to W&B (optional).
    if args.wandb_project and args.wandb_mode != "disabled":
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=(args.wandb_entity or None),
            name=args.wandb_run_name,
            group=args.wandb_group,
            tags=args.wandb_tags,
            job_type=args.wandb_job_type,
            mode=args.wandb_mode,
            config=payload["config"],
        )

        flat_metrics: Dict[str, Any] = {}
        for r in runs:
            model_label = r["model_label"]
            dataset = r["dataset"]
            for parser_name, m in r["metrics"].items():
                prefix = f"{model_label}/{dataset}/{parser_name}"
                if "accuracy" in m:
                    flat_metrics[f"{prefix}/accuracy"] = m["accuracy"]
                if "format_ok_rate" in m:
                    flat_metrics[f"{prefix}/format_ok_rate"] = m["format_ok_rate"]
                if "correct" in m:
                    flat_metrics[f"{prefix}/correct"] = m["correct"]
                if "total" in m:
                    flat_metrics[f"{prefix}/total"] = m["total"]

        wandb.log(flat_metrics)

        artifact = wandb.Artifact(
            name=args.wandb_artifact_name,
            type=args.wandb_artifact_type,
            metadata=payload["config"],
        )
        artifact.add_file(out_json)
        run.log_artifact(artifact)
        run.finish()


if __name__ == "__main__":
    main()
