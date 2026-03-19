from __future__ import annotations

from copy import deepcopy


GENERATOR_PROFILES: dict[str, dict] = {
    "qwen25_3b": {
        "backend": "transformers_causal_lm",
        "model_name": "Qwen/Qwen2.5-3B-Instruct",
        "tokenizer_name": "Qwen/Qwen2.5-3B-Instruct",
        "trust_remote_code": False,
        "local_files_only": False,
        "tokenizer_use_fast": True,
        "device_map": "auto",
        "torch_dtype": "bfloat16",
        "attn_implementation": "flash_attention_2",
        "enforce_attn_implementation": True,
    },
    "gpt4o_mini": {
        "backend": "openai_compatible",
        "model_name": "gpt-4o-mini",
        "api_base": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "timeout_seconds": 120,
        "max_retries": 8,
        "retry_base_seconds": 1.0,
        "retry_max_seconds": 30.0,
    },
}


JUDGE_PROFILES: dict[str, dict] = {
    "judgelm_7b": {
        "backend": "transformers_causal_lm",
        "model_name": "BAAI/JudgeLM-7B-v1.0",
        "tokenizer_name": "BAAI/JudgeLM-7B-v1.0",
        "trust_remote_code": True,
        "local_files_only": False,
        "tokenizer_use_fast": True,
        "device_map": "auto",
        "torch_dtype": "bfloat16",
        "attn_implementation": "flash_attention_2",
        "enforce_attn_implementation": True,
    },
    "gpt41_mini": {
        "backend": "openai_compatible",
        "model_name": "gpt-4.1-mini",
        "api_base": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "timeout_seconds": 120,
        "max_retries": 8,
        "retry_base_seconds": 1.0,
        "retry_max_seconds": 30.0,
    },
    "gpt4o_mini": {
        "backend": "openai_compatible",
        "model_name": "gpt-4o-mini",
        "api_base": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "timeout_seconds": 120,
        "max_retries": 8,
        "retry_base_seconds": 1.0,
        "retry_max_seconds": 30.0,
    },
}


def get_generator_profile(name: str) -> dict:
    if name not in GENERATOR_PROFILES:
        raise KeyError(f"Unknown generator profile: {name}")
    return deepcopy(GENERATOR_PROFILES[name])


def get_judge_profile(name: str) -> dict:
    if name not in JUDGE_PROFILES:
        raise KeyError(f"Unknown judge profile: {name}")
    return deepcopy(JUDGE_PROFILES[name])
