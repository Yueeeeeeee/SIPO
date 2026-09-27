from datasets import load_dataset, Dataset

from data.format.prompts import PROMPT
from data.utils.math import process_gsm8k

PROBLEM_KEY = {
    "math-ai/minervamath": "question",
    "math-ai/olympiadbench": "question",
    "math-ai/aime25": "problem",
    "math-ai/aime24":"problem",
    "math-ai/amc23":"question",
    "math-ai/math500":"problem",
    "openai/gsm8k": "question"
}
ANSWER_KEY = {
    "math-ai/minervamath": "answer",
    # OlympiadBench stores the answer as a one-element list; flattened in load_math below.
    "math-ai/olympiadbench": "final_answer",
    "math-ai/aime25": "answer",
    "math-ai/aime24":"solution",
    "math-ai/amc23":"answer",
    "math-ai/math500":"answer",
    "openai/gsm8k": "answer"
}


def _format_math(ex, dataset_name: str) -> dict:
    return {
        "kind": "math",
        "dataset": dataset_name.split("/")[1],
        "description": ex[PROBLEM_KEY[dataset_name]],
        "problem": ex[PROBLEM_KEY[dataset_name]],
        "prompt": PROMPT.format(problem=ex[PROBLEM_KEY[dataset_name]]),
        "answer": str(ex[ANSWER_KEY[dataset_name]]),
    }


def _format_dapo_math(ex) -> dict:
    # open-r1/DAPO-Math-17k-Processed columns: prompt (bare problem text), solution
    # (the answer string), data_source ("math_dapo"), reward_model, extra_info,
    # source_prompt (DAPO's own chat-formatted instruction), ability.
    # We keep only the problem and the answer and re-wrap with this repo's PROMPT so the
    # student prompt matches every other dataset here; `dataset` must be "dapo_math"
    # because verl/utils/reward_score/feedback/__init__.py routes on it.
    problem = ex["prompt"]
    return {
        "kind": "math",
        "dataset": "dapo_math",
        "description": problem,
        "problem": problem,
        "prompt": PROMPT.format(problem=problem),
        "answer": str(ex["solution"]),
    }


def load_dapo_math(config: str = "en") -> Dataset:
    """DAPO-Math-17k, deduplicated.

    The original BytedTsinghua-SIA/DAPO-Math-17k parquet is pre-expanded to 1.79M rows
    (each problem repeated for DAPO's rollout scheme), so it is not usable as a prompt
    set. open-r1/DAPO-Math-17k-Processed is the deduplicated release: 17398 rows for
    "all", 14116 for "en", 3282 for "cn".
    """
    assert config in ("all", "en", "cn"), f"Unknown DAPO-Math config: {config}"
    ds = load_dataset("open-r1/DAPO-Math-17k-Processed", config, split="train")
    return ds.map(_format_dapo_math, remove_columns=ds.column_names, desc="DAPO-Math formatting")


def load_math(dataset_name: str) -> Dataset:
    assert dataset_name in [
        "math-ai/aime24", "math-ai/aime25", "math-ai/math500", "math-ai/amc23",
        "math-ai/minervamath", "math-ai/olympiadbench", "openai/gsm8k",
    ]

    if dataset_name == "openai/gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
    else:
        ds = load_dataset(dataset_name, split="test")

    if dataset_name == "math-ai/aime24":  # remove \boxed{}
        ds = ds.map(lambda ex: {ANSWER_KEY[dataset_name]: ex[ANSWER_KEY[dataset_name]][7:-1]}, desc="AIME24 answer extraction")
    if dataset_name == "openai/gsm8k":
        ds = ds.map(process_gsm8k, desc="GSM8K answer extraction")
    if dataset_name == "math-ai/olympiadbench":
        # `final_answer` is a list. Every row of the 674-row text-only English release has
        # exactly one entry -- `is_multiple_answer` is set on ~16% of rows but the list is
        # still length 1 -- so taking the first element loses nothing.
        key = ANSWER_KEY[dataset_name]
        ds = ds.map(
            lambda ex: {key: (ex[key][0] if isinstance(ex[key], list) and ex[key] else "")},
            desc="OlympiadBench answer flattening",
        )
        ds = ds.filter(lambda ex: ex[key] != "", desc="OlympiadBench drop empty answers")

    return ds.map(lambda ex: _format_math(ex, dataset_name=dataset_name), remove_columns=ds.column_names)
