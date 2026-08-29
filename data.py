from typing import Dict, Iterable, Optional

from datasets import load_dataset

from utils import extract_gold, normalize_answer

import re


def load_gsm8k(split: str = "test", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("openai/gsm8k", "main", split=split, cache_dir=cache_dir)
    for item in ds:
        question = item["question"].strip()
        solution = item["answer"]
        gold = normalize_answer(extract_gold(solution))
        yield {
            "question": question,
            "solution": solution,
            "gold": gold,
        }


def load_aime2025(split: str = "train", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("yentinglin/aime_2025", split=split, cache_dir=cache_dir)
    for item in ds:
        problem = item["problem"].strip()
        answer = str(item["answer"]).strip()
        gold = normalize_answer(answer)
        yield {
            "question": problem,
            "solution": answer,
            "gold": gold,
        }


def load_aime2024(split: str = "train", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("HuggingFaceH4/aime_2024", split=split, cache_dir=cache_dir)
    for item in ds:
        problem = item["problem"].strip()
        answer = str(item["answer"]).strip()
        gold = normalize_answer(answer)
        yield {
            "question": problem,
            "solution": answer,
            "gold": gold,
        }


def load_gpqa_diamond(split: str = "test", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("fingertap/GPQA-Diamond", split=split, cache_dir=cache_dir)
    for item in ds:
        question = item["question"].strip()
        answer = item["answer"].strip()
        gold = normalize_answer(answer)
        yield {
            "question": question,
            "solution": answer,
            "gold": gold,
        }


def load_arc_easy(split: str = "test", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split=split, cache_dir=cache_dir)
    for item in ds:
        stem = item["question"].strip()
        choices = item["choices"]
        labels = choices["label"]
        texts = choices["text"]
        label_map = {"1": "a", "2": "b", "3": "c", "4": "d"}

        def map_label(l: str) -> str:
            s = str(l).strip()
            if s in label_map:
                return label_map[s]
            return s.lower()

                     
        formatted_choices = {}
        mapped_order = []
        for label, text in zip(labels, texts):
            mlabel = map_label(label)
            formatted_choices[mlabel] = text.strip()
            mapped_order.append(mlabel)

        ordered_lines = [f"{lab}: {formatted_choices[lab]}" for lab in mapped_order]
        question = stem + "\n" + "\n".join(ordered_lines)

                     
        raw_answer = item.get("answerKey", "").strip()
        mapped_answer = map_label(raw_answer) if raw_answer else ""
        gold = normalize_answer(mapped_answer)
        yield {
            "question": question,
            "solution": mapped_answer,
            "gold": gold,
        }


def load_arc_challenge(split: str = "test", cache_dir: Optional[str] = None) -> Iterable[Dict]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split, cache_dir=cache_dir)
    for item in ds:
        stem = item["question"].strip()
        choices = item["choices"]
        labels = choices["label"]
        texts = choices["text"]
        label_map = {"1": "a", "2": "b", "3": "c", "4": "d"}

        def map_label(l: str) -> str:
            s = str(l).strip()
            if s in label_map:
                return label_map[s]
            return s.lower()

        formatted_choices = {}
        mapped_order = []
        for label, text in zip(labels, texts):
            mlabel = map_label(label)
            formatted_choices[mlabel] = text.strip()
            mapped_order.append(mlabel)

        ordered_lines = [f"{lab}: {formatted_choices[lab]}" for lab in mapped_order]
        question = stem + "\n" + "\n".join(ordered_lines)

        raw_answer = item.get("answerKey", "").strip()
        mapped_answer = map_label(raw_answer) if raw_answer else ""
        gold = normalize_answer(mapped_answer)
        yield {
            "question": question,
            "solution": mapped_answer,
            "gold": gold,
        }


def load_winogrande(
    split: str = "validation",
    subset: str = "winogrande_debiased",
    cache_dir: Optional[str] = None,
) -> Iterable[Dict]:
    ds = load_dataset("allenai/winogrande", subset, split=split, cache_dir=cache_dir)
    for item in ds:
        ask_str = 'Pickout proper choice that fits the _ in the following sentence:'
        sentence = item["sentence"].strip()
        option1 = str(item["option1"]).strip()
        option2 = str(item["option2"]).strip()
        question = f"{ask_str}\n{sentence}\n1: {option1}\n2: {option2}"
        answer = str(item["answer"])
        gold = normalize_answer(answer)
        yield {
            "question": question,
            "solution": answer,
            "gold": gold,
        }


def load_mbppplus(
    split: str = "test",
    subset: str = None,
    cache_dir: Optional[str] = None,
) -> Iterable[Dict]:
    ds = load_dataset("evalplus/mbppplus", subset, split=split, cache_dir=cache_dir)
    for item in ds:
        question = f"""Please provide a self-contained Python script that solves the following problem in a markdown code block:\n```python\nYOUR_PYTHON_CODE\n```:
{item["prompt"]}
Your answer will be tested on test cases like:
{item["test_list"][0]}
{item["test_list"][1]}
{item["test_list"][2]}
"""

        answer = str(item["test"])
        gold = answer
        yield {
            "question": question,
            "solution": answer,
            "gold": gold,
        }


def load_humanevalplus(
    split: str = "test",
    subset: str = None,
    cache_dir: Optional[str] = None,
) -> Iterable[Dict]:
    ds = load_dataset("evalplus/humanevalplus", subset, split=split, cache_dir=cache_dir)
    for item in ds:
        question = f"""Please provide a self-contained Python script that solves the following problem in a markdown code block:\n```python\nYOUR_PYTHON_CODE\n```:
{item["prompt"]}
"""
        raw_answer = str(item["test"])
        answer = raw_answer.replace('candidate', item['entry_point'])
        answer += f'\n\ncheck({item["entry_point"]})'
        gold = answer
        yield {
            "question": question,
            "solution": answer,
            "gold": gold,
        }


                                                               
from typing import Iterable, Dict, Optional
from datasets import load_dataset

def load_medqa(split=None, subset=None, cache_dir=None):
       
    ds = load_dataset("json", data_files="./data/medqa.json", split="train")

    split_req = (split or "").strip().lower()
    choice_map = {"0": "a", "1": "b", "2": "c", "3": "d"}

    for item in ds:
                                                                
        if split_req and "split" in item:
            item_split = str(item.get("split", "")).strip().lower()
            if item_split and item_split != split_req:
                continue

        query = str(item.get("query", "")).strip()

        options = item.get("options", None)
        formatted_options = ""
        if isinstance(options, list) and len(options) >= 4:
            labels = ["A", "B", "C", "D"]
            lines = []
            for lab, opt in zip(labels, options[:4]):
                lines.append(f"{lab}: {str(opt).strip()}")
            formatted_options = "\n" + "\n".join(lines)

        question = query + formatted_options

        raw_answer = str(item.get("answer", "")).strip()

                                                   
        answer = None
        if re.fullmatch(r"[A-Da-d]", raw_answer):
            answer = raw_answer.lower()
        elif raw_answer in choice_map:
            answer = choice_map[raw_answer]
        elif isinstance(options, list) and options:
                                                                      
            for idx, opt in enumerate(options[:4]):
                if raw_answer and raw_answer in str(opt):
                    answer = choice_map[str(idx)]
                    break

        if answer is None:
                                                      
            maybe = str(item.get("label", "")).strip()
            if re.fullmatch(r"[A-Da-d]", maybe):
                answer = maybe.lower()

        gold = normalize_answer(answer) if answer is not None else None

        yield {
            "question": question,
            "solution": answer if answer is not None else "",
            "gold": gold,
        }
