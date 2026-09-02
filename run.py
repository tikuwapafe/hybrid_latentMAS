import argparse
import json
import os
import torch
from typing import Any, Dict, IO, List, Optional, Set, Tuple

from tqdm import tqdm

from data import (
    load_aime2024,
    load_aime2025,
    load_arc_easy,
    load_arc_challenge,
    load_gsm8k,
    load_gpqa_diamond,
    load_mbppplus,
    load_humanevalplus,
    load_medqa
)
# from methods.baseline import BaselineMethod
# from methods.latent_mas import LatentMASMethod
from methods.latent_mas_hybird import LatentMASMethod as HybridLatentMASMethod
from methods import default_agents as method_default_agents
# from methods.procrustes_latent_mas import ProcrustesLatentMASMethod
from methods.text_mas import TextMASMethod
# from methods.text_mas_c2c import TextMASC2CMethod
# from methods.Vision_latent_mas import VisionLatentMASMethod
# from methods.Vision_latent_mas_EX import VisionLatentMASMethodEX
# from methods.vision_latent_mas_proto import VisionLatentMASMethodPROTO
# from methods.vision_latent_mas_codec import VisionLatentMASMethodCodec
from methods.vision_latent_mas_codec_new import VisionLatentMASMethodCODECNew
# from methods.vision_latent_mas_codec_vllm import (
#     VisionLatentMASMethodCODECVLLM,
#     VLLMMultimodalWrapper,
# )
# from methods.vision_latent_mas_codec_sglang import VisionLatentMASMethodCODECSGLang
from methods.vision_latent_mas_ocr import VisionLatentMASMethodOCR
from infer_llm_wormhole import LLMWormhole2Agent
from models import ModelWrapper, _past_length
from utils import auto_device, extract_answer_with_meta, extract_markdown_python_block, normalize_answer, run_with_timeout, set_seed
import time


def evaluate(preds: List[Dict]) -> Tuple[float, int]:
    total = len(preds)
    correct = sum(1 for p in preds if p.get("correct", False))
    acc = correct / total if total > 0 else 0.0
    return acc, correct


def summarize_extraction(preds: List[Dict]) -> Dict:
    total = len(preds)
    unparsable = sum(1 for p in preds if p.get("extraction_source") == "unparsable")
    generated_tokens = [p.get("generated_tokens") for p in preds if p.get("generated_tokens") is not None]
    avg_generated = sum(generated_tokens) / len(generated_tokens) if generated_tokens else None
    stop_counts: Dict[str, int] = {}
    for p in preds:
        reason = p.get("stop_reason")
        if not reason:
            continue
        stop_counts[reason] = stop_counts.get(reason, 0) + 1
    return {
        "unparsable_rate": (unparsable / total) if total > 0 else 0.0,
        "avg_generated_tokens": avg_generated,
        "stop_reason_counts": stop_counts,
    }


def _parse_model_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


class _CodecNewWrapper(ModelWrapper):
    """
    Compatibility wrapper for the codec_new method.

    Notes:
    - The updated codec_new implementation no longer *requires* this wrapper, but keeping it
      preserves backwards compatibility with earlier experiments.
    """
    def prepare_chat_batch(
        self,
        batch_messages: List[List[Dict]],
        add_generation_prompt: bool = True,
    ) -> Tuple[List[str], torch.Tensor, torch.Tensor, List[List[str]]]:
        # Keep ModelWrapper contract: callers in codec_new unpack 4 return values.
        return super().prepare_chat_batch(
            batch_messages, add_generation_prompt=add_generation_prompt
        )

    def generate_latent_batch(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        *,
        inputs_embeds: Optional[torch.Tensor] = None,
        latent_steps: int,
        past_key_values: Optional[Tuple] = None,
        return_latent_embeds: bool = False,
        **kwargs,
    ) -> Tuple:
        # Ignore sampling kwargs like temperature/top_p/do_sample if present.
        if inputs_embeds is not None:
            if attention_mask is None:
                attention_mask = torch.ones(
                    inputs_embeds.shape[:2],
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
            forward_kwargs = dict(kwargs)
            forward_kwargs.pop("pixel_values", None)
            forward_kwargs.pop("input_ids", None)
            forward_kwargs.pop("inputs_embeds", None)
            forward_kwargs.pop("attention_mask", None)

            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
                **forward_kwargs,
            )
            past = outputs.past_key_values
            last_hidden = outputs.hidden_states[-1][:, -1, :]

            latents: List[torch.Tensor] = []
            for _ in range(int(latent_steps)):
                latent_vec = self._apply_latent_realignment(last_hidden, self.model)
                latents.append(latent_vec.detach())

                latent_embed = latent_vec.unsqueeze(1)
                past_len = _past_length(past)
                latent_mask = torch.ones(
                    (latent_embed.shape[0], past_len + 1),
                    dtype=attention_mask.dtype,
                    device=latent_embed.device,
                )
                try:
                    out2 = self.model(
                        inputs_embeds=latent_embed,
                        attention_mask=latent_mask,
                        past_key_values=past,
                        use_cache=True,
                        output_hidden_states=True,
                        return_dict=True,
                        **forward_kwargs,
                    )
                except Exception:
                    out2 = self.model(
                        inputs_embeds=latent_embed,
                        attention_mask=latent_mask,
                        past_key_values=past,
                        use_cache=True,
                        output_hidden_states=True,
                        return_dict=True,
                    )

                past = out2.past_key_values
                last_hidden = out2.hidden_states[-1][:, -1, :]

            if return_latent_embeds:
                if latents:
                    latent_stack = torch.stack(latents, dim=1)
                else:
                    latent_stack = torch.empty(
                        (inputs_embeds.shape[0], 0, inputs_embeds.shape[-1]),
                        device=inputs_embeds.device,
                        dtype=inputs_embeds.dtype,
                    )
                return past, latent_stack
            return past

        return super().generate_latent_batch(
            input_ids,
            attention_mask=attention_mask,
            latent_steps=latent_steps,
            past_key_values=past_key_values,
            return_latent_embeds=return_latent_embeds,
        )


def _wrap_codec_new_results(preds: List[Any], batch: List[Dict], task: str) -> List[Dict]:
    results: List[Dict] = []
    task = (task or "").lower()

    for pred_text, item in zip(preds, batch):
        final_text = "" if pred_text is None else str(pred_text)

        if task in ["mbppplus", "humanevalplus"]:
            pred = extract_markdown_python_block(final_text)
            gold = item.get("gold", "")
            if pred is None:
                ok = False
                error_msg = "python error: No python code block found"
            else:
                python_code_to_exe = pred + "\n" + gold
                ok, error_msg = run_with_timeout(python_code_to_exe, timeout=10)
            extraction_meta = {"extraction_source": None, "extracted_span": None}
        else:
            pred_raw, extraction_meta = extract_answer_with_meta(final_text, task)
            pred = normalize_answer(pred_raw)
            gold = str(item.get("gold", "")).strip()
            error_msg = None

            if task in ["aime2024", "aime2025"]:
                try:
                    pred_val = float(pred) if pred is not None else None
                    pred_int = int(pred_val) if pred_val is not None else None
                    gold_int = int(gold)
                    ok = pred_int is not None and abs(pred_val - pred_int) < 1e-6 and pred_int == gold_int
                except (ValueError, TypeError):
                    ok = False
                    error_msg = f"Value error in parsing answer. Pred: {pred}, Gold: {gold}"
            else:
                ok = (pred == gold) if (pred and gold) else False

        results.append(
            {
                "question": item.get("question", ""),
                "gold": gold if task not in ["mbppplus", "humanevalplus"] else item.get("gold", ""),
                "solution": item.get("solution", ""),
                "prediction": pred if task not in ["mbppplus", "humanevalplus"] else pred,
                "raw_prediction": final_text,
                "extraction_source": extraction_meta.get("extraction_source") if task not in ["mbppplus", "humanevalplus"] else None,
                "extracted_span": extraction_meta.get("extracted_span") if task not in ["mbppplus", "humanevalplus"] else None,
                "stop_reason": "unknown",
                "stop_span": None,
                "generated_tokens": None,
                "agents": [],
                "correct": ok,
                "error": error_msg,
            }
        )
    return results


def _load_role_model_map(raw: str) -> Dict[str, int]:
    if not raw:
        return {}
    payload = raw
    if os.path.exists(raw):
        with open(raw, "r", encoding="utf-8") as f:
            payload = f.read()
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("role_model_map must be a JSON object mapping role -> model index.")
    return {str(k).lower(): int(v) for k, v in data.items()}


def _build_hybrid_agent_models(
    model_pool: List[str],
    role_model_map: Optional[Dict[str, int]] = None,
) -> List[str]:
    if not model_pool:
        raise ValueError("latent_mas_hybrid requires at least one model name.")
    role_model_map = role_model_map or {}
    assigned: List[str] = []
    num_models = len(model_pool)
    for idx, agent in enumerate(method_default_agents()):
        role = str(agent.role).lower()
        model_idx = role_model_map.get(role, idx % num_models)
        if model_idx < 0 or model_idx >= num_models:
            raise ValueError(
                f"role_model_map index {model_idx} out of range for role {role} "
                f"(available model indices: 0..{num_models - 1})."
            )
        assigned.append(model_pool[model_idx])
    return assigned


def _write_jsonl_line(fh: IO[str], obj: Dict[str, Any]) -> None:
    fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
    fh.flush()


def _in_partition(problem_idx: int, num_partitions: int, partition_id: int) -> bool:
    if num_partitions <= 1:
        return True
    return ((problem_idx - 1) % num_partitions) == partition_id


def _partition_target_count(max_samples: int, num_partitions: int, partition_id: int) -> int:
    if max_samples <= 0:
        return 0
    if num_partitions <= 1:
        return max_samples
    start = partition_id + 1
    if start > max_samples:
        return 0
    return ((max_samples - start) // num_partitions) + 1


def _load_resume_preds_jsonl(
    path: str,
    max_samples: int,
    num_partitions: int = 1,
    partition_id: int = 0,
) -> Tuple[List[Dict], Set[int]]:
    """
    Load existing JSONL predictions for resume.

    Returns:
      - preds list in ascending problem_idx order (compatible with evaluate/summarize)
      - set of completed problem_idx
    """
    if not os.path.exists(path):
        return [], set()

    by_idx: Dict[int, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            idx = row.get("problem_idx")
            if not isinstance(idx, int) or idx <= 0:
                continue
            if max_samples > 0 and idx > max_samples:
                continue
            if not _in_partition(idx, num_partitions=num_partitions, partition_id=partition_id):
                continue
            by_idx[idx] = row  # last occurrence wins

    completed_ids = set(by_idx.keys())
    preds: List[Dict[str, Any]] = []
    for idx in sorted(completed_ids):
        row = by_idx[idx]
        preds.append(
            {
                "question": row.get("question", ""),
                "gold": row.get("gold", ""),
                "solution": "",
                "prediction": row.get("prediction"),
                "raw_prediction": row.get("llm_resp", ""),
                "extraction_source": None,
                "extracted_span": None,
                "stop_reason": row.get("stop_reason"),
                "stop_span": None,
                "generated_tokens": row.get("generated_tokens"),
                "agents": [],
                "correct": bool(row.get("ok", False)),
                "error": row.get("error"),
                "time_sec": row.get("time_sec"),
                "time_mode": row.get("time_mode"),
            }
        )
    return preds, completed_ids


# Main processing function for each batch
def process_batch(
    method,
    batch: List[Tuple[int, Dict]],
    processed: int,
    preds: List[Dict],
    progress,
    max_samples: int,
    args: argparse.Namespace,
    preds_jsonl_fh: Optional[IO[str]] = None,
    completed_problem_ids: Optional[Set[int]] = None,
) -> Tuple[int, List[Dict]]:
    remaining = max_samples - processed
    if remaining <= 0:
        return processed, preds
    current_batch = batch[:remaining]
    current_items = [item for _, item in current_batch]

    infer_t0 = getattr(method, "total_infer_time_sec", None)
    infer_n0 = getattr(method, "total_infer_items", None)
    batch_t0 = time.time()
    if args.method in {
        "vision_latent_mas_codec_new",
        "vision_latent_mas_codec_vllm",
        "vision_latent_mas_codec_sglang",
        "vision_latent_mas_rot",
        "vision_latent_mas_ocr",
        "llm_wormhole",
    }:
        raw_preds = method.run_batch(current_items)
        results = _wrap_codec_new_results(raw_preds, current_items, args.task)
    elif args.method == "latent_mas" and args.use_vllm:
        results = method.run_batch_vllm(current_items)
    else:
        results = method.run_batch(current_items)
    batch_elapsed = time.time() - batch_t0

    infer_per_item_elapsed: Optional[float] = None
    infer_n1 = getattr(method, "total_infer_items", None)
    infer_t1 = getattr(method, "total_infer_time_sec", None)
    if isinstance(infer_t0, (int, float)) and isinstance(infer_t1, (int, float)) and isinstance(infer_n0, (int, float)) and isinstance(infer_n1, (int, float)):
        infer_delta_n = int(infer_n1 - infer_n0)
        infer_delta_t = float(infer_t1 - infer_t0)
        if infer_delta_n > 0 and infer_delta_t >= 0.0:
            infer_per_item_elapsed = infer_delta_t / float(infer_delta_n)

    if len(results) > remaining:
        results = results[:remaining]
    per_item_elapsed = (batch_elapsed / max(1, len(results)))
    for (problem_idx, _), res in zip(current_batch, results):
        # Prefer inference-only timing when method exposes it.
        if infer_per_item_elapsed is not None:
            res["time_sec"] = infer_per_item_elapsed
            res["time_mode"] = "infer_batch_avg"
            res["time_wall_sec"] = per_item_elapsed
            res["time_eval_sec"] = max(0.0, per_item_elapsed - infer_per_item_elapsed)
        elif "time_sec" not in res:
            res["time_sec"] = per_item_elapsed
            res["time_mode"] = "batch_avg"
        else:
            res["time_mode"] = "reported"

        preds.append(res)
        if completed_problem_ids is not None:
            completed_problem_ids.add(problem_idx)
        print(f"\n==================== Problem #{problem_idx} ====================")
        print("Question:")
        print(res.get("question", "").strip())
        agents = res.get("agents", [])
        for a in agents:
            name = a.get("name", "Agent")
            role = a.get("role", "")
            agent_header = f"----- Agent: {name} ({role}) -----"
            print(agent_header)
            agent_input = a.get("input", "").rstrip()
            agent_output = a.get("output", "").rstrip()
            latent_steps = a.get("latent_steps", None)
            print("[To Tokenize]")
            print(agent_input)
            if latent_steps is not None:
                print("[Latent Steps]")
                print(latent_steps)
            print("[Output]")
            print(agent_output)
            print("----------------------------------------------")
        print(f"Result: Pred={res.get('prediction')} | Gold={res.get('gold')} | OK={res.get('correct')}")

        if preds_jsonl_fh is not None:
            _write_jsonl_line(
                preds_jsonl_fh,
                {
                    "problem_idx": problem_idx,
                    "task": args.task,
                    "method": args.method,
                    "question": res.get("question", ""),
                    "llm_resp": res.get("raw_prediction", ""),
                    "prediction": res.get("prediction"),
                    "gold": res.get("gold"),
                    "ok": bool(res.get("correct", False)),
                    "time_sec": float(res.get("time_sec", per_item_elapsed)),
                    "time_mode": res.get("time_mode", "batch_avg"),
                    "time_wall_sec": res.get("time_wall_sec"),
                    "time_eval_sec": res.get("time_eval_sec"),
                    "stop_reason": res.get("stop_reason"),
                    "generated_tokens": res.get("generated_tokens"),
                    "error": res.get("error"),
                },
            )

    processed += len(results)
    if progress is not None:
        progress.update(len(results))
    return processed, preds


def main():
    parser = argparse.ArgumentParser()

    # core args for experiments
    parser.add_argument(
        "--method",
        choices=[
            "baseline",
            "text_mas",
            "text_mas_c2c",
            "latent_mas",
            "latent_mas_hybrid",
            "procrustes_latent_mas",
            "vision_latent_mas",
            "vision_latent_mas_ex",
            "vision_latent_mas_proto",
            "vision_latent_mas_codec",
            "vision_latent_mas_codec_new",
            "vision_latent_mas_codec_vllm",
            "vision_latent_mas_codec_sglang",
            "vision_latent_mas_rot",
            "vision_latent_mas_ocr",
            "llm_wormhole",
        ],
        required=True,
        help="Which multi-agent method to run.",
    )
    parser.add_argument("--model_name", type=str, default="",
                        help="Model name to use for experiments (e.g. 'Qwen/Qwen3-14B').")
    parser.add_argument("--max_samples", type=int, default=-1, help="Number of questions to evaluate; set -1 to use all samples.")
    parser.add_argument("--task", choices=["gsm8k", "aime2024", "aime2025", "gpqa", "arc_easy", "arc_challenge", "mbppplus", 'humanevalplus', 'medqa'], default="gsm8k",
                        help="Dataset/task to evaluate. Controls which loader is used.")
    parser.add_argument("--prompt", type=str, choices=["sequential", "hierarchical"], default="sequential", help="Multi-agent system architecture: 'sequential' or 'hierarchical'.")

    # other args
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--latent_steps", type=int, default=0, help="Number of latent steps for LatentMAS method")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--generate_bs", type=int, default=20, help="Batch size for generation")
    parser.add_argument("--text_mas_context_length", type=int, default=-1, help="TextMAS context length limit")
    parser.add_argument("--think", action="store_true", help="Manually add think token in the prompt for LatentMAS")
    parser.add_argument("--latent_space_realign", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    # vLLM support
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM backend for generation")
    parser.add_argument("--enable_prefix_caching", action="store_true", help="Enable prefix caching in vLLM for latent_mas")
    parser.add_argument("--use_second_HF_model", action="store_true", help="Use a second HF model for latent generation in latent_mas")
    parser.add_argument("--device2", type=str, default="cuda:1")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="How many GPUs vLLM should shard the model across")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.45, help="Target GPU memory utilization for vLLM")
    parser.add_argument("--max_model_len", type=int, default=None, help="Max model length for vLLM")
    # procrustes (cross-family latent) setup
    parser.add_argument(
        "--agent_model_names",
        type=str,
        default="",
        help="Comma-separated list of model names for multi-model methods (text_mas, text_mas_c2c, procrustes_latent_mas, vision_latent_mas, vision_latent_mas_proto, vision_latent_mas_codec).",
    )
    parser.add_argument("--agent_models_json", type=str, default="", help="JSON list (or path) of model names for procrustes_latent_mas.")
    parser.add_argument("--anchors_path", type=str, default="", help="Anchor JSON for procrustes_latent_mas.")
    parser.add_argument("--pivot_agent_idx", type=int, default=0, help="Pivot model index for procrustes_latent_mas.")
    parser.add_argument("--procrustes_reg", type=float, default=1e-5, help="Ridge term added to Procrustes alignment.")
    parser.add_argument(
        "--prefix_postproc",
        type=str,
        default="mean_and_norm_match",
        choices=["none", "norm_match", "mean_and_norm_match"],
        help="Postprocess mapped prefixes: none|norm_match|mean_and_norm_match.",
    )
    parser.add_argument("--run_dir", type=str, default="", help="Output directory for metrics/preds. Defaults to runs/<method>_<task>_<timestamp>.")
    parser.add_argument("--save_preds", action="store_true", help="Save per-sample predictions to preds.json in the run dir.")
    parser.add_argument("--save_preds_jsonl", action="store_true", help="Stream per-sample results to JSONL (one line per question).")
    parser.add_argument("--preds_jsonl_path", type=str, default="", help="Optional explicit JSONL output path. Defaults to <run_dir>/preds.jsonl when --save_preds_jsonl is set.")
    parser.add_argument("--resume_preds_jsonl", type=int, default=1, help="Resume from existing JSONL and append new rows (1/0).")
    parser.add_argument(
        "--num_partitions",
        type=int,
        default=1,
        help="Partition count for sharded runs. Each process handles problem_idx where (idx-1) %% num_partitions == partition_id.",
    )
    parser.add_argument(
        "--partition_id",
        type=int,
        default=0,
        help="0-based partition id in [0, num_partitions-1] for sharded runs.",
    )
    parser.add_argument("--role_model_map", type=str, default="", help="Role-to-model index JSON (string or path) for multi-model text_mas.")
    parser.add_argument("--c2c_adapter_root", type=str, default="", help="Root directory of C2C adapter checkpoints for text_mas_c2c.")
    parser.add_argument("--c2c_align_strategy", type=str, default="first", help="Token alignment strategy for text_mas_c2c (e.g., first/last).")
    parser.add_argument("--agent_devices", type=str, default="", help="Comma-separated device list aligned with agent_model_names.")
    parser.add_argument("--vision_mapping_path", type=str, default="", help="Mapping checkpoint .pt for vision_latent_mas.")
    parser.add_argument("--prefix_len", type=int, default=0, help="Cap total injected latent prefix length (0 = no cap).")

    # vision_latent_mas_ex anchors + mapping options
    parser.add_argument("--vision_anchors_dir", type=str, default="", help="Directory of anchor images for vision_latent_mas_ex.")
    parser.add_argument("--vision_anchors_json", type=str, default="", help="JSON list (or path) of anchor image paths.")
    parser.add_argument("--vision_anchor_dataset", type=str, default="", help="HF dataset name for anchor images.")
    parser.add_argument("--vision_anchor_dataset_config", type=str, default="", help="HF dataset config for anchors.")
    parser.add_argument("--vision_anchor_split", type=str, default="validation", help="HF dataset split for anchors.")
    parser.add_argument("--vision_anchor_image_column", type=str, default="image", help="HF dataset image column name.")
    parser.add_argument("--vision_anchor_streaming", type=int, default=1, help="Use streaming for anchor dataset (1/0).")
    parser.add_argument("--vision_num_anchors", type=int, default=16, help="Number of anchors to sample/use.")
    parser.add_argument("--vision_anchor_seed", type=int, default=42, help="Seed for anchor sampling.")
    parser.add_argument("--vision_anchor_batch_size", type=int, default=8, help="Batch size for anchor token extraction.")
    parser.add_argument("--vision_anchor_cache", type=int, default=1, help="Enable anchor token cache (1/0).")
    parser.add_argument("--vision_anchor_cache_refresh", type=int, default=0, help="Rebuild anchor cache even if present (1/0).")
    parser.add_argument("--vision_anchor_cache_dir", type=str, default="", help="Directory for anchor token caches.")
    parser.add_argument("--vision_dummy_image_size", type=int, default=224, help="Dummy image size for vision span.")
    parser.add_argument("--vision_ot_eps", type=float, default=0.07, help="OT epsilon for vision token alignment.")
    parser.add_argument("--vision_ot_iters", type=int, default=50, help="OT iterations for vision token alignment.")
    parser.add_argument("--vision_map_lr", type=float, default=2e-3, help="Mapping learning rate.")
    parser.add_argument("--vision_map_steps", type=int, default=600, help="Mapping training steps.")
    parser.add_argument("--vision_map_batch", type=int, default=256, help="Mapping training batch size.")
    parser.add_argument("--vision_map_weight_decay", type=float, default=1e-4, help="Mapping weight decay.")
    parser.add_argument("--vision_max_tokens_per_anchor", type=int, default=256, help="Max tokens per anchor.")
    parser.add_argument("--vision_max_pairs_total", type=int, default=8192, help="Max token pairs total.")
    parser.add_argument("--vision_resample_mode", type=str, default="interpolate", help="Resample mode: interpolate|repeat|sample.")
    parser.add_argument("--vision_ablate_zero", action="store_true", help="Ablate vision tokens by zeroing.")
    parser.add_argument("--vision_ablate_shuffle", action="store_true", help="Ablate vision tokens by shuffling.")
    parser.add_argument("--vision_ablate_noise_std", type=float, default=0.0, help="Add noise to vision tokens.")
    parser.add_argument("--vision_debug", type=int, default=1, help="Enable vision_latent_mas_ex debug logs (1/0).")

    # vision_latent_mas_proto protocol + adapter options
    parser.add_argument(
        "--vision_comm_protocol_hint",
        type=str,
        default="The attached image is NOT a photo. It contains TEXT messages from other agents. Read the image as a note and use it as context.",
        help="Protocol hint injected into receiver prompts for vision_latent_mas_proto.",
    )
    parser.add_argument("--vision_comm_msg_max_new_tokens", type=int, default=128, help="Max tokens per agent message for vision_latent_mas_proto.")
    parser.add_argument(
        "--vision_ocr_msg_max_new_tokens",
        type=int,
        default=0,
        help="Per-agent message token budget for vision_latent_mas_ocr (<=0 uses latent_steps, then vision_comm_msg_max_new_tokens).",
    )
    parser.add_argument("--vision_comm_msg_temperature", type=float, default=None, help="Temperature for protocol messages (defaults to --temperature).")
    parser.add_argument("--vision_comm_msg_top_p", type=float, default=None, help="Top-p for protocol messages (defaults to --top_p).")
    parser.add_argument("--vision_comm_max_messages", type=int, default=6, help="Max retained protocol messages.")
    parser.add_argument("--vision_comm_max_chars", type=int, default=1400, help="Max chars to render into protocol image.")
    parser.add_argument("--vision_comm_alpha", type=float, default=1.0, help="Blend alpha for injected protocol tokens.")
    parser.add_argument("--vision_t2v_enabled", type=int, default=1, help="Enable text-to-vision adapter (1/0).")
    parser.add_argument("--vision_t2v_num_samples", type=int, default=128, help="Number of synthetic samples for adapter training.")
    parser.add_argument("--vision_t2v_steps", type=int, default=1500, help="Training steps for text-to-vision adapter.")
    parser.add_argument("--vision_t2v_batch", type=int, default=8, help="Batch size for adapter training.")
    parser.add_argument("--vision_t2v_lr", type=float, default=2e-4, help="Learning rate for adapter training.")
    parser.add_argument("--vision_t2v_weight_decay", type=float, default=1e-4, help="Weight decay for adapter training.")
    parser.add_argument("--vision_t2v_num_queries", type=int, default=64, help="Number of query tokens for adapter output.")
    parser.add_argument("--vision_t2v_anchor_batch", type=int, default=4, help="Batch size for adapter target extraction.")
    parser.add_argument("--vision_t2v_text_image_size", type=int, default=384, help="Rendered text-image size for adapter training.")
    parser.add_argument("--vision_t2v_max_chars_per_line", type=int, default=42, help="Max chars per line for rendered text images.")
    parser.add_argument("--vision_t2v_cache_dir", type=str, default="", help="Cache directory for text-to-vision adapters.")

    # vision_latent_mas_codec options (old codec)
    parser.add_argument("--vision_codec_text_encoder", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Text encoder for codec z space.")
    parser.add_argument("--vision_codec_text_device", type=str, default="cpu", help="Device for codec text encoder.")
    parser.add_argument("--vision_codec_cache_dir", type=str, default="runs/vision_codec_cache", help="Codec cache directory.")
    parser.add_argument("--vision_codec_ckpt", type=str, default="", help="Explicit codec checkpoint path.")
    parser.add_argument("--vision_codec_train", type=int, default=1, help="Train codec if cache is missing (1/0).")
    parser.add_argument("--vision_codec_strength", type=float, default=1.0, help="Injection strength for codec span tokens.")
    parser.add_argument("--vision_codec_agg", type=str, default="mean", help="Message aggregation: mean|concat|latest.")
    parser.add_argument("--vision_codec_stat_imgs", type=int, default=4, help="Number of dummy images for vision stats.")
    parser.add_argument("--vision_codec_anchor_texts_path", type=str, default="", help="Optional anchor texts file path.")
    parser.add_argument("--vision_codec_ridge_lam", type=float, default=1e-3, help="Ridge lambda for latent->z fit.")
    parser.add_argument("--vision_codec_anchor_batch", type=int, default=4, help="Anchor batch size for codec training.")
    parser.add_argument("--vision_codec_k_tokens", type=int, default=32, help="Decoder tokens for codec.")
    parser.add_argument("--vision_codec_layers", type=int, default=2, help="Decoder layers for codec.")
    parser.add_argument("--vision_codec_lr", type=float, default=5e-4, help="Decoder learning rate for codec.")
    parser.add_argument("--vision_codec_steps", type=int, default=200, help="Decoder training steps for codec.")
    parser.add_argument("--vision_codec_bs", type=int, default=2, help="Decoder training batch size for codec.")
    parser.add_argument("--vision_codec_reg_w", type=float, default=0.05, help="Decoder regularization weight.")
    parser.add_argument("--vision_codec_train_mix", type=float, default=1.0, help="Codec training mix strength.")

    # New codec_new-specific args (optional; the method has sane defaults even if these are unset)
    parser.add_argument("--vision_codec_path", type=str, default="", help="Checkpoint path for codec_new (if set, load/save).")
    parser.add_argument("--wormhole_codec_path", type=str, default="", help="Checkpoint path for llm_wormhole (produced by train_llm_wormhole.py).")
    parser.add_argument("--vision_codec_dim", type=int, default=256, help="Universal space dimension D for codec_new.")
    parser.add_argument("--vision_codec_tokens", type=int, default=16, help="Number of universal tokens K for codec_new (excluding extra tokens).")
    parser.add_argument("--vision_codec_img_tokens", type=int, default=256, help="Decoder query tokens for codec_new (K_img).")
    parser.add_argument("--vision_codec_heads", type=int, default=8, help="Number of attention heads in codec_new.")
    parser.add_argument("--vision_codec_dropout", type=float, default=0.0, help="Dropout in codec_new.")
    parser.add_argument("--vision_codec_gate_init_bias", type=float, default=-4.0, help="Gate init bias for codec_new.")
    parser.add_argument(
        "--vision_codec_decode_chunks",
        type=int,
        default=1,
        help="Decode memory in N chunks for codec_new (1 keeps old single-decode behavior).",
    )
    parser.add_argument("--vision_codec_train_steps", type=int, default=200, help="Training steps for codec_new.")
    parser.add_argument("--vision_codec_train_batch_size", type=int, default=8, help="Batch size for codec_new training.")
    parser.add_argument("--vision_codec_train_lr", type=float, default=5e-4, help="Learning rate for codec_new training.")
    parser.add_argument("--vision_codec_loss_mse", type=float, default=1.0, help="Weight for prompt-end MSE loss.")
    parser.add_argument("--vision_codec_loss_kl", type=float, default=0.25, help="Weight for next-token KL loss.")
    parser.add_argument("--vision_codec_loss_cka", type=float, default=0.1, help="Weight for CKA loss.")
    parser.add_argument("--vision_codec_loss_stats", type=float, default=0.1, help="Weight for injected-statistics loss.")
    parser.add_argument("--vision_codec_kl_temp", type=float, default=1.0, help="Temperature for KL distillation.")
    parser.add_argument("--vision_codec_ref_idx", type=int, default=0, help="Reference model index for universal alignment.")
    parser.add_argument("--vision_codec_dummy_image_size", type=int, default=224, help="Dummy image size for codec_new.")
    parser.add_argument("--vision_codec_dummy_image_count", type=int, default=1, help="Dummy image count per sample for codec_new.")
    parser.add_argument(
        "--vision_codec_dummy_image_counts",
        type=str,
        default="",
        help="Comma-separated dummy-image counts per model (aligned with --agent_model_names).",
    )
    parser.add_argument(
        "--vision_codec_dummy_image_sizes",
        type=str,
        default="",
        help="Comma-separated dummy-image sizes per model (aligned with --agent_model_names).",
    )
    parser.add_argument(
        "--vision_codec_dummy_image_spec_json",
        type=str,
        default="",
        help="JSON string/path mapping model_name -> {count,size} for per-model dummy images.",
    )
    parser.add_argument(
        "--vision_codec_check_dummy_img_tokens",
        type=int,
        default=0,
        help="If 1, check whether per-model dummy image token counts match and warn on mismatch.",
    )
    parser.add_argument(
        "--vision_codec_require_dummy_img_tokens_match",
        type=int,
        default=0,
        help="If 1 with --vision_codec_check_dummy_img_tokens=1, fail when dummy image token counts mismatch.",
    )
    parser.add_argument("--vision_codec_dummy_tokens_dir", type=str, default="checkpoints/dummy_img_tokens", help="Directory of per-model dummy_img_tokens (.pt) for codec_vllm.")
    parser.add_argument("--vision_codec_dummy_tokens_json", type=str, default="", help="JSON mapping of model_name->path to dummy_img_tokens (.pt).")
    parser.add_argument("--vision_codec_dummy_tokens", type=str, default="", help="Comma-separated list of dummy_img_tokens paths aligned with agent_model_names.")
    parser.add_argument("--vision_codec_dummy_tokens_map_json", type=str, default="", help="JSON mapping of model_name->path to dummy_img_tokens (.pt) for sglang.")
    parser.add_argument(
        "--vision_hier_comm_mode",
        type=str,
        default="chained",
        choices=["chained", "independent_join"],
        help="Hierarchical communication mode for vision_latent_mas_codec_new: chained sender conditioning or independent senders merged only at judger.",
    )
    parser.add_argument(
        "--vision_hier_join_agg",
        type=str,
        default="concat",
        choices=["concat", "mean"],
        help="How to merge sender memories for judger when --vision_hier_comm_mode=independent_join.",
    )
    parser.add_argument("--sglang_chat_template", type=str, default="")
    parser.add_argument("--sglang_chat_templates", type=str, default="")

    args = parser.parse_args()

    if args.num_partitions < 1:
        raise ValueError("--num_partitions must be >= 1.")
    if args.partition_id < 0 or args.partition_id >= args.num_partitions:
        raise ValueError("--partition_id must satisfy 0 <= partition_id < num_partitions.")

    if args.vision_codec_dummy_tokens_map_json and not getattr(args, "vision_codec_dummy_tokens_map", ""):
        args.vision_codec_dummy_tokens_map = args.vision_codec_dummy_tokens_map_json

    if args.method == "latent_mas" and args.use_vllm:
        args.use_second_HF_model = True
        args.enable_prefix_caching = True

    # codec_new relies on latent steps; default run.py sets latent_steps=0, which breaks.
    if args.method in {"vision_latent_mas_codec_new", "vision_latent_mas_codec_vllm", "vision_latent_mas_codec_sglang"} and (args.latent_steps is None or args.latent_steps <= 0):
        args.latent_steps = 8  # sensible default for codec_new

    set_seed(args.seed)
    device = auto_device(args.device)
    if args.method in {"baseline", "latent_mas"} and not args.model_name:
        raise ValueError("--model_name is required for baseline and latent_mas.")
    if args.method == "latent_mas_hybrid" and not args.model_name and not args.agent_model_names:
        raise ValueError("--model_name or --agent_model_names is required for latent_mas_hybrid.")
    if args.method in {"text_mas", "text_mas_c2c"} and not args.model_name and not args.agent_model_names:
        raise ValueError("--model_name is required for single-model text_mas (or use --agent_model_names).")
    if args.method in {
        "vision_latent_mas_codec_new",
        "vision_latent_mas_codec_vllm",
        "vision_latent_mas_codec_sglang",
        "vision_latent_mas_rot",
        "vision_latent_mas_ocr",
    } and not args.model_name and not args.agent_model_names:
        raise ValueError(
            "--model_name is required for vision_latent_mas_codec_new/vision_latent_mas_codec_vllm/vision_latent_mas_codec_sglang/vision_latent_mas_rot/vision_latent_mas_ocr (or use --agent_model_names)."
        )
    if args.method == "llm_wormhole":
        if not args.agent_model_names:
            raise ValueError(
                "--agent_model_names (2つ以上のモデル名をカンマ区切り) is required for llm_wormhole."
            )
        if len(_parse_model_list(args.agent_model_names)) < 2:
            raise ValueError("llm_wormhole requires at least 2 models in --agent_model_names (sender, receiver).")
        if not args.wormhole_codec_path:
            raise ValueError(
                "--wormhole_codec_path is required for llm_wormhole "
                "(checkpoint produced by train_llm_wormhole.py)."
            )
    model = None
    multi_models = None
    role_model_map = None
    hybrid_agent_models = None
    sglang_model_names: List[str] = []
    sglang_devices: List[str] = []
    if args.method in {
        "text_mas",
        "text_mas_c2c",
        "vision_latent_mas_codec_new",
        "vision_latent_mas_codec_vllm",
        "vision_latent_mas_codec_sglang",
        "vision_latent_mas_rot",
        "vision_latent_mas_ocr",
        "llm_wormhole",
    } and args.agent_model_names:
        model_names = _parse_model_list(args.agent_model_names)
        if not model_names:
            raise ValueError("--agent_model_names provided but no valid model names parsed.")
        role_model_map = _load_role_model_map(args.role_model_map)
        devices = []
        if args.agent_devices:
            devices = _parse_model_list(args.agent_devices)
            if len(devices) == 1:
                devices = devices * len(model_names)
            elif len(devices) != len(model_names):
                raise ValueError("--agent_devices must be a single device or match number of agent_model_names.")
        else:
            devices = [args.device] * len(model_names)

        if args.method == "vision_latent_mas_codec_sglang":
            sglang_model_names = model_names
            sglang_devices = devices
        elif args.method == "vision_latent_mas_codec_vllm":
            multi_models = [
                VLLMMultimodalWrapper(
                    name,
                    args=args,
                    tensor_parallel_size=getattr(args, "tensor_parallel_size", 1),
                    gpu_memory_utilization=getattr(args, "gpu_memory_utilization", 0.90),
                    max_model_len=getattr(args, "max_model_len", None),
                    max_num_seqs=getattr(args, "max_num_seqs", None),
                    enforce_eager=getattr(args, "vllm_enforce_eager", False),
                    trust_remote_code=getattr(args, "trust_remote_code", True),
                    dtype=getattr(args, "dtype", "auto"),
                    disable_log_stats=True,
                )
                for name in model_names
            ]
        else:
            # codec_new/ocr do not use vLLM (needs multimodal processors / hidden states)
            wrapper_cls = _CodecNewWrapper if args.method == "vision_latent_mas_codec_new" else ModelWrapper
            use_vllm = False if args.method in {
                "vision_latent_mas_codec_new",
                "vision_latent_mas_codec_vllm",
                "vision_latent_mas_rot",
                "vision_latent_mas_ocr",
                "text_mas_c2c",
                "llm_wormhole",
            } else args.use_vllm
            multi_models = [
                wrapper_cls(name, torch.device(dev), use_vllm=use_vllm, args=args)
                for name, dev in zip(model_names, devices)
            ]
    elif args.method == "vision_latent_mas_codec_new":
        wrapper_cls = _CodecNewWrapper
        multi_models = [wrapper_cls(args.model_name, device, use_vllm=False, args=args)]
    elif args.method == "vision_latent_mas_codec_vllm":
        multi_models = [
            VLLMMultimodalWrapper(
                args.model_name,
                args=args,
                tensor_parallel_size=getattr(args, "tensor_parallel_size", 1),
                gpu_memory_utilization=getattr(args, "gpu_memory_utilization", 0.90),
                max_model_len=getattr(args, "max_model_len", None),
                max_num_seqs=getattr(args, "max_num_seqs", None),
                enforce_eager=getattr(args, "vllm_enforce_eager", False),
                trust_remote_code=getattr(args, "trust_remote_code", True),
                dtype=getattr(args, "dtype", "auto"),
                disable_log_stats=True,
            )
        ]
    elif args.method == "vision_latent_mas_codec_sglang":
        sglang_model_names = [args.model_name]
        sglang_devices = [args.device]
    elif args.method in {"vision_latent_mas_rot", "vision_latent_mas_ocr"}:
        # OCR method is multimodal; avoid vLLM.
        multi_models = [ModelWrapper(args.model_name, device, use_vllm=False, args=args)]
    elif args.method == "latent_mas_hybrid":
        model_pool = _parse_model_list(args.agent_model_names) if args.agent_model_names else [args.model_name]
        if not model_pool:
            raise ValueError("latent_mas_hybrid could not parse model names from --model_name/--agent_model_names.")
        hybrid_role_model_map = _load_role_model_map(args.role_model_map)
        hybrid_agent_models = _build_hybrid_agent_models(model_pool, hybrid_role_model_map)
        model = ModelWrapper(model_pool[0], device, use_vllm=False, args=args)
    elif args.method not in {"procrustes_latent_mas", "vision_latent_mas", "vision_latent_mas_ex", "vision_latent_mas_proto", "vision_latent_mas_codec", "vision_latent_mas_codec_new"}:
        model = ModelWrapper(args.model_name, device, use_vllm=args.use_vllm, args=args)

    if args.method in {"vision_latent_mas_codec_new", "vision_latent_mas_codec_vllm"} and multi_models:
        for wrapper in multi_models:
            if hasattr(wrapper, "model") and not hasattr(wrapper.model.config, "hidden_size"):
                try:
                    hidden = wrapper.model.get_input_embeddings().weight.shape[1]
                    setattr(wrapper.model.config, "hidden_size", int(hidden))
                except Exception:
                    pass

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or os.path.join("runs", f"{args.method}_{args.task}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    preds_jsonl_path = ""
    if args.save_preds_jsonl:
        preds_jsonl_path = args.preds_jsonl_path or os.path.join(run_dir, "preds.jsonl")
        os.makedirs(os.path.dirname(preds_jsonl_path) or ".", exist_ok=True)

    start_time = time.time()

    common_kwargs = dict(
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # method selection 
    if args.method == "baseline":
        method = BaselineMethod(
            model,
            max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            use_vllm=args.use_vllm,
            args=args
        )
    elif args.method == "text_mas":
        if multi_models:
            method = TextMASMethod(
                model=None,
                models=multi_models,
                role_model_map=role_model_map,
                max_new_tokens_each=args.max_new_tokens,
                **common_kwargs,
                generate_bs=args.generate_bs,
                args=args,
            )
        else:
            method = TextMASMethod(
                model,
                max_new_tokens_each=args.max_new_tokens,
                **common_kwargs,
                generate_bs=args.generate_bs,
                args=args,
            )
    elif args.method == "text_mas_c2c":
        if multi_models:
            method = TextMASC2CMethod(
                model=None,
                models=multi_models,
                role_model_map=role_model_map,
                max_new_tokens_each=args.max_new_tokens,
                **common_kwargs,
                generate_bs=args.generate_bs,
                args=args,
            )
        else:
            method = TextMASC2CMethod(
                model,
                models=None,
                role_model_map=None,
                max_new_tokens_each=args.max_new_tokens,
                **common_kwargs,
                generate_bs=args.generate_bs,
                args=args,
            )
    elif args.method == 'latent_mas':
        method = LatentMASMethod(
            model,
            latent_steps=args.latent_steps,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs, 
            args=args,
        )
    elif args.method == "latent_mas_hybrid":
        if model is None:
            raise ValueError("latent_mas_hybrid failed to initialize base model.")
        method = HybridLatentMASMethod(
            model,
            agent_models=hybrid_agent_models,
            latent_steps=args.latent_steps,
            judger_max_new_tokens=args.max_new_tokens,
            **common_kwargs,
            generate_bs=args.generate_bs,
            args=args,
        )
    elif args.method == "vision_latent_mas_codec_new":
        if not multi_models:
            raise ValueError("vision_latent_mas_codec_new requires at least one model.")
        # Align codec_new's expected "mode" with existing --prompt.
        args.mode = getattr(args, "mode", None) or args.prompt
        method = VisionLatentMASMethodCODECNew(
            args=args,
            models=multi_models,
        )
    elif args.method in {"vision_latent_mas_rot", "vision_latent_mas_ocr"}:
        if not multi_models:
            raise ValueError("vision_latent_mas_rot/vision_latent_mas_ocr requires at least one model.")
        # Keep consistency with other multi-agent methods.
        args.mode = getattr(args, "mode", None) or args.prompt
        method = VisionLatentMASMethodOCR(
            args=args,
            models=multi_models,
        )
    elif args.method == "llm_wormhole":
        if not multi_models or len(multi_models) < 2:
            raise ValueError("llm_wormhole requires at least 2 models via --agent_model_names.")
        method = LLMWormhole2Agent(
            args=args,
            models=multi_models,
        )

    preds: List[Dict] = []
    completed_problem_ids: Set[int] = set()
    processed = 0
    batch: List[Tuple[int, Dict]] = []

    # dataset loading
    if args.task == "gsm8k":
        dataset_iter = load_gsm8k(split=args.split)
    elif args.task == "aime2024":
        dataset_iter = load_aime2024(split="train")
    elif args.task == "aime2025":
        dataset_iter = load_aime2025(split='train')
    elif args.task == "gpqa":
        dataset_iter = load_gpqa_diamond(split='test')
    elif args.task == "arc_easy":
        dataset_iter = load_arc_easy(split='test')
    elif args.task == "arc_challenge":
        dataset_iter = load_arc_challenge(split='test')
    elif args.task == "mbppplus":
        dataset_iter = load_mbppplus(split='test')
    elif args.task == "humanevalplus":
        dataset_iter = load_humanevalplus(split='test')
    elif args.task == "medqa":
        dataset_iter = load_medqa(split='test')
    else:
        raise ValueError(f'no {args.task} support')

    if args.max_samples == -1:
        dataset_iter = list(dataset_iter)  
        args.max_samples = len(dataset_iter)

    max_problem_idx = args.max_samples
    partition_target = _partition_target_count(
        max_samples=max_problem_idx,
        num_partitions=args.num_partitions,
        partition_id=args.partition_id,
    )
    if args.num_partitions > 1:
        print(
            f"[partition] partition_id={args.partition_id}/{args.num_partitions - 1} "
            f"target_samples={partition_target} max_problem_idx={max_problem_idx}"
        )

    if preds_jsonl_path and args.resume_preds_jsonl:
        resumed_preds, resumed_ids = _load_resume_preds_jsonl(
            preds_jsonl_path,
            max_problem_idx,
            num_partitions=args.num_partitions,
            partition_id=args.partition_id,
        )
        preds.extend(resumed_preds)
        completed_problem_ids |= resumed_ids
        processed = len(completed_problem_ids)
        if processed > 0:
            print(
                f"[resume] loaded {processed} completed sample(s) from {preds_jsonl_path}; "
                "will skip and append new rows."
            )

    progress = tqdm(total=partition_target)
    if processed > 0:
        progress.update(processed)

    preds_jsonl_fh: Optional[IO[str]] = None
    if preds_jsonl_path:
        file_mode = "a" if args.resume_preds_jsonl else "w"
        preds_jsonl_fh = open(preds_jsonl_path, file_mode, encoding="utf-8")

    try:
        for data_idx, item in enumerate(dataset_iter, start=1):
            if data_idx > max_problem_idx:
                break
            if processed >= partition_target:
                break
            if not _in_partition(data_idx, args.num_partitions, args.partition_id):
                continue
            if data_idx in completed_problem_ids:
                continue
            batch.append((data_idx, item))
            if len(batch) == args.generate_bs or processed + len(batch) == partition_target:
                processed, preds = process_batch(
                    method,
                    batch,
                    processed,
                    preds,
                    progress,
                    partition_target,
                    args,
                    preds_jsonl_fh=preds_jsonl_fh,
                    completed_problem_ids=completed_problem_ids,
                )
                batch = []
                if processed >= partition_target:
                    break

        if batch and processed < partition_target:
            processed, preds = process_batch(
                method,
                batch,
                processed,
                preds,
                progress,
                max_samples=partition_target,
                args=args,
                preds_jsonl_fh=preds_jsonl_fh,
                completed_problem_ids=completed_problem_ids,
            )
    finally:
        if preds_jsonl_fh is not None:
            preds_jsonl_fh.close()
    progress.close()

    total_time = time.time() - start_time

    acc, correct = evaluate(preds)
    extra = summarize_extraction(preds)

    summary = {
        "method": args.method,
        "model": args.model_name or None,
        "agent_model_names": args.agent_model_names,
        "role_model_map": args.role_model_map,
        "anchors_path": getattr(args, "anchors_path", ""),
        "vision_mapping_path": getattr(args, "vision_mapping_path", ""),
        "pivot_agent_idx": getattr(args, "pivot_agent_idx", None),
        "prefix_len": getattr(args, "prefix_len", None),
        "prefix_postproc": getattr(args, "prefix_postproc", None),
        "split": args.split,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "num_partitions": args.num_partitions,
        "partition_id": args.partition_id,
        "partition_samples": partition_target,
        "processed_samples": processed,
        "accuracy": acc,
        "correct": correct,
        "unparsable_rate": extra["unparsable_rate"],
        "avg_generated_tokens": extra["avg_generated_tokens"],
        "stop_reason_counts": extra["stop_reason_counts"],
        "total_time_sec": round(total_time, 4),
        "time_per_sample_sec": (round(total_time / partition_target, 4) if partition_target > 0 else None),
        "run_dir": run_dir,
        "preds_jsonl_path": preds_jsonl_path or None,
    }

    metrics_path = os.path.join(run_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if args.save_preds:
        preds_path = os.path.join(run_dir, "preds.json")
        with open(preds_path, "w", encoding="utf-8") as f:
            json.dump(preds, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()