"""
train_llm_wormhole.py

VW (Vision Wormhole) の Universal Codec を、VLM ではなく純粋な LLM 間の
潜在通信に転用するための学習スクリプト。

設計方針:
  - LatentToUniversalEncoder / UniversalToVisionDecoder はモダリティ非依存な
    Perceiver-Resampler なので無改造でそのまま再利用する
    (methods/vision_latent_mas_codec_new.py からインポート)。
  - 「dummy画像」の代わりに「dummy文章枠 (プレースホルダトークン列)」を使う。
    プレースホルダは新規specialトークンを追加するのではなく、既存の
    よく学習された低情報量トークン (デフォルトでは改行/空白系トークン) を
    再利用する。これにより base_dummy_embed が常に「分布内」の値になる
    (VLM版で白画像をvision encoderに通した出力を使うのと同じ発想)。
  - 送信側の latent rollout (LatentMASのlatent thoughts生成) は
    models.py の ModelWrapper.generate_latent_batch をそのまま使う。
  - 受信側への注入は、プレースホルダ位置のトークン埋め込みを
    base_dummy_embed + gate*delta で上書きし、1回のforwardで
    teacher (実テキスト) との出力分布を近づける (MSE + KL) ように
    Encoder/Decoderのみを学習する (受信側・送信側LLM本体はfreeze)。

2エージェント (--agent_model_names A,B) を主眼に置いているが、
3体以上でも動くように hub-and-spoke のridge alignmentは元のvision版と
同じロジックを流用している。
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from models import ModelWrapper, _past_length

# Encoder/Decoderは無改造で再利用する
from methods.vision_latent_mas_codec_new import (
    LatentToUniversalEncoder,
    UniversalToVisionDecoder,
)


# =============================================================================
# 基本ユーティリティ (train_vision_latent_mas_codec_new.py から流用・簡略化)
# =============================================================================


def _safe_int(x: Any, default: int) -> int:
    try:
        return default if x is None else int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float) -> float:
    try:
        return default if x is None else float(x)
    except Exception:
        return default


def _parse_model_list(raw: str) -> List[str]:
    return [s.strip() for s in (raw or "").split(",") if s.strip()]


def _sanitize_logits(logits: torch.Tensor, clip: float) -> torch.Tensor:
    x = torch.nan_to_num(logits.float(), nan=0.0, posinf=1e4, neginf=-1e4)
    if clip > 0:
        x = x.clamp(min=-clip, max=clip)
    return x


def _compute_kl_loss(
    *,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temp: float,
    logit_clip: float,
    topk: int,
) -> torch.Tensor:
    t_logits = _sanitize_logits(teacher_logits, clip=logit_clip)
    s_logits = _sanitize_logits(student_logits, clip=logit_clip)
    if topk > 0 and topk < int(t_logits.shape[-1]):
        idx = torch.topk(t_logits, k=int(topk), dim=-1).indices
        t_logits = torch.gather(t_logits, dim=-1, index=idx)
        s_logits = torch.gather(s_logits, dim=-1, index=idx)
    T = max(1e-6, float(temp))
    log_tgt = F.log_softmax(t_logits / T, dim=-1)
    logp = F.log_softmax(s_logits / T, dim=-1)
    loss = F.kl_div(logp, log_tgt, reduction="batchmean", log_target=True)
    return loss * (T * T)


def _resample_tokens(x: torch.Tensor, target_len: int) -> torch.Tensor:
    """トークン数の不一致を線形補間で吸収する (vision版と同じユーティリティ)。"""
    if target_len <= 0:
        raise ValueError("target_len must be > 0")
    orig_dtype = x.dtype
    xf = x.float()
    if xf.ndim == 2:
        y = F.interpolate(xf.unsqueeze(0).transpose(1, 2), size=target_len, mode="linear", align_corners=False)
        return y.transpose(1, 2)[0].to(dtype=orig_dtype)
    if xf.ndim == 3:
        y = F.interpolate(xf.transpose(1, 2), size=target_len, mode="linear", align_corners=False)
        return y.transpose(1, 2).to(dtype=orig_dtype)
    raise ValueError(f"bad shape: {tuple(x.shape)}")


def _ridge_fit(X: torch.Tensor, Y: torch.Tensor, ridge: float = 1e-3) -> Tuple[torch.Tensor, torch.Tensor]:
    """hub空間への affine alignment を closed-form ridge regression で解く。"""
    Xc = X.detach().float().cpu()
    Yc = Y.detach().float().cpu()
    D = Xc.shape[1]
    Xm = Xc.mean(0, keepdim=True)
    Ym = Yc.mean(0, keepdim=True)
    X0 = Xc - Xm
    Y0 = Yc - Ym
    XtX = X0.T @ X0 + float(ridge) * torch.eye(D, dtype=X0.dtype)
    W = torch.linalg.solve(XtX, X0.T @ Y0)
    b = (Ym - Xm @ W).squeeze(0)
    return W.float(), b.float()


def _infer_hidden_size(wrapper: ModelWrapper) -> int:
    cfg = getattr(wrapper.model, "config", None)
    h = getattr(cfg, "hidden_size", None) if cfg is not None else None
    if h is None:
        h = int(wrapper.model.get_input_embeddings().weight.shape[1])
    return int(h)


def _get_hidden_states_tuple(out: Any) -> Optional[Tuple[torch.Tensor, ...]]:
    return getattr(out, "hidden_states", None)


# =============================================================================
# ここが VLM 版との本質的な違い: dummy「文章」枠のユーティリティ
# =============================================================================


# プレースホルダとして再利用する既存トークン。新規specialトークンを追加して
# embeddingを再初期化するより、既に十分学習済みの低情報量トークンを
# 流用するほうが base_dummy_embed が分布内に収まりやすい。
_DEFAULT_PLACEHOLDER_STRINGS = ["\n", " ", ".", "..."]


def _pick_placeholder_token_id(tokenizer) -> int:
    """既存語彙の中から、単一トークンとして安定に解釈できる低情報量トークンを選ぶ。"""
    for s in _DEFAULT_PLACEHOLDER_STRINGS:
        ids = tokenizer(s, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            return int(ids[0])
    # フォールバック: pad/eosトークンを流用 (これも学習済みで分布内)
    if tokenizer.pad_token_id is not None:
        return int(tokenizer.pad_token_id)
    return int(tokenizer.eos_token_id)


@dataclass
class WormholeSlot:
    """1エージェント分のプレースホルダ設定。"""

    token_id: int
    n_tokens: int


def _build_dummy_text_ids(slot: WormholeSlot) -> torch.Tensor:
    """プレースホルダトークンを n_tokens 個並べた id 列を返す。

    VLM版の `_make_dummy_image` に相当。画像は実際にモデルに通す必要が
    あったが、テキストの場合は embedding lookup だけで済むので
    forward pass すら不要 (`_extract_dummy_text_tokens` 参照)。
    """
    return torch.full((slot.n_tokens,), int(slot.token_id), dtype=torch.long)


@torch.no_grad()
def _extract_dummy_text_tokens(wrapper: ModelWrapper, slot: WormholeSlot) -> torch.Tensor:
    """base_dummy_embed を取得する。

    VLM版 `_extract_dummy_image_tokens` は「白画像をvision encoderに通す」
    必要があったが、テキストの場合はトークン埋め込み層の lookup だけで
    「本来このプレースホルダが持つはずの埋め込み」が手に入る。
    """
    ids = _build_dummy_text_ids(slot).to(wrapper.device)
    emb = wrapper.model.get_input_embeddings()
    return emb(ids).detach().float()  # [n_tokens, H]


def _build_receiver_prompt_ids(
    wrapper: ModelWrapper,
    slot: WormholeSlot,
) -> Tuple[List[int], List[int]]:
    """受信側の学習用プロンプトを組み立てる。

    teacher側 (train_vision_latent_mas_codec_new.py の _train_one_model と
    同じ発想) と揃えるため、"Message:\\n<latent>\\n\\nAcknowledge." の
    <latent> 部分をプレースホルダのトークン列に置き換える。

    【重要】以前の実装は「プレースホルダ文字列をn_tokens回リピートしてから
    丸ごと再トークナイズし、同一トークンIDの連続runを検出する」方式だった。
    しかしBPEトークナイザは同じ文字(改行・空白等)が連続すると複数文字を
    まとめて1トークンにマージすることがあるため、「token_idがn_tokens回
    連続する」という前提が崩れ、run検出に失敗するケースがあった
    (実際にQwen3-1.7Bで発生)。

    このため、prefix/suffixを別々にトークナイズしたうえで、プレースホルダの
    token_idを「文字列として作ってから再トークナイズ」せず「idのリストとして
    直接spliceする」方式に変更した。これにより再トークナイズによる
    マージが原理的に起こり得ず、位置は検出ではなく構築時点で確定する。
    """
    messages_prefix = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Message:\n"},
    ]
    tpl = getattr(wrapper.tokenizer, "chat_template", None)
    if tpl:
        prefix_text = wrapper.tokenizer.apply_chat_template(
            messages_prefix, tokenize=False, add_generation_prompt=False
        )
    else:
        prefix_text = "<|system|>\nYou are a helpful assistant.\n</|system|>\n<|user|>\nMessage:\n"

    # suffix (= "\n\nAcknowledge." + chat templateのgeneration prompt部分) を、
    # 「generation_prompt込み全文」と「generation_prompt無し全文」の差分として
    # 安全に取り出す (文字列の手動編集をしないので破損しにくい)。
    messages_full = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Message:\n\n\nAcknowledge."},
    ]
    if tpl:
        full_no_gen = wrapper.tokenizer.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=False
        )
        full_with_gen = wrapper.tokenizer.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=True
        )
        gen_prompt_suffix = full_with_gen[len(full_no_gen):]
    else:
        gen_prompt_suffix = "\n<|assistant|>"
    suffix_text = "\n\nAcknowledge." + gen_prompt_suffix

    prefix_ids = wrapper.tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
    suffix_ids = wrapper.tokenizer(suffix_text, add_special_tokens=False)["input_ids"]
    placeholder_ids = [int(slot.token_id)] * int(slot.n_tokens)

    combined = list(prefix_ids) + placeholder_ids + list(suffix_ids)
    positions = list(range(len(prefix_ids), len(prefix_ids) + len(placeholder_ids)))
    return combined, positions


def _build_teacher_messages(text: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Message:\n{text}\n\nAcknowledge."},
    ]


# =============================================================================
# 学習本体 (2エージェント/N エージェント共通)
# =============================================================================


def _train_one_model(
    *,
    wrapper: ModelWrapper,
    slot: WormholeSlot,
    anchor_texts: List[str],
    cfg: argparse.Namespace,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], int]:
    device = wrapper.device
    H = _infer_hidden_size(wrapper)

    wrapper.model.eval()
    for p in wrapper.model.parameters():
        p.requires_grad_(False)

    enc = LatentToUniversalEncoder(
        h_in=H,
        d_univ=int(cfg.codec_dim),
        k_univ=int(cfg.codec_tokens),
        n_heads=int(cfg.codec_heads),
        n_layers=int(cfg.codec_layers),
        dropout=float(cfg.codec_dropout),
    ).to(device=device, dtype=torch.float32)

    dec = UniversalToVisionDecoder(
        d_univ=int(cfg.codec_dim),
        h_out=H,
        k_img=int(cfg.codec_slot_tokens),
        n_heads=int(cfg.codec_heads),
        n_layers=int(cfg.codec_layers),
        dropout=float(cfg.codec_dropout),
        gate_init_bias=float(cfg.codec_gate_init_bias),
    ).to(device=device, dtype=torch.float32)

    enc.train()
    dec.train()
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=float(cfg.lr))

    dummy_tokens = _extract_dummy_text_tokens(wrapper, slot)  # [n_tokens, H]
    dummy_rms = dummy_tokens.pow(2).mean().sqrt().clamp_min(1e-6)

    emb_layer = wrapper.model.get_input_embeddings()
    recv_ids, recv_pos = _build_receiver_prompt_ids(wrapper, slot)
    recv_ids_t = torch.tensor(recv_ids, dtype=torch.long, device=device)

    pbar = tqdm(range(int(cfg.steps)), desc=f"[llm-wormhole] {wrapper.model_name}")
    for step in pbar:
        batch_texts = random.sample(anchor_texts, k=min(int(cfg.batch_size), len(anchor_texts)))
        B = len(batch_texts)
        if B == 0:
            continue

        # --- teacher: 実テキストをそのまま渡した場合の最終トークン隠れ状態/logits ---
        teacher_msgs = [_build_teacher_messages(t) for t in batch_texts]
        _, teacher_ids, teacher_mask, _ = wrapper.prepare_chat_batch(teacher_msgs, add_generation_prompt=True)
        teacher_ids = teacher_ids.to(device)
        teacher_mask = teacher_mask.to(device)

        with torch.no_grad():
            out_t = wrapper.model(
                input_ids=teacher_ids,
                attention_mask=teacher_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            hs_t = _get_hidden_states_tuple(out_t)
            last_pos = (teacher_mask.sum(dim=1) - 1).long()
            teacher_h = hs_t[-1][torch.arange(B, device=device), last_pos, :]
            teacher_logits = out_t.logits[torch.arange(B, device=device), last_pos, :]

        # --- 送信側の latent rollout (LatentMASそのもの) ---
        _, lat = wrapper.generate_latent_batch(
            teacher_ids,
            attention_mask=teacher_mask,
            latent_steps=int(cfg.latent_steps),
            return_latent_embeds=True,
        )
        lat = torch.nan_to_num(lat.detach().float(), nan=0.0, posinf=1e4, neginf=-1e4)
        if cfg.latent_clip > 0:
            lat = lat.clamp(min=-cfg.latent_clip, max=cfg.latent_clip)

        # --- Encoder -> Decoder ---
        U = torch.nan_to_num(enc(lat), nan=0.0, posinf=1e4, neginf=-1e4)
        delta, gate = dec(U)
        delta = torch.nan_to_num(delta, nan=0.0, posinf=1e4, neginf=-1e4)
        inj = gate * delta  # [B, n_tokens, H]
        if cfg.inj_clip > 0:
            inj = inj.clamp(min=-cfg.inj_clip, max=cfg.inj_clip)

        # --- 受信側: dummy文章枠に注入して1回のforward ---
        recv_ids_b = recv_ids_t.unsqueeze(0).expand(B, -1)
        base_embeds = emb_layer(recv_ids_b).detach()
        inputs_embeds = base_embeds.clone()
        add = _resample_tokens(inj, len(recv_pos))
        inputs_embeds[:, recv_pos, :] = base_embeds[:, recv_pos, :] + add.to(base_embeds.dtype)
        attn_s = torch.ones_like(recv_ids_b, dtype=teacher_mask.dtype)

        out_s = wrapper.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_s,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hs_s = _get_hidden_states_tuple(out_s)
        student_h = hs_s[-1][:, -1, :]
        student_logits = out_s.logits[:, -1, :]

        teacher_h = torch.nan_to_num(teacher_h.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        student_h = torch.nan_to_num(student_h.float(), nan=0.0, posinf=1e4, neginf=-1e4)

        # --- loss: MSE + KL + stats (VLM版と同じ3項構成) ---
        loss = torch.zeros((), device=device)
        loss_mse = F.mse_loss(student_h, teacher_h)
        loss = loss + cfg.loss_mse * loss_mse

        loss_kl = _compute_kl_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            temp=float(cfg.kl_temp),
            logit_clip=float(cfg.kl_logit_clip),
            topk=int(cfg.kl_topk),
        )
        loss = loss + cfg.loss_kl * loss_kl

        inj_rms = inj.float().pow(2).mean().sqrt().clamp_min(1e-6)
        loss_stats = F.mse_loss(inj_rms, dummy_rms.expand_as(inj_rms))
        loss = loss + cfg.loss_stats * loss_stats

        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(enc.parameters()) + list(dec.parameters()), max_norm=1.0, error_if_nonfinite=False
        )
        opt.step()

        if (step + 1) % max(1, int(cfg.log_every)) == 0:
            pbar.set_postfix(
                {
                    "loss": f"{float(loss.detach()):.4f}",
                    "mse": f"{float(loss_mse.detach()):.4f}",
                    "kl": f"{float(loss_kl.detach()):.4f}",
                    "gate": f"{float(gate.mean().detach()):.3f}",
                }
            )

    enc_sd = {k: v.detach().float().cpu() for k, v in enc.state_dict().items()}
    dec_sd = {k: v.detach().float().cpu() for k, v in dec.state_dict().items()}
    return enc_sd, dec_sd, int(dummy_tokens.shape[0])


@torch.no_grad()
def _collect_anchor_U(
    *, wrapper: ModelWrapper, enc_sd: Dict[str, torch.Tensor], anchor_texts: List[str], cfg: argparse.Namespace
) -> torch.Tensor:
    """hub空間へのridge alignmentのため、各モデルのU出力をanchor text集合で収集する。"""
    device = wrapper.device
    H = _infer_hidden_size(wrapper)
    enc = LatentToUniversalEncoder(
        h_in=H,
        d_univ=int(cfg.codec_dim),
        k_univ=int(cfg.codec_tokens),
        n_heads=int(cfg.codec_heads),
        n_layers=int(cfg.codec_layers),
        dropout=float(cfg.codec_dropout),
    ).to(device=device, dtype=torch.float32)
    enc.load_state_dict(enc_sd, strict=True)
    enc.eval()

    out_list = []
    bs = max(1, min(4, int(cfg.batch_size)))
    for s in tqdm(range(0, len(anchor_texts), bs), desc=f"[align-U] {wrapper.model_name}", leave=False):
        texts = anchor_texts[s : s + bs]
        msgs = [_build_teacher_messages(t) for t in texts]
        _, ids, mask, _ = wrapper.prepare_chat_batch(msgs, add_generation_prompt=True)
        ids, mask = ids.to(device), mask.to(device)
        _, lat = wrapper.generate_latent_batch(
            ids, attention_mask=mask, latent_steps=int(cfg.latent_steps), return_latent_embeds=True
        )
        U = enc(lat.detach().float()).detach().float().cpu()
        out_list.append(U)
    return torch.cat(out_list, dim=0) if out_list else torch.empty(
        (0, int(cfg.codec_tokens) + 2, int(cfg.codec_dim)), dtype=torch.float32
    )


def _load_anchor_texts(path: str) -> List[str]:
    default = [
        "Summarize the following in one sentence: The mitochondrion is the powerhouse of the cell.",
        "Give a step-by-step plan to solve a two-digit multiplication problem.",
        "Explain what 'gradient descent' is in simple terms.",
        "List three potential failure modes in multi-agent reasoning systems.",
        "State the Pythagorean theorem and one practical use-case.",
        "You are given: A=17, B=5. Compute A*B and show your reasoning.",
        "Define Bayes' rule and describe one intuition for it.",
        "Explain what an embedding is in machine learning.",
    ]
    if not path or not os.path.exists(path):
        return default
    if path.endswith(".jsonl"):
        out: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, str):
                    out.append(obj)
                elif isinstance(obj, dict):
                    for k in ("text", "prompt", "message"):
                        if k in obj:
                            out.append(obj[k])
                            break
        return out if len(out) >= 4 else default
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, list) and len(obj) >= 4 else default


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--agent_model_names", type=str, required=True, help="2つ以上のモデル名をカンマ区切りで")
    p.add_argument("--codec_path", type=str, required=True)
    p.add_argument("--anchor_texts_path", type=str, default="")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--latent_steps", type=int, default=32)
    p.add_argument(
        "--latent_space_realign",
        type=int,
        default=1,
        help=(
            "1 (推奨): LatentMASのInput-Output Alignment (W_a) を有効にし、"
            "latent thoughtsを有効な入力埋め込み空間に再射影する。"
            "0: 恒等写像 (再射影なし、LatentMAS本来のオプションと同じ挙動)。"
        ),
    )
    p.add_argument("--codec_dim", type=int, default=256)
    p.add_argument("--codec_tokens", type=int, default=64)
    p.add_argument("--codec_slot_tokens", type=int, default=32, help="dummy文章枠のトークン数 (VLM版のcodec_img_tokensに相当)")
    p.add_argument("--codec_heads", type=int, default=8)
    p.add_argument("--codec_layers", type=int, default=4)
    p.add_argument("--codec_dropout", type=float, default=0.0)
    p.add_argument("--codec_gate_init_bias", type=float, default=-4.0)

    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--loss_mse", type=float, default=1.0)
    p.add_argument("--loss_kl", type=float, default=0.25)
    p.add_argument("--loss_stats", type=float, default=0.1)
    p.add_argument("--kl_temp", type=float, default=1.0)
    p.add_argument("--kl_logit_clip", type=float, default=80.0)
    p.add_argument("--kl_topk", type=int, default=0)
    p.add_argument("--latent_clip", type=float, default=50.0)
    p.add_argument("--inj_clip", type=float, default=20.0)
    p.add_argument("--log_every", type=int, default=10)

    p.add_argument("--ref_idx", type=int, default=0, help="hub空間として使う参照モデルのindex")
    p.add_argument("--ridge", type=float, default=1e-3)

    args = p.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_names = _parse_model_list(args.agent_model_names)
    if len(model_names) < 2:
        raise ValueError("--agent_model_names には2つ以上のモデルを指定してください")

    anchor_texts = _load_anchor_texts(args.anchor_texts_path)
    print(f"Loaded {len(anchor_texts)} anchor texts.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    encoders: Dict[str, Dict[str, torch.Tensor]] = {}
    decoders: Dict[str, Dict[str, torch.Tensor]] = {}
    slot_by_name: Dict[str, WormholeSlot] = {}

    for name in model_names:
        print(f"\n===== Training wormhole codec for {name} =====")
        wrapper = ModelWrapper(name, device, use_vllm=False, args=args)
        token_id = _pick_placeholder_token_id(wrapper.tokenizer)
        slot = WormholeSlot(token_id=token_id, n_tokens=int(args.codec_slot_tokens))
        slot_by_name[name] = slot

        enc_sd, dec_sd, n_tok = _train_one_model(wrapper=wrapper, slot=slot, anchor_texts=anchor_texts, cfg=args)
        encoders[name] = enc_sd
        decoders[name] = dec_sd
        print(f"[done] {name}: placeholder_token_id={token_id} slot_tokens={n_tok}")

        del wrapper
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- hub空間へのridge alignment (2体なら片方をrefにするだけで十分だが、
    #     N体への拡張性のため元のvision版と同じロジックを流用) ---
    ref_idx = max(0, min(int(args.ref_idx), len(model_names) - 1))
    ref_name = model_names[ref_idx]

    U_by_name: Dict[str, torch.Tensor] = {}
    for name in model_names:
        wrapper = ModelWrapper(name, device, use_vllm=False, args=args)
        U_by_name[name] = _collect_anchor_U(wrapper=wrapper, enc_sd=encoders[name], anchor_texts=anchor_texts, cfg=args)
        del wrapper
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    D = int(args.codec_dim)
    X_ref = U_by_name[ref_name].reshape(-1, D)
    out_map, in_map = {}, {}
    for name, U in U_by_name.items():
        X = U.reshape(-1, D)
        W_out, b_out = _ridge_fit(X, X_ref, ridge=float(args.ridge))
        W_in, b_in = _ridge_fit(X_ref, X, ridge=float(args.ridge))
        out_map[name] = {"W": W_out, "b": b_out}
        in_map[name] = {"W": W_in, "b": b_in}

    ckpt = {
        "version": 1,
        "models": model_names,
        "ref_model_name": ref_name,
        "config": {
            "codec_dim": int(args.codec_dim),
            "codec_tokens": int(args.codec_tokens),
            "codec_slot_tokens": int(args.codec_slot_tokens),
            "codec_heads": int(args.codec_heads),
            "codec_layers": int(args.codec_layers),
            "codec_dropout": float(args.codec_dropout),
            "codec_gate_init_bias": float(args.codec_gate_init_bias),
        },
        "encoders": encoders,
        "decoders": decoders,
        "align": {"ref_idx": ref_idx, "ref_model_name": ref_name, "out": out_map, "in": in_map},
        "placeholder": {
            name: {"token_id": slot_by_name[name].token_id, "n_tokens": slot_by_name[name].n_tokens}
            for name in model_names
        },
    }
    os.makedirs(os.path.dirname(args.codec_path) or ".", exist_ok=True)
    torch.save(ckpt, args.codec_path)
    print(f"\nSaved LLM wormhole codec checkpoint to: {args.codec_path}")


if __name__ == "__main__":
    main()