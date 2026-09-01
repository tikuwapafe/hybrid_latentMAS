"""
linear_alignment_ablation.py

目的:
    VWのUniversal Visual Codec (Perceiver型encoder/decoder, 蒸留学習が必要) の代わりに、
    LatentMASのW_a (models.py の _build_latent_realign_matrix / _apply_latent_realignment)
    と同じ閉形式ridge回帰だけで
        rollout hidden states (送信元の潜在思考) -> 受信側の継続プロンプトスロット
    への写像 (W_cross) を作り、どこまで性能が出るかを検証するためのスクリプト。

前提:
    - 011リポジトリ (Vision Wormhole) 直下に配置し、`from models import ModelWrapper` で
      既存のモデルロード・rollout生成ロジックをそのまま流用する。
    - target_mode="slot" (デフォルト): テキストのみのLLM同士の異種間通信を想定。
      注入先はVLMのvision-token spanではなく、受信側プロンプト中の
      「継続プロンプトスロット」(固定の中立文 filler_text の埋め込み) に residual 加算する。
    - target_mode="vision" は元のVLM向け設計を残してあるが、get_dummy_image_embedding は
      未実装のまま (011のvision-side実装を見てから埋める)。target_delta の構成も
      vision向けには未対応 (下記の重要な訂正を参照)。

重要な訂正 (以前のバージョンのバグ):
    fit_linear_map の目的変数を、誤って全アンカーで共通の定数 x_baseline にしていた。
    これだと W_cross は「入力(H_rollout)に関わらず同じ出力を返す」方向に最適化されてしまい、
    メッセージ内容の情報がほぼ伝わらない退化した写像になる。
    正しくは、各アンカーごとに異なる目的変数 target_delta_i を使う必要がある:

        target_delta_i = (受信側モデル自身によるtext_iの埋め込み) - x_baseline

    こうすることで、LatentMASのW_a (各語彙トークンごとに異なるペアで回帰する) と
    同じ「content-dependentなペア回帰」という構造になる。
    get_baseline_slot_embedding は任意のテキストを受け取れる汎用関数なので、
    filler_textの代わりにアンカーテキスト自身を渡すことでこの target_delta を構成する。

メモリ設計:
    デフォルトで sequential_loading=True。sender/receiverを同時にGPUへロードせず、
        (1) receiverのみロード -> 基準点構築 + no-opチェック + 各アンカーの内容埋め込み計算
            (埋め込み参照のみで軽量、フォワードパス不要) -> reference_norm計算 -> アンロード
        (2) senderのみロード -> 全アンカーのrollout収集 -> アンロード
        (3) 収集済みテンソルのみでclosed-form fit (モデル不要)
    の順に処理する。単一GPUに2つの大きめモデルの重みを同時に乗せると
    アクティベーション用の余裕がほぼ無くなりOOMしやすいため。

使い方の想定フロー:
    1. (推奨) --check_filler_candidates で filler_text の内容依存性を先に確認する
    2. 本実行: collect_alignment_pairs_sequential() でペア収集
       (--val_fraction > 0 なら一部をhold-outし、再構成診断を先に表示する)
    3. fit_linear_map() で全データを使い最終的な W_cross を閉形式で解く
    4. build_injection() で実際の埋め込みを作り、既存の評価パイプラインに差し込む
"""

from __future__ import annotations

import gc
import json
import random
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F

from models import ModelWrapper  # 011リポジトリ直下に配置する前提


# ---------------------------------------------------------------------------
# 0. filler_text (X_slot構築用の中立文) の選定について
# ---------------------------------------------------------------------------
#
# 制約:
#   1. 評価ベンチマークのドメインと非重複
#      (GSM8K/AIME=数学, GPQA/MedQA=科学・医療, MBPP+/HumanEval+=コード, ARC=常識)
#   2. anchorコーパスと非重複 (VWデフォルト: cos_e=常識QA, OpenCodeReasoning=コード, PRM800K=数学)
#   3. 短いフレーズの反復ではなく、自然な連続文であること (周期パターンを避ける)
#   4. 想定する最大 K をカバーできる十分な長さを1つの連続文で用意し、そこから切り出す
#
# 実測メモ: K=32でこの文を使ったno-opチェックはKL=0.0337 (閾値0.05を通過)。
# 別候補 (灯台守の文) はKL=0.0864で不合格だった。文の途中で不自然に切れている
# 可能性が交絡因子として残っているため、下流精度評価がおかしければここに戻ること。
DEFAULT_FILLER_TEXT = (
    "The region's traditional pottery techniques developed gradually over several "
    "centuries, shaped by the availability of local clay deposits and the seasonal "
    "patterns of nearby rivers. Artisans typically gathered materials in early autumn, "
    "when the riverbanks were driest and the clay easiest to extract. Workshops were "
    "often built near natural windbreaks to protect drying pottery from sudden gusts, "
    "and firing kilns were positioned according to the prevailing wind direction to "
    "manage smoke and temperature evenly across a batch. Later, apprentices learned to "
    "judge firing readiness by the color of the smoke rising from the kiln vents, a "
    "skill passed down through direct observation rather than written instruction."
)


# ---------------------------------------------------------------------------
# 1. データ収集: rollout hidden states と 基準点・目的変数(target_delta) を集める
# ---------------------------------------------------------------------------

@dataclass
class AlignmentPair:
    """1つのアンカーテキストから得られる (rollout, baseline, target_delta) のペア。"""
    h_rollout: torch.Tensor     # shape: [T, d_h_sender]  (送信元モデルの最終層 hidden state rollout)
    x_baseline: torch.Tensor    # shape: [K, d_h_receiver] (注入時の残差ベース; 全ペアで共通の固定値)
    target_delta: torch.Tensor  # shape: [K, d_h_receiver] (= receiverでのtext_i埋め込み - x_baseline;
                                 #        フィット対象。アンカーごとに異なる)


def load_anchor_texts(anchor_path: str, limit: int | None = None) -> list[str]:
    """VWのanchor jsonl (例: data/vision_codec_anchor_text/mixed_cose_ocr_prm800k.jsonl) を読む。"""
    texts = []
    with open(anchor_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # VWのjsonlのキー名に合わせて調整すること (例: "text" / "message" など)
            texts.append(obj.get("text", obj.get("message", "")))
            if limit is not None and len(texts) >= limit:
                break
    return texts


def generate_latent_rollout(sender_wrapper: ModelWrapper, prompt: str, T: int) -> torch.Tensor:
    """
    ModelWrapper.generate_latent_batch をそのまま流用する。

    generate_latent_batch は内部で:
      - プロンプトをforwardして最終層hidden stateを得る
      - _apply_latent_realignment (= LatentMAS W_a + NormMatch相当) を適用
      - それを次ステップの inputs_embeds としてT回autoregressiveに繰り返す
    を行い、return_latent_embeds=True で各ステップの (アライン済み) 潜在ベクトルを
    [1, T, d_h] で返す。ここではW_cross入力用にそのまま使う。
    """
    input_ids = sender_wrapper.tokenize_text(prompt)  # [1, L], add_special_tokens=False
    attention_mask = torch.ones_like(input_ids)
    _, latent_stack = sender_wrapper.generate_latent_batch(
        input_ids,
        attention_mask,
        latent_steps=T,
        return_latent_embeds=True,
    )
    return latent_stack[0]  # [T, d_h_sender] (batch=1を取り出す)


def get_dummy_image_embedding(receiver_wrapper: ModelWrapper, dummy_image_path: str) -> torch.Tensor:
    """
    [VLM向け, target_mode="vision"] 受信側VLMに固定のダミー画像を通し、
    実際のvision projector出力 X̄_img を得る。
    形状: [L_img, d_h_receiver]
    """
    # TODO: 011リポジトリのVLM画像処理パス (AutoProcessor経由の画像埋め込み取得方法) に
    # 合わせて実装する。ModelWrapperには現状これに対応する公開メソッドがないため、
    # モデルごとのvision encoder + projectorへの直接アクセスが必要。
    # さらに vision モードでは target_delta の構成方法 (「テキストをvision空間でどう
    # 表現するか」に相当するもの) も未検討であり、slotモードとは別途設計が必要。
    raise NotImplementedError


def get_text_embedding(
    receiver_wrapper: ModelWrapper,
    text: str,
    length_K: int,
) -> torch.Tensor:
    """
    受信側モデル自身の入力埋め込み層を使って、任意のテキストをK個のトークン分だけ埋め込む。

    2つの用途で使う汎用関数:
      (a) filler_text (中立文) を渡す -> 基準点 x_baseline の構築
      (b) アンカーテキスト自身を渡す -> target_delta 構成用の「受信側での内容表現」

    実在のトークン列の埋め込みなので、内容が何であれ定義上 W_in の行そのもの (on-manifold)。
    埋め込み参照のみでフォワードパスを伴わないため、(b)の用途で全アンカーに対して
    呼んでも計算コストは軽い。

    Returns:
        [K, d_h_receiver]
    """
    text_model = receiver_wrapper._get_text_model()
    embed_layer = text_model.get_input_embeddings()

    ids = receiver_wrapper.tokenizer(
        text, add_special_tokens=False, return_tensors="pt"
    )["input_ids"][0]

    # length_K に満たない場合は text を繰り返して継ぎ足す
    while ids.shape[0] < length_K:
        extra = receiver_wrapper.tokenizer(
            text, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0]
        ids = torch.cat([ids, extra], dim=0)
    ids = ids[:length_K].to(receiver_wrapper.device)

    with torch.no_grad():
        emb = embed_layer(ids)  # [K, d_h_receiver]
    return emb.detach()


# 後方互換のためのエイリアス (以前の名称)
def get_baseline_slot_embedding(receiver_wrapper: ModelWrapper, filler_text: str, slot_length_K: int) -> torch.Tensor:
    return get_text_embedding(receiver_wrapper, filler_text, slot_length_K)


def compute_reference_norm(receiver_wrapper: ModelWrapper) -> float:
    """
    NormMatch (models.py の _build_latent_realign_matrix 内の target_norm と同じ計算) を、
    受信側モデルに対して行う。build_injection の gate や、Δの明示的なノルムクリップに使う。
    """
    text_model = receiver_wrapper._get_text_model()
    input_weight = text_model.get_input_embeddings().weight.detach()
    return input_weight.norm(dim=1).mean().item()


def check_baseline_is_noop(
    receiver_wrapper: ModelWrapper,
    prompt: str,
    x_baseline: torch.Tensor,
    insertion_position: int,
) -> float:
    """
    Δを加える前の必須サニティチェック。
    基準点 x_baseline を insertion_position に挿入した場合 (Δなし) と、
    何も挿入しない通常のプロンプトの場合とで、次トークン分布がどれだけ変わるかを
    KLダイバージェンスで測る。

    目的はKL=0の達成ではなく、「内容による寄与」と「挿入によるトークン数増加
    (位置ズレ)による寄与」を切り分けること。check_filler_content_sensitivity と
    組み合わせて、候補間でKLの大きさが揃っているかを確認すること。

    Returns:
        kl_divergence (float)
    """
    text_model = receiver_wrapper._get_text_model()
    embed_layer = text_model.get_input_embeddings()

    input_ids = receiver_wrapper.tokenize_text(prompt)  # [1, L]

    # teacher: 通常のプロンプト (挿入なし)
    attn_teacher = torch.ones_like(input_ids)
    with torch.no_grad():
        out_teacher = text_model(
            input_ids=input_ids, attention_mask=attn_teacher, use_cache=False, return_dict=True
        )
    logits_teacher = out_teacher.logits[0, -1, :]

    # student: insertion_position に x_baseline (基準点のみ, Δなし) を挿入
    with torch.no_grad():
        token_embeds = embed_layer(input_ids)[0]  # [L, d_h]
    pos = max(0, min(insertion_position, token_embeds.shape[0]))
    x_baseline = x_baseline.to(device=token_embeds.device, dtype=token_embeds.dtype)
    combined = torch.cat([token_embeds[:pos], x_baseline, token_embeds[pos:]], dim=0).unsqueeze(0)
    attn_student = torch.ones((1, combined.shape[1]), dtype=torch.long, device=combined.device)
    with torch.no_grad():
        out_student = text_model(
            inputs_embeds=combined, attention_mask=attn_student, use_cache=False, return_dict=True
        )
    logits_student = out_student.logits[0, -1, :]

    kl = F.kl_div(
        F.log_softmax(logits_student, dim=-1).unsqueeze(0),
        F.softmax(logits_teacher, dim=-1).unsqueeze(0),
        reduction="batchmean",
    )
    return kl.item()


def check_filler_content_sensitivity(
    receiver_wrapper: ModelWrapper,
    prompt: str,
    candidate_filler_texts: list[str],
    slot_length_K: int,
    insertion_position: int,
) -> dict[str, float]:
    """
    複数の中立文候補それぞれについて check_baseline_is_noop 相当のKLを測り、
    候補間でKLの大きさがどれだけ揃っているかを見る。
    """
    results: dict[str, float] = {}
    for filler_text in candidate_filler_texts:
        x_baseline = get_text_embedding(receiver_wrapper, filler_text, length_K=slot_length_K)
        kl = check_baseline_is_noop(
            receiver_wrapper, prompt=prompt, x_baseline=x_baseline, insertion_position=insertion_position,
        )
        results[filler_text[:40] + "..."] = kl
    return results


def _unload_model_wrapper(wrapper: ModelWrapper) -> None:
    """モデル重みをGPUから解放する。sender/receiverを同時にロードしないための補助関数。"""
    for attr in ("model", "HF_model", "vllm_engine"):
        if hasattr(wrapper, attr):
            try:
                delattr(wrapper, attr)
            except AttributeError:
                pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def collect_alignment_pairs_sequential(
    sender_model_name: str,
    receiver_model_name: str,
    sender_device: str,
    receiver_device: str,
    sender_latent_space_realign: bool,
    anchor_texts: list[str],
    latent_steps_T: int,
    target_mode: str = "slot",
    dummy_image_path: str | None = None,
    filler_text: str | None = None,
    insertion_position: int = 0,
    kl_noop_threshold: float = 0.05,
) -> Tuple[list[AlignmentPair], float]:
    """
    [逐次ロード版] sender/receiverを同時にGPUへロードしない。

        (1) receiverのみロード ->
            基準点 x_baseline 構築 + no-opチェック +
            各アンカーの target_delta (= receiverでのtext_i埋め込み - x_baseline) 計算 +
            reference_norm計算 -> アンロード
        (2) senderのみロード -> 全アンカーのrollout収集 -> アンロード

    target_delta の計算は埋め込み参照のみ (フォワードパス不要) なので、
    アンカー数が多くても軽量。

    Returns:
        (pairs, reference_norm)
    """
    if target_mode != "slot":
        raise NotImplementedError(
            "target_delta の構成は現状 target_mode='slot' のみ対応。"
            "'vision' はget_dummy_image_embeddingの実装と合わせて別途設計が必要。"
        )
    assert filler_text is not None

    # --- Phase 1: receiver のみロード ---
    receiver_wrapper = _load_model_wrapper(receiver_model_name, receiver_device, latent_space_realign=False)

    x_baseline = get_text_embedding(receiver_wrapper, filler_text, length_K=latent_steps_T)
    kl = check_baseline_is_noop(
        receiver_wrapper, prompt=anchor_texts[0], x_baseline=x_baseline,
        insertion_position=insertion_position,
    )
    if kl > kl_noop_threshold:
        _unload_model_wrapper(receiver_wrapper)
        raise RuntimeError(
            f"基準点 X_slot の挿入だけでKLダイバージェンスが {kl:.4f} "
            f"(閾値 {kl_noop_threshold}) を超えました。filler_text / slot_length_K / "
            "insertion_position を見直してください。"
        )
    print(f"[sanity check] baseline no-op KL = {kl:.4f} (<= {kl_noop_threshold} でOK)")

    reference_norm = compute_reference_norm(receiver_wrapper)
    x_baseline_cpu = x_baseline.detach().cpu()

    print(f"[target] {len(anchor_texts)}件のアンカーについて、受信側での内容embedding "
          "(target_deltaの元) を計算中 (埋め込み参照のみ)")
    target_deltas_cpu: list[torch.Tensor] = []
    for text in anchor_texts:
        content_emb = get_text_embedding(receiver_wrapper, text, length_K=latent_steps_T)
        target_deltas_cpu.append((content_emb - x_baseline).detach().cpu())

    _unload_model_wrapper(receiver_wrapper)
    del receiver_wrapper
    print("[memory] receiverをアンロードしました。senderをロードします。")

    # --- Phase 2: sender のみロード ---
    sender_wrapper = _load_model_wrapper(sender_model_name, sender_device, sender_latent_space_realign)

    pairs: list[AlignmentPair] = []
    for i, text in enumerate(anchor_texts):
        h_rollout = generate_latent_rollout(sender_wrapper, text, T=latent_steps_T)
        assert h_rollout.shape[0] == x_baseline_cpu.shape[0], (
            f"T ({h_rollout.shape[0]}) must equal target length ({x_baseline_cpu.shape[0]}); "
            "set latent_steps_T accordingly."
        )
        pairs.append(AlignmentPair(
            h_rollout=h_rollout.detach().cpu(),
            x_baseline=x_baseline_cpu,
            target_delta=target_deltas_cpu[i],
        ))
        if (i + 1) % 50 == 0:
            print(f"[rollout] {i + 1}/{len(anchor_texts)} 件処理済み")

    _unload_model_wrapper(sender_wrapper)
    del sender_wrapper
    print("[memory] senderをアンロードしました。")

    return pairs, reference_norm


# ---------------------------------------------------------------------------
# 2. 閉形式ridge回帰: LatentMAS W_a (models.py _build_latent_realign_matrix) と同形式
# ---------------------------------------------------------------------------

def fit_linear_map(
    pairs: list[AlignmentPair],
    ridge_lambda: float = 1e-2,
    device: str | None = None,
) -> torch.Tensor:
    """
    W_cross = argmin_W || H W - Y ||_F^2 + lambda ||W||_F^2
    ここで Y は各アンカーごとに異なる target_delta (内容依存)。

    閉形式解: W = (H^T H + lambda I)^{-1} H^T Y

    models.py の _build_latent_realign_matrix と同じ形式だが、
    LatentMASのW_aは各語彙トークンごとに異なるペア (W_out行 <-> W_in行) で回帰するのに対し、
    ここでは各アンカーテキストごとに異なるペア (sender rollout <-> receiver内容embedding差分)
    で回帰する。目的変数が全ペアで同一だと、入力に関わらず同じ出力を返す退化した写像になる
    ので、target_delta が message-dependent であることが必須。

    device を指定すると、その上でsolveを実行する (モデルは既にアンロード済みなので
    どちらのGPUを使ってもよい)。未指定の場合はpairsが乗っているデバイス (通常CPU) で解く。
    """
    H = torch.cat([p.h_rollout for p in pairs], dim=0).to(torch.float32)     # [N*T, d_h_sender]
    Y = torch.cat([p.target_delta for p in pairs], dim=0).to(torch.float32)  # [N*T, d_h_receiver]

    if device is not None:
        H = H.to(device)
        Y = Y.to(device)

    d_h = H.shape[1]
    I = torch.eye(d_h, dtype=H.dtype, device=H.device)
    HtH = H.T @ H
    HtY = H.T @ Y
    W_cross = torch.linalg.solve(HtH + ridge_lambda * I, HtY)
    return W_cross.cpu()  # shape: [d_h_sender, d_h_receiver]


def evaluate_reconstruction(pairs: list[AlignmentPair], W_cross: torch.Tensor) -> dict:
    """
    held-outペアに対して、H_rollout @ W_cross が target_delta をどれだけ再現できているかを測る。
    以前のバージョン (目的変数が定数) では、この診断をやっても無意味だったため未実装だった。
    フィット対象を content-dependent に直した今、初めて意味のある診断になる。

    Returns:
        {"relative_frobenius_error": ..., "mean_cosine_similarity": ...}
    """
    H = torch.cat([p.h_rollout for p in pairs], dim=0).to(torch.float32)
    Y = torch.cat([p.target_delta for p in pairs], dim=0).to(torch.float32)
    pred = H @ W_cross.to(torch.float32)
    rel_err = (pred - Y).norm() / Y.norm().clamp_min(1e-8)
    cos = F.cosine_similarity(pred, Y, dim=-1).mean()
    return {
        "relative_frobenius_error": rel_err.item(),
        "mean_cosine_similarity": cos.item(),
    }


# ---------------------------------------------------------------------------
# 3. 推論時の注入: VW Eq.1 と同じ残差注入形式
# ---------------------------------------------------------------------------

def build_injection(
    h_rollout: torch.Tensor,   # [T, d_h_sender]  (T == K)
    x_baseline: torch.Tensor,  # [K, d_h_receiver]
    W_cross: torch.Tensor,     # [d_h_sender, d_h_receiver]
    gate: float = 0.5,         # VWのgでは学習されるが、ここでは固定 or sweep対象のハイパラ
    reference_norm: float | None = None,  # compute_reference_norm() の結果を渡すとノルムクリップできる
) -> torch.Tensor:
    """
    X_slot = X̄_slot + gate * (H_rollout @ W_cross)

    VW Eq.1: X_img = X̄_img + g * Resample(Delta, L_img) と同じ残差注入の形式。
    W_cross は target_delta (= 受信側での内容embedding - x_baseline) を予測するよう
    フィットされているので、Delta = H_rollout @ W_cross は message-dependent な
    内容予測になっているはず (以前のバージョンとの違い)。

    reference_norm を渡した場合、Deltaのノルムが基準点の典型ノルムを大きく超えないように
    簡易クリップする。
    """
    delta = h_rollout.to(torch.float32) @ W_cross.to(torch.float32)  # [K, d_h_receiver]
    if reference_norm is not None:
        delta_norm = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        max_norm = reference_norm
        scale = torch.clamp(max_norm / delta_norm, max=1.0)
        delta = delta * scale
    return x_baseline.to(torch.float32) + gate * delta


# ---------------------------------------------------------------------------
# CLI: 実験の骨格
# ---------------------------------------------------------------------------

def _load_model_wrapper(model_name: str, device_str: str, latent_space_realign: bool) -> ModelWrapper:
    """簡易args名前空間を作ってModelWrapperを構築する。"""
    device = torch.device(device_str)
    ns = argparse.Namespace(
        latent_space_realign=latent_space_realign,
        use_second_HF_model=False,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=False,
        method="linear_alignment_ablation",
    )
    return ModelWrapper(model_name, device, use_vllm=False, args=ns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sender_model_name", required=True)
    parser.add_argument("--receiver_model_name", required=True)
    parser.add_argument("--sender_device", default="cuda:0")
    parser.add_argument("--receiver_device", default="cuda:0")
    parser.add_argument("--sender_latent_space_realign", action="store_true",
                         help="送信元自身のW_a(自己アライメント)を有効にしてrolloutを生成する。"
                              "無効の場合はNormMatchのみ適用された生に近いhidden stateになる。")
    parser.add_argument("--anchor_texts_path", required=True)
    parser.add_argument("--anchor_limit", type=int, default=300)
    parser.add_argument("--latent_steps_T", type=int, required=True,
                         help="自由に選べるスロット長 K (送信元rollout長Tと一致させる)")
    parser.add_argument("--target_mode", choices=["slot"], default="slot",
                         help="現状 slot (LLMのみの継続プロンプトスロット注入) のみ対応")
    parser.add_argument("--insertion_position", type=int, default=0,
                         help="スロット挿入位置 (トークンオフセット)。TextMASプロンプトテンプレート"
                              "(prompts.py)で<CONTEXT>が入る位置に揃えることを推奨")
    parser.add_argument("--filler_text", default=None,
                         help="基準点構築に使う中立文。未指定時は DEFAULT_FILLER_TEXT")
    parser.add_argument("--check_filler_candidates", action="store_true",
                         help="複数の中立文候補でKLの大きさを比較する事前チェックのみ実行して終了する"
                              "(receiverのみロードされる)")
    parser.add_argument("--kl_noop_threshold", type=float, default=0.05)
    parser.add_argument("--ridge_lambda", type=float, default=1e-2)
    parser.add_argument("--fit_device", default=None,
                         help="closed-form solveを実行するデバイス。未指定ならCPUで解く")
    parser.add_argument("--val_fraction", type=float, default=0.1,
                         help="hold-outして再構成診断に使う割合。0にすると診断をスキップする")
    parser.add_argument("--val_seed", type=int, default=42)
    parser.add_argument("--gate_sweep", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 1.0])
    parser.add_argument("--out_dir", default="checkpoints/linear_alignment_ablation")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    filler_text = args.filler_text or DEFAULT_FILLER_TEXT
    anchor_texts = load_anchor_texts(args.anchor_texts_path, limit=args.anchor_limit)

    # --- filler候補比較のみ実行する場合: receiverのみロードすれば足りる ---
    if args.check_filler_candidates:
        receiver_wrapper = _load_model_wrapper(args.receiver_model_name, args.receiver_device, latent_space_realign=False)
        candidates = [DEFAULT_FILLER_TEXT]
        if filler_text != DEFAULT_FILLER_TEXT:
            candidates.append(filler_text)
        results = check_filler_content_sensitivity(
            receiver_wrapper, anchor_texts[0],
            candidate_filler_texts=candidates,
            slot_length_K=args.latent_steps_T,
            insertion_position=args.insertion_position,
        )
        print("[filler content sensitivity check]")
        for text_preview, kl in results.items():
            print(f"  KL={kl:.4f}  filler='{text_preview}'")
        print("候補間でKLの大きさが揃っていれば、内容自体は無害と判断してよい。")
        _unload_model_wrapper(receiver_wrapper)
        return

    # --- 本実行 ---
    pairs, reference_norm = collect_alignment_pairs_sequential(
        sender_model_name=args.sender_model_name,
        receiver_model_name=args.receiver_model_name,
        sender_device=args.sender_device,
        receiver_device=args.receiver_device,
        sender_latent_space_realign=args.sender_latent_space_realign,
        anchor_texts=anchor_texts,
        latent_steps_T=args.latent_steps_T,
        target_mode=args.target_mode,
        filler_text=filler_text,
        insertion_position=args.insertion_position,
        kl_noop_threshold=args.kl_noop_threshold,
    )

    # --- held-out再構成診断 (target_deltaがmessage-dependentになったので初めて意味を持つ) ---
    if args.val_fraction > 0 and len(pairs) >= 10:
        rng = random.Random(args.val_seed)
        indices = list(range(len(pairs)))
        rng.shuffle(indices)
        n_val = max(1, int(len(pairs) * args.val_fraction))
        val_idx = set(indices[:n_val])
        train_pairs = [p for i, p in enumerate(pairs) if i not in val_idx]
        val_pairs = [p for i, p in enumerate(pairs) if i in val_idx]

        W_cross_train_only = fit_linear_map(train_pairs, ridge_lambda=args.ridge_lambda, device=args.fit_device)
        diag = evaluate_reconstruction(val_pairs, W_cross_train_only)
        print(f"[held-out diagnostic] n_train={len(train_pairs)} n_val={len(val_pairs)}")
        print(f"  relative Frobenius error = {diag['relative_frobenius_error']:.4f}  (低いほど良い、0が完全再現)")
        print(f"  mean cosine similarity   = {diag['mean_cosine_similarity']:.4f}  (高いほど良い、1が完全一致)")
    else:
        diag = None

    # --- 最終的な W_cross は全データでフィットする ---
    W_cross = fit_linear_map(pairs, ridge_lambda=args.ridge_lambda, device=args.fit_device)

    torch.save(
        {
            "W_cross": W_cross,
            "reference_norm": reference_norm,
            "held_out_diagnostic": diag,
            "gate_sweep": args.gate_sweep,
            "args": vars(args),
        },
        Path(args.out_dir) / "linear_map.pt",
    )
    print(f"[done] W_cross saved to {args.out_dir}/linear_map.pt, shape={tuple(W_cross.shape)}")
    print(f"[info] reference_norm (受信側embeddingの典型ノルム) = {reference_norm:.4f}")
    print("次のステップ: gate_sweepの各値でGSM8K等のend-to-end評価を実行し、")
    print("TextMAS / VW(非線形コーデック) / 本手法(線形写像) の3系統を比較すること。")


if __name__ == "__main__":
    main()