"""
infer_llm_wormhole.py

train_llm_wormhole.py で学習した codec (Encoder/Decoder + hub alignment) を使い、
2体の異種LLMエージェントに実際に「潜在通信」させる最小推論スクリプト。

VW本家の VisionLatentMASMethodCODECNew.run_item() に相当するが、
- planner/critic/refiner/judger の4役ロールを、送信側(sender)/受信側(receiver)の
  2エージェントに単純化
- dummy画像への注入ではなく dummy文章枠への注入
という2点を変更している。

流れ (2エージェント・sequentialの最小形):
  1. sender が質問に対して "latent thoughts" を自己回帰的に生成 (LatentMASのlatent rollout)
  2. sender専用の Encoder で hub空間 U_ref にエンコード (+ affine alignment)
  3. receiver専用の Decoder で hub空間から receiver空間へデコード (+ affine alignment)
     -> 注入ベクトル (delta * gate)
  4. receiver のプロンプト中の dummy文章枠にその注入ベクトルを加算し、
     質問文と合わせて通常どおり autoregressive に最終回答を生成
"""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional, Tuple

import torch

from models import ModelWrapper
from methods.vision_latent_mas_codec_new import (
    LatentToUniversalEncoder,
    UniversalToVisionDecoder,
    _apply_affine,
)
from train_llm_wormhole import (
    WormholeSlot,
    _resample_tokens,
    _infer_hidden_size,
    _get_hidden_states_tuple,
)


SENDER_PROMPT_TEMPLATE = """You are a Planner Agent. Given an input question, think through how to solve it.

Question: {question}

Do not produce the final answer. Just think through the problem.
"""

RECEIVER_PROMPT_TEMPLATE_PREFIX = """You are a helpful assistant. You are given latent information from another agent's reasoning process, provided below as a sequence of special tokens, followed by a target question.

Latent information:
"""

RECEIVER_PROMPT_TEMPLATE_SUFFIX = """

Target Question: {question}

The latent information might contain irrelevant contents; ignore it if unhelpful.
Reason step by step and give your final answer clearly at the end.
"""


class LLMWormhole2Agent:
    """2エージェント異種LLM向けの最小Vision-Wormhole推論クラス。

    run.py の既存メソッド (VisionLatentMASMethodCODECNew等) と同じ流儀で、
    ModelWrapper のインスタンスは呼び出し側 (run.py) が構築して渡す。
    ここでモデルを自前でロードし直すと、run.py 側の --agent_devices や
    --latent_space_realign などの設定と食い違ったり、二重ロードで
    メモリを無駄にしたりするため。
    """

    def __init__(self, args: argparse.Namespace, models: List[ModelWrapper], ckpt_path: str = ""):
        self.args = args
        ckpt_path = ckpt_path or getattr(args, "wormhole_codec_path", "")
        if not ckpt_path:
            raise ValueError("--wormhole_codec_path is required for llm_wormhole method")
        ckpt = torch.load(ckpt_path, map_location="cpu")

        self.model_names: List[str] = list(ckpt["models"])
        if len(self.model_names) < 2:
            raise ValueError("codec checkpoint must contain at least 2 models")

        name_to_wrapper = {w.model_name: w for w in models}
        missing = [n for n in self.model_names if n not in name_to_wrapper]
        if missing:
            raise ValueError(
                f"codec checkpoint expects models {self.model_names}, but these were not passed in "
                f"via --agent_model_names: {missing}"
            )
        self.device = next(iter(name_to_wrapper.values())).model.device

        cfg = ckpt["config"]
        self.codec_dim = int(cfg["codec_dim"])
        self.codec_tokens = int(cfg["codec_tokens"])
        self.codec_slot_tokens = int(cfg["codec_slot_tokens"])
        self.codec_heads = int(cfg["codec_heads"])
        self.codec_layers = int(cfg["codec_layers"])
        self.codec_dropout = float(cfg["codec_dropout"])
        self.codec_gate_init_bias = float(cfg["codec_gate_init_bias"])

        # --- モデル(既存インスタンスを再利用)・Encoder/Decoder・alignment・プレースホルダをロード ---
        self.wrappers: Dict[str, ModelWrapper] = {}
        self.encoders: Dict[str, LatentToUniversalEncoder] = {}
        self.decoders: Dict[str, UniversalToVisionDecoder] = {}
        self.slots: Dict[str, WormholeSlot] = {}

        placeholder_cfg = ckpt.get("placeholder", {})
        align = ckpt["align"]
        self.align_out: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.align_in: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

        # run_batch()のインターフェース (List[Dict] -> List[str]) を
        # run.py の process_batch から呼べるようにするための計測用フィールド
        # (他メソッドと同様、_wrap_codec_new_results経由の集計に使われる)
        self.total_infer_time_sec = 0.0
        self.total_infer_batches = 0
        self.total_infer_items = 0

        for name in self.model_names:
            wrapper = name_to_wrapper[name]
            self.wrappers[name] = wrapper

            H = _infer_hidden_size(wrapper)
            enc = LatentToUniversalEncoder(
                h_in=H,
                d_univ=self.codec_dim,
                k_univ=self.codec_tokens,
                n_heads=self.codec_heads,
                n_layers=self.codec_layers,
                dropout=self.codec_dropout,
            ).to(device=self.device, dtype=torch.float32)
            enc.load_state_dict(ckpt["encoders"][name])
            enc.eval()
            self.encoders[name] = enc

            dec = UniversalToVisionDecoder(
                d_univ=self.codec_dim,
                h_out=H,
                k_img=self.codec_slot_tokens,
                n_heads=self.codec_heads,
                n_layers=self.codec_layers,
                dropout=self.codec_dropout,
                gate_init_bias=self.codec_gate_init_bias,
            ).to(device=self.device, dtype=torch.float32)
            dec.load_state_dict(ckpt["decoders"][name])
            dec.eval()
            self.decoders[name] = dec

            pc = placeholder_cfg.get(name, {})
            self.slots[name] = WormholeSlot(
                token_id=int(pc.get("token_id")),
                n_tokens=int(pc.get("n_tokens", self.codec_slot_tokens)),
            )

            out_map = align["out"].get(name)
            in_map = align["in"].get(name)
            self.align_out[name] = (out_map["W"].to(self.device), out_map["b"].to(self.device))
            self.align_in[name] = (in_map["W"].to(self.device), in_map["b"].to(self.device))

    # ------------------------------------------------------------------
    # 送信側: latent rollout -> hub空間へのエンコード
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _encode_sender(self, sender_name: str, question: str, latent_steps: int) -> torch.Tensor:
        wrapper = self.wrappers[sender_name]
        prompt = SENDER_PROMPT_TEMPLATE.format(question=question)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        _, ids, mask, _ = wrapper.prepare_chat_batch([messages], add_generation_prompt=True)
        ids, mask = ids.to(self.device), mask.to(self.device)

        _, lat = wrapper.generate_latent_batch(
            ids, attention_mask=mask, latent_steps=latent_steps, return_latent_embeds=True
        )  # [1, latent_steps, H_sender]

        U = self.encoders[sender_name](lat.detach().float())  # [1, K, d_univ]
        W_out, b_out = self.align_out[sender_name]
        U_ref = _apply_affine(U, W_out.to(U.dtype), b_out.to(U.dtype))
        return U_ref.detach()

    # ------------------------------------------------------------------
    # 受信側: hub空間からのデコード -> dummy文章枠への注入 -> 生成
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _decode_receiver(self, receiver_name: str, U_ref: torch.Tensor) -> torch.Tensor:
        W_in, b_in = self.align_in[receiver_name]
        U_i = _apply_affine(U_ref.to(self.device), W_in.to(U_ref.dtype), b_in.to(U_ref.dtype))
        delta, gate = self.decoders[receiver_name](U_i)
        return (gate * delta).detach()  # [1, k_img, H_receiver]

    @torch.no_grad()
    def _generate_receiver_answer(
        self,
        receiver_name: str,
        question: str,
        inj: torch.Tensor,
        max_new_tokens: int = 512,
    ) -> str:
        """dummy文章枠に注入して最終回答を生成する。

        【重要】train_llm_wormhole.py と同じ理由で、プレースホルダ文字列を
        リピートしてから丸ごと再トークナイズする方式は使わない
        (BPEのマージにより「token_idがn_tokens回連続する」前提が崩れるため)。
        prefix/suffixを別々にトークナイズし、プレースホルダのtoken_idを
        直接spliceする。
        """
        wrapper = self.wrappers[receiver_name]
        slot = self.slots[receiver_name]

        messages_prefix = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": RECEIVER_PROMPT_TEMPLATE_PREFIX},
        ]
        tpl = getattr(wrapper.tokenizer, "chat_template", None)
        if tpl:
            prefix_text = wrapper.tokenizer.apply_chat_template(
                messages_prefix, tokenize=False, add_generation_prompt=False
            )
        else:
            prefix_text = (
                f"<|system|>\nYou are a helpful assistant.\n</|system|>\n"
                f"<|user|>\n{RECEIVER_PROMPT_TEMPLATE_PREFIX}"
            )

        suffix_body = RECEIVER_PROMPT_TEMPLATE_SUFFIX.format(question=question)
        messages_full = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": RECEIVER_PROMPT_TEMPLATE_PREFIX + suffix_body},
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
        suffix_text = suffix_body + gen_prompt_suffix

        prefix_ids = wrapper.tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
        suffix_ids = wrapper.tokenizer(suffix_text, add_special_tokens=False)["input_ids"]
        placeholder_ids = [int(slot.token_id)] * int(slot.n_tokens)

        ids = list(prefix_ids) + placeholder_ids + list(suffix_ids)
        positions = list(range(len(prefix_ids), len(prefix_ids) + len(placeholder_ids)))

        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        emb = wrapper.model.get_input_embeddings()
        base_embeds = emb(input_ids).detach()
        inputs_embeds = base_embeds.clone()

        add = _resample_tokens(inj, len(positions)).to(base_embeds.dtype)
        inputs_embeds[:, positions, :] = base_embeds[:, positions, :] + add
        attention_mask = torch.ones_like(input_ids)

        gen_out = wrapper.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=wrapper.tokenizer.pad_token_id,
        )
        # inputs_embeds経由で生成した場合、生成トークンのみがsequencesとして返る
        # (プロンプト部分はembedsで与えているためトークンIDには含まれない)
        text = wrapper.tokenizer.decode(gen_out[0], skip_special_tokens=True)
        return text.strip()

    # ------------------------------------------------------------------
    # エンドツーエンド
    # ------------------------------------------------------------------

    def run(self, question: str, latent_steps: Optional[int] = None, max_new_tokens: Optional[int] = None) -> Dict[str, Any]:
        latent_steps = latent_steps if latent_steps is not None else max(1, int(getattr(self.args, "latent_steps", 0) or 32))
        max_new_tokens = max_new_tokens if max_new_tokens is not None else int(getattr(self.args, "max_new_tokens", 512))
        sender_name, receiver_name = self.model_names[0], self.model_names[1]

        zero_injection = bool(getattr(self.args, "wormhole_zero_injection", False))
        if zero_injection:
            # --- アブレーション用ベースライン ---
            # プロンプト構造 (dummy文章枠を含む) は本番と完全に同一のまま、
            # 注入ベクトルだけをゼロにする。これにより「wormholeの注入内容」が
            # 精度にどれだけ寄与しているかを、プロンプト構造の違いという
            # 交絡要因なしに切り分けられる (単にreceiverモデル単体に素の質問を
            # 投げる場合との比較よりも公平な比較になる)。
            slot = self.slots[receiver_name]
            H = _infer_hidden_size(self.wrappers[receiver_name])
            inj = torch.zeros((1, slot.n_tokens, H), device=self.device, dtype=torch.float32)
        else:
            U_ref = self._encode_sender(sender_name, question, latent_steps=latent_steps)
            inj = self._decode_receiver(receiver_name, U_ref)

        answer = self._generate_receiver_answer(receiver_name, question, inj, max_new_tokens=max_new_tokens)

        return {
            "sender": sender_name,
            "receiver": receiver_name,
            "question": question,
            "answer": answer,
            "zero_injection": zero_injection,
        }

    # ------------------------------------------------------------------
    # run.py 連携用インターフェース
    # ------------------------------------------------------------------

    def run_item(self, item: Dict[str, Any]) -> str:
        question = item.get("question", item.get("prompt", ""))
        return self.run(question)["answer"]

    def run_batch(self, batch: List[Dict[str, Any]]) -> List[str]:
        """run.py の process_batch から呼ばれるエントリポイント。

        VisionLatentMASMethodCODECNew.run_batch と同じインターフェース
        (List[Dict] -> List[str]) にしているため、run.py 側の
        _wrap_codec_new_results をそのまま流用してスコアリングできる。
        """
        import time

        infer_start = time.time()
        outputs = [self.run_item(item) for item in batch]
        infer_total = time.time() - infer_start
        self.total_infer_time_sec += infer_total
        self.total_infer_batches += 1
        self.total_infer_items += len(batch)
        return outputs


# ============================================================================
# スタンドアロンCLI (run.pyを介さず単独で動作確認したい場合用)
# ============================================================================


def _standalone_main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--codec_path", type=str, required=True)
    p.add_argument("--question", type=str, required=True)
    p.add_argument("--latent_steps", type=int, default=32)
    p.add_argument("--max_new_tokens", type=int, default=512)
    ns = p.parse_args()

    ckpt = torch.load(ns.codec_path, map_location="cpu")
    model_names = list(ckpt["models"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    wrapper_args = argparse.Namespace(latent_space_realign=True)
    models = [ModelWrapper(name, device, use_vllm=False, args=wrapper_args) for name in model_names]

    full_args = argparse.Namespace(
        wormhole_codec_path=ns.codec_path,
        latent_steps=ns.latent_steps,
        max_new_tokens=ns.max_new_tokens,
    )
    mas = LLMWormhole2Agent(args=full_args, models=models, ckpt_path=ns.codec_path)
    result = mas.run(ns.question, latent_steps=ns.latent_steps, max_new_tokens=ns.max_new_tokens)

    print(f"\n[sender={result['sender']}] -> [receiver={result['receiver']}]")
    print(f"Question: {result['question']}")
    print(f"Answer:\n{result['answer']}")


if __name__ == "__main__":
    _standalone_main()