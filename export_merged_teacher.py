"""
Sanity-check bridge script (vlpuda-sanity branch): bakes phase-1's trained
LoRA deltas into a PLAIN CLIP checkpoint (no LoRA wrapper classes at all),
loadable by the original, unmodified VLP-UDA repo (vlpuda_pure/) via
`model.load_state_dict(...)` on its own `clip.load(...)`-built backbone.

Why: to test the hypothesis that our own CMKD reimplementation is what's
underperforming, not the (already-adapted, 76.3%-accuracy) teacher itself --
by handing the SAME teacher to VLP-UDA's own, unmodified classifier_layer +
CMKD training loop and seeing whether IT gets good results starting from it.

Only nn.MultiheadAttention modules were ever LoRA-wrapped (see
loralib/utils.py's _apply_vit_lora/_apply_text_lora -- PARAMS=['q','k','v'],
out_proj/'o' untouched), so unwrapping is scoped to exactly those.

Usage:
    python export_merged_teacher.py \\
        --root /kaggle/working/data --domains a-c --backbone ViT-B/16 \\
        --dataset-config-file configs/datasets/officehome.yaml \\
        --config-file configs/trainers/PHPL/b32_ep10_officehome.yaml \\
        --phase1-dir output/PHPLMOMENTUM/1/officehome/a-c \\
        --out merged_teacher_clip.pt
"""
import argparse

import torch
import torch.nn as nn

from dassl.data import DataManager

from utils.clip_part import load_clip_to_cpu
from loralib.utils import apply_lora, apply_lora_rn, load_lora
from loralib.layers import PlainMultiheadAttentionLoRA, LinearLoRA
from trainers.da.phpl_momentum import _FrozenTeacherCLIP
from train_cmkd import build_cfg
import os.path as osp


@torch.no_grad()
def evaluate_cosine(model, test_loader, device):
    """Plain zero-shot/cosine-similarity accuracy (model(image) -> logits,
    feat), same metric phase 1 reports for teacher_now."""
    model.eval()
    correct, total = 0, 0
    for batch in test_loader:
        image = batch["img"].to(device)
        label = batch["label"].to(device)
        logits, _ = model(image)
        correct += (logits.argmax(dim=-1) == label).sum().item()
        total += label.size(0)
    return 100.0 * correct / max(total, 1)


def _merged_weight(proj):
    """proj is either a plain nn.Linear (never had LoRA, e.g. when 'o' isn't
    in PARAMS) or a LinearLoRA -- merge if the latter and not already merged
    (merge_lora_param() permanently overwrites .weight.data in place; NOT
    followed by sub_lora_data(), unlike LinearLoRA.forward()'s own
    just-in-time merge/unmerge -- this bake-in is meant to be permanent)."""
    if isinstance(proj, LinearLoRA) and proj.r > 0 and not proj.merged:
        proj.merge_lora_param()
        proj.merged = True
    weight = proj.weight.data.clone()
    bias = proj.bias.data.clone() if proj.bias is not None else None
    return weight, bias


def _to_plain_mha(lora_attn):
    """Reconstructs a standard nn.MultiheadAttention (in_proj_weight/bias +
    out_proj) from a PlainMultiheadAttentionLoRA, with every LoRA delta
    already baked into the returned weights."""
    qw, qb = _merged_weight(lora_attn.q_proj)
    kw, kb = _merged_weight(lora_attn.k_proj)
    vw, vb = _merged_weight(lora_attn.v_proj)
    ow, ob = _merged_weight(lora_attn.proj)

    has_bias = qb is not None
    mha = nn.MultiheadAttention(
        lora_attn.embed_dim, lora_attn.num_heads, bias=has_bias, batch_first=lora_attn.batch_first
    )
    with torch.no_grad():
        mha.in_proj_weight.copy_(torch.cat([qw, kw, vw], dim=0).float())
        if has_bias:
            mha.in_proj_bias.copy_(torch.cat([qb, kb, vb], dim=0).float())
        mha.out_proj.weight.copy_(ow.float())
        if ob is not None:
            mha.out_proj.bias.copy_(ob.float())
    return mha.to(dtype=qw.dtype)


def _unwrap_lora_attention(transformer):
    for block in transformer.resblocks:
        for name, submodule in list(block.named_children()):
            if isinstance(submodule, PlainMultiheadAttentionLoRA):
                setattr(block, name, _to_plain_mha(submodule))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--domains", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="ViT-B/16")
    parser.add_argument("--dataset-config-file", type=str, required=True)
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="/tmp/export_merged_teacher")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=2)

    parser.add_argument("--phase1-dir", type=str, required=True,
                         help="phase 1's output dir (contains a TeacherNow/ subdir)")
    parser.add_argument("--phase1-checkpoint", choices=["last", "best"], default="last")
    parser.add_argument("--out", type=str, required=True, help="output .pt path for the merged CLIP state_dict")

    parser.add_argument("--skip-eval", action="store_true",
                         help="skip the before/after accuracy sanity check (faster, less safe)")

    args = parser.parse_args()
    cfg = build_cfg(args)
    device = torch.device(f"cuda:{cfg.GPU}" if torch.cuda.is_available() else "cpu")

    dm = DataManager(cfg)
    classnames = dm.dataset.classnames
    test_loader = dm.test_loader

    print(f"Loading TeacherNow checkpoint from {args.phase1_dir}")
    clip_model = load_clip_to_cpu(cfg)
    if cfg.TRAINER.PHPLMOMENTUM.PREC in ("fp32", "amp"):
        clip_model.float()
    teacher_now = _FrozenTeacherCLIP(cfg, classnames, clip_model)

    is_vit = cfg.MODEL.BACKBONE.NAME.split('-')[0] == 'ViT'
    apply_fn = apply_lora if is_vit else apply_lora_rn
    list_lora_layers = apply_fn(cfg, teacher_now)
    filename = "LoRA-last" if args.phase1_checkpoint == "last" else "LoRA-best"
    load_lora(cfg, list_lora_layers, osp.join(args.phase1_dir, "TeacherNow"), filename=filename)
    teacher_now.to(device)

    if not args.skip_eval:
        acc_before = evaluate_cosine(teacher_now, test_loader, device)
        print(f"[sanity] accuracy BEFORE merging LoRA: {acc_before:.2f}% (should match phase 1's own eval)")

    print(f"Merging LoRA into {len(list_lora_layers)} attention layers")
    # teacher_now.image_encoder IS clip_model.visual, and text_encoder.transformer
    # IS clip_model.transformer (Simple_TextEncoder holds references, not copies) --
    # mutating through teacher_now mutates clip_model itself.
    _unwrap_lora_attention(teacher_now.image_encoder.transformer)
    _unwrap_lora_attention(teacher_now.text_encoder.transformer)

    remaining = sum(
        1 for m in clip_model.modules() if isinstance(m, PlainMultiheadAttentionLoRA)
    )
    assert remaining == 0, f"{remaining} PlainMultiheadAttentionLoRA modules were not unwrapped"

    if not args.skip_eval:
        acc_after = evaluate_cosine(teacher_now, test_loader, device)
        print(f"[sanity] accuracy AFTER merging LoRA:  {acc_after:.2f}% (should match the BEFORE value closely)")
        if abs(acc_after - acc_before) > 0.5:
            print("[sanity] WARNING: >0.5pp gap -- the merge likely has a bug, don't trust the exported checkpoint")

    torch.save(clip_model.state_dict(), args.out)
    print(f"Saved merged, plain CLIP state_dict to {args.out}")


if __name__ == "__main__":
    main()
