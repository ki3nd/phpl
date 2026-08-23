"""
Eval-only: loads an ALREADY-merged CLIP checkpoint (from
export_merged_teacher.py's --out) and reports its zero-shot/cosine-similarity
accuracy. No LoRA merging here -- the checkpoint is already plain.

Usage:
    python eval_merged_teacher.py \\
        --root /kaggle/working/data --domains a-c --backbone ViT-B/16 \\
        --dataset-config-file configs/datasets/officehome.yaml \\
        --config-file configs/trainers/PHPL/b32_ep10_officehome.yaml \\
        --checkpoint merged_teacher_clip.pt
"""
import argparse

import torch

from dassl.data import DataManager

from utils.clip_part import load_clip_to_cpu
from trainers.da.phpl_momentum import _FrozenTeacherCLIP
from train_cmkd import build_cfg
from export_merged_teacher import evaluate_cosine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--domains", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="ViT-B/16")
    parser.add_argument("--dataset-config-file", type=str, required=True)
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="/tmp/eval_merged_teacher")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=2)

    parser.add_argument("--checkpoint", type=str, required=True,
                         help="a plain, already-merged CLIP state_dict (.pt) from export_merged_teacher.py")

    args = parser.parse_args()
    cfg = build_cfg(args)
    device = torch.device(f"cuda:{cfg.GPU}" if torch.cuda.is_available() else "cpu")

    dm = DataManager(cfg)
    classnames = dm.dataset.classnames
    test_loader = dm.test_loader

    clip_model = load_clip_to_cpu(cfg)
    if cfg.TRAINER.PHPLMOMENTUM.PREC in ("fp32", "amp"):
        clip_model.float()
    clip_model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    model = _FrozenTeacherCLIP(cfg, classnames, clip_model).to(device)
    acc = evaluate_cosine(model, test_loader, device)
    print(f"accuracy = {acc:.2f}%")


if __name__ == "__main__":
    main()
