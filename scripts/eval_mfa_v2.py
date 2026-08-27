"""Offline evaluation/diagnosis of a train_mfa_v2.py run from its saved
checkpoints -- reproduces evaluate()'s own Teacher1/Teacher2/Ensemble
numbers and then digs into WHY the ensemble lands where it does.

Motivation: a run whose Ensemble accuracy sits BETWEEN the two branches
(e.g. Teacher1 81.15 < Ensemble 81.83 < Teacher2 82.20) is not a healthy
ensemble -- averaging two diverse models should beat both. This script
measures the things that distinguish the possible causes:

  - Is there any headroom at all? -> ORACLE ceiling (either branch right).
    If the oracle is barely above the better branch, the branches make the
    same mistakes and NO fusion rule can help; the complementarity premise
    itself is what needs revisiting.
  - Do the branches even disagree? -> disagreement rate, plus each
    branch's accuracy ON the disagreement subset (who wins the arguments).
  - Are the two confidence scales comparable? -> mean max-prob / mean
    entropy per source. Branch 1's cosine logits are logit_scale(~100)-
    scaled and saturate toward 0/1, while branch 2's learned head is
    label-smoothed and diffuse; a 0.5/0.5 average of two such distributions
    is NOT a 50/50 vote, and the effective weight is an accident of those
    two unrelated scales.
  - Would a different fusion rule fix it? -> a weight sweep plus
    logit-space, per-sample z-scored, and entropy-matched-temperature
    fusions, all computed offline from the SAME cached logits.

Because the forward pass is the expensive part (two ViT-B/16 backbones over
the whole target set), it runs ONCE and caches every source's logits to an
.npz; every analysis above is then instant and re-runnable without touching
the models. Pass --recompute to force a fresh forward pass.

THREE prediction sources are cached, not two -- Teacher2's cosine branch
comes free (its features are already computed for the head), and it is what
lets "branch" be separated from "modality": T1-cosine and T2-cosine are both
language-side, T2-head is vision-side.

IMPORTANT -- Teacher2 here is NOT bit-identical to what evaluate() scored:
train_mfa_v2.py keeps Teacher2's EMA head (teacher_classifier_layer) as a
plain local variable, NOT a submodule of model2, and saves only
model2.state_dict() -- so the EMA head is in no checkpoint. What IS saved is
the EMA backbone (model2.teacher_model) and the LIVE head
(model2.classifier_layer), so that pairing is what gets scored here. Expect
Teacher2 to land near, but not exactly on, the number the training run
printed. Teacher1 IS exact (LoRA weights are saved in full, backbone frozen).

Requires CUDA: vlpuda_pure/models/backbone.py hardcodes device="cuda" and
.cuda() (and make_model.py:33 does .cuda() on the backbone), so branch 2
cannot be built on CPU. Branch 1 has no such constraint -- use
--skip-branch2 to score it alone on a CPU-only machine.

Usage:
    python scripts/eval_mfa_v2.py \\
        --root /home/pc1175/DA-Research/data_root --domains a-c \\
        --dataset-config-file configs/datasets/officehome.yaml \\
        --config-file configs/trainers/PHPL/b32_ep10_officehome.yaml \\
        --lora-path  .../MFA/officehome/a-c/Teacher1/LoRA-last.pt \\
        --model2-path .../MFA/officehome/a-c/model2-last.pt \\
        --cache /tmp/mfa_v2_a-c_logits.npz

(--root must contain an `office_home/` directory laid out as
<root>/office_home/<domain>/<class>/*.jpg -- a symlink is fine.)
"""
import argparse
import os.path as osp
import sys

import numpy as np
import torch
from torch.nn import functional as F
from torchvision.transforms import Compose, Normalize, Resize, ToTensor
from torchvision.transforms.functional import InterpolationMode

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from dassl.data import DataManager  # noqa: E402
from dassl.utils import set_random_seed  # noqa: E402
from loralib.utils import load_lora  # noqa: E402
from train_cmkd import build_cfg  # noqa: E402
from train_mfa_v2 import TransferNet, _build_lora_pair  # noqa: E402

# Cached logits are keyed by source name; SOURCES defines both the order
# they're reported in and which pair the "current" ensemble uses.
T1 = "t1_cosine"      # Teacher1: CLIP+LoRA cosine (branch 1) -- exact
T2_HEAD = "t2_head"   # Teacher2: EMA backbone -> LIVE head (branch 2)
T2_COS = "t2_cosine"  # Teacher2: EMA backbone -> cosine (free, language-side)


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _entropy(p):
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=-1)


def _acc(pred, label):
    return 100.0 * (pred == label).mean()


@torch.no_grad()
def extract_logits(args, cfg, device):
    """One forward pass over the test set; returns {source: logits} + labels."""
    normalize = Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    crop = cfg.INPUT.SIZE[0]
    # VLP-UDA's own TEST transform, identical to train_mfa_v2.py's _tfm_test:
    # a direct Resize to the crop size, NO CenterCrop. Must match exactly or
    # the numbers aren't comparable to the ones training printed.
    tfm_test = Compose([
        Resize([crop, crop], interpolation=InterpolationMode.BILINEAR),
        ToTensor(),
        normalize,
    ])
    # Same transform for train -- the train loaders are built by DataManager
    # but never iterated here.
    dm = DataManager(cfg, custom_tfm_train=tfm_test, custom_tfm_test=tfm_test)
    test_loader = dm.test_loader
    classnames = dm.dataset.classnames
    num_classes = dm.num_classes
    is_vit = cfg.MODEL.BACKBONE.NAME.split("-")[0] == "ViT"
    print(f"test set: {len(test_loader.dataset)} images, {num_classes} classes")

    print(f"Building Teacher1 (CLIP+LoRA) and loading {args.lora_path}")
    student1, teacher1, _, lora1_t = _build_lora_pair(cfg, is_vit, classnames, device)
    # _build_lora_pair builds the student too (it's how the pair is kept
    # structurally identical); nothing here needs it.
    del student1
    lora_dir, lora_file = osp.dirname(args.lora_path), osp.basename(args.lora_path)
    if lora_file.endswith(".pt"):
        lora_file = lora_file[:-3]
    # load_lora validates r/alpha/encoder against cfg and raises on mismatch.
    load_lora(cfg, lora1_t, lora_dir, lora_file)
    teacher1.eval()

    model2 = None
    if not args.skip_branch2:
        print(f"Building Teacher2 (vlpuda_pure TransferNet) and loading {args.model2_path}")
        vlpuda_args = argparse.Namespace(
            datasets="office_home", model_name="VIT-B", num_class=num_classes,
            baseline=False, pda=False, fixmatch=False, rst=False,
            label_smoothing=0.1, lambda1=0.25, lambda2=0.1, lambda3=0.025,
            multiple_lr_classifier=1000, max_iter=1,
        )
        # train=False skips the loss modules -- none of them hold parameters
        # that matter here, and forward() is never called (only
        # forward_features/forward_head/classifier_layer are).
        model2 = TransferNet(vlpuda_args, train=False).to(device)
        state = torch.load(args.model2_path, map_location=device)
        missing, unexpected = model2.load_state_dict(state, strict=False)
        # Reported rather than swallowed: a silently-unloaded backbone would
        # look like a plausible-but-wrong accuracy instead of an error.
        missing = [k for k in missing]
        unexpected = [k for k in unexpected]
        print(f"  load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected")
        for k in missing[:5]:
            print(f"    missing:    {k}")
        for k in unexpected[:5]:
            print(f"    unexpected: {k}")
        if missing:
            raise SystemExit(
                "Refusing to score a partially-loaded model2 -- missing keys above."
            )
        model2.eval()

    out = {T1: [], T2_HEAD: [], T2_COS: []}
    labels = []
    for i, batch in enumerate(test_loader):
        image = batch["img"].to(device)
        labels.append(batch["label"].numpy())

        logits1, _ = teacher1(image)
        out[T1].append(logits1.float().cpu().numpy())

        if model2 is not None:
            # Both branch-2 sources share ONE feature computation -- the
            # cosine head is genuinely free on top of the classifier head.
            feat2 = model2.teacher_model.forward_features(image)
            out[T2_HEAD].append(model2.classifier_layer(feat2).float().cpu().numpy())
            out[T2_COS].append(model2.teacher_model.forward_head(feat2).float().cpu().numpy())

        if (i + 1) % 10 == 0:
            print(f"  batch {i + 1}/{len(test_loader)}")

    labels = np.concatenate(labels)
    res = {k: np.concatenate(v) for k, v in out.items() if v}
    return res, labels


def analyze(logits, labels, args):
    names = [n for n in (T1, T2_HEAD, T2_COS) if n in logits]
    probs = {n: _softmax(logits[n]) for n in names}
    preds = {n: probs[n].argmax(-1) for n in names}

    print("\n" + "=" * 72)
    print("PER-SOURCE ACCURACY")
    print("=" * 72)
    for n in names:
        print(f"  {n:10s} {_acc(preds[n], labels):6.2f}%")

    if T1 not in probs or T2_HEAD not in probs:
        print("\n(branch 2 absent -- skipping every two-branch diagnostic)")
        return

    p1, p2 = probs[T1], probs[T2_HEAD]
    c1, c2 = preds[T1] == labels, preds[T2_HEAD] == labels

    ens = (0.5 * (p1 + p2)).argmax(-1)
    print(f"\n  {'ensemble':10s} {_acc(ens, labels):6.2f}%   "
          f"(current rule: 0.5*(p1+p2), what evaluate() computes)")

    print("\n" + "=" * 72)
    print("HEADROOM")
    print("=" * 72)
    both = (c1 & c2).mean() * 100
    either = (c1 | c2).mean() * 100
    neither = (~c1 & ~c2).mean() * 100
    print(f"  both right          {both:6.2f}%")
    print(f"  ORACLE (either)     {either:6.2f}%   <- ceiling of ANY per-sample fusion")
    print(f"  neither right       {neither:6.2f}%   <- unreachable by fusion")
    print(f"  oracle - best single{either - max(_acc(preds[T1], labels), _acc(preds[T2_HEAD], labels)):6.2f}pp"
          f"   <- the entire prize on offer")

    dis = preds[T1] != preds[T2_HEAD]
    print("\n" + "=" * 72)
    print("DISAGREEMENT (the only samples fusion can change)")
    print("=" * 72)
    print(f"  disagreement rate   {dis.mean() * 100:6.2f}%  ({dis.sum()} samples)")
    if dis.sum():
        print(f"  on those, T1 right  {c1[dis].mean() * 100:6.2f}%")
        print(f"  on those, T2 right  {c2[dis].mean() * 100:6.2f}%")
        print(f"  on those, neither   {(~c1[dis] & ~c2[dis]).mean() * 100:6.2f}%")

    print("\n" + "=" * 72)
    print("CALIBRATION  (are the two confidence scales comparable?)")
    print("=" * 72)
    print(f"  {'source':10s} {'mean max-p':>11s} {'mean entropy':>13s} {'p>0.85':>8s}")
    for n in names:
        mp = probs[n].max(-1)
        print(f"  {n:10s} {mp.mean():11.4f} {_entropy(probs[n]).mean():13.4f} "
              f"{(mp >= 0.85).mean() * 100:7.2f}%")

    print("\n" + "=" * 72)
    print("ALTERNATIVE FUSION RULES  (offline, same cached logits)")
    print("=" * 72)
    z1, z2 = logits[T1], logits[T2_HEAD]

    best_w, best_acc = None, -1
    for w in np.arange(0, 1.0001, 0.05):
        a = _acc((w * p1 + (1 - w) * p2).argmax(-1), labels)
        if a > best_acc:
            best_w, best_acc = w, a
    print(f"  prob average, best weight   w={best_w:.2f} -> {best_acc:6.2f}%   "
          f"(w=1 is pure T1, w=0 pure T2)")

    # Geometric mean of the two distributions == arithmetic mean in log space.
    geo = _acc((np.log(np.clip(p1, 1e-12, None)) + np.log(np.clip(p2, 1e-12, None))).argmax(-1), labels)
    print(f"  geometric mean (log-prob)   {geo:6.2f}%")

    # Per-sample standardization of each branch's logits: kills the scale
    # difference with no fitted parameter at all.
    def _z(z):
        return (z - z.mean(-1, keepdims=True)) / (z.std(-1, keepdims=True) + 1e-8)
    zs = _acc(_softmax(0.5 * (_z(z1) + _z(z2))).argmax(-1), labels)
    print(f"  per-sample z-scored logits  {zs:6.2f}%")

    # Entropy matching: one scalar temperature per branch, chosen so both
    # branches carry the SAME mean predictive entropy -- unsupervised, needs
    # no target labels, just makes "confidence" mean the same thing on both.
    def _mean_ent(z, T):
        return _entropy(_softmax(z / T)).mean()

    def _fit_T(z, target_ent):
        lo, hi = 1e-3, 1e3
        for _ in range(60):
            mid = (lo * hi) ** 0.5
            if _mean_ent(z, mid) < target_ent:
                lo = mid
            else:
                hi = mid
        return (lo * hi) ** 0.5

    target = 0.5 * (_entropy(p1).mean() + _entropy(p2).mean())
    t1_T, t2_T = _fit_T(z1, target), _fit_T(z2, target)
    em = _acc((0.5 * (_softmax(z1 / t1_T) + _softmax(z2 / t2_T))).argmax(-1), labels)
    print(f"  entropy-matched temps       {em:6.2f}%   (T1={t1_T:.3f}, T2={t2_T:.3f})")

    # Per-sample pick-the-more-confident, on the entropy-matched scales --
    # a gate is only meaningful once confidence is comparable.
    q1, q2 = _softmax(z1 / t1_T), _softmax(z2 / t2_T)
    pick = np.where((q1.max(-1) >= q2.max(-1))[:, None], q1, q2)
    print(f"  argmax-confidence gate      {_acc(pick.argmax(-1), labels):6.2f}%   "
          f"(after entropy matching)")

    if T2_COS in probs:
        p3 = probs[T2_COS]
        print(f"  3-way average (+T2 cosine)  "
              f"{_acc(((p1 + p2 + p3) / 3).argmax(-1), labels):6.2f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=str, required=True,
                        help="dassl dataset root -- must contain office_home/<domain>/<class>/")
    parser.add_argument("--domains", type=str, required=True, help="e.g. a-c")
    parser.add_argument("--backbone", type=str, default="ViT-B/16")
    parser.add_argument("--dataset-config-file", type=str, required=True)
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--lora-path", type=str, required=True,
                        help="Teacher1's saved LoRA weights, e.g. .../Teacher1/LoRA-last.pt")
    parser.add_argument("--model2-path", type=str, default=None,
                        help="Teacher2's saved TransferNet state_dict, e.g. .../model2-last.pt "
                             "(required unless --skip-branch2)")
    parser.add_argument("--cache", type=str, default=None,
                        help="path to an .npz of cached logits; reused if it exists (the "
                             "forward pass is the only expensive part), written if it doesn't")
    parser.add_argument("--recompute", action="store_true",
                        help="ignore an existing --cache and redo the forward pass")
    parser.add_argument("--skip-branch2", action="store_true",
                        help="score Teacher1 alone -- the only mode that runs without CUDA, "
                             "since vlpuda_pure hardcodes .cuda() (see module docstring)")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="/tmp/eval_mfa_v2",
                        help="unused by this script; build_cfg just wants one")
    args = parser.parse_args()

    if not args.skip_branch2 and args.model2_path is None:
        parser.error("--model2-path is required unless --skip-branch2 is set")

    if args.cache and osp.exists(args.cache) and not args.recompute:
        print(f"Loading cached logits from {args.cache} (pass --recompute to redo)")
        z = np.load(args.cache)
        labels = z["labels"]
        logits = {k: z[k] for k in z.files if k != "labels"}
    else:
        cfg = build_cfg(args)
        if cfg.SEED >= 0:
            set_random_seed(cfg.SEED)
        if args.skip_branch2:
            device = torch.device(f"cuda:{cfg.GPU}" if torch.cuda.is_available() else "cpu")
        else:
            if not torch.cuda.is_available():
                raise SystemExit(
                    "CUDA is required for branch 2 (vlpuda_pure hardcodes device=\"cuda\" in "
                    "models/backbone.py and .cuda() in models/make_model.py). Use "
                    "--skip-branch2 to score Teacher1 alone on this machine."
                )
            device = torch.device(f"cuda:{cfg.GPU}")
        print(f"device: {device}")
        logits, labels = extract_logits(args, cfg, device)
        if args.cache:
            np.savez_compressed(args.cache, labels=labels, **logits)
            print(f"cached logits -> {args.cache}")

    analyze(logits, labels, args)


if __name__ == "__main__":
    main()
