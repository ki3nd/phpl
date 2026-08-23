"""
MFA-style cross-model co-training between two HETEROGENEOUS teacher-student
pairs (branch: mfa-cross). Inspired by Multiple Fusion Adaptation (Zhang et
al., https://github.com/KaiiZhang/MFA) -- their "cross-model fusion" (each
net's teacher supervises the OTHER net's student, not just its own) and
"temporal fusion" (each net's own EMA teacher), applied here to two
STRUCTURALLY DIFFERENT architectures instead of MFA's two identical nets:

  - Student 1 / Teacher 1: PHPL-style -- LoRA-adapted CLIP, cosine-similarity
    classification against text embeddings (trainers/da/phpl_momentum.py's
    CustomCLIP/_FrozenTeacherCLIP).
  - Student 2 / Teacher 2: VLP-UDA-style -- a SEPARATE LoRA-adapted CLIP plus
    a learned classifier_layer (train_cmkd.py's _ClassifierHead: BatchNorm1d
    -> LayerNorm -> Linear), trained jointly (not the frozen-backbone phase-2
    split train_cmkd.py uses -- here the backbone trains too).

Each teacher is its own model's EMA (temporal fusion, unchanged from
elsewhere in this project). Each student's target-domain loss has TWO parts,
ADDED together, never replacing one another:
  - "self":  supervised by its OWN teacher (as usual).
  - "cross": supervised by the OTHER pair's teacher (MFA's cross-model term).
Student 1's cross/self loss (CONFI hard-threshold mask + CE) uses the SAME
fixed threshold (cfg.TRAINER.PHPLMOMENTUM.CONFI, default 0.85) for both --
no CBST/ramping-alpha à la MFA's own online-pseudo-label filter. Student 2's
cross/self loss (CMKD's gini-impurity task/distill terms) has no hard
threshold at all -- continuous, weighted by the agreement coefficient, same
as its "self" form.

A flat, iteration-based loop (not Dassl's TrainerXU/BaseDA) -- see
train_cmkd.py's own docstring for why. Source/target batches are SHARED
between both students each iteration (matches MFA's own trainer.py).

Usage:
    python train_mfa.py \\
        --root /kaggle/working/data --domains a-c --backbone ViT-B/16 \\
        --dataset-config-file configs/datasets/officehome.yaml \\
        --config-file configs/trainers/PHPL/b32_ep10_officehome.yaml \\
        --output-dir output/MFA/officehome/a-c \\
        --total-iters 2000 --print-freq 50 --eval-freq 200
"""
import argparse
import math
import os.path as osp

import torch
from torch.nn import functional as F

from dassl.data import DataManager
from dassl.utils import mkdir_if_missing, set_random_seed

from utils.clip_part import load_clip_to_cpu
from loralib.utils import apply_lora, apply_lora_rn, save_lora
from trainers.da.phpl_momentum import (
    CustomCLIP, _FrozenTeacherCLIP, _copy_lora_params, _ema_update_lora_params,
)
from train_cmkd import build_cfg, CyclingLoader, _gini_impurity, _calibrated_coefficient, _ClassifierHead


@torch.no_grad()
def _ema_update_module(ema_module, src_module, momentum):
    """Generic parameter+buffer-wise EMA for a plain nn.Module (Student 2's
    classifier head isn't LoRA, so _ema_update_lora_params doesn't apply)."""
    ema_state = ema_module.state_dict()
    for k, v in src_module.state_dict().items():
        ema_state[k].mul_(momentum).add_(v, alpha=1.0 - momentum)


def _cmkd_lamb(cur_iter, max_iter, gamma):
    p = min(cur_iter / max_iter, 1.0)
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0


def _build_lora_pair(cfg, is_vit, classnames, device):
    """One (student, teacher, list_lora_layers) triple -- a fresh CLIP
    backbone, LoRA-wrapped, teacher hard-copied from student's (fresh) init,
    both frozen except student's own LoRA params."""
    apply_fn = apply_lora if is_vit else apply_lora_rn

    clip_student = load_clip_to_cpu(cfg)
    clip_teacher = load_clip_to_cpu(cfg)
    if cfg.TRAINER.PHPLMOMENTUM.PREC in ("fp32", "amp"):
        clip_student.float()
        clip_teacher.float()

    student = CustomCLIP(cfg, classnames, clip_student)
    teacher = _FrozenTeacherCLIP(cfg, classnames, clip_teacher)
    list_lora_student = apply_fn(cfg, student)
    list_lora_teacher = apply_fn(cfg, teacher)

    for param in student.parameters():
        param.requires_grad_(False)
    for name, param in student.named_parameters():
        if "lora" in name:
            param.requires_grad_(True)
    for param in teacher.parameters():
        param.requires_grad_(False)

    _copy_lora_params(student, teacher)
    student.to(device)
    teacher.to(device)
    teacher.eval()
    return student, teacher, list_lora_student, list_lora_teacher


@torch.no_grad()
def evaluate(teacher1, teacher2_backbone, teacher2_head, test_loader, device):
    teacher2_head.eval()
    correct1, correct2, correct_ens, total = 0, 0, 0, 0
    for batch in test_loader:
        image = batch["img"].to(device)
        label = batch["label"].to(device)

        logits1, _ = teacher1(image)
        prob1 = F.softmax(logits1, dim=-1)

        _, feat2 = teacher2_backbone(image)
        logits2 = teacher2_head(feat2.float())
        prob2 = F.softmax(logits2, dim=-1)

        prob_ens = 0.5 * (prob1 + prob2)

        correct1 += (prob1.argmax(dim=-1) == label).sum().item()
        correct2 += (prob2.argmax(dim=-1) == label).sum().item()
        correct_ens += (prob_ens.argmax(dim=-1) == label).sum().item()
        total += label.size(0)
    teacher2_head.train()
    return (
        100.0 * correct1 / max(total, 1),
        100.0 * correct2 / max(total, 1),
        100.0 * correct_ens / max(total, 1),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=str, required=True)
    parser.add_argument("--domains", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="ViT-B/16")
    parser.add_argument("--dataset-config-file", type=str, required=True)
    parser.add_argument("--config-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=2)

    parser.add_argument("--total-iters", type=int, default=2000)
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--eval-freq", type=int, default=200)

    # Student 1 (PHPL-style) -- its own LoRA LR. Plain SGD, matching PHPL's
    # own recipe.
    parser.add_argument("--s1-lr", type=float, default=0.0035)
    parser.add_argument("--s1-momentum", type=float, default=0.9)
    parser.add_argument("--s1-weight-decay", type=float, default=5e-4)

    # Student 2 (VLP-UDA-style) -- TWO separate LRs (backbone LoRA vs
    # classifier head), same order of magnitude as Student 1's LoRA LR for
    # the backbone (NOT VLP-UDA's own tiny 3e-6 -- too low to converge in a
    # short budget, confirmed the hard way earlier in this project) and a
    # much larger one for the freshly-initialized classifier head.
    parser.add_argument("--s2-lora-lr", type=float, default=0.0035)
    parser.add_argument("--s2-lora-momentum", type=float, default=0.9)
    parser.add_argument("--s2-lora-weight-decay", type=float, default=5e-4)
    parser.add_argument("--s2-clf-optim", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--s2-clf-lr", type=float, default=0.003)
    parser.add_argument("--s2-clf-weight-decay", type=float, default=5e-4)

    parser.add_argument("--ema-momentum", type=float, default=0.996,
                         help="Teacher1 and Teacher2-backbone's LoRA EMA momentum")
    parser.add_argument("--head-ema-momentum", type=float, default=0.996,
                         help="Teacher2-head's EMA momentum, active AFTER --head-warmup-iters")
    parser.add_argument("--head-warmup-iters", type=int, default=100,
                         help="Teacher2-head hard-copies Student 2's head every step for this "
                              "many iterations (momentum=0) before switching to --head-ema-momentum "
                              "-- mirrors the epoch-1 hard-copy lesson learned for a randomly "
                              "initialized head elsewhere in this project.")

    parser.add_argument("--lambda1", type=float, default=0.25, help="Student 2's CMKD task/distill weight")
    parser.add_argument("--lamb-gamma", type=float, default=10.0, help="Student 2's lamb ramp gamma")
    parser.add_argument("--cross-weight", type=float, default=1.0,
                         help="weight on BOTH students' cross-teaching loss term")

    # Warmup: both students train INDEPENDENTLY (no self/cross split, no
    # cross-teaching at all) against a single shared FROZEN zero-shot CLIP
    # (no LoRA) for this many iterations first -- at iteration 0, neither
    # Teacher1 nor Teacher2 has anything meaningful to teach yet (Teacher2's
    # classifier head starts genuinely random), so bootstrapping from a
    # stable, always-reasonable frozen reference avoids each student
    # poisoning the other via cross-teaching before either is ready. AFTER
    # warmup, this frozen reference is dropped ENTIRELY (not blended in,
    # confirmed by prior experimentation on the PHPL branch: dropping the
    # frozen teacher outright from epoch 2 on beat a beta-blend with it) and
    # replaced by each student's own (by-then-adapted) EMA teacher for self,
    # plus the other pair's teacher for cross.
    parser.add_argument("--warmup-iters", type=int, default=200)
    # Warmup LR is a separate, smaller CONSTANT (PHPL's own WARMUP_CONS_LR
    # convention) for each of the 3 optimizers, not the main cosine schedule.
    parser.add_argument("--s1-warmup-lr", type=float, default=1e-5)
    parser.add_argument("--s2-lora-warmup-lr", type=float, default=1e-5)
    parser.add_argument("--s2-clf-warmup-lr", type=float, default=1e-5)

    args = parser.parse_args()
    cfg = build_cfg(args)
    device = torch.device(f"cuda:{cfg.GPU}" if torch.cuda.is_available() else "cpu")
    mkdir_if_missing(args.output_dir)
    mkdir_if_missing(osp.join(args.output_dir, "Teacher1"))
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    print("Building data loaders")
    dm = DataManager(cfg)
    train_loader_x = CyclingLoader(dm.train_loader_x)
    train_loader_u = CyclingLoader(dm.train_loader_u)
    test_loader = dm.test_loader
    classnames = dm.dataset.classnames
    num_classes = dm.num_classes
    is_vit = cfg.MODEL.BACKBONE.NAME.split('-')[0] == 'ViT'

    print("Building Student1/Teacher1 (PHPL-style)")
    student1, teacher1, lora1_s, lora1_t = _build_lora_pair(cfg, is_vit, classnames, device)

    print("Building Student2/Teacher2 (VLP-UDA-style)")
    student2_backbone, teacher2_backbone, lora2_s, lora2_t = _build_lora_pair(cfg, is_vit, classnames, device)
    feat_dim = student2_backbone.text_encoder.text_projection.shape[1]
    student2_head = _ClassifierHead(feat_dim, num_classes).to(device)
    teacher2_head = _ClassifierHead(feat_dim, num_classes).to(device)
    teacher2_head.load_state_dict(student2_head.state_dict())
    for param in teacher2_head.parameters():
        param.requires_grad_(False)

    print("Building frozen zero-shot CLIP (warmup reference only)")
    clip_frozen = load_clip_to_cpu(cfg)
    if cfg.TRAINER.PHPLMOMENTUM.PREC in ("fp32", "amp"):
        clip_frozen.float()
    teacher_frozen = _FrozenTeacherCLIP(cfg, classnames, clip_frozen).to(device)
    for param in teacher_frozen.parameters():
        param.requires_grad_(False)
    teacher_frozen.eval()

    optim1 = torch.optim.SGD(
        [p for p in student1.parameters() if p.requires_grad],
        lr=args.s1_lr, momentum=args.s1_momentum, weight_decay=args.s1_weight_decay,
    )
    optim2_backbone = torch.optim.SGD(
        [p for p in student2_backbone.parameters() if p.requires_grad],
        lr=args.s2_lora_lr, momentum=args.s2_lora_momentum, weight_decay=args.s2_lora_weight_decay,
    )
    if args.s2_clf_optim == "adam":
        optim2_head = torch.optim.Adam(student2_head.parameters(), lr=args.s2_clf_lr,
                                        weight_decay=args.s2_clf_weight_decay)
    else:
        optim2_head = torch.optim.SGD(student2_head.parameters(), lr=args.s2_clf_lr,
                                       momentum=0.9, nesterov=True, weight_decay=args.s2_clf_weight_decay)

    # Cosine schedules span the POST-warmup budget only -- during warmup, LR
    # is held at a separate constant instead (see the loop below), and the
    # scheduler is never .step()'d until warmup ends.
    post_warmup_iters = max(args.total_iters - args.warmup_iters, 1)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(optim1, T_max=post_warmup_iters)
    sched2_backbone = torch.optim.lr_scheduler.CosineAnnealingLR(optim2_backbone, T_max=post_warmup_iters)
    sched2_head = torch.optim.lr_scheduler.CosineAnnealingLR(optim2_head, T_max=post_warmup_iters)

    confi = cfg.TRAINER.PHPLMOMENTUM.CONFI
    best_acc1, best_acc2, best_acc_ens = 0.0, 0.0, 0.0

    for it in range(args.total_iters):
        batch_x = train_loader_x.next()
        batch_u = train_loader_u.next()
        image_x = batch_x["img"].to(device)
        label_x = batch_x["label"].to(device)
        image_u = batch_u["img"].to(device)
        label_u = batch_u["label"].to(device)

        in_warmup = it < args.warmup_iters

        # ---- source CE, both students ----
        logits1_x, _ = student1(image_x)
        loss_x1 = F.cross_entropy(logits1_x, label_x)

        _, feat2_x = student2_backbone(image_x)
        logits2_x = student2_head(feat2_x.float())
        loss_x2 = F.cross_entropy(logits2_x, label_x)

        def _masked_ce(logits, prob_ref):
            max_probs, pseudo_label = torch.max(prob_ref, dim=-1)
            mask = max_probs.ge(confi).float()
            epsilon = 1e-8
            return (F.cross_entropy(logits, pseudo_label, reduction="none") * mask).sum() / (mask.sum() + epsilon)

        logits1_u, _ = student1(image_u)
        _, feat2_u = student2_backbone(image_u)
        logits2_u = student2_head(feat2_u.float())
        pred2_u = F.softmax(logits2_u, dim=1)

        if in_warmup:
            # Single shared frozen reference, no self/cross split, no
            # cross-teaching at all yet.
            with torch.no_grad():
                logits_frozen_u, _ = teacher_frozen(image_u)
                prob_frozen_u = F.softmax(logits_frozen_u, dim=-1)

            loss_u1_self = _masked_ce(logits1_u, prob_frozen_u)
            loss_u1_cross = torch.tensor(0.0, device=device)
            loss1 = loss_x1 + loss_u1_self

            coe = _calibrated_coefficient(pred2_u, prob_frozen_u)
            pred_mix = 0.5 * (pred2_u + prob_frozen_u)
            lamb = 1.0  # no ramp during warmup -- there's only one (frozen) reference
            loss_u2_self = args.lambda1 * (
                _gini_impurity(pred2_u, coe) + _gini_impurity(pred_mix, 1 - coe)
            )
            loss_u2_cross = torch.tensor(0.0, device=device)
            loss2 = loss_x2 + loss_u2_self
        else:
            # ---- both teachers' target-domain predictions (frozen, no_grad) ----
            with torch.no_grad():
                logits_t1_u, _ = teacher1(image_u)
                prob_t1_u = F.softmax(logits_t1_u, dim=-1)

                _, feat_t2_u = teacher2_backbone(image_u)
                logits_t2_u = teacher2_head(feat_t2_u.float())
                prob_t2_u = F.softmax(logits_t2_u, dim=-1)

            # ---- Student 1: self (Teacher1) + cross (Teacher2) ----
            loss_u1_self = _masked_ce(logits1_u, prob_t1_u)
            loss_u1_cross = _masked_ce(logits1_u, prob_t2_u)
            loss1 = loss_x1 + loss_u1_self + args.cross_weight * loss_u1_cross

            # ---- Student 2: self (Teacher2) + cross (Teacher1) ----
            lamb = _cmkd_lamb(it - args.warmup_iters, post_warmup_iters, args.lamb_gamma)

            def _task_distill(pred_ref):
                coe = _calibrated_coefficient(pred2_u, pred_ref)
                pred_mix = 0.5 * (pred2_u + pred_ref)
                task = args.lambda1 * lamb * _gini_impurity(pred2_u, coe)
                distill = args.lambda1 * lamb * _gini_impurity(pred_mix, 1 - coe)
                return task + distill

            loss_u2_self = _task_distill(prob_t2_u)
            loss_u2_cross = _task_distill(prob_t1_u)
            loss2 = loss_x2 + loss_u2_self + args.cross_weight * loss_u2_cross

        if in_warmup:
            for pg in optim1.param_groups:
                pg["lr"] = args.s1_warmup_lr
            for pg in optim2_backbone.param_groups:
                pg["lr"] = args.s2_lora_warmup_lr
            for pg in optim2_head.param_groups:
                pg["lr"] = args.s2_clf_warmup_lr

        optim1.zero_grad()
        loss1.backward()
        optim1.step()

        optim2_backbone.zero_grad()
        optim2_head.zero_grad()
        loss2.backward()
        optim2_backbone.step()
        optim2_head.step()

        if not in_warmup:
            sched1.step()
            sched2_backbone.step()
            sched2_head.step()

        _ema_update_lora_params(teacher1, student1, lambda k: args.ema_momentum)
        _ema_update_lora_params(teacher2_backbone, student2_backbone, lambda k: args.ema_momentum)
        head_momentum = 0.0 if it < args.head_warmup_iters else args.head_ema_momentum
        _ema_update_module(teacher2_head, student2_head, head_momentum)

        if (it + 1) % args.print_freq == 0:
            acc_x1 = (logits1_x.argmax(-1) == label_x).float().mean().item() * 100
            acc_x2 = (logits2_x.argmax(-1) == label_x).float().mean().item() * 100
            print(
                f"iter [{it + 1}/{args.total_iters}] "
                f"loss1 {loss1.item():.4f} (x {loss_x1.item():.4f} self {loss_u1_self.item():.4f} "
                f"cross {loss_u1_cross.item():.4f}) acc_x1 {acc_x1:.2f} | "
                f"loss2 {loss2.item():.4f} (x {loss_x2.item():.4f} self {loss_u2_self.item():.4f} "
                f"cross {loss_u2_cross.item():.4f}) acc_x2 {acc_x2:.2f} lamb {lamb:.4f}"
            )

        if (it + 1) % args.eval_freq == 0 or (it + 1) == args.total_iters:
            acc1, acc2, acc_ens = evaluate(teacher1, teacher2_backbone, teacher2_head, test_loader, device)
            print(f"[eval] iter {it + 1}: Teacher1 {acc1:.2f}% | Teacher2 {acc2:.2f}% | Ensemble {acc_ens:.2f}%")

            save_lora(cfg, lora1_t, osp.join(args.output_dir, "Teacher1"), filename="LoRA-last")
            torch.save(teacher2_backbone.state_dict(), osp.join(args.output_dir, "teacher2_backbone-last.pt"))
            torch.save(teacher2_head.state_dict(), osp.join(args.output_dir, "teacher2_head-last.pt"))

            if acc1 > best_acc1:
                best_acc1 = acc1
                save_lora(cfg, lora1_t, osp.join(args.output_dir, "Teacher1"), filename="LoRA-best")
            if acc2 > best_acc2:
                best_acc2 = acc2
                torch.save(teacher2_backbone.state_dict(), osp.join(args.output_dir, "teacher2_backbone-best.pt"))
                torch.save(teacher2_head.state_dict(), osp.join(args.output_dir, "teacher2_head-best.pt"))
            if acc_ens > best_acc_ens:
                best_acc_ens = acc_ens
                print(f"  new best ensemble ({best_acc_ens:.2f}%)")

    print(f"Done. best Teacher1={best_acc1:.2f}% Teacher2={best_acc2:.2f}% Ensemble={best_acc_ens:.2f}%")


if __name__ == "__main__":
    main()
