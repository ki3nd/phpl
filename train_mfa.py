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
  - Student 2 / Teacher 2: VLP-UDA-style -- a SEPARATE CLIP whose vision
    encoder is finetuned DIRECTLY (no LoRA, matching VLP-UDA's own
    make_model.py) plus a learned classifier_layer (train_cmkd.py's
    _ClassifierHead: BatchNorm1d -> LayerNorm -> Linear), trained jointly
    (not the frozen-backbone phase-2 split train_cmkd.py uses -- here the
    backbone trains too).

Per-branch hyperparameters live in a YAML file (--hparams-config, default
configs/mfa/hparams.yaml) with "s1"/"s2" sections, loaded as argparse
defaults -- any flag passed explicitly on the command line still overrides
the YAML value.

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
train_cmkd.py's own docstring for why. Student 1 and Student 2 train at
DIFFERENT rates (see --s1-total-iters/--s2-per-s1 below), each pulling its
own batches from the same underlying CyclingLoader stream -- NOT the same
literal batch each step, unlike MFA's own trainer.py (which runs both nets
1:1 and does share a batch); see the "macro-step" structure below for why.

Usage:
    python train_mfa.py \\
        --root /kaggle/working/data --domains a-c --backbone ViT-B/16 \\
        --dataset-config-file configs/datasets/officehome.yaml \\
        --config-file configs/trainers/PHPL/b32_ep10_officehome.yaml \\
        --output-dir output/MFA/officehome/a-c \\
        --s1-total-iters 1000 --s2-per-s1 10 --print-freq 50 --eval-freq 200
"""
import argparse
import math
import os.path as osp

import torch
import yaml
from torch.nn import functional as F
from torchvision.transforms import (
    Compose, Normalize, RandomCrop, RandomHorizontalFlip, Resize, ToTensor,
)
from torchvision.transforms.functional import InterpolationMode

from dassl.data import DataManager
from dassl.utils import mkdir_if_missing, set_random_seed

from utils.MK_MMD import MK_MMD

from utils.clip_part import load_clip_to_cpu
from loralib.utils import apply_lora, apply_lora_rn, save_lora
from trainers.da.phpl_momentum import (
    CustomCLIP, _FrozenTeacherCLIP, _copy_lora_params, _ema_update_lora_params,
)
from train_cmkd import build_cfg, CyclingLoader, _gini_impurity, _calibrated_coefficient, _ClassifierHead


def _load_hparams_as_defaults(parser, path):
    """Load the s1/s2 sections of --hparams-config as argparse defaults --
    any flag passed explicitly on the command line still overrides these
    (argparse only falls back to a default when the flag is omitted).
    parser.set_defaults() silently accepts unknown dest names (they'd just
    never be read), so a YAML key that doesn't match any registered
    --s1-*/--s2-* flag (typo, or a flag renamed later without updating the
    YAML) is validated here instead of failing silently."""
    known = {action.dest for action in parser._actions}
    with open(path) as f:
        hp = yaml.safe_load(f) or {}
    flat = {}
    for branch in ("s1", "s2"):
        for k, v in (hp.get(branch) or {}).items():
            dest = f"{branch}_{k}"
            if dest not in known:
                raise ValueError(
                    f"{path}: '{branch}.{k}' does not match any registered "
                    f"--{branch}-{k.replace('_', '-')} flag -- typo, or the "
                    f"flag was renamed without updating this YAML?"
                )
            flat[dest] = v
    parser.set_defaults(**flat)


@torch.no_grad()
def _ema_update_module(ema_module, src_module, momentum):
    """Generic parameter+buffer-wise EMA for a plain nn.Module (Student 2's
    classifier head isn't LoRA, so _ema_update_lora_params doesn't apply).
    BatchNorm1d's state_dict includes num_batches_tracked, a Long counter --
    not a learned value, can't take a float EMA -- hard-copied instead."""
    ema_state = ema_module.state_dict()
    for k, v in src_module.state_dict().items():
        if not torch.is_floating_point(v):
            ema_state[k].copy_(v)
        else:
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


def _build_finetune_pair(cfg, classnames, device):
    """Student2/Teacher2 backbone -- plain CustomCLIP, NO LoRA. Matches
    VLP-UDA's own make_model.py: only the vision encoder (image_encoder =
    clip_model.visual) is finetuned directly; text_encoder/logit_scale stay
    frozen (VLP-UDA doesn't train them either). Teacher is a full-weight EMA
    of the vision encoder (temporal fusion), hard-copied from student at
    init."""
    clip_student = load_clip_to_cpu(cfg)
    clip_teacher = load_clip_to_cpu(cfg)
    if cfg.TRAINER.PHPLMOMENTUM.PREC in ("fp32", "amp"):
        clip_student.float()
        clip_teacher.float()

    student = CustomCLIP(cfg, classnames, clip_student)
    teacher = _FrozenTeacherCLIP(cfg, classnames, clip_teacher)

    for param in student.parameters():
        param.requires_grad_(False)
    for param in student.image_encoder.parameters():
        param.requires_grad_(True)
    for param in teacher.parameters():
        param.requires_grad_(False)

    teacher.image_encoder.load_state_dict(student.image_encoder.state_dict())
    student.to(device)
    teacher.to(device)
    teacher.eval()
    return student, teacher


@torch.no_grad()
def evaluate(teacher1, teacher2_backbone, teacher2_head, test_loader, device):
    """teacher2_head is expected to already be permanently in eval mode (set
    once at construction, in main()) -- NOT toggled here, unlike a typical
    eval() helper, since it must also run in eval mode as a reference DURING
    training (see main()'s loop), not just here."""
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
    parser.add_argument("--seed", type=int, default=42)

    # Student 1 (PHPL) typically converges in ~750-1000 iterations; Student 2
    # (VLP-UDA-style, learning its classifier_layer from scratch) wants far
    # more -- VLP-UDA's own recipe uses 10000. Rather than force both onto one
    # shared iteration count (either starving Student 2 or over-training
    # Student 1 well past its usual sweet spot), each "macro-step" runs:
    #   1. Snapshot BOTH teachers' current target-domain predictions BEFORE
    #      either student updates this macro-step -- simulates running both
    #      branches in parallel (matches MFA's own trainer.py: it queries
    #      both mean_models before backward/step on EITHER optimizer, even
    #      though MFA itself only ever runs both nets at the same 1:1 rate).
    #   2. Student 2 runs --s2-per-s1 micro-iterations, EACH with its own
    #      immediate EMA update (never batched/delayed) -- ALL of them using
    #      the SAME snapshotted Teacher1 prediction from step 1 (Teacher1
    #      truly hasn't changed meanwhile -- not an artificial cache).
    #   3. Student 1 runs exactly 1 iteration, using the SAME snapshotted
    #      Teacher2 prediction from step 1 (NOT Teacher2's post-burst state --
    #      that would make Student 1 see the future relative to Student 2's
    #      snapshot of Teacher1), then its own immediate EMA update.
    parser.add_argument("--s1-total-iters", type=int, default=1000)
    parser.add_argument("--s2-per-s1", type=int, default=10)
    parser.add_argument("--print-freq", type=int, default=50, help="in macro-steps")
    parser.add_argument("--eval-freq", type=int, default=200, help="in macro-steps")
    parser.add_argument("--hparams-config", type=str, default="configs/mfa/hparams.yaml",
                         help="YAML file with s1/s2 per-branch hyperparameter defaults -- "
                              "see _load_hparams_as_defaults()")

    # Student 1 (PHPL-style, LoRA). Plain SGD, matching PHPL's own recipe.
    # Defaults come from --hparams-config's "s1" section (see below);
    # add_argument's own default= here is just a fallback if that file is
    # missing/edited, not the real source of truth.
    parser.add_argument("--s1-lr", type=float, default=0.0035)
    parser.add_argument("--s1-momentum", type=float, default=0.9)
    parser.add_argument("--s1-weight-decay", type=float, default=5e-4)
    # PHPL's own default (MMD_WEIGHT=1.0, always on) -- a domain-alignment
    # term between Student1's source/target features.
    parser.add_argument("--s1-mmd-weight", type=float, default=1.0)
    parser.add_argument("--s1-warmup-lr", type=float, default=0.001)
    parser.add_argument("--s1-cross-weight", type=float, default=0.5,
                         help="weight on Student 1's cross-teaching loss term")
    parser.add_argument("--s1-ema-momentum", type=float, default=0.996,
                         help="Teacher1's LoRA EMA momentum")

    # Student 2 (VLP-UDA-style, full backbone finetune -- no LoRA). Defaults
    # come from --hparams-config's "s2" section, matching VLP-UDA's own
    # configs/office_home.yaml (ViT-B) -- NOT its argparse script defaults,
    # which are for a different config entirely.
    parser.add_argument("--s2-lr", type=float, default=3e-6, help="vision-encoder LR")
    parser.add_argument("--s2-multiple-lr-classifier", type=float, default=1000,
                         help="classifier head LR = --s2-lr * this")
    parser.add_argument("--s2-lr-gamma", type=float, default=0.0003)
    parser.add_argument("--s2-lr-decay", type=float, default=0.75)
    parser.add_argument("--s2-weight-decay", type=float, default=5e-4)
    parser.add_argument("--s2-momentum", type=float, default=0.9)
    parser.add_argument("--s2-label-smoothing", type=float, default=0.1,
                         help="applied to loss_x2 (classifier CE on source) only, not reg_loss's "
                              "cosine-branch CE -- matches VLP-UDA's own clf_loss/regularization_term")
    parser.add_argument("--s2-lambda1", type=float, default=0.25, help="Student 2's CMKD task/distill weight")
    # CMKD's own reg_loss (models/cmkd.py's regularization_term) -- trains the
    # cosine branch itself (CE on source with its own label + entropy-min on
    # target).
    parser.add_argument("--s2-lambda2", type=float, default=0.1, help="Student 2's CMKD reg_loss: CE weight on source cosine branch")
    parser.add_argument("--s2-lambda3", type=float, default=0.025, help="Student 2's CMKD reg_loss: entropy-min weight on target cosine branch")
    # VLP-UDA's own LambdaSheduler default -- lamb never reaches 1.0 with
    # this (maxes out at ~0.46, at the very end of training).
    parser.add_argument("--s2-lamb-gamma", type=float, default=1.0, help="Student 2's lamb ramp gamma")
    parser.add_argument("--s2-no-lamb-ramp", action="store_true",
                         help="skip the lamb ramp entirely -- lamb=1.0 for the whole post-warmup "
                              "run, so the entropy-min terms are just a constant lambda1 weight")
    parser.add_argument("--s2-cross-weight", type=float, default=0.5,
                         help="weight on Student 2's cross-teaching loss term")
    parser.add_argument("--s2-self-from-student", action="store_true",
                         help="self-loss reference is student2_backbone's OWN live cosine branch "
                              "(detached) instead of Teacher2's EMA -- matches pure VLP-UDA's real "
                              "design exactly (models/cmkd.py + make_model.py: self.teacher_model "
                              "is declared but never used in forward(), only for --rst checkpoint "
                              "saving). Off by default (Teacher2 EMA reference, current behavior). "
                              "Teacher2's own EMA update still runs either way -- this only changes "
                              "which one feeds the self-loss.")
    parser.add_argument("--s2-cross-mode", choices=["mask", "gini"], default="mask",
                         help="Student 2's cross-teaching mechanism: 'mask' = Student1-style "
                              "CONFI hard-threshold mask + CE (current default); 'gini' = CMKD's "
                              "own task/distill gini-impurity formula (same as Student 2's self "
                              "loss), using Teacher1's prediction as the reference -- Teacher1's "
                              "cosine-similarity branch is naturally sharp (logit_scale~100), so "
                              "it fits the sharp-reference role CMKD's calibrated_coefficient "
                              "expects.")
    parser.add_argument("--s2-backbone-ema-momentum", type=float, default=0.996,
                         help="Teacher2-backbone's full-weight (no LoRA) EMA momentum")
    parser.add_argument("--s2-head-ema-momentum", type=float, default=0.996,
                         help="Teacher2-head's EMA momentum, active AFTER --s2-head-warmup-iters")
    parser.add_argument("--s2-head-warmup-iters", type=int, default=100,
                         help="Teacher2-head hard-copies Student 2's head every step for this "
                              "many iterations (momentum=0) before switching to "
                              "--s2-head-ema-momentum -- mirrors the epoch-1 hard-copy lesson "
                              "learned for a randomly initialized head elsewhere in this project.")

    # Warmup: both students train INDEPENDENTLY (no self/cross split, no
    # cross-teaching at all) against a single shared FROZEN zero-shot CLIP
    # for this many macro-steps first -- at iteration 0, neither Teacher1 nor
    # Teacher2 has anything meaningful to teach yet (Teacher2's classifier
    # head starts genuinely random), so bootstrapping from a stable,
    # always-reasonable frozen reference avoids each student poisoning the
    # other via cross-teaching before either is ready. AFTER warmup, this
    # frozen reference is dropped ENTIRELY (not blended in, confirmed by
    # prior experimentation on the PHPL branch: dropping the frozen teacher
    # outright from epoch 2 on beat a beta-blend with it) and replaced by
    # each student's own (by-then-adapted) EMA teacher for self, plus the
    # other pair's teacher for cross.
    # Student 1 gets a separate CONSTANT warmup LR (--s1-warmup-lr, not its
    # main cosine schedule) -- Student 2 does NOT: it keeps stepping its own
    # VLP-UDA-style LR schedule (--s2-lr-gamma/--s2-lr-decay) from iteration
    # 0 straight through warmup, since VLP-UDA's own recipe has no separate
    # warmup-LR concept at all.
    # In macro-steps -- 100 for Student 1, which (at the default --s2-per-s1=10)
    # naturally gives Student 2 exactly 1000 warmup micro-iterations too.
    parser.add_argument("--warmup-iters", type=int, default=100)

    # Both students share ONE DataManager/transform (see below) -- there's no
    # way to give Student 2 VLP-UDA's own augmentation without also changing
    # Student 1's (which is already learning well on dassl's default
    # random_resized_crop). This flag switches BOTH to VLP-UDA's own
    # augmentation as an experiment (off by default -- dassl's current
    # transform, unchanged, for both).
    parser.add_argument("--vlpuda-augment", action="store_true",
                         help="use VLP-UDA's own train/test transforms (Resize(256,256)->"
                              "RandomCrop(224)->RandomHorizontalFlip, bilinear; test: direct "
                              "Resize(224,224), no CenterCrop) for BOTH students, instead of "
                              "dassl's default (random_resized_crop scale=(0.08,1.0), bicubic; "
                              "test: Resize+CenterCrop)")

    _load_hparams_as_defaults(parser, parser.parse_known_args()[0].hparams_config)
    args = parser.parse_args()
    cfg = build_cfg(args)
    device = torch.device(f"cuda:{cfg.GPU}" if torch.cuda.is_available() else "cpu")
    mkdir_if_missing(args.output_dir)
    mkdir_if_missing(osp.join(args.output_dir, "Teacher1"))
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    print("Building data loaders")
    if args.vlpuda_augment:
        print("Using VLP-UDA's own train/test transforms (--vlpuda-augment)")
        normalize = Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
        crop_size = cfg.INPUT.SIZE[0]
        tfm_train = Compose([
            Resize([256, 256], interpolation=InterpolationMode.BILINEAR),
            RandomCrop(crop_size),
            RandomHorizontalFlip(),
            ToTensor(),
            normalize,
        ])
        tfm_test = Compose([
            Resize([crop_size, crop_size], interpolation=InterpolationMode.BILINEAR),
            ToTensor(),
            normalize,
        ])
        dm = DataManager(cfg, custom_tfm_train=tfm_train, custom_tfm_test=tfm_test)
    else:
        dm = DataManager(cfg)
    train_loader_x = CyclingLoader(dm.train_loader_x)
    train_loader_u = CyclingLoader(dm.train_loader_u)
    test_loader = dm.test_loader
    classnames = dm.dataset.classnames
    num_classes = dm.num_classes
    is_vit = cfg.MODEL.BACKBONE.NAME.split('-')[0] == 'ViT'

    print("Building Student1/Teacher1 (PHPL-style)")
    student1, teacher1, lora1_s, lora1_t = _build_lora_pair(cfg, is_vit, classnames, device)

    print("Building Student2/Teacher2 (VLP-UDA-style, full backbone finetune)")
    student2_backbone, teacher2_backbone = _build_finetune_pair(cfg, classnames, device)
    feat_dim = student2_backbone.text_encoder.text_projection.shape[1]
    student2_head = _ClassifierHead(feat_dim, num_classes).to(device)
    teacher2_head = _ClassifierHead(feat_dim, num_classes).to(device)
    teacher2_head.load_state_dict(student2_head.state_dict())
    for param in teacher2_head.parameters():
        param.requires_grad_(False)
    # Permanently in eval mode -- it's used as a frozen reference (self/cross)
    # DURING training too (see the main loop below), not just at eval time.
    # Left in .train() mode, its BatchNorm1d would normalize against each
    # reference call's own batch statistics instead of stable running
    # statistics, and would keep updating those running stats from every
    # such call -- both wrong for something meant to be a stable teacher.
    teacher2_head.eval()

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
    # Single SGD, 2 param groups, matching VLP-UDA's own get_optimizer()/
    # get_parameters() exactly (nesterov=True for BOTH groups).
    optim2 = torch.optim.SGD(
        [
            {"params": student2_backbone.image_encoder.parameters(), "lr": args.s2_lr},
            {"params": student2_head.parameters(), "lr": args.s2_multiple_lr_classifier * args.s2_lr},
        ],
        momentum=args.s2_momentum, weight_decay=args.s2_weight_decay, nesterov=True,
    )

    s2_total_iters = args.s1_total_iters * args.s2_per_s1
    s1_post_warmup = max(args.s1_total_iters - args.warmup_iters, 1)

    # Student 1's cosine schedule spans its OWN post-warmup budget -- during
    # warmup, LR is held at a separate constant instead (see the loop below),
    # and the scheduler is never .step()'d until warmup ends.
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(optim1, T_max=s1_post_warmup)
    # Student 2's schedule is VLP-UDA's own LambdaLR decay ((1+gamma*step)^
    # -decay), stepped every micro-iteration from iteration 0 -- no warmup
    # pause, matching VLP-UDA's own recipe (it has no warmup-LR concept).
    sched2 = torch.optim.lr_scheduler.LambdaLR(
        optim2, lr_lambda=lambda step: (1.0 + args.s2_lr_gamma * step) ** (-args.s2_lr_decay)
    )

    confi = cfg.TRAINER.PHPLMOMENTUM.CONFI
    best_acc1, best_acc2, best_acc_ens = 0.0, 0.0, 0.0

    def _masked_ce(logits, prob_ref):
        max_probs, pseudo_label = torch.max(prob_ref, dim=-1)
        mask = max_probs.ge(confi).float()
        epsilon = 1e-8
        return (F.cross_entropy(logits, pseudo_label, reduction="none") * mask).sum() / (mask.sum() + epsilon)

    def _task_distill(pred_own, pred_ref, lamb):
        coe = _calibrated_coefficient(pred_own, pred_ref)
        pred_mix = 0.5 * (pred_own + pred_ref)
        task = args.s2_lambda1 * lamb * _gini_impurity(pred_own, coe)
        distill = args.s2_lambda1 * lamb * _gini_impurity(pred_mix, 1 - coe)
        return task + distill

    s2_it_global = 0
    for macro in range(args.s1_total_iters):
        in_warmup = macro < args.warmup_iters

        # ---- Student 1's own batch, used both for its own step below AND to
        # snapshot Teacher2's cross-reference BEFORE Student 2's burst runs
        # (simulates both branches running in parallel -- see the CLI help
        # text above for --s1-total-iters). ----
        batch_x1 = train_loader_x.next()
        batch_u1 = train_loader_u.next()
        image_x1 = batch_x1["img"].to(device)
        label_x1 = batch_x1["label"].to(device)
        image_u1 = batch_u1["img"].to(device)

        with torch.no_grad():
            if in_warmup:
                logits_cross_for_s1, _ = teacher_frozen(image_u1)
            else:
                _, feat_cross_for_s1 = teacher2_backbone(image_u1)
                logits_cross_for_s1 = teacher2_head(feat_cross_for_s1.float())
            prob_cross_for_s1 = F.softmax(logits_cross_for_s1, dim=-1)

        # ---- Student 2 burst: --s2-per-s1 micro-iterations, own immediate
        # EMA update after EACH one -- Teacher1 is untouched throughout (it
        # only updates after Student 1's single step below), so every
        # micro-step here naturally uses the SAME Teacher1 snapshot. ----
        for _ in range(args.s2_per_s1):
            batch_x2 = train_loader_x.next()
            batch_u2 = train_loader_u.next()
            image_x2 = batch_x2["img"].to(device)
            label_x2 = batch_x2["label"].to(device)
            image_u2 = batch_u2["img"].to(device)

            logits2_x2_clip, feat2_x2 = student2_backbone(image_x2)
            logits2_x2 = student2_head(feat2_x2.float())
            # label_smoothing matches VLP-UDA's own clf_loss -- applies ONLY
            # to this classifier CE, not reg_loss's cosine-branch CE below.
            loss_x2 = F.cross_entropy(logits2_x2, label_x2, label_smoothing=args.s2_label_smoothing)

            logits2_u2_clip, feat2_u2 = student2_backbone(image_u2)
            logits2_u2 = student2_head(feat2_u2.float())
            pred2_u2 = F.softmax(logits2_u2, dim=1)

            # Student 2's self/reg loss is IDENTICAL during and after warmup
            # -- VLP-UDA's own recipe has no warmup concept at all, and
            # Teacher2 (EMA) is already a stable, near-zero-shot reference
            # from iteration 0 (no LoRA now, so it starts as an exact copy of
            # pretrained CLIP, same as teacher_frozen would give). Only
            # cross-teaching (Teacher1) is skipped during warmup -- Teacher1
            # itself is still bootstrapping in this window (see Student 1's
            # own warmup below), not a reliable reference yet.
            #
            # Self-loss reference is Teacher2's (EMA) cosine branch, NOT
            # student2_backbone's own live one -- both are detached either
            # way (no gradient flows through the self-loss reference
            # regardless), so this only affects target stability: Teacher2's
            # EMA-smoothed weights give a steadier target than the live
            # student's (which shifts slightly every micro-step on a fresh
            # batch) -- "temporal fusion" applied to Student2's self loss.
            # reg_loss still trains student2_backbone's OWN cosine branch
            # directly (pred_self2_clip_live, undetached) -- unaffected.
            pred_self2_clip_live = F.softmax(logits2_u2_clip, dim=-1)
            if args.s2_self_from_student:
                # --s2-self-from-student: matches pure VLP-UDA exactly (no
                # teacher/EMA involved in self-training at all).
                prob_self2 = pred_self2_clip_live.detach()
            else:
                with torch.no_grad():
                    logits_self2_teacher, _ = teacher2_backbone(image_u2)
                    prob_self2 = F.softmax(logits_self2_teacher, dim=-1)

            # Continuous ramp across the WHOLE run (including warmup) from
            # iteration 0, matching VLP-UDA's own LambdaSheduler -- no reset
            # at warmup's end.
            lamb = 1.0 if args.s2_no_lamb_ramp else _cmkd_lamb(s2_it_global, s2_total_iters, args.s2_lamb_gamma)
            loss_u2_self = _task_distill(pred2_u2, prob_self2, lamb)
            # CMKD's own reg_loss (models/cmkd.py's regularization_term):
            # trains the cosine branch itself directly -- CE on source with
            # its true label, plus entropy-min on target.
            reg_loss = (
                args.s2_lambda2 * F.cross_entropy(logits2_x2_clip, label_x2)
                + args.s2_lambda3 * lamb * _gini_impurity(pred_self2_clip_live)
            )

            if in_warmup:
                loss_u2_cross = torch.tensor(0.0, device=device)
                loss2 = loss_x2 + loss_u2_self + reg_loss
            else:
                with torch.no_grad():
                    logits_t1_u2, _ = teacher1(image_u2)
                    prob_cross2 = F.softmax(logits_t1_u2, dim=-1)
                if args.s2_cross_mode == "gini":
                    # CMKD's own task/distill formula, same as self, but with
                    # Teacher1 (naturally sharp -- logit_scale~100) as the
                    # reference instead of Teacher2's own EMA.
                    loss_u2_cross = _task_distill(pred2_u2, prob_cross2, lamb)
                else:
                    # Student1-style CONFI=0.85 hard-threshold mask + CE.
                    loss_u2_cross = _masked_ce(logits2_u2, prob_cross2)
                loss2 = loss_x2 + loss_u2_self + args.s2_cross_weight * loss_u2_cross + reg_loss

            optim2.zero_grad()
            loss2.backward()
            optim2.step()
            # No warmup pause -- VLP-UDA's own LR schedule runs continuously
            # from iteration 0 (see --s2-lr-gamma/--s2-lr-decay above).
            sched2.step()

            _ema_update_module(teacher2_backbone.image_encoder, student2_backbone.image_encoder,
                                args.s2_backbone_ema_momentum)
            head_momentum = 0.0 if s2_it_global < args.s2_head_warmup_iters else args.s2_head_ema_momentum
            _ema_update_module(teacher2_head, student2_head, head_momentum)
            s2_it_global += 1

        # ---- Student 1's own step, using image_u1/prob_cross_for_s1 computed
        # BEFORE the Student 2 burst above. ----
        logits1_x, feat_x1 = student1(image_x1)
        loss_x1 = F.cross_entropy(logits1_x, label_x1)
        logits1_u, feat_u1 = student1(image_u1)
        loss_mmd1 = MK_MMD(feat_x1, feat_u1)

        if in_warmup:
            loss_u1_self = _masked_ce(logits1_u, prob_cross_for_s1)
            loss_u1_cross = torch.tensor(0.0, device=device)
            loss1 = loss_x1 + loss_u1_self + args.s1_mmd_weight * loss_mmd1
        else:
            with torch.no_grad():
                logits_t1_self, _ = teacher1(image_u1)
                prob_self1 = F.softmax(logits_t1_self, dim=-1)
            loss_u1_self = _masked_ce(logits1_u, prob_self1)
            loss_u1_cross = _masked_ce(logits1_u, prob_cross_for_s1)
            loss1 = (
                loss_x1 + loss_u1_self + args.s1_cross_weight * loss_u1_cross
                + args.s1_mmd_weight * loss_mmd1
            )

        if in_warmup:
            for pg in optim1.param_groups:
                pg["lr"] = args.s1_warmup_lr

        optim1.zero_grad()
        loss1.backward()
        # PHPL's own model_backward_and_update_with_gradient_monitoring always
        # clips to max_norm=20.0 (trainers/baseda.py) -- missing here before.
        torch.nn.utils.clip_grad_norm_(
            [p for p in student1.parameters() if p.requires_grad], max_norm=20.0
        )
        optim1.step()
        if not in_warmup:
            sched1.step()

        _ema_update_lora_params(teacher1, student1, lambda k: args.s1_ema_momentum)

        if (macro + 1) % args.print_freq == 0:
            acc_x1 = (logits1_x.argmax(-1) == label_x1).float().mean().item() * 100
            acc_x2 = (logits2_x2.argmax(-1) == label_x2).float().mean().item() * 100
            print(
                f"macro [{macro + 1}/{args.s1_total_iters}] (s2 iter {s2_it_global}/{s2_total_iters}) "
                f"loss1 {loss1.item():.4f} (x {loss_x1.item():.4f} self {loss_u1_self.item():.4f} "
                f"cross {loss_u1_cross.item():.4f} mmd {loss_mmd1.item():.4f}) acc_x1 {acc_x1:.2f} | "
                f"loss2 {loss2.item():.4f} (x {loss_x2.item():.4f} self {loss_u2_self.item():.4f} "
                f"cross {loss_u2_cross.item():.4f} reg {reg_loss.item():.4f}) acc_x2 {acc_x2:.2f} lamb {lamb:.4f}"
            )

        if (macro + 1) % args.eval_freq == 0 or (macro + 1) == args.s1_total_iters or (macro + 1) == args.warmup_iters:
            acc1, acc2, acc_ens = evaluate(teacher1, teacher2_backbone, teacher2_head, test_loader, device)
            tag = " [end of warmup]" if (macro + 1) == args.warmup_iters else ""
            print(f"[eval] macro {macro + 1}{tag}: Teacher1 {acc1:.2f}% | Teacher2 {acc2:.2f}% | Ensemble {acc_ens:.2f}%")

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
