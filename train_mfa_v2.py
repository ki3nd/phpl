"""
Diagnostic v2 of the MFA-style cross-model co-training trainer (see
train_mfa.py's own docstring for the full architecture/rationale) -- the
ONLY thing that changes here is Student 2 / Teacher 2: instead of
reimplementing VLP-UDA's model/loss (train_mfa.py's _ClassifierHead/
_task_distill/_gini_impurity/_calibrated_coefficient, all hand-ported into
this project), Student 2 here IS the real, untouched
vlpuda_pure/models/make_model.py's TransferNet + models/cmkd.py's CMKD,
imported directly from the vlpuda_pure/ clone (github.com/Wenlve-Zhou/
VLP-UDA) sitting next to this file. This exists to eliminate any doubt
that a reimplementation bug (rather than the cross-model architecture
itself) explains Student 2 underperforming a pure VLP-UDA run.

Student 1 / Teacher 1 (PHPL-style LoRA) is COMPLETELY UNCHANGED from
train_mfa.py -- same classes, same loss, same optimizer, own dassl
DataManager. Student 2 / Teacher 2 now:
  - Model: vlpuda_pure's own TransferNet (CLIP ViT-B/16 backbone finetuned
    directly, no LoRA, same as train_mfa.py's design -- but here it's
    their actual class, not a reimplementation).
  - Self-loss: TransferNet.forward()'s own (clf_loss, transfer_loss) --
    called AS-IS, untouched. This already IS VLP-UDA's real CMKD self-
    training (no teacher/EMA involved at all in this loss, by design --
    see cmkd.py).
  - Cross-loss (Teacher1 -> Student2, Teacher2 -> Student1): NOT part of
    CMKD -- added on top here, same mechanism as train_mfa.py (CONFI
    hard-threshold mask + CE, --s2-cross-mode mask/gini).
  - "Teacher2" (temporal fusion / EMA): vlpuda_pure's own teacher_model
    (models/make_model.py) was hard-copied at init but NEVER updated in
    the original codebase (only read by --rst) -- vlpuda_pure/utils/
    tools.py's ema_update_teacher (added in this project) gives it a real
    EMA update every Student 2 micro-iteration. vlpuda_pure's own design
    has no separate classifier-head EMA at all (only one classifier_layer
    exists) -- this project adds one anyway (teacher_classifier_layer, a
    deepcopy of model2.classifier_layer), so "Teacher2" = EMA-backbone
    features fed through this separate EMA head, not the live one. Both
    the backbone's and the head's EMA share the SAME --s2-ema-warmup-iters
    window (hard-copy together, then switch to their own real momentum
    together) -- otherwise Teacher2 would be a smoothed backbone paired
    with an instantaneously-tracking head (or vice versa) during that
    window, since the head is trained against the LIVE backbone's
    features, not the lagging EMA one.
  - Data: both branches use dassl's own DataManager (each its own SEPARATE
    instance, own independent shuffled stream -- see train_mfa.py's own
    loader-independence fix), with VLP-UDA's own transform (Resize(256,256)
    -> RandomCrop(224) -> RandomHorizontalFlip, bilinear; test: direct
    Resize, no CenterCrop) always on for both, matching train_mfa.py's
    --vlpuda-augment. dassl reads the exact same on-disk class-subfolder
    layout vlpuda_pure's own ImageFolder-based loader would, so this needed
    no changes to how the dataset is organized on disk. evaluate() uses ONE
    shared test_loader for BOTH branches, so Teacher1/Teacher2 are
    compared/ensembled on the exact same image each time.

vlpuda_pure/'s own top-level package names (`models`, `utils`, `clip`)
collide with this project's own same-named packages -- see
_import_vlpuda_pure() below for how that's kept from clobbering either
side's imports.

Usage:
    python train_mfa_v2.py \\
        --root /kaggle/working/data --domains a-c --backbone ViT-B/16 \\
        --dataset-config-file configs/datasets/officehome.yaml \\
        --config-file configs/trainers/PHPL/b32_ep10_officehome.yaml \\
        --output-dir output/MFA_v2/officehome/a-c \\
        --s1-total-iters 1000 --s2-per-s1 10 --print-freq 50 --eval-freq 200
"""
import argparse
import copy
import importlib
import os.path as osp
import sys

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
from train_cmkd import build_cfg, CyclingLoader

_VLPUDA_DIR = osp.join(osp.dirname(osp.abspath(__file__)), "vlpuda_pure")
_SHADOWED_PREFIXES = ("models", "utils", "clip")


def _import_vlpuda_pure():
    """vlpuda_pure/'s internal absolute imports (`from models... import`,
    `from utils... import`, `import clip`) use the SAME top-level names as
    this project's own `utils`/`clip` packages (this project has no top-
    level `models` package, but guard for it anyway). Importing naively
    with vlpuda_pure/ on sys.path would either fail outright (Python
    reuses whatever `utils`/`clip` module this project already cached in
    sys.modules) or, worse, silently bind THIS project's later `import
    utils.MK_MMD`/`import clip` calls to vlpuda_pure's fork instead of its
    own -- so this does the import in an isolated sys.path/sys.modules
    scope, and restores exactly what was there before on the way out."""
    def _matches(name):
        return name in _SHADOWED_PREFIXES or name.split(".", 1)[0] in _SHADOWED_PREFIXES

    saved_modules = {k: v for k, v in sys.modules.items() if _matches(k)}
    for k in saved_modules:
        del sys.modules[k]

    saved_path = sys.path[:]
    sys.path.insert(0, _VLPUDA_DIR)
    try:
        make_model = importlib.import_module("models.make_model")
        tools = importlib.import_module("utils.tools")
    finally:
        sys.path[:] = saved_path
        for k in [k for k in sys.modules if _matches(k)]:
            del sys.modules[k]
        sys.modules.update(saved_modules)

    return make_model, tools


_vlpuda_make_model, _vlpuda_tools = _import_vlpuda_pure()
TransferNet = _vlpuda_make_model.TransferNet
ema_update_teacher = _vlpuda_tools.ema_update_teacher


def _load_hparams_as_defaults(parser, path):
    """Same mechanism as train_mfa.py's own helper -- see there for the
    full rationale. Kept as a separate copy (not imported from train_mfa.py)
    since this script is meant to have nothing to do with train_mfa.py."""
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


def _build_lora_pair(cfg, is_vit, classnames, device):
    """Student 1 / Teacher 1 -- identical to train_mfa.py's own helper."""
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
def evaluate(teacher1, model2, teacher_classifier_layer, test_loader, device):
    """Both Teacher1 and Teacher2 (model2.teacher_model's EMA backbone
    features through teacher_classifier_layer, its own EMA head) look at
    the exact same image each batch (ONE shared dassl test_loader), so the
    ensemble is a fair average."""
    correct1, correct2, correct_ens, total = 0, 0, 0, 0
    for batch in test_loader:
        image = batch["img"].to(device)
        label = batch["label"].to(device)

        logits1, _ = teacher1(image)
        prob1 = F.softmax(logits1, dim=-1)

        feat2_teacher = model2.teacher_model.forward_features(image)
        logits2 = teacher_classifier_layer(feat2_teacher)
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

    parser.add_argument("--s1-total-iters", type=int, default=1000)
    parser.add_argument("--s2-per-s1", type=int, default=10)
    parser.add_argument("--print-freq", type=int, default=50, help="in macro-steps")
    parser.add_argument("--eval-freq", type=int, default=200, help="in macro-steps")
    parser.add_argument("--hparams-config", type=str, default="configs/mfa/hparams_v2.yaml")
    parser.add_argument("--warmup-iters", type=int, default=100)

    # Student 1 -- identical set of flags to train_mfa.py.
    parser.add_argument("--s1-lr", type=float, default=0.0035)
    parser.add_argument("--s1-momentum", type=float, default=0.9)
    parser.add_argument("--s1-weight-decay", type=float, default=5e-4)
    parser.add_argument("--s1-mmd-weight", type=float, default=1.0)
    parser.add_argument("--s1-warmup-lr", type=float, default=0.001)
    parser.add_argument("--s1-cross-weight", type=float, default=0.5)
    parser.add_argument("--s1-ema-momentum", type=float, default=0.996)

    # Student 2 -- now TransferNet/CMKD's own real hyperparameters (see
    # configs/mfa/hparams_v2.yaml for VLP-UDA's actual office_home.yaml
    # values, loaded as defaults below).
    parser.add_argument("--s2-lr", type=float, default=3e-6, help="vision-encoder LR")
    parser.add_argument("--s2-multiple-lr-classifier", type=float, default=1000)
    parser.add_argument("--s2-lr-gamma", type=float, default=0.0003)
    parser.add_argument("--s2-lr-decay", type=float, default=0.75)
    parser.add_argument("--s2-weight-decay", type=float, default=5e-4)
    parser.add_argument("--s2-momentum", type=float, default=0.9)
    parser.add_argument("--s2-label-smoothing", type=float, default=0.1)
    parser.add_argument("--s2-lambda1", type=float, default=0.25)
    parser.add_argument("--s2-lambda2", type=float, default=0.1)
    parser.add_argument("--s2-lambda3", type=float, default=0.025)
    parser.add_argument("--s2-cross-weight", type=float, default=0.5,
                         help="weight on Student 2's cross-teaching loss term "
                              "(NOT part of CMKD -- added on top, same as train_mfa.py)")
    parser.add_argument("--s2-cross-mode", choices=["mask", "gini"], default="mask")
    parser.add_argument("--s2-ema-momentum", type=float, default=0.99,
                         help="ONE shared EMA momentum for BOTH model2.teacher_model "
                              "(backbone) and teacher_classifier_layer (head), active AFTER "
                              "--s2-ema-warmup-iters")
    parser.add_argument("--s2-ema-warmup-iters", type=int, default=100,
                         help="BOTH teacher_model and teacher_classifier_layer hard-copy "
                              "model2.base_network/classifier_layer every step for this many "
                              "iterations (momentum=0) before switching to --s2-ema-momentum "
                              "together -- kept in sync so Teacher2 is never a smoothed "
                              "backbone paired with an instantaneously-tracking head or vice "
                              "versa")
    parser.add_argument("--s2-batch-size", type=int, default=32)
    parser.add_argument("--s2-num-workers", type=int, default=8)
    parser.add_argument("--disable-s1", action="store_true",
                         help="Train Student2 alone, with NOTHING from dassl (no Student1, "
                              "no shared test_loader) -- to check Student2 in isolation "
                              "against a standalone VLP-UDA run. Forces --s2-cross-weight to "
                              "0.0 regardless of what's set (there's no Teacher1 to cross-teach "
                              "from/to), and evaluate()s on vlpuda_pure's own native "
                              "target_test_loader instead of dassl's, reporting Teacher2 only.")

    _load_hparams_as_defaults(parser, parser.parse_known_args()[0].hparams_config)
    args = parser.parse_args()
    if args.disable_s1 and args.s2_cross_weight != 0.0:
        print(f"--disable-s1: forcing --s2-cross-weight 0.0 (was {args.s2_cross_weight}) -- "
              f"no Teacher1 exists to cross-teach with")
        args.s2_cross_weight = 0.0
    print("Resolved args (CLI overrides applied on top of --hparams-config defaults):")
    for k, v in sorted(vars(args).items()):
        print(f"  {k} = {v}")
    cfg = build_cfg(args)
    device = torch.device(f"cuda:{cfg.GPU}" if torch.cuda.is_available() else "cpu")
    mkdir_if_missing(args.output_dir)
    mkdir_if_missing(osp.join(args.output_dir, "Teacher1"))
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)

    print("Building Student1/Teacher1's data loader (dassl, VLP-UDA-style transform)")
    # Both branches use dassl's own DataManager (each its own SEPARATE
    # instance -- see dm2 below -- own independent shuffled stream, same
    # principle as train_mfa.py's loader-independence fix), with VLP-UDA's
    # own transform (matching train_mfa.py's --vlpuda-augment, just
    # always-on instead of a flag, since v2's whole point is fidelity to
    # VLP-UDA and there's no "compare against dassl's own aug" motive here).
    _normalize = Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    _crop_size = cfg.INPUT.SIZE[0]
    _tfm_train = Compose([
        Resize([256, 256], interpolation=InterpolationMode.BILINEAR),
        RandomCrop(_crop_size),
        RandomHorizontalFlip(),
        ToTensor(),
        _normalize,
    ])
    _tfm_test = Compose([
        Resize([_crop_size, _crop_size], interpolation=InterpolationMode.BILINEAR),
        ToTensor(),
        _normalize,
    ])
    dm1 = DataManager(cfg, custom_tfm_train=_tfm_train, custom_tfm_test=_tfm_test)
    train_loader_x1 = CyclingLoader(dm1.train_loader_x)
    train_loader_u1 = CyclingLoader(dm1.train_loader_u)
    test_loader = dm1.test_loader
    classnames = dm1.dataset.classnames
    num_classes = dm1.num_classes
    is_vit = cfg.MODEL.BACKBONE.NAME.split('-')[0] == 'ViT'

    print("Building Student1/Teacher1 (PHPL-style)")
    student1, teacher1, lora1_s, lora1_t = _build_lora_pair(cfg, is_vit, classnames, device)

    print("Building Student2/Teacher2's data loader (dassl, same VLP-UDA-style transform)")
    # Per user's explicit request: keep using dassl's DataManager for
    # Student 2 too (their on-disk folder layout is already set up for it,
    # and it reads the exact same class-subfolder convention vlpuda_pure's
    # own ImageFolder-based loader would anyway) -- just with a SEPARATE
    # instance from dm1 (own independent shuffled stream, same principle as
    # train_mfa.py's loader-independence fix), same transform as dm1 above.
    dm2 = DataManager(cfg, custom_tfm_train=_tfm_train, custom_tfm_test=_tfm_test)
    train_loader_x2 = CyclingLoader(dm2.train_loader_x)
    train_loader_u2 = CyclingLoader(dm2.train_loader_u)

    s2_total_iters = args.s1_total_iters * args.s2_per_s1
    vlpuda_args = argparse.Namespace(
        datasets="office_home",
        model_name="VIT-B",
        num_class=num_classes,
        baseline=False,
        pda=False,
        fixmatch=False,
        rst=False,
        label_smoothing=args.s2_label_smoothing,
        lambda1=args.s2_lambda1,
        lambda2=args.s2_lambda2,
        lambda3=args.s2_lambda3,
        multiple_lr_classifier=args.s2_multiple_lr_classifier,
        max_iter=s2_total_iters,
    )

    print("Building Student2/Teacher2 (vlpuda_pure's own TransferNet)")
    model2 = TransferNet(vlpuda_args, train=True).to(device)
    # vlpuda_pure's own design has no classifier-head EMA at all (only
    # base_network gets a teacher_model copy) -- the classifier head needs
    # its own teacher too, same as train_mfa.py's teacher2_head, since it's
    # a freshly-initialized (near-random, small-std) module just like
    # there. Hard-copied every step for --s2-ema-warmup-iters (mirrors the
    # epoch-1 hard-copy lesson learned elsewhere in this project for a
    # randomly initialized head), then a real EMA after that.
    teacher_classifier_layer = copy.deepcopy(model2.classifier_layer).to(device)
    for param in teacher_classifier_layer.parameters():
        param.requires_grad_(False)
    teacher_classifier_layer.eval()

    optim1 = torch.optim.SGD(
        [p for p in student1.parameters() if p.requires_grad],
        lr=args.s1_lr, momentum=args.s1_momentum, weight_decay=args.s1_weight_decay,
    )
    # Matches vlpuda_pure/main.py's own get_optimizer/get_lr_scheduler
    # exactly (initial_lr=1.0 baked into the param groups via
    # get_parameters(), the LR value itself lives in the LambdaLR).
    optim2 = torch.optim.SGD(
        model2.get_parameters(initial_lr=1.0),
        lr=args.s2_lr, momentum=args.s2_momentum, weight_decay=args.s2_weight_decay, nesterov=True,
    )
    sched2 = torch.optim.lr_scheduler.LambdaLR(
        optim2, lr_lambda=lambda step: args.s2_lr * (1.0 + args.s2_lr_gamma * float(step)) ** (-args.s2_lr_decay)
    )

    s1_post_warmup = max(args.s1_total_iters - args.warmup_iters, 1)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(optim1, T_max=s1_post_warmup)

    print("Building frozen zero-shot CLIP (Student1's warmup reference only)")
    clip_frozen = load_clip_to_cpu(cfg)
    if cfg.TRAINER.PHPLMOMENTUM.PREC in ("fp32", "amp"):
        clip_frozen.float()
    teacher_frozen = _FrozenTeacherCLIP(cfg, classnames, clip_frozen).to(device)
    for param in teacher_frozen.parameters():
        param.requires_grad_(False)
    teacher_frozen.eval()

    confi = cfg.TRAINER.PHPLMOMENTUM.CONFI
    best_acc1, best_acc2, best_acc_ens = 0.0, 0.0, 0.0

    def _masked_ce(logits, prob_ref):
        max_probs, pseudo_label = torch.max(prob_ref, dim=-1)
        mask = max_probs.ge(confi).float()
        epsilon = 1e-8
        return (F.cross_entropy(logits, pseudo_label, reduction="none") * mask).sum() / (mask.sum() + epsilon)

    s2_it_global = 0
    for macro in range(args.s1_total_iters):
        in_warmup = macro < args.warmup_iters

        if not in_warmup and teacher_frozen is not None:
            # teacher_frozen is only ever read inside `if in_warmup:` blocks
            # -- once warmup ends it's dead weight (a whole extra CLIP model)
            # sitting on the GPU for the rest of the run. Free it right here
            # instead of holding it until the process exits.
            del teacher_frozen, clip_frozen
            teacher_frozen = None
            torch.cuda.empty_cache()

        batch_x1 = train_loader_x1.next()
        batch_u1 = train_loader_u1.next()
        image_x1 = batch_x1["img"].to(device)
        label_x1 = batch_x1["label"].to(device)
        image_u1 = batch_u1["img"].to(device)

        with torch.no_grad():
            if in_warmup:
                logits_cross_for_s1, _ = teacher_frozen(image_u1)
            else:
                # "Teacher2" for Student1's cross term: EMA backbone
                # (model2.teacher_model) through its own EMA head
                # (teacher_classifier_layer) -- both are temporal-fusion
                # EMAs now, no live parameters involved on Teacher2's side.
                feat_cross_for_s1 = model2.teacher_model.forward_features(image_u1)
                logits_cross_for_s1 = teacher_classifier_layer(feat_cross_for_s1)
            prob_cross_for_s1 = F.softmax(logits_cross_for_s1, dim=-1)

        for _ in range(args.s2_per_s1):
            batch_x2 = train_loader_x2.next()
            batch_u2 = train_loader_u2.next()
            data_x2 = batch_x2["img"].to(device)
            label_x2 = batch_x2["label"].to(device)
            data_u2 = batch_u2["img"].to(device)

            # model2.train() would ALSO flip teacher_model into train mode
            # (nn.Module.train() recurses into every submodule) -- it must
            # stay permanently eval (see make_model.py's own __init__ and
            # this project's ema_update_teacher). Set base_network/
            # classifier_layer's mode directly instead.
            model2.base_network.train()
            model2.classifier_layer.train()
            # TransferNet.forward() now returns target_logits too (a small,
            # additive change to vlpuda_pure/models/make_model.py -- it was
            # already computed internally for transfer_loss/cmkd, just not
            # returned before). This already includes clf_loss
            # (label_smoothing baked into model2.clf_loss) AND the full CMKD
            # self-training loss (task_loss + distill_loss + reg_loss, see
            # models/cmkd.py) -- no teacher/EMA involved in this loss at
            # all, by design.
            clf_loss, transfer_loss, target_logits2 = model2(data_x2, data_u2, label_x2)
            loss2_self = clf_loss + transfer_loss

            if in_warmup:
                loss2_cross = torch.tensor(0.0, device=device)
                loss2 = loss2_self
            else:
                # Cross-teaching is NOT part of CMKD -- added on top here,
                # same mechanism as train_mfa.py. Reuses target_logits2
                # above instead of a separate model2.predict(data_u2) call
                # -- that used to cost a WHOLE EXTRA forward pass through
                # the full CLIP backbone (a real OOM contributor on top of
                # the 5 CLIP model copies already in play), and it mutated
                # classifier_layer's BatchNorm1d running stats an extra
                # time regardless of --s2-cross-weight's value. Neither
                # concern applies anymore -- target_logits2 is free.
                with torch.no_grad():
                    logits_t1_u2, _ = teacher1(data_u2)
                    prob_cross2 = F.softmax(logits_t1_u2, dim=-1)
                loss2_cross = _masked_ce(target_logits2, prob_cross2)
                loss2 = loss2_self + args.s2_cross_weight * loss2_cross

            optim2.zero_grad()
            loss2.backward()
            optim2.step()
            sched2.step()

            # Backbone and head EMA share the SAME warmup window
            # (--s2-ema-warmup-iters) -- both hard-copy (momentum=0)
            # together at first, then both switch to their own real
            # momentum together. Giving the backbone a real momentum from
            # step 0 while the head hard-copies would leave Teacher2 as a
            # smoothed backbone paired with an instantaneously-tracking
            # head for that whole window -- a mismatch, since the head is
            # trained (via TransferNet.forward()) against the LIVE
            # backbone's features, not the lagging EMA one.
            ema_momentum = 0.0 if s2_it_global < args.s2_ema_warmup_iters else args.s2_ema_momentum
            ema_update_teacher(model2.teacher_model, model2.base_network, ema_momentum)
            ema_update_teacher(teacher_classifier_layer, model2.classifier_layer, ema_momentum)
            s2_it_global += 1

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
        torch.nn.utils.clip_grad_norm_(
            [p for p in student1.parameters() if p.requires_grad], max_norm=20.0
        )
        optim1.step()
        if not in_warmup:
            sched1.step()

        _ema_update_lora_params(teacher1, student1, lambda k: args.s1_ema_momentum)

        if (macro + 1) % args.print_freq == 0:
            acc_x1 = (logits1_x.argmax(-1) == label_x1).float().mean().item() * 100
            with torch.no_grad():
                acc_x2 = (torch.max(model2.predict(data_x2), 1)[1] == label_x2).float().mean().item() * 100
            print(
                f"macro [{macro + 1}/{args.s1_total_iters}] (s2 iter {s2_it_global}/{s2_total_iters}) "
                f"loss1 {loss1.item():.4f} (x {loss_x1.item():.4f} self {loss_u1_self.item():.4f} "
                f"cross {loss_u1_cross.item():.4f} mmd {loss_mmd1.item():.4f}) acc_x1 {acc_x1:.2f} | "
                f"loss2 {loss2.item():.4f} (clf {clf_loss.item():.4f} transfer {transfer_loss.item():.4f} "
                f"cross {loss2_cross.item():.4f}) acc_x2 {acc_x2:.2f}"
            )

        if (macro + 1) % args.eval_freq == 0 or (macro + 1) == args.s1_total_iters or (macro + 1) == args.warmup_iters:
            model2.eval()
            acc1, acc2, acc_ens = evaluate(teacher1, model2, teacher_classifier_layer, test_loader, device)
            tag = " [end of warmup]" if (macro + 1) == args.warmup_iters else ""
            print(f"[eval] macro {macro + 1}{tag}: Teacher1 {acc1:.2f}% | Teacher2 {acc2:.2f}% | Ensemble {acc_ens:.2f}%")

            save_lora(cfg, lora1_t, osp.join(args.output_dir, "Teacher1"), filename="LoRA-last")
            torch.save(model2.state_dict(), osp.join(args.output_dir, "model2-last.pt"))

            if acc1 > best_acc1:
                best_acc1 = acc1
                save_lora(cfg, lora1_t, osp.join(args.output_dir, "Teacher1"), filename="LoRA-best")
            if acc2 > best_acc2:
                best_acc2 = acc2
                torch.save(model2.state_dict(), osp.join(args.output_dir, "model2-best.pt"))
            if acc_ens > best_acc_ens:
                best_acc_ens = acc_ens
                print(f"  new best ensemble ({best_acc_ens:.2f}%)")

    print(f"Done. best Teacher1={best_acc1:.2f}% Teacher2={best_acc2:.2f}% Ensemble={best_acc_ens:.2f}%")


if __name__ == "__main__":
    main()
