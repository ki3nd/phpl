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
  - Self-loss: TransferNet.forward()'s own (clf_loss, transfer_loss). By
    default this already IS VLP-UDA's real CMKD self-training (no
    teacher/EMA involved in the self-consistency reference at all, by
    design -- see cmkd.py). --s2-self-from-teacher swaps that reference to
    Teacher2's EMA cosine branch instead (make_model.py/cmkd.py gained an
    optional self_ref_logit_clip param for this, default None preserving
    the original behavior exactly) -- reg_loss's entropy-min term still
    always uses the LIVE cosine branch regardless, so it keeps training it.
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
    features fed through this separate EMA head, not the live one. By
    default BOTH get a DACS-style (Tarvainen & Valpola / vikolss/DACS
    update_ema_variables) ramp-up momentum every step from s2_it_global=0:
    momentum_t = min(t / (t+1), --s2-ema-momentum). This degenerates to a
    hard-copy at t=0 (teacher := student's CURRENT, already-gradient-
    stepped weights, never the raw random init -- the EMA update runs
    AFTER optimizer.step()) and ramps smoothly up to the target momentum
    over ~momentum/(1-momentum) steps (~99 steps for the 0.99 default),
    with no arbitrary warmup-length hyperparameter and no discontinuous
    jump. The old hard-copy-for-N-steps-then-jump behavior is still
    available behind --s2-ema-warmup (off by default) for ablation, using
    --s2-ema-warmup-iters as the hard-copy window. Either way, the
    backbone's and the head's EMA share the exact same momentum_t each
    step -- otherwise Teacher2 would be a smoothed backbone paired with an
    instantaneously-tracking head (or vice versa), since the head is
    trained against the LIVE backbone's features, not the lagging EMA one.
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
    ColorJitter, Compose, Normalize, RandomCrop, RandomHorizontalFlip,
    RandomResizedCrop, Resize, ToTensor,
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
        # backbone too, only for its `clip` module -- re-tokenizing Student 2's
        # class prompts (see --clarify-classnames) has to use the SAME clip
        # fork that built them, not this project's own top-level `clip`.
        backbone = importlib.import_module("models.backbone")
    finally:
        sys.path[:] = saved_path
        for k in [k for k in sys.modules if _matches(k)]:
            del sys.modules[k]
        sys.modules.update(saved_modules)

    return make_model, tools, backbone


_vlpuda_make_model, _vlpuda_tools, _vlpuda_backbone = _import_vlpuda_pure()
TransferNet = _vlpuda_make_model.TransferNet
ema_update_teacher = _vlpuda_tools.ema_update_teacher

# --clarify-classnames: OfficeHome class names that are ambiguous as BARE
# WORDS, so CLIP's text embedding lands on the wrong sense. Chosen purely from
# what each name denotes in the dataset (visible in the LABELED source domain)
# and from the word's other English senses -- never from target-domain
# accuracy, which would need target labels this setting does not have. Classes
# whose bare name is already unambiguous are deliberately left alone: renaming
# something that already works can only cost. Keys are dassl's own classnames
# (lowercased directory names, underscores already turned into spaces).
_CLASSNAME_CLARIFY = {
    "mouse": "computer mouse",            # vs. the animal
    "monitor": "computer monitor",        # vs. a person who monitors
    "computer": "desktop computer",       # bare "computer" is a category, not an object
    "tv": "television",                   # abbreviation
    "notebook": "paper notebook",         # vs. a laptop
    "keyboard": "computer keyboard",      # vs. a musical keyboard
    "glasses": "eyeglasses",              # vs. drinking glasses
    "speaker": "loudspeaker",             # vs. a person speaking
    "fan": "electric fan",                # vs. a sports fan
    "marker": "marker pen",               # vs. a landmark or a mark
    "soda": "soda can",                   # vs. baking soda / soda water
    "pan": "frying pan",                  # vs. the verb / the proper noun
    "drill": "power drill",               # vs. a drill exercise
    "ruler": "measuring ruler",           # vs. a monarch
    "sink": "kitchen sink",               # vs. the verb
    "folder": "file folder",              # vs. a filesystem folder
    "postit notes": "post-it notes",      # dataset spells it without the hyphen
    "flipflops": "flip-flops",            # dataset spells it as one word
    "clipboards": "clipboard",            # odd plural; also vs. the OS clipboard
    "lamp shade": "lampshade",            # standard spelling
    "file cabinet": "filing cabinet",     # standard usage
}


def _clarify_classnames(classnames, enabled):
    """Map dassl classnames through _CLASSNAME_CLARIFY.

    Disabled: returns the list completely untouched (underscores and all), so
    every existing call path behaves exactly as before.

    Enabled: returns names with underscores turned into spaces AND the
    ambiguous ones replaced. The underscore normalization matters -- dassl's
    classnames come straight from directory names ("alarm_clock"), and while
    branch 1 strips underscores itself (Base_CustomCLIP does
    `c.replace("_", " ")`), branch 2's prompts are built from this list
    directly, so a name left as "alarm_clock" would reach CLIP verbatim."""
    if not enabled:
        return list(classnames)
    out, changed = [], []
    for c in classnames:
        plain = c.replace("_", " ").lower()
        new = _CLASSNAME_CLARIFY.get(plain, plain)
        if new != plain:
            changed.append(f"{plain} -> {new}")
        out.append(new)
    print(f"--clarify-classnames: rewrote {len(changed)}/{len(classnames)} class names")
    for line in changed:
        print(f"    {line}")
    return out


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
    parser.add_argument("--s1-warmup-iters", type=int, default=50,
                         help="macro-steps (Student1's own cadence) before Student1 switches "
                              "from the frozen zero-shot CLIP self-reference to its own "
                              "teacher1 EMA + starts pulling the cross term from Teacher2. "
                              "Separate from --s2-warmup-iters so the two branches' "
                              "cross-teaching onset can be ablated independently.")
    parser.add_argument("--s2-warmup-iters", type=int, default=500,
                         help="s2 micro-steps (s2_it_global, Student2's own cadence -- "
                              "--s2-per-s1 times denser than macro-steps) before Student2 "
                              "starts adding the cross term from Teacher1. Default (500) is "
                              "--s1-warmup-iters's default (50) times the default --s2-per-s1 "
                              "(10), so both branches end their warmup at roughly the same "
                              "macro-step by default -- separate from --s1-warmup-iters, so "
                              "the two can still be set independently to ablate cross-teaching "
                              "onset per branch.")

    # Student 1 -- identical set of flags to train_mfa.py.
    parser.add_argument("--s1-lr", type=float, default=0.0035)
    parser.add_argument("--s1-momentum", type=float, default=0.9)
    parser.add_argument("--s1-weight-decay", type=float, default=5e-4)
    parser.add_argument("--s1-mmd-weight", type=float, default=1.0,
                         help="Weight on Student1's MK-MMD source/target feature-alignment "
                              "loss (default 1.0, i.e. ON). Pass 0.0 to turn it OFF ENTIRELY: "
                              "both the MK_MMD call and the target-side forward pass that "
                              "exists only to feed it are skipped, so it then costs nothing. "
                              "NOTE: configs/mfa/hparams_v2.yaml's s1.mmd_weight overrides "
                              "this argparse default (see _load_hparams_as_defaults), so both "
                              "are kept at 1.0 -- change either one and the other no longer "
                              "governs.")
    parser.add_argument("--s1-warmup-lr", type=float, default=0.001)
    parser.add_argument("--s1-cross-weight", type=float, default=0.5)
    parser.add_argument("--s1-ema-momentum", type=float, default=0.996)

    # Student 2 -- now TransferNet/CMKD's own real hyperparameters (see
    # configs/mfa/hparams_v2.yaml for VLP-UDA's actual office_home.yaml
    # values, loaded as defaults below).
    parser.add_argument("--s2-model-name", choices=["VIT-B", "RN50", "RN101"], default="VIT-B",
                         help="Branch 2's CLIP backbone (vlpuda_pure/models/backbone.py picks "
                              "the checkpoint and the feature width from this). Branch 1 is "
                              "always the --backbone CLIP+LoRA, so setting this to RN50/RN101 "
                              "makes the two branches ARCHITECTURALLY heterogeneous -- a CNN's "
                              "local receptive fields against ViT's global attention -- which "
                              "is the point: two branches sharing one architecture inherit one "
                              "error set and have little left to teach each other. RN101 keeps "
                              "output_num=512, the same width as VIT-B, so classifier_layer is "
                              "unchanged; RN50 is 1024 and resizes it. NOTE: --s2-lr's default "
                              "(3e-6) is VLP-UDA's tuned value for full ViT fine-tuning and is "
                              "almost certainly far too small for a ResNet -- expect to sweep "
                              "it. Also note TransferNet.forward()'s fix_bn, a no-op on ViT, "
                              "starts actually freezing BatchNorm once this is a ResNet.")
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
    parser.add_argument("--s2-cross-mode", choices=["mask", "gini"], default="mask",
                         help="Form of Student2's cross loss (Teacher1 -> Student2). 'mask' "
                              "(default): CONFI hard-threshold + CE on the argmax pseudo-label, "
                              "at a CONSTANT --s2-cross-weight. 'gini': CMKD's own task/distill "
                              "pair (model2.cmkd's own gini_impurity/calibrated_coefficient, so "
                              "the arithmetic is identical to the self loss) with Teacher1 as "
                              "the reference -- no argmax, no threshold, every target sample "
                              "contributes weighted by coe=exp(-KL), and the term is scaled by "
                              "lambda1 * lamb so it ramps on the SAME schedule as the self loss "
                              "instead of being diluted by it. Measured on this run's logits at "
                              "batch 32, the two modes are the same order of magnitude but "
                              "follow different curves -- gini goes 0.05 -> 0.93 as lamb ramps "
                              "0.025 -> 0.462, while mask sits flat near 0.08 -- so a single "
                              "--s2-cross-weight does not mean the same thing in both; sweep it "
                              "per mode. --s1-threshold has no effect on this loss in 'gini' "
                              "mode.")
    parser.add_argument("--s2-self-from-teacher", action="store_true",
                         help="Self-loss's reference (coe/mix in CMKD's task_loss/distill_loss) "
                              "comes from Teacher2's EMA cosine branch (model2.teacher_model, "
                              "detached) instead of the live student's own -- vlpuda_pure's own "
                              "cmkd.py/make_model.py gained an optional self_ref_logit_clip param "
                              "for this (default None preserves the original behavior exactly). "
                              "reg_loss's entropy-min term still always uses the LIVE cosine "
                              "branch (unaffected), so it keeps training it regardless.")
    parser.add_argument("--s2-ema-momentum", type=float, default=0.99,
                         help="ONE shared EMA momentum target for BOTH model2.teacher_model "
                              "(backbone) and teacher_classifier_layer (head). By default this "
                              "is the ASYMPTOTIC target of a DACS-style ramp-up "
                              "(min(t/(t+1), momentum), t=s2_it_global) applied every step from "
                              "t=0 -- see --s2-ema-warmup to use a fixed hard-copy-then-jump "
                              "schedule instead.")
    parser.add_argument("--s2-ema-warmup", action="store_true",
                         help="Use the old hard-copy-for-N-steps-then-jump EMA schedule for "
                              "Teacher2 (momentum=0 for --s2-ema-warmup-iters steps, then a "
                              "discontinuous jump to --s2-ema-momentum) instead of the default "
                              "DACS-style continuous ramp-up. Off by default -- kept for "
                              "ablation only.")
    parser.add_argument("--s2-ema-warmup-iters", type=int, default=100,
                         help="Only used when --s2-ema-warmup is set. BOTH teacher_model and "
                              "teacher_classifier_layer hard-copy model2.base_network/"
                              "classifier_layer every step for this many iterations "
                              "(momentum=0) before switching to --s2-ema-momentum together -- "
                              "kept in sync so Teacher2 is never a smoothed backbone paired "
                              "with an instantaneously-tracking head or vice versa")
    parser.add_argument("--s2-batch-size", type=int, default=32)
    parser.add_argument("--s2-num-workers", type=int, default=8)
    parser.add_argument("--strong-aug", action="store_true",
                         help="FixMatch-style weak/strong split for the TARGET (unlabeled) "
                              "image stream, for BOTH branches: teachers/self-reference/cross-"
                              "reference keep seeing the current (weak) view unchanged; each "
                              "student's OWN forward (whose output feeds self+cross loss) sees "
                              "a mildly harder 'strong' view instead (slightly more aggressive "
                              "crop + light color jitter -- NOT VLP-UDA's own ColorJitter(0.4)+"
                              "RandAugment(m10) fixmatch recipe, which is CIFAR/ImageNet-CNN-"
                              "scale aggressive and risks knocking CLIP's own pretrained "
                              "semantics off-distribution). Off by default -- when on, Student2 "
                              "gets an EXTRA full-gradient backbone forward pass per micro-"
                              "iteration (the strong view, on top of the weak one already used "
                              "for its self-reference/reg_loss) -- this is the same shape of "
                              "cost that caused the earlier OOM (removed then, reintroduced "
                              "here on purpose) -- reduce --s2-batch-size if it OOMs again. "
                              "Student1 (LoRA) has no such cost -- just swaps which image its "
                              "own forward call uses, for free. evaluate() is entirely "
                              "unaffected (test_loader keeps its own separate, deterministic "
                              "transform, never touched by this flag).")
    parser.add_argument("--clarify-classnames", action="store_true",
                         help="Replace ambiguous OfficeHome class names with unambiguous ones "
                              "in BOTH branches (see _CLASSNAME_CLARIFY): 'mouse' -> 'computer "
                              "mouse', 'monitor' -> 'computer monitor', 'notebook' -> 'paper "
                              "notebook', etc. CLIP scores images against the class NAME, so a "
                              "name whose dominant English sense is the wrong object (the animal "
                              "mouse, a person who monitors, a laptop) puts the text embedding "
                              "in the wrong place and both branches inherit that error. Each "
                              "branch keeps its OWN prompt template (branch 1: utils/templates."
                              "py's 'a photo of a {}.'; branch 2: vlpuda_pure's 'an image of a "
                              "{}') -- only the class name inside it changes. The rewrites are "
                              "justified by what the name denotes in the LABELED source domain "
                              "plus the word's other senses, never by target-domain accuracy "
                              "(that would need target labels). Off by default.")
    parser.add_argument("--s1-threshold", type=float, default=None,
                         help="ONE shared confidence threshold for the pseudo-label mask on "
                              "EVERY reference distribution, both branches (Student1's warmup "
                              "self-reference from teacher_frozen, its post-warmup "
                              "self-reference from teacher1, its cross-reference from "
                              "teacher_classifier_layer, and Student2's cross-reference from "
                              "teacher1) -- exactly the same single cutoff the code already "
                              "used, just settable now. Defaults to "
                              "cfg.TRAINER.PHPLMOMENTUM.CONFI (0.85), i.e. unchanged behavior.")
    parser.add_argument("--s1-loss-u-mode", choices=["mask", "ratio"], default="mask",
                         help="How Student1's SELF loss (loss_u1_self -- reference is "
                              "teacher_frozen during warmup, teacher1 after) reduces its "
                              "per-sample CE. 'mask' (default, PHPL's own): average CE over "
                              "ONLY the samples clearing --s1-threshold, i.e. divide by the "
                              "COUNT that pass -- loss magnitude is independent of how few "
                              "pass, so it gets high-variance when that count is small. "
                              "'ratio' (DACS-style, cf. LOSS_U_MODE in phpl_momentum.py): CE "
                              "over ALL samples (argmax pseudo-label for every one, confident "
                              "or not), scaled by the FRACTION that pass -- smooth curriculum, "
                              "lower gradient variance, but the term shrinks proportionally "
                              "when few pass. Both are exactly 0 when nothing passes.")
    parser.add_argument("--s1-loss-cross-mode", choices=["mask", "ratio"], default="mask",
                         help="Same choice, for Student1's CROSS loss (loss_u1_cross -- "
                              "reference is teacher_classifier_layer, branch 2's learned, "
                              "label-smoothed head). Separate from --s1-loss-u-mode because "
                              "this reference is far softer than branch 1's "
                              "logit_scale(~100)-saturated cosine outputs, so the two losses "
                              "sit in very different mask-sparsity regimes.")
    parser.add_argument("--use-debias", action="store_true",
                         help="Logit-adjustment debiasing (UniMoS/DebiasPL-style, same "
                              "mechanism as cfg.TRAINER.PHPLMOMENTUM.USE_DEBIAS in train.py) "
                              "against CLIP's own zero-shot class bias, applied to the "
                              "genuine CLIP-cosine teacher predictions on BOTH branches -- NOT "
                              "to teacher_classifier_layer's output (Teacher2's learned-head "
                              "cross-reference to Student1, which isn't a CLIP prediction). "
                              "Two INDEPENDENT EMA trackers (qhat1, qhat2), one per branch: "
                              "qhat1 covers teacher_frozen (Student1's warmup self-reference) "
                              "and teacher1 (self-reference post-warmup + cross-reference to "
                              "Student2); qhat2 covers Teacher2's cosine branch "
                              "(self_ref_logit_clip) ONLY when --s2-self-from-teacher is set "
                              "(the only place it's computed outside vlpuda_pure's own "
                              "forward()).")
    parser.add_argument("--debias-tau", type=float, default=0.5,
                         help="Only used when --use-debias is set -- see its help text.")
    parser.add_argument("--debias-momentum", type=float, default=0.99,
                         help="Only used when --use-debias is set -- see its help text.")
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
    # --strong-aug: mild "strong" view (a bit more aggressive crop + light
    # color jitter -- NOT VLP-UDA's own ColorJitter(0.4)+RandAugment(m10)
    # fixmatch recipe, deliberately toned down for a CLIP-pretrained model,
    # see the flag's own help text). dassl's DatasetWrapper natively
    # supports a LIST of transforms -- each produces its own output["img"],
    # output["img2"], ... key, no custom dataset wrapper needed. This list
    # is used for BOTH train_x and train_u (dassl applies the same
    # transform list to both loaders) -- img2 is simply unused for train_x
    # (source), no behavior change there.
    if args.strong_aug:
        _tfm_strong = Compose([
            RandomResizedCrop(_crop_size, scale=(0.5, 1.0), interpolation=InterpolationMode.BILINEAR),
            RandomHorizontalFlip(),
            ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0),
            ToTensor(),
            _normalize,
        ])
        _tfm_train_list = [_tfm_train, _tfm_strong]
    else:
        _tfm_train_list = _tfm_train
    dm1 = DataManager(cfg, custom_tfm_train=_tfm_train_list, custom_tfm_test=_tfm_test)
    train_loader_x1 = CyclingLoader(dm1.train_loader_x)
    train_loader_u1 = CyclingLoader(dm1.train_loader_u)
    test_loader = dm1.test_loader
    classnames = _clarify_classnames(dm1.dataset.classnames, args.clarify_classnames)
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
    dm2 = DataManager(cfg, custom_tfm_train=_tfm_train_list, custom_tfm_test=_tfm_test)
    train_loader_x2 = CyclingLoader(dm2.train_loader_x)
    train_loader_u2 = CyclingLoader(dm2.train_loader_u)

    s2_total_iters = args.s1_total_iters * args.s2_per_s1
    vlpuda_args = argparse.Namespace(
        datasets="office_home",
        model_name=args.s2_model_name,
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
    if args.clarify_classnames:
        # vlpuda_pure/models/backbone.py hardcodes its own class_list with no
        # hook to override, so retokenize AFTER construction instead of
        # editing that file. `text`/`text_features` are plain tensor
        # attributes (not buffers/parameters), so they never appear in
        # state_dict -- overwriting them here is both safe and permanent, and
        # loading a checkpoint later cannot silently restore the old ones.
        # BOTH base_network and teacher_model need it: TransferNet.__init__
        # deepcopies the former into the latter, so each holds its own copy.
        # The template is vlpuda's own ("an image of a {}"), only the class
        # name inside it changes. Its text tower is frozen (get_parameters
        # optimizes only model.visual + classifier_layer) so these embeddings
        # stay valid for the whole run.
        # Guard: this rewrite assumes dassl's class ORDER (sorted directory
        # names) is the same order as backbone.py's hardcoded office_home
        # class_list -- verified true for OfficeHome's 65 classes. On any
        # other dataset the two orders could differ and silently scramble
        # every text embedding, which would look like a bad result rather
        # than a bug, so refuse instead of guessing.
        if cfg.DATASET.NAME != "OfficeHome" or len(classnames) != 65:
            raise SystemExit(
                f"--clarify-classnames is only verified for OfficeHome's 65 classes "
                f"(got {cfg.DATASET.NAME} with {len(classnames)}): _CLASSNAME_CLARIFY's "
                f"entries and the dassl/vlpuda class-order match are both "
                f"OfficeHome-specific."
            )
        prompts2 = [f"an image of a {c}" for c in classnames]
        tokens2 = _vlpuda_backbone.clip.tokenize(prompts2).to(device)
        for net in (model2.base_network, model2.teacher_model):
            net.text = tokens2
            with torch.no_grad():
                tf = net.encode_text().detach()
            net.text_features = tf / tf.norm(dim=1, keepdim=True)
        print(f"--clarify-classnames: retokenized Student2's {len(prompts2)} prompts, e.g. "
              + ", ".join(repr(p) for p in prompts2[:2]))
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

    s1_post_warmup = max(args.s1_total_iters - args.s1_warmup_iters, 1)
    sched1 = torch.optim.lr_scheduler.CosineAnnealingLR(optim1, T_max=s1_post_warmup)

    print("Building frozen zero-shot CLIP (Student1's warmup reference only)")
    clip_frozen = load_clip_to_cpu(cfg)
    if cfg.TRAINER.PHPLMOMENTUM.PREC in ("fp32", "amp"):
        clip_frozen.float()
    teacher_frozen = _FrozenTeacherCLIP(cfg, classnames, clip_frozen).to(device)
    for param in teacher_frozen.parameters():
        param.requires_grad_(False)
    teacher_frozen.eval()

    # ONE shared threshold for every reference distribution on both branches
    # (see --s1-threshold) -- same single cutoff the code already used, just
    # settable. Defaults to cfg.TRAINER.PHPLMOMENTUM.CONFI, so passing nothing
    # preserves the old behavior exactly.
    confi = args.s1_threshold if args.s1_threshold is not None else cfg.TRAINER.PHPLMOMENTUM.CONFI
    print(f"Pseudo-label threshold (shared, both branches): {confi} "
          f"(cfg.TRAINER.PHPLMOMENTUM.CONFI={cfg.TRAINER.PHPLMOMENTUM.CONFI})")
    best_acc1, best_acc2, best_acc_ens = 0.0, 0.0, 0.0

    def _pseudo_label_loss(logits, prob_ref, mode="mask"):
        """mode "mask" (default) is the original behavior, unchanged; "ratio"
        is phpl_momentum.py's own LOSS_U_MODE "ratio" (DACS-style). See
        --s1-loss-u-mode's help text for the difference."""
        max_probs, pseudo_label = torch.max(prob_ref, dim=-1)
        mask = max_probs.ge(confi).float()
        if mode == "ratio":
            # No sample dropped -- CE over the WHOLE batch (argmax
            # pseudo-label for every sample), scaled by the fraction that
            # clears `confi`.
            return F.cross_entropy(logits, pseudo_label) * mask.mean()
        epsilon = 1e-8
        return (F.cross_entropy(logits, pseudo_label, reduction="none") * mask).sum() / (mask.sum() + epsilon)

    # --use-debias: see its own help text for scope (teacher1/teacher_frozen
    # only for qhat1, Teacher2's cosine branch only for qhat2). Matches
    # cfg.TRAINER.PHPLMOMENTUM.USE_DEBIAS's own update order in
    # trainers/da/phpl_momentum.py: correct `logits` using qhat as of
    # BEFORE this call, then update qhat from `logits`' own RAW
    # (pre-correction) prediction -- never the other way around.
    def _debias_correct(logits, qhat):
        prob_raw = F.softmax(logits, dim=-1)
        corrected = logits - args.debias_tau * torch.log(qhat)
        qhat.mul_(args.debias_momentum).add_(prob_raw.mean(dim=0), alpha=1.0 - args.debias_momentum)
        return corrected

    qhat1 = torch.full((num_classes,), 1.0 / num_classes, device=device) if args.use_debias else None
    qhat2 = torch.full((num_classes,), 1.0 / num_classes, device=device) if args.use_debias else None

    s2_it_global = 0
    for macro in range(args.s1_total_iters):
        # Student1's own warmup (macro-step cadence) and Student2's own
        # warmup (s2_it_global cadence, --s2-per-s1 times denser) are
        # tracked separately so each branch's cross-teaching onset can be
        # ablated independently -- they are NOT required to line up.
        in_warmup1 = macro < args.s1_warmup_iters

        if not in_warmup1 and teacher_frozen is not None:
            # teacher_frozen is only ever read inside `if in_warmup1:` blocks
            # -- once Student1's warmup ends it's dead weight (a whole extra
            # CLIP model) sitting on the GPU for the rest of the run. Free
            # it right here instead of holding it until the process exits.
            del teacher_frozen, clip_frozen
            teacher_frozen = None
            torch.cuda.empty_cache()

        batch_x1 = train_loader_x1.next()
        batch_u1 = train_loader_u1.next()
        image_x1 = batch_x1["img"].to(device)
        label_x1 = batch_x1["label"].to(device)
        # weak = current/default view, used for every TEACHER-side
        # computation below (self-reference, cross-reference) -- unchanged.
        # strong = --strong-aug's mildly harder view, used ONLY for
        # Student1's OWN forward further down (feeds both self+cross loss)
        # -- free for Student1 (LoRA), just swaps which image tensor its
        # forward call uses, no extra pass.
        image_u1 = batch_u1["img"].to(device)
        image_u1_strong = batch_u1["img2"].to(device) if args.strong_aug else image_u1

        with torch.no_grad():
            if in_warmup1:
                logits_cross_for_s1, _ = teacher_frozen(image_u1)
                if args.use_debias:
                    # teacher_frozen IS a genuine CLIP zero-shot prediction
                    # (same qhat1 tracker teacher1 uses post-warmup, so it
                    # carries over seamlessly once teacher1 takes over).
                    logits_cross_for_s1 = _debias_correct(logits_cross_for_s1, qhat1)
            else:
                # "Teacher2" for Student1's cross term: EMA backbone
                # (model2.teacher_model) through its own EMA head
                # (teacher_classifier_layer) -- both are temporal-fusion
                # EMAs now, no live parameters involved on Teacher2's side.
                feat_cross_for_s1 = model2.teacher_model.forward_features(image_u1)
                logits_cross_for_s1 = teacher_classifier_layer(feat_cross_for_s1)
            prob_cross_for_s1 = F.softmax(logits_cross_for_s1, dim=-1)

        for _ in range(args.s2_per_s1):
            in_warmup2 = s2_it_global < args.s2_warmup_iters
            batch_x2 = train_loader_x2.next()
            batch_u2 = train_loader_u2.next()
            data_x2 = batch_x2["img"].to(device)
            label_x2 = batch_x2["label"].to(device)
            data_u2 = batch_u2["img"].to(device)
            # weak (data_u2) feeds every TEACHER-side computation below
            # (self-reference, reg_loss, cross-reference to Teacher1) --
            # unchanged. strong feeds ONLY the classifier's own prediction
            # (own_pred_target_img below) -- see --strong-aug's own help
            # text for the OOM-risk tradeoff this reintroduces here.
            data_u2_strong = batch_u2["img2"].to(device) if args.strong_aug else data_u2

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
            # models/cmkd.py) -- no teacher/EMA involved in this loss by
            # default (real CMKD design), UNLESS --s2-self-from-teacher.
            self_ref_logit_clip = None
            if args.s2_self_from_teacher:
                with torch.no_grad():
                    teacher_feat_u2 = model2.teacher_model.forward_features(data_u2)
                    self_ref_logit_clip = model2.teacher_model.forward_head(teacher_feat_u2).detach()
                    if args.use_debias:
                        # Teacher2's cosine branch IS a genuine CLIP
                        # prediction (unlike teacher_classifier_layer) --
                        # its own qhat2 tracker, separate from qhat1. Only
                        # reachable here since this is the only place
                        # Teacher2's cosine branch is computed OUTSIDE
                        # vlpuda_pure's own (untouched) forward() --
                        # self-from-student's internal reference isn't
                        # debiased (out of scope for now).
                        self_ref_logit_clip = _debias_correct(self_ref_logit_clip, qhat2)
            # Read the CMKD ramp BEFORE the forward: cmkd.py's forward reads
            # lamb at its top and calls self.lamb.step() at its bottom, so
            # reading it afterwards would give the NEXT step's value, not the
            # one the self-loss just used. --s2-cross-mode gini wants the same
            # lamb the self term got.
            lamb_cross = model2.cmkd.lamb.lamb()
            clf_loss, transfer_loss, target_logits2 = model2(
                data_x2, data_u2, label_x2, self_ref_logit_clip=self_ref_logit_clip,
                own_pred_target_img=(data_u2_strong if args.strong_aug else None),
            )
            loss2_self = clf_loss + transfer_loss

            if in_warmup2:
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
                    if args.use_debias:
                        logits_t1_u2 = _debias_correct(logits_t1_u2, qhat1)
                    prob_cross2 = F.softmax(logits_t1_u2, dim=-1)
                if args.s2_cross_mode == "gini":
                    # CMKD's own task/distill pair, but with Teacher1 as the
                    # reference instead of Student2's own cosine branch.
                    # Calls model2.cmkd's OWN methods rather than a copy, so
                    # this is byte-identical arithmetic to the self term
                    # (train_cmkd.py's hand-ported _gini_impurity /
                    # _calibrated_coefficient differ slightly -- an extra
                    # clamp and epsilon -- and using those here would put a
                    # second variable into the mask-vs-gini comparison).
                    # lambda1 * lamb is applied here, so unlike "mask" this
                    # term ramps with the same schedule as the self loss
                    # instead of staying at a constant weight. No confidence
                    # threshold either: every target sample contributes,
                    # weighted by coe = exp(-KL(teacher || student)).
                    pred_own = F.softmax(target_logits2, dim=1)
                    pred_ref = prob_cross2.detach()
                    coe = model2.cmkd.calibrated_coefficient(pred_own, pred_ref)
                    pred_mix = 0.5 * (pred_own + pred_ref)
                    loss2_cross = (
                        args.s2_lambda1 * lamb_cross * model2.cmkd.gini_impurity(pred_own, coe)
                        + args.s2_lambda1 * lamb_cross * model2.cmkd.gini_impurity(pred_mix, 1 - coe)
                    )
                else:
                    # PHPL-style CONFI hard-threshold mask + CE. Note this
                    # branch is NOT ramped -- it sits at a constant
                    # --s2-cross-weight while the self loss ramps up, so its
                    # relative share shrinks over training.
                    loss2_cross = _pseudo_label_loss(target_logits2, prob_cross2)
                loss2 = loss2_self + args.s2_cross_weight * loss2_cross

            optim2.zero_grad()
            loss2.backward()
            optim2.step()
            sched2.step()

            # Backbone and head EMA always share the SAME momentum this
            # step -- giving the backbone a real momentum while the head
            # hard-copies (or vice versa) would leave Teacher2 as a
            # smoothed backbone paired with an instantaneously-tracking
            # head -- a mismatch, since the head is trained (via
            # TransferNet.forward()) against the LIVE backbone's features,
            # not the lagging EMA one.
            if args.s2_ema_warmup:
                # Ablation-only: hard-copy (momentum=0) for
                # --s2-ema-warmup-iters steps, then a discontinuous jump to
                # --s2-ema-momentum.
                ema_momentum = 0.0 if s2_it_global < args.s2_ema_warmup_iters else args.s2_ema_momentum
            else:
                # Default: DACS-style ramp-up (vikolss/DACS's
                # update_ema_variables), min(t/(t+1), momentum) every step
                # from t=0. At t=0 this is 0/(0+1)=0 -- a hard-copy, but of
                # the student's CURRENT (post-optimizer.step()) weights,
                # never the raw random init -- then ramps continuously up
                # to --s2-ema-momentum over ~momentum/(1-momentum) steps
                # instead of jumping there after a fixed warmup window.
                ema_momentum = min(s2_it_global / (s2_it_global + 1), args.s2_ema_momentum)
            ema_update_teacher(model2.teacher_model, model2.base_network, ema_momentum)
            ema_update_teacher(teacher_classifier_layer, model2.classifier_layer, ema_momentum)
            s2_it_global += 1

        logits1_x, feat_x1 = student1(image_x1)
        loss_x1 = F.cross_entropy(logits1_x, label_x1)
        # ON by default (--s1-mmd-weight 1.0); passing 0.0 skips the WHOLE
        # thing, including the target-side forward pass that exists only to
        # feed MK-MMD, not just the multiply (same gating as MMD_WEIGHT in
        # trainers/da/phpl_momentum.py). MK-MMD stays weak-vs-weak (source has
        # no strong view either) -- comparing it against a strong-augmented
        # target would confound domain shift with augmentation-strength shift,
        # which isn't what MK-MMD is meant to measure. Cheap for Student1
        # (LoRA), so just an extra forward call rather than reusing logits1_u's
        # own feat.
        if args.s1_mmd_weight > 0:
            _, feat_u1_weak = student1(image_u1)
            loss_mmd1 = MK_MMD(feat_x1, feat_u1_weak)
        else:
            loss_mmd1 = torch.tensor(0.0, device=device)
        # self+cross loss use the STRONG view instead (see --strong-aug).
        logits1_u, _ = student1(image_u1_strong)

        if in_warmup1:
            loss_u1_self = _pseudo_label_loss(logits1_u, prob_cross_for_s1, args.s1_loss_u_mode)
            loss_u1_cross = torch.tensor(0.0, device=device)
            loss1 = loss_x1 + loss_u1_self + args.s1_mmd_weight * loss_mmd1
        else:
            with torch.no_grad():
                logits_t1_self, _ = teacher1(image_u1)
                if args.use_debias:
                    logits_t1_self = _debias_correct(logits_t1_self, qhat1)
                prob_self1 = F.softmax(logits_t1_self, dim=-1)
            loss_u1_self = _pseudo_label_loss(logits1_u, prob_self1, args.s1_loss_u_mode)
            loss_u1_cross = _pseudo_label_loss(logits1_u, prob_cross_for_s1, args.s1_loss_cross_mode)
            loss1 = (
                loss_x1 + loss_u1_self + args.s1_cross_weight * loss_u1_cross
                + args.s1_mmd_weight * loss_mmd1
            )

        if in_warmup1:
            for pg in optim1.param_groups:
                pg["lr"] = args.s1_warmup_lr

        optim1.zero_grad()
        loss1.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in student1.parameters() if p.requires_grad], max_norm=20.0
        )
        optim1.step()
        if not in_warmup1:
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

        # s2_it_global has already been advanced past this macro-step's
        # s2_per_s1 micro-iterations by here, so ">= s2_warmup_iters" (not
        # "==") catches the macro-step Student2's own warmup ends in, even
        # though it isn't tied 1:1 to macro-steps like Student1's is.
        s2_warmup_just_ended = (
            s2_it_global >= args.s2_warmup_iters
            and s2_it_global - args.s2_per_s1 < args.s2_warmup_iters
        )
        if (
            (macro + 1) % args.eval_freq == 0
            or (macro + 1) == args.s1_total_iters
            or (macro + 1) == args.s1_warmup_iters
            or s2_warmup_just_ended
        ):
            model2.eval()
            acc1, acc2, acc_ens = evaluate(teacher1, model2, teacher_classifier_layer, test_loader, device)
            tags = []
            if (macro + 1) == args.s1_warmup_iters:
                tags.append("end of s1 warmup")
            if s2_warmup_just_ended:
                tags.append("end of s2 warmup")
            tag = f" [{', '.join(tags)}]" if tags else ""
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
