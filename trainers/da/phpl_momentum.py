"""
PHPLMOMENTUM: Mean-Teacher self-training on top of PHPL's LoRA-CLIP architecture.

Differences vs. the original PHPL trainer (trainers/da/phpl.py):
  - Three CLIP instances: student and teacher_now are LoRA-CLIP (apply_lora applied),
    teacher_init is a PLAIN CLIP model that apply_lora is never called on -- exactly
    like PHPL's own clip_model_teacher, so its attention layers stay PyTorch's native
    nn.MultiheadAttention forever, never replaced by PlainMultiheadAttentionLoRA's
    (loralib/layers.py) reimplemented attention path. This was previously optimized
    away (teacher_now with its LoRA contribution suppressed via a "bypass" trick,
    mathematically equal to teacher_init's pretrained weights) to save a backbone copy
    of memory, but that still routed through PlainMultiheadAttentionLoRA's from-scratch
    attention implementation -- an observed confidence gap vs PHPL's own teacher
    (mask_ratio ~0.34 in PHPL at epoch 0 vs ~0.00 for the bypassed anchor) suggested a
    numerical discrepancy there, so teacher_init is back as a real separate,
    untouched-by-LoRA model for a fair, apples-to-apples comparison against PHPL.
  - teacher_now's LoRA weights are updated by EMA/momentum from student's LoRA after every
    step (no gradient, no optimizer) -- classic Mean-Teacher. teacher_init is a
    permanent, never-touched snapshot of the original pretrained CLIP. Optionally
    (cfg.TRAINER.PHPLMOMENTUM.EMA_MOMENTUM_RAMP, default [] = disabled, a single
    EMA_MOMENTUM everywhere), different transformer-block depths can use different
    momentum values (same breakpoint style as loralib's RANK_RAMP, own separate
    list) -- experimental, no literature precedent found for per-depth EMA
    momentum specifically (unlike per-layer LR decay, which is standard).
  - Pseudo-label for the student comes from a logit-space blend of teacher_now and
    teacher_init -- same formula as PHPL's CustomCLIP.forward:
        beta = epoch / (max_epoch - 1)
        logits_fusion = beta * teacher_now_logits + (1 - beta) * teacher_init_logits
        pseudo_label = argmax(softmax(logits_fusion))
    beta ramps 0 -> 1 across training, so pseudo-labels lean on the stable anchor early
    and on the adapting teacher_now later.
  - loss_u weighting is configurable via cfg.TRAINER.PHPLMOMENTUM.LOSS_U_MODE:
      "mask" (default): PHPL's own strategy -- average CE only over target samples
        whose fused teacher probability's max class exceeds CONFI (default 0.85);
        can hit loss_u == 0 entirely if none clear it in a batch.
      "ratio": no target sample is dropped from the cross-entropy; instead the WHOLE
        batch's loss_u is scaled by mask_ratio (the fraction of the batch that clears
        CONFI), so confidence still gates how much loss_u matters but never zeroes it
        out just because no single sample individually clears CONFI. Opt-in only.
      "kd": bypasses hard pseudo-labels/thresholding entirely (THRESHOLD_MODE has no
        effect in this mode). CLIP's own logit_scale (~100) already saturates softmax
        toward 0/1, so any confidence threshold ends up cutting an already near-binary
        distribution. Instead distills the student toward the fused-teacher
        distribution, softened by cfg.TRAINER.PHPLMOMENTUM.KD_TEMPERATURE (default 4.0,
        >1 softens), via KL-divergence -- every target sample contributes, weighted
        implicitly by student/teacher agreement, no hard cutoff needed. Same idea as
        EKDA's own student-training KD loss.
  - The confidence mask itself follows cfg.TRAINER.PHPLMOMENTUM.THRESHOLD_MODE:
      "fixed" (default): a single CONFI for every class, every epoch (unchanged).
      "flexmatch": from epoch 2 onward, each class gets its own threshold
        tau_c = beta_c*CONFI, where beta_c is that class's share of a running,
        never-reset count of samples confidently (>= the fixed CONFI) assigned to
        it -- FlexMatch-style Curriculum Pseudo Labeling. This count starts
        accumulating from epoch 1 (even though epoch 1's mask always uses the
        plain fixed threshold), so epoch 2 doesn't start every class's tau_c at 0.
        cfg.TRAINER.PHPLMOMENTUM.FLEXMATCH_MAPPING picks how beta_c maps to tau_c:
        "linear" (default, tau_c = beta_c*CONFI) or "convex" (FlexMatch's own
        M(beta)=beta/(2-beta) -- RISES SLOWER than linear while beta_c is still
        low-to-moderate, only catching up sharply near beta_c=1; see train.py's
        FLEXMATCH_MAPPING comment for the full explanation and a caveat).
      "flexmatch_v2": faithful to the actual FlexMatch paper (Zhang et al., NeurIPS
        2021) -- unlike "flexmatch" above, which is a simplified approximation.
        Keeps one slot per unlabeled sample (self.selected_label, size =
        len(train_u)), OVERWRITTEN (not accumulated) with that sample's current
        pseudo-label whenever confidence clears the fixed CONFI, so a sample
        revisited across many epochs is only ever counted once, at its most recent
        assignment -- fixes "flexmatch"'s tendency for tau_c to drift upward forever
        with epoch count. sigma(c) = count of slots == c; beta_c = sigma(c) /
        max(max_c sigma(c), N - sum(sigma)) if cfg.TRAINER.PHPLMOMENTUM.FLEXMATCH_WARMUP
        (default True, the paper's own warm-up term, active from iteration 1, no
        separate epoch-1 gate needed) else beta_c = sigma(c) / max_c sigma(c). The
        warm-up term can leave tau_c near 0 for an entire short training run if
        sum(sigma) never approaches N -- set FLEXMATCH_WARMUP False to skip it.
      "cbst": Class-Balanced Self-Training (Zou et al., ECCV 2018). Unlike the
        flexmatch modes (relative between classes), CBST independently picks each
        class's OWN top `portion` fraction of confidences among samples currently
        predicted as that class -- never collapses toward 0 for a rarely-confident
        class. Recomputed once per epoch (before_epoch), scanning the WHOLE target
        set with teacher_init (epoch 1) or teacher_now (epoch 2+); a class with no
        predictions that epoch falls back to CONFI. `portion` ramps linearly from
        CBST_PORTION_START to CBST_PORTION_MAX, hitting MAX at the last epoch. The
        scan uses the SAME weak-aug transform image_u_weak uses in training (not a
        clean/deterministic test-style transform), so it measures confidence the
        same way the per-batch teacher forward does.
        New threshold strategies should be added as new modes here, not new flags.
  - Student sees strong-augmented target/source images; the teacher sees a weak-augmented
    target view (asymmetric-view self-training, reduces confirmation bias) -- but only if
    cfg.TRAINER.PHPLMOMENTUM.USE_STRONG_AUG is set True. Default (False) is no weak/strong
    split at all: student and teachers all see PHPL's own single augmentation pipeline.
  - loss_mmd (Multi-Kernel MMD between source/target student features) is kept, same as
    PHPL, weighted by cfg.TRAINER.PHPLMOMENTUM.MMD_WEIGHT (default 1.0, unchanged; 0.0
    disables it entirely, skipping the computation).
  - Optional CutMix (cfg.TRAINER.PHPLMOMENTUM.USE_CUTMIX, default off): cuts a random box
    out of image_u_strong and pastes it into image_x (both already strong-aug), and adds
    an extra loss_mix = lam*CE(pred_mix, label_x) + (1-lam)*CE(pred_mix, pseudo_label) to
    the total loss, where lam is the fraction of the mixed image that stayed source.
  - Optional debiasing (cfg.TRAINER.PHPLMOMENTUM.USE_DEBIAS, default off): a single EMA
    tracker (self.qhat, shared between teacher_now and teacher_init) estimates how often
    the teachers' own predictions land on each class; DEBIAS_TAU * log(qhat) is subtracted
    from BOTH teachers' logits before fusion, to counteract systematic class bias inherited
    from CLIP's zero-shot behavior (not the true target-domain class distribution).
  - Optional Self-Consistency Loss (cfg.TRAINER.PHPLMOMENTUM.USE_SCL, default off,
    EKDA/PromptSRC-style): loss_scl = L1(student_text, teacher_init_text)*SCL_TEXT_WEIGHT +
    L1(feat_x, teacher_init_image_features)*SCL_IMAGE_WEIGHT, computed on image_x --
    regularizes the student's text/source-image features against drifting too far
    from pure zero-shot CLIP, independent of teacher_init's (fading) role in
    pseudo-labeling. No KL term on logits (EKDA also has one; opted out here).
  - Optional CMKD (cfg.TRAINER.PHPLMOMENTUM.USE_CMKD, default off, VLP-UDA-style): a
    two-phase curriculum, NOT trained jointly with LoRA from epoch 0. Phase 1 (epoch <
    CMKD_START_EPOCH) is exactly the ordinary LoRA/EMA-teacher training above, unchanged.
    From epoch >= CMKD_START_EPOCH (phase 2), Student/TeacherNow/TeacherInit are frozen --
    no optimizer step, no EMA update -- and only a small learned MLP (student_mlp) on top
    of feat_x/feat_u is trained, using TeacherNow's own (now-converged, "zero-shot" in the
    sense of no longer being trained) softmax prediction as the SOLE reference distribution
    (no more beta-blend with teacher_init -- one fixed teacher). Loss is mlp_loss_x (plain
    CE on source) plus an agreement-weighted entropy-min (Gini-impurity) objective on
    target: samples where the MLP agrees with the teacher get their OWN prediction
    sharpened (task_loss); samples where they disagree get the 50/50-averaged prediction
    sharpened instead (distill_loss) -- never a hard CE-to-pseudo-label. Both terms are
    additionally ramped by `lamb` (_cmkd_lamb, a DANN-style sigmoid 0->~1 over phase 2's own
    iteration count, gamma=CMKD_LAMB_GAMMA), so the MLP leans on plain source CE early and
    only takes the target-domain terms seriously once source performance is established.
    No regularization term on the cosine branch itself (VLP-UDA's own reg_loss) -- LoRA is
    already frozen and done adapting by phase 2. No teacher_mlp/EMA counterpart -- test()
    evaluates student_mlp directly.
  - Evaluation uses teacher_now (the adapting EMA teacher), not the student. save_model/
    load_model persist teacher_now's own LoRA snapshot directly (a separate "TeacherNow"
    checkpoint dir, alongside "Student") -- reconstructing it by re-copying student's
    weights at load time would give a DIFFERENT (raw, non-EMA'd) model than whatever
    teacher_now's EMA state actually was when that epoch's eval score was recorded.
"""
import math
import os.path as osp
import random
import re

import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY
from dassl.data import DataManager
from dassl.data.data_manager import build_data_loader
from dassl.data.transforms import build_transform
from dassl.metrics import compute_accuracy
from dassl.utils import mkdir_if_missing
from dassl.optim import build_optimizer, build_lr_scheduler

from trainers.baseda import *
from utils.clip_part import *
from utils.MK_MMD import MK_MMD
from utils.templates import CUSTOM_TEMPLATES
from loralib.utils import apply_lora, apply_lora_rn, save_lora, load_lora


class CustomCLIP(Base_CustomCLIP):
    """Same as Base_CustomCLIP.forward, but also returns image_features (needed for MK-MMD)."""

    def forward(self, image):
        text_features = self.text_encoder(self.tokenized_prompts.to(self.logit_scale.device))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits, image_features


class _FrozenTeacherCLIP(CustomCLIP):
    """teacher_now never receives gradients (its LoRA is written directly via
    EMA, never via an optimizer step) and must always run deterministically,
    always recomputing LoRA live from the current
    lora_A/lora_B rather than a stale merged snapshot (see LinearLoRA.train()/
    lora_train() in loralib/layers.py: calling .train(mode) on a LinearLoRA
    either bakes the current LoRA delta into the frozen weight once and locks
    out future updates, or re-enables dropout -- both wrong for a teacher).

    Overriding train() here to unconditionally force every submodule's
    `.training = False`, regardless of who calls .train()/.eval() or what
    mode is requested, makes this hold no matter what -- independent of
    tracking every call site in Dassl (set_model_mode fires at the start of
    every epoch and after every test()) or any call site added later."""

    def train(self, mode=True):
        for m in self.modules():
            m.training = False
        return self


def _rand_bbox(height, width, lam):
    """One random box whose area is (1 - lam) fraction of the image, same
    formula as the original CutMix paper's rand_bbox."""
    cut_rat = (1.0 - lam) ** 0.5
    cut_h = int(height * cut_rat)
    cut_w = int(width * cut_rat)

    cy = random.randint(0, height - 1)
    cx = random.randint(0, width - 1)

    y1 = max(cy - cut_h // 2, 0)
    y2 = min(cy + cut_h // 2, height)
    x1 = max(cx - cut_w // 2, 0)
    x2 = min(cx + cut_w // 2, width)
    return y1, y2, x1, x2


def _cutmix(image_a, image_b, alpha=1.0):
    """Cut a random box out of image_b and paste it into image_a (both
    [B, C, H, W], same shape). Returns (mixed_image, lam), where lam is the
    actual fraction of pixels that stayed from image_a (not the raw Beta
    sample, since the box is snapped to integer pixels)."""
    lam = float(torch.distributions.Beta(alpha, alpha).sample())
    height, width = image_a.shape[-2:]
    y1, y2, x1, x2 = _rand_bbox(height, width, lam)

    mixed = image_a.clone()
    mixed[:, :, y1:y2, x1:x2] = image_b[:, :, y1:y2, x1:x2]

    box_area = (y2 - y1) * (x2 - x1)
    lam = 1.0 - box_area / (height * width)
    return mixed, lam


def _text_features(model):
    """Encode the class-name prompts through `model`'s own text encoder,
    L2-normalized -- the same computation CustomCLIP.forward does internally,
    exposed standalone here since adding it to forward()'s return signature
    would require touching every call site (student(), teacher_now(),
    teacher_init(), test())."""
    text_features = model.text_encoder(model.tokenized_prompts.to(model.logit_scale.device))
    return text_features / text_features.norm(dim=-1, keepdim=True)


class _MLPHead(nn.Module):
    """Small learned classifier on top of image features -- CMKD's own
    (VLP-UDA-style) alternative to CLIP's fixed cosine-similarity classification.
    Stays fp32 (default nn.Linear dtype, never cast to half) -- same reasoning as
    keeping LoRA A/B fp32 despite the frozen CLIP backbone being fp16: a small,
    freshly-initialized, EMA-updated component benefits from full precision.
    Callers must cast the incoming image feature to fp32 (.float()) before
    calling this, since CLIP's own image_features are fp16 by default."""

    def __init__(self, feat_dim, hidden_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def _gini_impurity(pred, coe=1.0):
    """Class-balanced entropy-minimization term (CMKD/VLP-UDA): sharpens `pred`
    (a softmax distribution), normalized per-class by that class's total mass in
    the batch (sum_dim) to avoid the degenerate "always predict the majority
    class" failure mode plain entropy minimization is prone to. `coe` is a
    per-sample weight (e.g. agreement with another branch)."""
    sum_dim = pred.sum(dim=0, keepdim=True).detach().clamp(min=1e-8)
    return torch.sum(coe * (1 - torch.sum(pred ** 2 / sum_dim, dim=-1)))


def _calibrated_coefficient(pred, pred_reference):
    """CMKD's per-sample agreement weight between two predicted distributions:
    exp(-KL(pred, pred_reference)), close to 1 when they agree, close to 0 when
    they disagree. Detached -- used only as a fixed weight, not backprop'd through."""
    epsilon = 1e-8
    distance = F.kl_div((pred + epsilon).log(), pred_reference, reduction="none").sum(-1)
    return torch.exp(-distance).detach()


def _cmkd_lamb(cur_iter, max_iter, gamma=10.0):
    """DANN-style sigmoid ramp 0 -> ~1 over max_iter steps (Ganin & Lempitsky).
    VLP-UDA's own LambdaSheduler hardcodes gamma=1.0, which only reaches ~0.46
    by cur_iter==max_iter and rises slowly throughout -- appropriate for their
    ~10000-iteration training budget, not our much shorter one. gamma=10 (the
    classic DANN value) reaches ~0.9 by 30% of the way through and ~1.0 by the
    end, giving a faster ramp and a full-strength ceiling instead."""
    p = min(cur_iter / max_iter, 1.0)
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0


def _lora_param_items(model):
    return [(k, v) for k, v in model.state_dict().items() if "lora_" in k]


@torch.no_grad()
def _copy_lora_params(src_model, dst_model):
    dst_state = dst_model.state_dict()
    for k, v in _lora_param_items(src_model):
        dst_state[k].copy_(v)


_BLOCK_IDX_RE = re.compile(r"resblocks\.(\d+)\.")


def _block_index(key):
    """Transformer block index parsed out of a LoRA state_dict key (e.g.
    "text_encoder.transformer.resblocks.3.attn.q_proj.w_lora_A" -> 3), or None
    if the key doesn't match that pattern."""
    m = _BLOCK_IDX_RE.search(key)
    return int(m.group(1)) if m else None


def _momentum_for_key(key, base_momentum, boundaries, ramp):
    """Per-depth EMA momentum override -- see train.py's EMA_MOMENTUM_RAMP
    comment. Same breakpoint style as loralib/utils.py's _compute_rank (a block
    whose index >= boundaries[i] uses ramp[i]), but a separate, independently
    configurable list. Falls back to base_momentum if ramp is empty (the
    default/disabled case) or the key has no resolvable block index."""
    if not ramp:
        return base_momentum
    block_idx = _block_index(key)
    if block_idx is None:
        return base_momentum
    for boundary, momentum in zip(reversed(boundaries), reversed(ramp)):
        if block_idx >= boundary:
            return momentum
    return base_momentum


@torch.no_grad()
def _ema_update_lora_params(ema_model, src_model, momentum_fn):
    """LoRA A/B are fp32 (see loralib/layers.py's _half_except_lora), so this
    plain in-place EMA doesn't underflow the way it would on fp16 leaves.
    `momentum_fn(key)` returns the momentum to use for that specific param --
    pass e.g. `lambda k: 0.996` for a single constant momentum everywhere."""
    ema_state = ema_model.state_dict()
    for k, v in _lora_param_items(src_model):
        m = momentum_fn(k)
        ema_state[k].mul_(m).add_(v, alpha=1.0 - m)


@TRAINER_REGISTRY.register()
class PHPLMOMENTUM(BaseDA):

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.domains = cfg.DOMAINS
        self.save = cfg.SAVE_MODEL
        self.ema_momentum = cfg.TRAINER.PHPLMOMENTUM.EMA_MOMENTUM
        self.ema_momentum_ramp_boundaries = cfg.TRAINER.PHPLMOMENTUM.EMA_MOMENTUM_RAMP_BOUNDARIES
        self.ema_momentum_ramp = cfg.TRAINER.PHPLMOMENTUM.EMA_MOMENTUM_RAMP
        if self.ema_momentum_ramp and len(self.ema_momentum_ramp) != len(self.ema_momentum_ramp_boundaries):
            raise ValueError(
                "TRAINER.PHPLMOMENTUM.EMA_MOMENTUM_RAMP must be the same length as "
                f"EMA_MOMENTUM_RAMP_BOUNDARIES ({len(self.ema_momentum_ramp_boundaries)}), "
                f"got {len(self.ema_momentum_ramp)}"
            )
        self.confi = cfg.TRAINER.PHPLMOMENTUM.CONFI
        self.loss_u_mode = cfg.TRAINER.PHPLMOMENTUM.LOSS_U_MODE
        if self.loss_u_mode not in ("mask", "ratio", "kd"):
            raise ValueError(
                f"Unknown TRAINER.PHPLMOMENTUM.LOSS_U_MODE: {self.loss_u_mode!r} (expected 'mask', 'ratio', or 'kd')"
            )
        self.kd_temperature = cfg.TRAINER.PHPLMOMENTUM.KD_TEMPERATURE
        self.use_cutmix = cfg.TRAINER.PHPLMOMENTUM.USE_CUTMIX
        self.cutmix_alpha = cfg.TRAINER.PHPLMOMENTUM.CUTMIX_ALPHA
        self.beta_power = cfg.TRAINER.PHPLMOMENTUM.BETA_POWER

        self.use_debias = cfg.TRAINER.PHPLMOMENTUM.USE_DEBIAS
        self.debias_tau = cfg.TRAINER.PHPLMOMENTUM.DEBIAS_TAU
        self.debias_momentum = cfg.TRAINER.PHPLMOMENTUM.DEBIAS_MOMENTUM
        if self.use_debias:
            # Single EMA tracker shared between teacher_now and teacher_init.
            self.qhat = torch.full((self.num_classes,), 1.0 / self.num_classes, device=self.device)

        self.mmd_weight = cfg.TRAINER.PHPLMOMENTUM.MMD_WEIGHT

        self.use_scl = cfg.TRAINER.PHPLMOMENTUM.USE_SCL
        self.scl_text_weight = cfg.TRAINER.PHPLMOMENTUM.SCL_TEXT_WEIGHT
        self.scl_image_weight = cfg.TRAINER.PHPLMOMENTUM.SCL_IMAGE_WEIGHT

        self.threshold_mode = cfg.TRAINER.PHPLMOMENTUM.THRESHOLD_MODE
        if self.threshold_mode not in ("fixed", "flexmatch", "flexmatch_v2", "cbst"):
            raise ValueError(
                f"Unknown TRAINER.PHPLMOMENTUM.THRESHOLD_MODE: {self.threshold_mode!r} "
                "(expected 'fixed', 'flexmatch', 'flexmatch_v2', or 'cbst')"
            )
        if self.threshold_mode == "flexmatch":
            # Running per-class confident-prediction count, never reset -- see
            # train.py's THRESHOLD_MODE comment.
            self.class_confident_count = torch.zeros(self.num_classes, device=self.device)
        if self.threshold_mode in ("flexmatch", "flexmatch_v2"):
            self.flexmatch_mapping = cfg.TRAINER.PHPLMOMENTUM.FLEXMATCH_MAPPING
            if self.flexmatch_mapping not in ("linear", "convex"):
                raise ValueError(
                    f"Unknown TRAINER.PHPLMOMENTUM.FLEXMATCH_MAPPING: {self.flexmatch_mapping!r} (expected 'linear' or 'convex')"
                )
        if self.threshold_mode == "flexmatch_v2":
            # One slot per unlabeled sample, -1 = never confidently assigned yet --
            # see train.py's THRESHOLD_MODE comment for the exact FlexMatch formula.
            num_unlabeled = len(self.dm.dataset.train_u)
            self.selected_label = torch.full((num_unlabeled,), -1, dtype=torch.long, device=self.device)
            self.flexmatch_warmup = cfg.TRAINER.PHPLMOMENTUM.FLEXMATCH_WARMUP
        if self.threshold_mode == "cbst":
            self.cbst_portion_start = cfg.TRAINER.PHPLMOMENTUM.CBST_PORTION_START
            self.cbst_portion_max = cfg.TRAINER.PHPLMOMENTUM.CBST_PORTION_MAX
            # Fallback threshold for any class with zero predictions in an epoch's scan.
            self.cbst_tau = torch.full((self.num_classes,), self.confi, device=self.device)
            # Match whatever transform image_u_weak actually uses in training (weak-aug,
            # still has random_resized_crop/flip -- NOT a clean/deterministic test-style
            # transform), so CBST's scan measures confidence the same way forward_backward's
            # own per-batch teacher forward does. is_train=False is passed to
            # build_data_loader itself (not the transform) only to keep drop_last=False,
            # so the WHOLE target set gets scanned, none dropped.
            if cfg.TRAINER.PHPLMOMENTUM.USE_STRONG_AUG:
                tfm_scoring = build_transform(
                    cfg, is_train=True,
                    choices=["random_resized_crop", "random_flip", "normalize"],
                )
            else:
                tfm_scoring = build_transform(cfg, is_train=True)
            self.cbst_scoring_loader = build_data_loader(
                cfg,
                sampler_type="SequentialSampler",
                data_source=self.dm.dataset.train_u,
                batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
                tfm=tfm_scoring,
                is_train=False,
            )

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME}) x3 (student, teacher_now, teacher_init)")
        clip_model_student = load_clip_to_cpu(cfg)
        clip_model_teacher_now = load_clip_to_cpu(cfg)
        clip_model_teacher_init = load_clip_to_cpu(cfg)

        if cfg.TRAINER.PHPLMOMENTUM.PREC in ("fp32", "amp"):
            clip_model_student.float()
            clip_model_teacher_now.float()
            clip_model_teacher_init.float()

        print("Building student / teacher_now / teacher_init CLIP wrappers")
        self.student = CustomCLIP(cfg, classnames, clip_model_student)
        self.teacher_now = _FrozenTeacherCLIP(cfg, classnames, clip_model_teacher_now)
        # teacher_init is a PLAIN CustomCLIP -- apply_lora is deliberately never called
        # on it below, so its image_encoder/text_encoder keep PyTorch's native
        # nn.MultiheadAttention (never replaced by PlainMultiheadAttentionLoRA),
        # exactly matching how PHPL builds its own clip_model_teacher.
        self.teacher_init = CustomCLIP(cfg, classnames, clip_model_teacher_init)

        is_vit = cfg.MODEL.BACKBONE.NAME.split('-')[0] == 'ViT'
        apply_fn = apply_lora if is_vit else apply_lora_rn
        self.list_lora_layers = apply_fn(cfg, self.student)
        # Keep teacher_now's own LoRA layer list too -- save_model/load_model need
        # it to persist/restore teacher_now's actual EMA state directly (test()
        # evaluates teacher_now, not student, so that's what "best model" must mean).
        self.list_lora_layers_teacher_now = apply_fn(cfg, self.teacher_now)

        print("Freezing everything except student's LoRA parameters...")
        for model in (self.student, self.teacher_now, self.teacher_init):
            for param in model.parameters():
                param.requires_grad_(False)
        for name, param in self.student.named_parameters():
            if "lora" in name:
                param.requires_grad_(True)

        # teacher_now starts as an exact copy of student's (freshly-initialized) LoRA weights.
        # teacher_init has no LoRA params at all -- nothing to copy into it.
        _copy_lora_params(self.student, self.teacher_now)

        Total_Memory = 0
        for name, param in self.student.named_parameters():
            if param.requires_grad:
                Total_Memory += param.numel() * param.element_size() / (1024 ** 2)
        print(f"Student trainable (LoRA) memory: {Total_Memory:.3f}MB")

        self.student.to(self.device)
        self.teacher_now.to(self.device)
        self.teacher_init.to(self.device)

        # _FrozenTeacherCLIP.train() forces `.training = False` on every
        # submodule no matter what -- call it once here (any subsequent call
        # from Dassl's set_model_mode, e.g. at the start of every epoch, is a
        # harmless no-op). teacher_init has no LoRA layers and CLIP's native
        # nn.MultiheadAttention is built with dropout=0, so its .train()/.eval()
        # state genuinely never affects its output -- no special handling needed.
        self.teacher_now.eval()

        len_train_loader_x = len(self.train_loader_x)
        len_train_loader_u = len(self.train_loader_u)
        if self.cfg.TRAIN.COUNT_ITER == "train_x":
            self.num_batches = len_train_loader_x
        elif self.cfg.TRAIN.COUNT_ITER == "train_u":
            self.num_batches = len_train_loader_u
        elif self.cfg.TRAIN.COUNT_ITER == "smaller_one":
            self.num_batches = min(len_train_loader_x, len_train_loader_u)
        else:
            raise ValueError('Training batch name is wrong!')

        self.use_cmkd = cfg.TRAINER.PHPLMOMENTUM.USE_CMKD
        if self.use_cmkd:
            feat_dim = clip_model_student.text_projection.shape[1]
            self.student_mlp = _MLPHead(feat_dim, cfg.TRAINER.PHPLMOMENTUM.CMKD_HIDDEN_DIM, self.num_classes).to(self.device)
            self.cmkd_lambda1 = cfg.TRAINER.PHPLMOMENTUM.CMKD_LAMBDA1
            self.cmkd_lamb_gamma = cfg.TRAINER.PHPLMOMENTUM.CMKD_LAMB_GAMMA
            self.cmkd_start_epoch = cfg.TRAINER.PHPLMOMENTUM.CMKD_START_EPOCH
            self.cmkd_iter = 0
            # lamb ramps over phase 2's OWN iteration budget, not the whole run --
            # phase 1 (epoch < cmkd_start_epoch) never touches cmkd_iter.
            self.cmkd_max_iter = max((self.max_epoch - self.cmkd_start_epoch) * self.num_batches, 1)

        self.optim = build_optimizer(self.student, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("Student", self.student, self.optim, self.sched)
        # Registered without an optimizer/scheduler: EMA-updated, never trained by gradient.
        self.register_model("TeacherNow", self.teacher_now, None, None)
        # No gradient, no EMA -- a permanent, never-touched snapshot.
        self.register_model("TeacherInit", self.teacher_init, None, None)

        if self.use_cmkd:
            # student_mlp gets its OWN optimizer/scheduler, cloned from cfg.OPTIM but
            # with a different LR, no warmup, and MAX_EPOCH shrunk to phase 2's own
            # duration -- its scheduler is only ever .step()'d during phase 2 (see
            # forward_backward/update_lr), so the cosine curve should span exactly
            # that window, not the whole (phase 1 + phase 2) run.
            mlp_optim_cfg = cfg.OPTIM.clone()
            mlp_optim_cfg.defrost()
            mlp_optim_cfg.LR = cfg.TRAINER.PHPLMOMENTUM.CMKD_LR
            mlp_optim_cfg.WARMUP_EPOCH = 0
            mlp_optim_cfg.MAX_EPOCH = max(self.max_epoch - cfg.TRAINER.PHPLMOMENTUM.CMKD_START_EPOCH, 1)
            self.mlp_optim = build_optimizer(self.student_mlp, mlp_optim_cfg)
            self.mlp_sched = build_lr_scheduler(self.mlp_optim, mlp_optim_cfg)
            self.register_model("StudentMLP", self.student_mlp, self.mlp_optim, self.mlp_sched)

        self.t_sne_path = osp.join(self.output_dir, "tsne")
        mkdir_if_missing(self.t_sne_path)

    def build_data_loader(self):
        """Override: student gets a strong-aug view ("img"), both teachers get a weak-aug
        view ("img2") of the same image. train_loader_x also carries both views for
        uniformity, but only "img" is used for the source CE loss.

        If TRAINER.PHPLMOMENTUM.USE_STRONG_AUG is False (default), no custom_tfm_train
        tuple is passed at all -- DataManager falls back to its own single default
        transform (cfg.INPUT.TRANSFORMS), producing only an "img" key with ONE random
        augmentation draw per sample, exactly like PHPL. A tuple of two transforms
        (even if both are literally the same object) always makes Dassl's
        DatasetWrapper.__getitem__ apply the transform TWICE independently, producing
        two DIFFERENT random augmentations of the same image under "img"/"img2" --
        not a match for PHPL's single-draw behavior."""
        cfg = self.cfg
        if cfg.TRAINER.PHPLMOMENTUM.USE_STRONG_AUG:
            strong_tfm = build_transform(
                cfg, is_train=True,
                choices=["random_resized_crop", "random_flip", "randaugment", "normalize"],
            )
            weak_tfm = build_transform(
                cfg, is_train=True,
                choices=["random_resized_crop", "random_flip", "normalize"],
            )
            dm = DataManager(cfg, custom_tfm_train=(strong_tfm, weak_tfm))
        else:
            dm = DataManager(cfg)

        self.train_loader_x = dm.train_loader_x
        self.train_loader_u = dm.train_loader_u
        self.val_loader = dm.val_loader
        self.test_loader = dm.test_loader

        self.num_classes = dm.num_classes
        self.num_source_domains = dm.num_source_domains
        self.lab2cname = dm.lab2cname

        self.dm = dm

    def before_epoch(self):
        if self.threshold_mode == "cbst":
            self._update_cbst_thresholds()

    @torch.no_grad()
    def _update_cbst_thresholds(self):
        """Class-Balanced Self-Training (Zou et al., ECCV 2018) -- see the class
        docstring's THRESHOLD_MODE section. Scans the WHOLE target set once, then
        for each class independently sets tau_c to the confidence value at the
        top `portion` fraction among only the samples currently predicted as that
        class. A class with zero predictions this epoch keeps its CONFI fallback
        (self.cbst_tau was initialized to CONFI and is never reset to it here, so a
        class that recovers later just gets overwritten again)."""
        denom = max(self.max_epoch - 1, 1)
        portion = min(
            self.cbst_portion_start + self.epoch * (self.cbst_portion_max - self.cbst_portion_start) / denom,
            self.cbst_portion_max,
        )
        scoring_model = self.teacher_init if self.epoch == 0 else self.teacher_now

        all_max_probs, all_pred_labels = [], []
        for batch in self.cbst_scoring_loader:
            images = batch["img"].to(self.device)
            logits, _ = scoring_model(images)
            probs = F.softmax(logits, dim=-1)
            max_probs, pred_labels = torch.max(probs, dim=-1)
            all_max_probs.append(max_probs)
            all_pred_labels.append(pred_labels)
        all_max_probs = torch.cat(all_max_probs)
        all_pred_labels = torch.cat(all_pred_labels)

        for c in range(self.num_classes):
            conf_c = all_max_probs[all_pred_labels == c]
            if conf_c.numel() == 0:
                continue  # keep the existing (CONFI-initialized) fallback for this class
            sorted_conf, _ = torch.sort(conf_c, descending=True)
            idx = min(int(sorted_conf.numel() * portion), sorted_conf.numel() - 1)
            self.cbst_tau[c] = sorted_conf[idx]

        print(
            f"[cbst] epoch {self.epoch + 1}: portion={portion:.3f}, "
            f"tau_c=[{', '.join(f'{v:.3f}' for v in self.cbst_tau.tolist())}]"
        )

    def parse_batch_train(self, batch_x, batch_u):
        image_x = batch_x["img"].to(self.device)
        label_x = batch_x["label"].to(self.device)
        image_u_strong = batch_u["img"].to(self.device)
        if self.cfg.TRAINER.PHPLMOMENTUM.USE_STRONG_AUG:
            image_u_weak = batch_u["img2"].to(self.device)
        else:
            # No "img2" key exists (see build_data_loader) -- teacher and student
            # see the exact same single augmented draw, matching PHPL exactly.
            image_u_weak = image_u_strong
        label_u = batch_u["label"].to(self.device)
        # Per-sample dataset index -- only used by THRESHOLD_MODE "flexmatch_v2"'s
        # per-sample selected_label slots (Dassl's DatasetWrapper always includes it).
        index_u = batch_u["index"].to(self.device)
        return image_x, label_x, image_u_strong, image_u_weak, label_u, index_u

    def forward_backward(self, batch_x, batch_u):
        image_x, label_x, image_u_strong, image_u_weak, label_u, index_u = self.parse_batch_train(batch_x, batch_u)

        if self.use_cmkd and self.epoch >= self.cmkd_start_epoch:
            # Phase 2: Student/TeacherNow/TeacherInit are frozen -- see
            # _forward_backward_cmkd and the class docstring's CMKD section.
            return self._forward_backward_cmkd(image_x, label_x, image_u_weak, label_u)

        self.student.train()
        # teacher_now is a _FrozenTeacherCLIP -- its train() unconditionally
        # forces eval-like, no-merge behavior, so no explicit call is needed
        # (or harmful) here.

        logits_x, feat_x = self.student(image_x)
        loss_x = F.cross_entropy(logits_x, label_x)

        with torch.no_grad():
            logits_teacher_now, _ = self.teacher_now(image_u_weak)
            logits_teacher_init, _ = self.teacher_init(image_u_weak)

            denom = max(self.max_epoch - 1, 1)
            # BETA_POWER=1.0 (default) is the original linear ramp; <1.0 (e.g. 0.5,
            # sqrt) makes beta rise faster early on, so teacher_init's influence
            # drops off sooner than the linear schedule.
            beta = (self.epoch / denom) ** self.beta_power

            if self.use_debias:
                # Logit-adjustment debiasing (UniMoS/DebiasPL-style) against CLIP's
                # own zero-shot class bias -- see train.py's USE_DEBIAS comment.
                # Update qhat from the RAW (pre-debias) predictions of both teachers,
                # weighted by how much each actually contributes to the fused
                # pseudo-label (the same beta used below), then apply the SAME
                # correction (from BEFORE this update) to both teachers' logits.
                prob_now_raw = F.softmax(logits_teacher_now, dim=-1)
                prob_init_raw = F.softmax(logits_teacher_init, dim=-1)

                logits_teacher_now = logits_teacher_now - self.debias_tau * torch.log(self.qhat)
                logits_teacher_init = logits_teacher_init - self.debias_tau * torch.log(self.qhat)

                combined_prob = beta * prob_now_raw + (1 - beta) * prob_init_raw
                self.qhat = self.debias_momentum * self.qhat + (1 - self.debias_momentum) * combined_prob.mean(dim=0)

            # Fuse in logit space, softmax once -- same as PHPL's CustomCLIP.forward
            # (softmax(a*x+b*y) is a weighted GEOMETRIC mean of softmax(x)/softmax(y),
            # not an arithmetic mean of two already-softmaxed distributions; CONFI was
            # calibrated against this sharper geometric-mean formula).
            logits_fusion = beta * logits_teacher_now + (1 - beta) * logits_teacher_init
            prob_fusion = F.softmax(logits_fusion, dim=-1)
            max_probs, pseudo_label = torch.max(prob_fusion, dim=-1)

            # Samples confidently (>= the FIXED CONFI) predicted as each class --
            # used both for the plain fixed-threshold mask below and, in "flexmatch"
            # mode, to update class_confident_count.
            confident = max_probs.ge(self.confi).float()

            if self.threshold_mode == "flexmatch":
                # Accumulate class_confident_count from epoch 1 onward (even though
                # epoch 1's mask below still uses the plain fixed threshold) so epoch
                # 2 doesn't start from an all-zero cold start -- warm-starting this
                # way means tau_c isn't 0 for every class the moment epoch 2 begins,
                # cutting down the noisy "everything passes" window FlexMatch would
                # otherwise have right after the per-class thresholds switch on.
                self.class_confident_count.scatter_add_(0, pseudo_label, confident)

            if self.threshold_mode == "flexmatch" and self.epoch >= 1:
                # Epoch 1 (warmup) always uses the plain fixed threshold, same as
                # "fixed" mode -- FlexMatch's per-class thresholds only kick in from
                # epoch 2.
                count_max = self.class_confident_count.max().clamp(min=1.0)
                beta_c = self.class_confident_count / count_max
                if self.flexmatch_mapping == "convex":
                    # FlexMatch's own M(beta) = beta/(2-beta) -- <= beta_c everywhere
                    # except at the 0/1 endpoints, so tau_c rises SLOWER than linear
                    # while confidence is still low-to-moderate (see train.py's
                    # FLEXMATCH_MAPPING comment).
                    beta_c = beta_c / (2.0 - beta_c)
                tau_c = beta_c * self.confi
                mask = max_probs.ge(tau_c[pseudo_label]).float()

                if (self.batch_idx + 1) % 10 == 0:
                    tau_list = ", ".join(f"{v:.3f}" for v in tau_c.tolist())
                    print(f"[flexmatch tau_c] epoch {self.epoch + 1} batch {self.batch_idx + 1}: [{tau_list}]")
            elif self.threshold_mode == "flexmatch_v2":
                # Overwrite (not accumulate) each confidently-predicted sample's own
                # slot with its current pseudo-label -- a sample revisited across many
                # epochs only ever counts once, at its most recent assignment.
                confident_bool = confident.bool()
                self.selected_label[index_u[confident_bool]] = pseudo_label[confident_bool]

                sigma = torch.bincount(
                    self.selected_label[self.selected_label >= 0], minlength=self.num_classes
                ).float()
                if self.flexmatch_warmup:
                    # Paper's own warm-up (Eq. 11): while most samples are still
                    # unassigned, N-sum(sigma) dominates the denominator, keeping
                    # every beta_c small instead of one class spiking to 1 from a
                    # few early hits. Can leave tau_c near 0 for the WHOLE run if
                    # sum(sigma) never gets close to N within a short training
                    # budget -- see FLEXMATCH_WARMUP in train.py.
                    n_unused = (self.selected_label < 0).sum().float()
                    denom = torch.maximum(sigma.max(), n_unused).clamp(min=1.0)
                else:
                    denom = sigma.max().clamp(min=1.0)
                beta_c = sigma / denom
                if self.flexmatch_mapping == "convex":
                    beta_c = beta_c / (2.0 - beta_c)
                tau_c = beta_c * self.confi
                mask = max_probs.ge(tau_c[pseudo_label]).float()

                if (self.batch_idx + 1) % 10 == 0:
                    tau_list = ", ".join(f"{v:.3f}" for v in tau_c.tolist())
                    print(f"[flexmatch_v2 tau_c] epoch {self.epoch + 1} batch {self.batch_idx + 1}: [{tau_list}]")
            elif self.threshold_mode == "cbst":
                # self.cbst_tau was computed once for the whole epoch in
                # before_epoch() -- no per-batch update here.
                mask = max_probs.ge(self.cbst_tau[pseudo_label]).float()
            else:
                mask = confident

            mask_ratio = mask.mean()

        logits_u, feat_u = self.student(image_u_strong)

        if self.loss_u_mode == "ratio":
            # DACS-style: never drop samples from the loss -- scale the WHOLE
            # batch's loss_u by the fraction that clears CONFI.
            loss_u = F.cross_entropy(logits_u, pseudo_label) * mask_ratio
        elif self.loss_u_mode == "kd":
            # Bypasses hard pseudo-labels/thresholding entirely -- distill the
            # student toward the (softened) fused-teacher distribution directly.
            # logits_fusion was computed under torch.no_grad() above, so only
            # logits_u (student) carries gradient here, as intended.
            T = self.kd_temperature
            loss_u = F.kl_div(
                F.log_softmax(logits_u / T, dim=1),
                F.softmax(logits_fusion / T, dim=1),
                reduction="batchmean",
            ) * (T * T)
        else:
            # PHPL's own strategy: average CE only over samples that clear CONFI.
            # Can hit "0 samples pass -> loss_u == 0" early in training.
            epsilon = 1e-8
            loss_u = (F.cross_entropy(logits_u, pseudo_label, reduction="none") * mask).sum() / (mask.sum() + epsilon)

        if self.mmd_weight > 0:
            loss_mmd = MK_MMD(feat_x, feat_u)
        else:
            loss_mmd = torch.tensor(0.0, device=self.device)

        if self.use_cutmix:
            # Cut a box out of image_u_strong and paste it into image_x (both
            # already strong-aug); truncate to the smaller batch in case the
            # two loaders' batch sizes ever differ (e.g. a partial last batch).
            n = min(image_x.size(0), image_u_strong.size(0))
            mixed_image, lam = _cutmix(image_x[:n], image_u_strong[:n], alpha=self.cutmix_alpha)
            logits_mix, _ = self.student(mixed_image)
            loss_mix = (
                lam * F.cross_entropy(logits_mix, label_x[:n])
                + (1 - lam) * F.cross_entropy(logits_mix, pseudo_label[:n])
            )
        else:
            loss_mix = torch.tensor(0.0, device=self.device)

        if self.use_scl:
            # Self-Consistency Loss (EKDA/PromptSRC-style): pull the student's text
            # and source-image features toward teacher_init's (pure zero-shot CLIP)
            # matching features -- computed on image_x, mirroring EKDA's own SCL
            # (which runs in its "train teacher" phase, also on the source batch).
            with torch.no_grad():
                _, zs_image_features = self.teacher_init(image_x)
                zs_text_features = _text_features(self.teacher_init)
            text_features = _text_features(self.student)
            # Raw (unweighted) values logged separately below -- SCL_TEXT_WEIGHT/
            # SCL_IMAGE_WEIGHT are EKDA/PromptSRC's own values, not verified to be
            # well-calibrated for LoRA; check these against loss_x/loss_u's scale
            # before trusting the weighted loss_scl.
            loss_scl_text_raw = F.l1_loss(text_features, zs_text_features)
            loss_scl_image_raw = F.l1_loss(feat_x, zs_image_features)
            loss_scl = loss_scl_text_raw * self.scl_text_weight + loss_scl_image_raw * self.scl_image_weight
        else:
            loss_scl_text_raw = torch.tensor(0.0, device=self.device)
            loss_scl_image_raw = torch.tensor(0.0, device=self.device)
            loss_scl = torch.tensor(0.0, device=self.device)

        loss = loss_x + loss_u + self.mmd_weight * loss_mmd + loss_mix + loss_scl
        self.model_backward_and_update_with_gradient_monitoring(
            loss,
            names="Student",
            max_norm=20.0,
            monitor_interval=10,
            clip_on_explosion=True,
        )

        if self.ema_momentum_ramp:
            momentum_fn = lambda k: _momentum_for_key(  # noqa: E731
                k, self.ema_momentum, self.ema_momentum_ramp_boundaries, self.ema_momentum_ramp
            )
        else:
            momentum_fn = lambda k: self.ema_momentum  # noqa: E731
        _ema_update_lora_params(self.teacher_now, self.student, momentum_fn)

        loss_summary = {
            "loss": loss.item(),
            "loss_x": loss_x.item(),
            "loss_u": loss_u.item(),
            "loss_mmd": loss_mmd.item(),
            "loss_mix": loss_mix.item(),
            "loss_scl": loss_scl.item(),
            "loss_scl_text_raw": loss_scl_text_raw.item(),
            "loss_scl_image_raw": loss_scl_image_raw.item(),
            "beta": beta,
            "mask_ratio": mask_ratio.item(),
            "acc_source": compute_accuracy(logits_x, label_x)[0].item(),
            "acc_target_pseudo": compute_accuracy(logits_u, pseudo_label)[0].item(),
            "acc_target_true": compute_accuracy(logits_u, label_u)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            # Explicit name -- StudentMLP's scheduler (registered once use_cmkd is
            # on) must NOT advance during phase 1, or its LR would already be
            # decayed by the time phase 2 actually starts training it.
            self.update_lr(names="Student")

        return loss_summary

    def _forward_backward_cmkd(self, image_x, label_x, image_u, label_u):
        """CMKD phase 2 (see class docstring): Student/TeacherNow are frozen --
        forward-only, under no_grad -- and only student_mlp is trained. Features come
        from teacher_now, not student -- by this point they're close (EMA-converged)
        but not identical, and test() evaluates student_mlp on teacher_now's features,
        so training on the same source avoids a train/eval feature mismatch. TeacherNow's
        own softmax prediction is the sole reference distribution (no beta-blend with
        teacher_init anymore, no EMA teacher_mlp counterpart)."""
        with torch.no_grad():
            _, feat_x = self.teacher_now(image_x)
            logits_teacher_now, feat_u = self.teacher_now(image_u)
            pred_teacher_u = F.softmax(logits_teacher_now, dim=-1)

        mlp_logits_x = self.student_mlp(feat_x.float())
        mlp_loss_x = F.cross_entropy(mlp_logits_x, label_x)

        mlp_logits_u = self.student_mlp(feat_u.float())
        pred_mlp_u = F.softmax(mlp_logits_u, dim=1)
        coe = _calibrated_coefficient(pred_mlp_u, pred_teacher_u)
        pred_mix_u = 0.5 * (pred_mlp_u + pred_teacher_u)

        lamb = _cmkd_lamb(self.cmkd_iter, self.cmkd_max_iter, self.cmkd_lamb_gamma)
        task_loss = self.cmkd_lambda1 * lamb * _gini_impurity(pred_mlp_u, coe)
        distill_loss = self.cmkd_lambda1 * lamb * _gini_impurity(pred_mix_u, 1 - coe)
        loss_cmkd = mlp_loss_x + task_loss + distill_loss
        self.cmkd_iter += 1

        self.model_backward_and_update_with_gradient_monitoring(
            loss_cmkd,
            names="StudentMLP",
            max_norm=20.0,
            monitor_interval=10,
            clip_on_explosion=True,
        )

        loss_summary = {
            "loss_cmkd": loss_cmkd.item(),
            "mlp_loss_x": mlp_loss_x.item(),
            "task_loss": task_loss.item(),
            "distill_loss": distill_loss.item(),
            "lamb": lamb,
            "acc_source_mlp": compute_accuracy(mlp_logits_x, label_x)[0].item(),
            "acc_target_mlp": compute_accuracy(mlp_logits_u, label_u)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr(names="StudentMLP")

        return loss_summary

    @torch.no_grad()
    def test(self, split=None):
        """Evaluation uses teacher_now (the adapting EMA teacher), not the student.
        The official return value (used for best-model selection) is still teacher_now's
        cosine-similarity accuracy; when USE_CMKD is on, student_mlp's own accuracy is
        also computed and logged (print + write_scalar) purely for comparison -- it's
        just random-init noise before cmkd_start_epoch, meaningful only afterward."""
        # No .eval() call needed -- _FrozenTeacherCLIP.train() (see build_model)
        # already forces this permanently.
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        data_loader = self.test_loader
        print("Do evaluation on test set (teacher_now)")

        mlp_correct, mlp_total = 0, 0
        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            logits, feat = self.teacher_now(input)
            self.evaluator.process(logits, label)

            if self.use_cmkd:
                # Diagnostic only -- not the official metric, not used for
                # best-model selection. Manual accuracy, not the full evaluator
                # machinery (per-class/confusion-matrix stats not needed here).
                mlp_logits = self.student_mlp(feat.float())
                mlp_correct += (mlp_logits.argmax(dim=-1) == label).sum().item()
                mlp_total += label.size(0)

        if self.cfg.DATASET.NAME == "VisDA17":
            results, accs = self.evaluator.evaluate()
        else:
            results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

        if self.use_cmkd:
            mlp_acc = 100.0 * mlp_correct / max(mlp_total, 1)
            print(f"[cmkd] epoch {self.epoch + 1}: student_mlp accuracy = {mlp_acc:.2f}%")
            self.write_scalar("{}/mlp_accuracy".format(split), mlp_acc, self.epoch)

        return results["accuracy"]

    def save_model(self, epoch, directory, is_best=False, model_name=""):
        filename = "LoRA-best" if model_name != "" else "LoRA-last"

        student_dir = osp.join(directory, "Student")
        mkdir_if_missing(student_dir)
        save_lora(self.cfg, self.list_lora_layers, student_dir, filename=filename)

        # test()/eval evaluates teacher_now, not student -- "best model" must mean
        # teacher_now's actual EMA state at that point, not student's raw weights.
        teacher_now_dir = osp.join(directory, "TeacherNow")
        mkdir_if_missing(teacher_now_dir)
        save_lora(self.cfg, self.list_lora_layers_teacher_now, teacher_now_dir, filename=filename)

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return
        filename = "LoRA-last" if epoch is not None else "LoRA-best"
        load_lora(self.cfg, self.list_lora_layers, osp.join(directory, "Student"), filename=filename)
        load_lora(self.cfg, self.list_lora_layers_teacher_now, osp.join(directory, "TeacherNow"), filename=filename)
