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
    permanent, never-touched snapshot of the original pretrained CLIP.
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
    EKDA/PromptSRC-style): loss_scl = (L1(student_text, teacher_init_text) +
    L1(feat_x, teacher_init_image_features)) * SCL_WEIGHT, computed on image_x --
    regularizes the student's text/source-image features against drifting too far
    from pure zero-shot CLIP, independent of teacher_init's (fading) role in
    pseudo-labeling. No KL term on logits (EKDA also has one; opted out here).
  - Evaluation uses teacher_now (the adapting EMA teacher), not the student. save_model/
    load_model persist teacher_now's own LoRA snapshot directly (a separate "TeacherNow"
    checkpoint dir, alongside "Student") -- reconstructing it by re-copying student's
    weights at load time would give a DIFFERENT (raw, non-EMA'd) model than whatever
    teacher_now's EMA state actually was when that epoch's eval score was recorded.
"""
import os.path as osp
import random

import torch
import torch.nn as nn
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY
from dassl.data import DataManager
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


def _lora_param_items(model):
    return [(k, v) for k, v in model.state_dict().items() if "lora_" in k]


@torch.no_grad()
def _copy_lora_params(src_model, dst_model):
    dst_state = dst_model.state_dict()
    for k, v in _lora_param_items(src_model):
        dst_state[k].copy_(v)


@torch.no_grad()
def _ema_update_lora_params(ema_model, src_model, momentum):
    """LoRA A/B are fp32 (see loralib/layers.py's _half_except_lora), so this
    plain in-place EMA doesn't underflow the way it would on fp16 leaves."""
    ema_state = ema_model.state_dict()
    for k, v in _lora_param_items(src_model):
        ema_state[k].mul_(momentum).add_(v, alpha=1.0 - momentum)


@TRAINER_REGISTRY.register()
class PHPLMOMENTUM(BaseDA):

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.domains = cfg.DOMAINS
        self.save = cfg.SAVE_MODEL
        self.ema_momentum = cfg.TRAINER.PHPLMOMENTUM.EMA_MOMENTUM
        self.confi = cfg.TRAINER.PHPLMOMENTUM.CONFI
        self.loss_u_mode = cfg.TRAINER.PHPLMOMENTUM.LOSS_U_MODE
        if self.loss_u_mode not in ("mask", "ratio"):
            raise ValueError(f"Unknown TRAINER.PHPLMOMENTUM.LOSS_U_MODE: {self.loss_u_mode!r} (expected 'mask' or 'ratio')")
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
        self.scl_weight = cfg.TRAINER.PHPLMOMENTUM.SCL_WEIGHT

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

        self.optim = build_optimizer(self.student, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("Student", self.student, self.optim, self.sched)
        # Registered without an optimizer/scheduler: EMA-updated, never trained by gradient.
        self.register_model("TeacherNow", self.teacher_now, None, None)
        # No gradient, no EMA -- a permanent, never-touched snapshot.
        self.register_model("TeacherInit", self.teacher_init, None, None)

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
        return image_x, label_x, image_u_strong, image_u_weak, label_u

    def forward_backward(self, batch_x, batch_u):
        image_x, label_x, image_u_strong, image_u_weak, label_u = self.parse_batch_train(batch_x, batch_u)

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
            mask = max_probs.ge(self.confi).float()
            mask_ratio = mask.mean()

        logits_u, feat_u = self.student(image_u_strong)

        if self.loss_u_mode == "ratio":
            # DACS-style: never drop samples from the loss -- scale the WHOLE
            # batch's loss_u by the fraction that clears CONFI.
            loss_u = F.cross_entropy(logits_u, pseudo_label) * mask_ratio
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
            loss_scl_text = F.l1_loss(text_features, zs_text_features)
            loss_scl_image = F.l1_loss(feat_x, zs_image_features)
            loss_scl = (loss_scl_text + loss_scl_image) * self.scl_weight
        else:
            loss_scl = torch.tensor(0.0, device=self.device)

        loss = loss_x + loss_u + self.mmd_weight * loss_mmd + loss_mix + loss_scl
        self.model_backward_and_update_with_gradient_monitoring(
            loss,
            names="Student",
            max_norm=20.0,
            monitor_interval=10,
            clip_on_explosion=True,
        )

        _ema_update_lora_params(self.teacher_now, self.student, self.ema_momentum)

        loss_summary = {
            "loss": loss.item(),
            "loss_x": loss_x.item(),
            "loss_u": loss_u.item(),
            "loss_mmd": loss_mmd.item(),
            "loss_mix": loss_mix.item(),
            "loss_scl": loss_scl.item(),
            "beta": beta,
            "mask_ratio": mask_ratio.item(),
            "acc_source": compute_accuracy(logits_x, label_x)[0].item(),
            "acc_target_pseudo": compute_accuracy(logits_u, pseudo_label)[0].item(),
            "acc_target_true": compute_accuracy(logits_u, label_u)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    @torch.no_grad()
    def test(self, split=None):
        """Evaluation uses teacher_now (the adapting EMA teacher), not the student."""
        # No .eval() call needed -- _FrozenTeacherCLIP.train() (see build_model)
        # already forces this permanently.
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        data_loader = self.test_loader
        print("Do evaluation on test set (teacher_now)")

        for batch_idx, batch in enumerate(data_loader):
            input, label = self.parse_batch_test(batch)
            logits, _ = self.teacher_now(input)
            self.evaluator.process(logits, label)

        if self.cfg.DATASET.NAME == "VisDA17":
            results, accs = self.evaluator.evaluate()
        else:
            results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = "{}/{}".format(split, k)
            self.write_scalar(tag, v, self.epoch)

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
