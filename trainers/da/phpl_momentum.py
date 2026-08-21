"""
PHPLMOMENTUM: Mean-Teacher self-training on top of PHPL's LoRA-CLIP architecture.

Differences vs. the original PHPL trainer (trainers/da/phpl.py):
  - No separate "teacher backbone": student, teacher_now and teacher_init are three
    independent LoRA-CLIP instances built from the SAME backbone (cfg.MODEL.BACKBONE.NAME).
    (Note: this duplicates the frozen CLIP backbone weights 3x in memory -- a correctness-
    first version. Sharing the frozen base weights across the three LoRA instances is a
    possible follow-up optimization, not done here.)
  - teacher_now's LoRA weights are updated by EMA/momentum from student's LoRA after every
    step (no gradient, no optimizer) -- classic Mean-Teacher.
  - teacher_init is a LoRA snapshot taken once at t=0 and frozen forever -- a stable anchor.
  - Pseudo-label for the student comes from a probability-space blend of the two teachers:
        beta = epoch / (max_epoch - 1)
        prob_fusion = beta * softmax(teacher_now_logits) + (1 - beta) * softmax(teacher_init_logits)
        pseudo_label = argmax(prob_fusion)
    beta ramps 0 -> 1 across training, so pseudo-labels lean on the stable teacher_init early
    and on the adapting teacher_now later.
  - No confidence threshold / FixMatch-style masking for now (dropped on purpose, per current
    experiment plan) -- loss_u is a plain CE over the whole target batch.
  - Student sees strong-augmented target/source images; both teachers see weak-augmented
    target images (asymmetric-view self-training, reduces confirmation bias).
  - loss_mmd (Multi-Kernel MMD between source/target student features) is kept, same as PHPL.
  - Evaluation uses teacher_now (the adapting EMA teacher), not the student.
"""
import os.path as osp

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
        self.teacher_now = CustomCLIP(cfg, classnames, clip_model_teacher_now)
        self.teacher_init = CustomCLIP(cfg, classnames, clip_model_teacher_init)

        is_vit = cfg.MODEL.BACKBONE.NAME.split('-')[0] == 'ViT'
        apply_fn = apply_lora if is_vit else apply_lora_rn
        self.list_lora_layers = apply_fn(cfg, self.student)
        apply_fn(cfg, self.teacher_now)
        apply_fn(cfg, self.teacher_init)

        print("Freezing everything except student's LoRA parameters...")
        for model in (self.student, self.teacher_now, self.teacher_init):
            for param in model.parameters():
                param.requires_grad_(False)
        for name, param in self.student.named_parameters():
            if "lora" in name:
                param.requires_grad_(True)

        # teacher_now starts as an exact copy of student's (freshly-initialized) LoRA weights;
        # teacher_init is a permanent snapshot of that same starting point, never touched again.
        _copy_lora_params(self.student, self.teacher_now)
        _copy_lora_params(self.student, self.teacher_init)

        Total_Memory = 0
        for name, param in self.student.named_parameters():
            if param.requires_grad:
                Total_Memory += param.numel() * param.element_size() / (1024 ** 2)
        print(f"Student trainable (LoRA) memory: {Total_Memory:.3f}MB")

        self.student.to(self.device)
        self.teacher_now.to(self.device)
        self.teacher_init.to(self.device)

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
        self.register_model("TeacherInit", self.teacher_init, None, None)

        self.t_sne_path = osp.join(self.output_dir, "tsne")
        mkdir_if_missing(self.t_sne_path)

    def build_data_loader(self):
        """Override: student gets a strong-aug view ("img"), both teachers get a weak-aug
        view ("img2") of the same image. train_loader_x also carries both views for
        uniformity, but only "img" is used for the source CE loss."""
        cfg = self.cfg
        strong_tfm = build_transform(
            cfg, is_train=True,
            choices=["random_resized_crop", "random_flip", "randaugment", "normalize"],
        )
        weak_tfm = build_transform(
            cfg, is_train=True,
            choices=["random_resized_crop", "random_flip", "normalize"],
        )
        dm = DataManager(cfg, custom_tfm_train=(strong_tfm, weak_tfm))

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
        image_u_weak = batch_u["img2"].to(self.device)
        label_u = batch_u["label"].to(self.device)
        return image_x, label_x, image_u_strong, image_u_weak, label_u

    def forward_backward(self, batch_x, batch_u):
        image_x, label_x, image_u_strong, image_u_weak, label_u = self.parse_batch_train(batch_x, batch_u)

        self.student.train()
        self.teacher_now.eval()
        self.teacher_init.eval()

        logits_x, feat_x = self.student(image_x)
        loss_x = F.cross_entropy(logits_x, label_x)

        with torch.no_grad():
            logits_teacher_now, _ = self.teacher_now(image_u_weak)
            logits_teacher_init, _ = self.teacher_init(image_u_weak)

            prob_now = F.softmax(logits_teacher_now, dim=-1)
            prob_init = F.softmax(logits_teacher_init, dim=-1)

            denom = max(self.max_epoch - 1, 1)
            beta = self.epoch / denom
            prob_fusion = beta * prob_now + (1 - beta) * prob_init
            pseudo_label = prob_fusion.argmax(dim=-1)

        logits_u, feat_u = self.student(image_u_strong)
        loss_u = F.cross_entropy(logits_u, pseudo_label)

        loss_mmd = MK_MMD(feat_x, feat_u)

        loss = loss_x + loss_u + loss_mmd
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
            "beta": beta,
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
        self.teacher_now.eval()
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
        save_dir = osp.join(directory, "Student")
        mkdir_if_missing(save_dir)
        filename = "LoRA-best" if model_name != "" else "LoRA-last"
        save_lora(self.cfg, self.list_lora_layers, save_dir, filename=filename)

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return
        filename = "LoRA-last" if epoch is not None else "LoRA-best"
        load_lora(self.cfg, self.list_lora_layers, osp.join(directory, "Student"), filename=filename)
        _copy_lora_params(self.student, self.teacher_now)
