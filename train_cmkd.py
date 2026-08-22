"""
Standalone CMKD phase-2 trainer.

Trains ONLY student_mlp (a small classifier head on top of image features)
against a FROZEN, already-adapted TeacherNow checkpoint from phase 1 (the
ordinary LoRA/EMA-teacher training in trainers/da/phpl_momentum.py). No LoRA
training, no EMA, no teacher_init, no student CLIP wrapper -- teacher_now is
the only CLIP model needed, loaded once and never updated again.

Deliberately NOT built on Dassl's TrainerXU/BaseDA (run_epoch/before_epoch/
after_epoch/save_model, tangled with the LoRA pipeline in phpl_momentum.py) --
just a flat, iteration-based loop. This also means the train dataloaders are
never torn down and recreated mid-run: train_loader_x/train_loader_u are each
wrapped in a persistent iterator (CyclingLoader) built ONCE up front, unlike
Dassl's run_epoch() which calls iter(loader) fresh every epoch (an expensive
worker respawn with persistent_workers=False). Periodic eval() only touches
the separate test_loader -- it never disturbs the train iterators.

Usage:
    python train_cmkd.py \\
        --root /path/to/data --domains a-c --backbone ViT-B/16 \\
        --dataset-config-file configs/datasets/officehome.yaml \\
        --config-file configs/trainers/PHPL/b32_ep10_officehome.yaml \\
        --phase1-dir output/PHPLMOMENTUM/1/officehome/a-c \\
        --output-dir output/PHPLMOMENTUM/2/officehome/a-c \\
        --print-freq 50 --eval-freq 500
    (all other flags default to VLP-UDA's own office_home.yaml values --
    total-iters=10000, optim=sgd lr=0.003 momentum=0.9 nesterov weight-decay=5e-4,
    lr-schedule=vlpuda, lambda1=0.25, lamb-gamma=1.0)
"""
import argparse
import math
import os.path as osp

import torch
from torch.nn import functional as F

from dassl.data import DataManager
from dassl.utils import mkdir_if_missing, set_random_seed

from utils.clip_part import load_clip_to_cpu
from loralib.utils import apply_lora, apply_lora_rn, load_lora
from trainers.da.phpl_momentum import _FrozenTeacherCLIP, _MLPHead
from train import setup_cfg


def _gini_impurity(pred, coe=1.0):
    """Class-balanced entropy-minimization term (CMKD/VLP-UDA) -- see
    trainers/da/phpl_momentum.py's own copy for the full explanation."""
    sum_dim = pred.sum(dim=0, keepdim=True).detach().clamp(min=1e-8)
    return torch.sum(coe * (1 - torch.sum(pred ** 2 / sum_dim, dim=-1)))


def _calibrated_coefficient(pred, pred_reference):
    epsilon = 1e-8
    distance = F.kl_div((pred + epsilon).log(), pred_reference, reduction="none").sum(-1)
    return torch.exp(-distance).detach()


def _cmkd_lamb(cur_iter, max_iter, gamma):
    p = min(cur_iter / max_iter, 1.0)
    return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0


def build_cfg(args):
    """Reuses train.py's own config plumbing (dataset/trainer yaml files,
    TRAINER.PHPLMOMENTUM.* fields, LoRA R/ALPHA/PARAMS) instead of
    reimplementing it -- setup_cfg() expects a train.py-shaped args
    Namespace, so fields this script doesn't expose get harmless defaults."""
    ns = argparse.Namespace(
        root=args.root,
        output_dir=args.output_dir,
        config_file=args.config_file,
        dataset_config_file=args.dataset_config_file,
        model_dir="",  # unused -- phase-1's checkpoint is loaded manually below
        domains=args.domains,
        source_domains=None,
        target_domains=None,
        trainer="PHPLMOMENTUM",
        backbone=args.backbone,
        head="",
        transforms=None,
        resume="",
        load_epoch=None,
        no_train=False,
        eval_only=False,
        gpu=args.gpu,
        seed=args.seed,
        save=False,
        opts=[],
    )
    return setup_cfg(ns)


class CyclingLoader:
    """A DataLoader wrapped in ONE persistent iterator, re-created only when
    it's actually exhausted -- never torn down/rebuilt on any other schedule
    (in particular, never tied to "epoch" or eval boundaries)."""

    def __init__(self, loader):
        self.loader = loader
        self._it = iter(loader)

    def next(self):
        try:
            return next(self._it)
        except StopIteration:
            self._it = iter(self.loader)
            return next(self._it)


@torch.no_grad()
def evaluate(student_mlp, teacher_now, test_loader, device):
    student_mlp.eval()
    correct, total = 0, 0
    for batch in test_loader:
        image = batch["img"].to(device)
        label = batch["label"].to(device)
        _, feat = teacher_now(image)
        logits = student_mlp(feat.float())
        correct += (logits.argmax(dim=-1) == label).sum().item()
        total += label.size(0)
    student_mlp.train()
    return 100.0 * correct / max(total, 1)


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

    parser.add_argument("--phase1-dir", type=str, required=True,
                         help="phase 1's output dir (contains a TeacherNow/ subdir)")
    parser.add_argument("--phase1-checkpoint", choices=["last", "best"], default="last")

    parser.add_argument("--total-iters", type=int, default=10000)
    parser.add_argument("--print-freq", type=int, default=50)
    parser.add_argument("--eval-freq", type=int, default=200)

    parser.add_argument("--hidden-dim", type=int, default=256)
    # Adam, not SGD: a randomly-initialized small head trained on frozen features
    # has naturally small, per-parameter-uneven mean-CE gradients (confirmed via a
    # real run -- SGD at lr=0.01 left mlp_loss_x pinned at ln(num_classes) after
    # 200 iterations). Adam's per-parameter adaptivity is the standard choice for
    # this kind of linear-probe-style training, unlike LoRA's own SGD. Defaults
    # below match VLP-UDA's own office_home.yaml as closely as possible instead
    # (optim=sgd, lr=0.003 = their backbone lr 3e-6 * multiple_lr_classifier
    # 1000, momentum=0.9 nesterov=True, weight_decay=5e-4, lr-schedule=vlpuda,
    # lambda1=0.25, lamb-gamma=1.0, total-iters=10000 = their n_epoch(20) *
    # n_iter_per_epoch(500)) -- pass --optim adam/--lr 1e-3 to use the
    # confirmed-working alternative instead (see the commit that added Adam:
    # SGD at a naive lr=0.01 left mlp_loss_x pinned at ln(num_classes) after
    # 200 iterations in an earlier, less faithfully-matched attempt).
    parser.add_argument("--optim", choices=["sgd", "adam", "adamw"], default="sgd")
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    # "vlpuda": their own decay shape (1+lr_gamma*step)^(-lr_decay), stepped every
    # iteration -- NOT multiplying by --lr again inside the lambda (LambdaLR
    # already does base_lr * lambda(step); their own code appears to double this
    # up, since their optimizer's base lr is already args.lr too).
    parser.add_argument("--lr-schedule", choices=["none", "cosine", "vlpuda"], default="vlpuda")
    parser.add_argument("--lr-gamma", type=float, default=0.0003, help="vlpuda schedule only")
    parser.add_argument("--lr-decay", type=float, default=0.75, help="vlpuda schedule only")

    parser.add_argument("--lambda1", type=float, default=0.25)
    parser.add_argument("--lamb-gamma", type=float, default=1.0)

    args = parser.parse_args()

    cfg = build_cfg(args)
    mkdir_if_missing(args.output_dir)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED)
    device = torch.device(f"cuda:{cfg.GPU}" if torch.cuda.is_available() else "cpu")

    print("Building data loaders")
    dm = DataManager(cfg)
    train_loader_x = CyclingLoader(dm.train_loader_x)
    train_loader_u = CyclingLoader(dm.train_loader_u)
    test_loader = dm.test_loader
    classnames = dm.dataset.classnames
    num_classes = dm.num_classes

    print(f"Loading frozen TeacherNow checkpoint from {args.phase1_dir}")
    clip_model = load_clip_to_cpu(cfg)
    if cfg.TRAINER.PHPLMOMENTUM.PREC in ("fp32", "amp"):
        clip_model.float()
    teacher_now = _FrozenTeacherCLIP(cfg, classnames, clip_model)
    is_vit = cfg.MODEL.BACKBONE.NAME.split('-')[0] == 'ViT'
    apply_fn = apply_lora if is_vit else apply_lora_rn
    list_lora_layers = apply_fn(cfg, teacher_now)
    filename = "LoRA-last" if args.phase1_checkpoint == "last" else "LoRA-best"
    load_lora(cfg, list_lora_layers, osp.join(args.phase1_dir, "TeacherNow"), filename=filename)
    for param in teacher_now.parameters():
        param.requires_grad_(False)
    teacher_now.to(device)
    teacher_now.eval()

    feat_dim = clip_model.text_projection.shape[1]
    student_mlp = _MLPHead(feat_dim, args.hidden_dim, num_classes).to(device)
    n_params = sum(p.numel() for p in student_mlp.parameters())
    print(f"student_mlp: {n_params:,} parameters")

    if args.optim == "sgd":
        optimizer = torch.optim.SGD(
            student_mlp.parameters(), lr=args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay, nesterov=True,
        )
    elif args.optim == "adamw":
        optimizer = torch.optim.AdamW(student_mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(student_mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = None
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_iters)
    elif args.lr_schedule == "vlpuda":
        lr_gamma, lr_decay = args.lr_gamma, args.lr_decay
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: (1.0 + lr_gamma * step) ** (-lr_decay)
        )

    best_acc = 0.0
    for it in range(args.total_iters):
        batch_x = train_loader_x.next()
        batch_u = train_loader_u.next()
        image_x = batch_x["img"].to(device)
        label_x = batch_x["label"].to(device)
        image_u = batch_u["img"].to(device)
        label_u = batch_u["label"].to(device)

        with torch.no_grad():
            _, feat_x = teacher_now(image_x)
            logits_teacher_u, feat_u = teacher_now(image_u)
            pred_teacher_u = F.softmax(logits_teacher_u, dim=-1)

        mlp_logits_x = student_mlp(feat_x.float())
        mlp_loss_x = F.cross_entropy(mlp_logits_x, label_x)

        mlp_logits_u = student_mlp(feat_u.float())
        pred_mlp_u = F.softmax(mlp_logits_u, dim=1)
        coe = _calibrated_coefficient(pred_mlp_u, pred_teacher_u)
        pred_mix_u = 0.5 * (pred_mlp_u + pred_teacher_u)

        lamb = _cmkd_lamb(it, args.total_iters, args.lamb_gamma)
        task_loss = args.lambda1 * lamb * _gini_impurity(pred_mlp_u, coe)
        distill_loss = args.lambda1 * lamb * _gini_impurity(pred_mix_u, 1 - coe)
        loss = mlp_loss_x + task_loss + distill_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        if (it + 1) % args.print_freq == 0:
            acc_source = (mlp_logits_x.argmax(-1) == label_x).float().mean().item() * 100
            acc_target = (mlp_logits_u.argmax(-1) == label_u).float().mean().item() * 100
            print(
                f"iter [{it + 1}/{args.total_iters}] loss {loss.item():.4f} "
                f"mlp_loss_x {mlp_loss_x.item():.4f} task_loss {task_loss.item():.4f} "
                f"distill_loss {distill_loss.item():.4f} lamb {lamb:.4f} "
                f"acc_source {acc_source:.2f} acc_target {acc_target:.2f} "
                f"lr {optimizer.param_groups[0]['lr']:.6e}"
            )

        if (it + 1) % args.eval_freq == 0 or (it + 1) == args.total_iters:
            acc = evaluate(student_mlp, teacher_now, test_loader, device)
            print(f"[eval] iter {it + 1}: student_mlp accuracy = {acc:.2f}%")
            torch.save(student_mlp.state_dict(), osp.join(args.output_dir, "student_mlp-last.pt"))
            if acc > best_acc:
                best_acc = acc
                torch.save(student_mlp.state_dict(), osp.join(args.output_dir, "student_mlp-best.pt"))
                print(f"  new best ({best_acc:.2f}%), saved to student_mlp-best.pt")

    print(f"Done. best student_mlp accuracy = {best_acc:.2f}%")


if __name__ == "__main__":
    main()
