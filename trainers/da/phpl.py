'''
@inproceedings{khattak2023maple,
  title={PHPL: Multi-modal prompt learning},
  author={Khattak, Muhammad Uzair and Rasheed, Hanoona and Maaz, Muhammad and Khan, Salman and Khan, Fahad Shahbaz},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19113--19122},
  year={2023}
}

Adapted from https://github.com/muzairkhattak/multimodal-prompt-learning
'''
import os.path as osp
import sys
import json

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms as T
from PIL import Image

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, count_num_param, mkdir_if_missing, read_image
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.data.transforms import INTERPOLATION_MODES

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from trainers.baseda import *
from utils.MK_MMD import MK_MMD
from utils.clip_part import *
from utils.sora import SoraModel
from utils.templates import CUSTOM_TEMPLATES
from loralib.utils import mark_only_lora_as_trainable, apply_lora, get_lora_parameters, lora_state_dict, save_lora, load_lora,apply_lora_rn
from matplotlib import pyplot as plt
from itertools import chain
_tokenizer = _Tokenizer()

class CustomCLIP(Base_CustomCLIP):
    def __init__(self, cfg, classnames, clip_model, clip_model_teacher):
        super().__init__(cfg, classnames, clip_model)

        self.text_encoder = Simple_TextEncoder(clip_model)
        self.cfg = cfg
        if cfg.MODEL.BACKBONE.NAME.split('-')[0] == 'ViT':
            self.image_encoder = clip_model.visual
        else:  # RN50, RN101
            self.image_encoder = clip_model.visual
             
        
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        prompt_prefix = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [prompt_prefix.format(c.replace("_", " ")) for c in classnames]
        self.tokenized_prompts = clip.tokenize(prompts)

        self.clip_model_teacher = clip_model_teacher.to(self.logit_scale.device)
        self.confi = cfg.TRAINER.PHPL.CONFI

        self.dim = clip_model.ln_final.weight.shape[0]
        self.epoch = cfg.OPTIM.MAX_EPOCH

    def forward(self, image, label=None, epoch=None, source=False, train=False):

        text_features = self.text_encoder(self.tokenized_prompts.to(self.logit_scale.device))
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        image_features = self.image_encoder(image.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        if self.cfg.MODEL.BACKBONE.NAME.split('-')[0] != 'ViT':
            compute_dtype = self.logit_scale.dtype
            text_features = text_features.to(compute_dtype)
            image_features = image_features.to(compute_dtype)
            logit_scale = self.logit_scale.to(compute_dtype).exp()
            logits = logit_scale * image_features @ text_features.t()
        
        else:
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * image_features @ text_features.t()

        if train:
            if source:
                loss = F.cross_entropy(logits, label)
                return loss, logits, image_features
            else:
                logits_u, _ = self.clip_model_teacher(image.type(self.dtype), self.tokenized_prompts.to(self.logit_scale.device))
                beta = epoch / self.epoch
                logits_fusion = beta * logits + (1 - beta) * logits_u
                pseudo_label = torch.softmax(logits_fusion, dim=-1)
                max_probs, label_p = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(self.confi).float()
                epsilon = 1e-8
                if self.cfg.MODEL.BACKBONE.NAME.split('-')[0] != 'ViT':
                    epsilon = 1e-8
                loss = (F.cross_entropy(logits, label_p, reduction="none") * mask).sum() / (mask.sum() + epsilon)
                return loss, logits, image_features
        else:
            return logits
class CLIPGradCAMWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        return self.model(image, train=False)
@TRAINER_REGISTRY.register()
class PHPL(BaseDA):
   
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        self.domains = cfg.DOMAINS
        self.save = cfg.SAVE_MODEL

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        clip_model_teacher = load_clip_to_cpu(cfg, teacher=True)

        if cfg.TRAINER.PHPL.PREC == "fp32" or cfg.TRAINER.PHPL.PREC == "amp":
            clip_model.float()  # CLIP's default precision is fp16


        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model, clip_model_teacher)
        if cfg.MODEL.BACKBONE.NAME.split('-')[0] == 'ViT':
            self.list_lora_layers = apply_lora(cfg, self.model)
        else:
            self.list_lora_layers = apply_lora_rn(cfg, self.model)
        
        print("Turning off gradients in both the image and the text encoder...")
        for _, param in self.model.named_parameters():
            param.requires_grad_(False)
            if "lora" in _ :
                param.requires_grad_(True)
            
        Total_Memory = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                Total_Memory += param.numel() * param.element_size() / (1024 ** 2)
                print(str(name) + " " + str(param.requires_grad) + " " + str(
                    (param.numel() * param.element_size()) / (1024 ** 2)) + "MB")
        print("Model Total Memory : " + str(Total_Memory) + "MB")

        self.model.to(self.device)
        self.optim = build_optimizer(self.model, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("LoRA", self.model, self.optim, self.sched)
        self.scaler = GradScaler() if cfg.TRAINER.PHPL.PREC == "amp" else None
        self.t_sne_path = osp.join(self.output_dir, "tsne")
        mkdir_if_missing(self.t_sne_path)
    

    def forward_backward(self, batch_x, batch_u):
        image_x, label, image_u,label_u = self.parse_batch_train(batch_x, batch_u)

        label = label.to(image_x.device)
        label_u = label_u.to(image_u.device)

        loss_x, logits_x, source_features = self.model(image_x, label, epoch=self.epoch, source=True, train=True)
        loss_u, logits_u, target_features = self.model(image_u, epoch=self.epoch, source=False, train=True)

        loss_mmd = MK_MMD(source_features, target_features)

        loss = loss_x + loss_mmd + loss_u
        self.model_backward_and_update_with_gradient_monitoring(
            loss,
            max_norm=20.0,
            monitor_interval=10,
            clip_on_explosion=True,
        )
        loss_summary = {
            "loss": loss.item(),
            "loss_x": loss_x.item(),
            "loss_u": loss_u.item(),
            "loss_mmd": loss_mmd.item(),
            "acc_source": compute_accuracy(logits_x, label)[0].item(),
            "acc_target": compute_accuracy(logits_u, label_u)[0].item(),
        }
        
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
    

    def save_model(self, epoch, directory, is_best=False, model_name=""):
        names = self.get_model_names()

        for name in names:
            save_dir = osp.join(directory, name)
            mkdir_if_missing(save_dir)
            if model_name != "":
                save_lora(self.cfg, self.list_lora_layers, save_dir, filename='LoRA-best')
            else:
                save_lora(self.cfg, self.list_lora_layers, save_dir, filename='LoRA-last')

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        for name in names:
            if epoch is not None:
                load_lora(self.cfg, self.list_lora_layers, osp.join(directory, name), filename='LoRA-last')
            else:
                load_lora(self.cfg, self.list_lora_layers, osp.join(directory, name), filename='LoRA-best')


