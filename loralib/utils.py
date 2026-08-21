import os
import warnings

import torch
import torch.nn as nn

from typing import Dict

from .layers import LoRALayer, PlainMultiheadAttentionLoRA, Conv2d, LinearLoRA, BatchNormLoRA

INDEX_POSITIONS_TEXT = {
    'top1': [11],
    'top2': [10, 11],
    'top3': [9, 10, 11],
    'bottom': [0, 1, 2, 3],
    'mid': [4, 5, 6, 7],
    'up': [8, 9, 10, 11],
    'half-up': [6, 7, 8, 9, 10, 11],
    'half-bottom': [0, 1, 2, 3, 4, 5],
    'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}


INDEX_POSITIONS_VISION = {
    'ViT-B/16': {
        'top': [11],
        'top3': [9, 10, 11],
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},
    'ViT-B/32': {
        'bottom': [0, 1, 2, 3],
        'mid': [4, 5, 6, 7],
        'up': [8, 9, 10, 11],
        'half-up': [6, 7, 8, 9, 10, 11],
        'half-bottom': [0, 1, 2, 3, 4, 5],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]},

    'ViT-L/14': {
        'half-up': [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        'half-bottom': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'all': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]}
}


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                    hasattr(m, 'bias') and \
                    m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


def get_lora_parameters(model, bias='none'):
    params = []
    for name, param in model.named_parameters():
        if bias == 'none':
            if 'lora_' in name:
                params.append(param)
        elif bias == 'all':
            if 'lora_' in name or 'bias' in name:
                params.append(param)
        elif bias == 'lora_only':
            if 'lora_' in name:
                params.append(param)
                bias_name = name.split('lora_')[0] + 'bias'
                if bias_name in model.state_dict():
                    bias_param = dict(model.named_parameters())[bias_name]
                    params.append(bias_param)
        else:
            raise NotImplementedError
    return params


def apply_lora(cfg, model, encoder='both'):
    list_lora_layers = []
    if encoder in ('both', 'text'):
        _apply_text_lora(cfg, model, list_lora_layers)
    if encoder in ('both', 'vision', 'image'):
        _apply_vit_lora(cfg, model, list_lora_layers)
    return list_lora_layers


 

def _build_conv2d_lora(conv_module, rank, cfg):
    """Create a Conv2d LoRA wrapper compatible with both new and legacy implementations."""
    kwargs = dict(
        r=rank,
        lora_alpha=cfg.TRAINER[cfg.TRAINER.NAME].ALPHA,
        dropout_rate=cfg.TRAINER[cfg.TRAINER.NAME].DROPOUT_RATE,
    )
    try:
        new_conv = Conv2d(conv_module, **kwargs)
    except TypeError:
        # legacy Conv2d signature, rebuild using raw dimensions
        kernel_size = conv_module.kernel_size
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        legacy_kwargs = dict(
            stride=conv_module.stride,
            padding=conv_module.padding,
            dilation=conv_module.dilation,
            groups=conv_module.groups,
            bias=conv_module.bias is not None,
            padding_mode=conv_module.padding_mode,
        )
        new_conv = Conv2d(
            in_channels=conv_module.in_channels,
            out_channels=conv_module.out_channels,
            kernel_size=kernel_size,
            r=rank,
            lora_alpha=cfg.TRAINER[cfg.TRAINER.NAME].ALPHA,
            **legacy_kwargs,
        )
        new_conv.to(device=conv_module.weight.device, dtype=conv_module.weight.dtype)
        with torch.no_grad():
            new_conv.weight.data.copy_(conv_module.weight.data)
            if conv_module.bias is not None and new_conv.bias is not None:
                new_conv.bias.data.copy_(conv_module.bias.data)
        if hasattr(new_conv, "init_lora_param"):
            new_conv.init_lora_param()
    return new_conv

def _build_depthwise_conv2d_lora(conv_module, rank, cfg):
    """Depthwise Conv2d LoRA wrapper that enforces groups==in_channels."""
    if not isinstance(conv_module, nn.Conv2d):
        raise TypeError("Depthwise LoRA expects an nn.Conv2d module.")
    if conv_module.groups != conv_module.in_channels:
        raise ValueError("The provided conv layer is not depthwise (groups != in_channels).")

    depthwise_rank = getattr(cfg.TRAINER[cfg.TRAINER.NAME], "DEPTHWISE_R", rank)
    depthwise_alpha = getattr(cfg.TRAINER[cfg.TRAINER.NAME], "DEPTHWISE_ALPHA", cfg.TRAINER[cfg.TRAINER.NAME].ALPHA)
    depthwise_dropout = getattr(cfg.TRAINER[cfg.TRAINER.NAME], "DEPTHWISE_DROPOUT", cfg.TRAINER[cfg.TRAINER.NAME].DROPOUT_RATE)

    kwargs = dict(
        r=depthwise_rank,
        lora_alpha=depthwise_alpha,
        dropout_rate=depthwise_dropout,
    )

    try:
        new_conv = Conv2d(conv_module, **kwargs)
    except TypeError:
        kernel_size = conv_module.kernel_size
        if isinstance(kernel_size, tuple) and len(kernel_size) > 0:
            kernel_size = kernel_size[0]
        legacy_kwargs = dict(
            stride=conv_module.stride,
            padding=conv_module.padding,
            dilation=conv_module.dilation,
            groups=conv_module.groups,
            bias=conv_module.bias is not None,
            padding_mode=conv_module.padding_mode,
        )
        new_conv = Conv2d(
            in_channels=conv_module.in_channels,
            out_channels=conv_module.out_channels,
            kernel_size=kernel_size,
            r=depthwise_rank,
            lora_alpha=depthwise_alpha,
            dropout_rate=depthwise_dropout,
            **legacy_kwargs,
        )
        new_conv.to(device=conv_module.weight.device, dtype=conv_module.weight.dtype)
        with torch.no_grad():
            new_conv.weight.data.copy_(conv_module.weight.data)
            if conv_module.bias is not None and new_conv.bias is not None:
                new_conv.bias.data.copy_(conv_module.bias.data)
        if hasattr(new_conv, "init_lora_param"):
            new_conv.init_lora_param()
    return new_conv

# --------------------------------------------------
# ResNet-specific helpers
# --------------------------------------------------

def _iter_resnet_blocks(resnet_module):
    """Yield (stage_name, block) pairs for layer1-4 in order."""
    for stage_name in ["layer1", "layer2", "layer3", "layer4"]:
        stage = getattr(resnet_module, stage_name, None)
        if stage is None:
            continue
        for block in stage:
            yield stage_name, block

def _apply_stem_lora(cfg, model, list_lora_layers):
    """Optionally inject LoRA into RN stem conv1/conv2/conv3."""
    stem_flags = getattr(cfg.TRAINER[cfg.TRAINER.NAME], "RN_STEM_CONVS", [])
    if not stem_flags:
        return

    rank = 3
    valid_convs = ("conv1","conv2")
    for conv_name in valid_convs:
        if conv_name not in stem_flags:
            continue
        stem_layer = getattr(model.image_encoder, conv_name, None)
        if not isinstance(stem_layer, nn.Conv2d):
            continue
        new_conv = _build_depthwise_conv2d_lora(stem_layer, rank, cfg)
        setattr(model.image_encoder, conv_name, new_conv)
        list_lora_layers.append(new_conv)

def _apply_stage_lora(cfg, model, list_lora_layers):
    """Inject LoRA into RN bottleneck conv layers (plus optional downsample)."""
    target_stages = getattr(
        cfg.TRAINER[cfg.TRAINER.NAME],
        "RN_LORA_STAGES",
        ["layer3","layer4"],
    )
    target_convs = getattr(
        cfg.TRAINER[cfg.TRAINER.NAME],
        "RN_STAGE_CONVS",
        ["conv3"],
    )
    # 2 82.1
    enable_downsample = getattr(
        cfg.TRAINER[cfg.TRAINER.NAME],
        "RN_ENABLE_DOWNSAMPLE_LORA",
        False,
    )
    downsample_stages = getattr(
        cfg.TRAINER[cfg.TRAINER.NAME],
        "RN_DOWNSAMPLE_STAGES",
        target_stages,
    )
    # 
    block_idx = 0
    rank_map = { "layer3": 3, "layer4": 3}
    for stage_name, block in _iter_resnet_blocks(model.image_encoder):
        rank = rank_map.get(stage_name, cfg.TRAINER[cfg.TRAINER.NAME].R)
        print(f"Applying LoRA rank {rank} to ResNet stage {stage_name}, block {block_idx}")
        if stage_name in target_stages:
            conv_names = list(target_convs)
            for conv_name in conv_names:
                submodule = getattr(block, conv_name, None)
                if isinstance(submodule, nn.Conv2d):
                    new_conv_lora = _build_conv2d_lora(submodule, rank, cfg)
                    setattr(block, conv_name, new_conv_lora)
                    list_lora_layers.append(new_conv_lora)
            if enable_downsample:
                _apply_downsample_lora(
                    cfg,
                    block,
                    stage_name,
                    downsample_stages,
                    list_lora_layers,
                    rank,
                )
            block_idx += 1
    return block_idx


def _apply_downsample_lora(cfg, block, stage_name, allowed_stages, list_lora_layers, rank):
    """Inject LoRA into downsample Sequential convs for specified ResNet stages."""
    if stage_name not in allowed_stages:
        return

    downsample = getattr(block, "downsample", None)
    if not isinstance(downsample, nn.Sequential):
        return

    target_conv_indices = getattr(
        cfg.TRAINER[cfg.TRAINER.NAME],
        "RN_DOWNSAMPLE_CONVS",
        None,
    )

    for name, submodule in list(downsample._modules.items()):
        if target_conv_indices is not None and name not in target_conv_indices:
            continue
        if isinstance(submodule, nn.Conv2d):
            new_conv_lora = _build_conv2d_lora(submodule, rank, cfg)
            downsample._modules[name] = new_conv_lora
            list_lora_layers.append(new_conv_lora)
        elif isinstance(submodule, nn.BatchNorm2d):
            bn_lora = BatchNormLoRA(submodule)
            downsample._modules[name] = bn_lora
            list_lora_layers.append(bn_lora)

# 32:92.8
def _apply_attnpool_lora(cfg, vision_encoder, list_lora_layers):
    """Inject LoRA into RN AttentionPool2d linear projections."""
    attnpool = getattr(vision_encoder, 'attnpool', None)
    if attnpool is None:
        print("Warning: RN backbone missing attention pooling module. Skipping LoRA injection for attnpool.")
        return

    attn_rank = 3
    print("Rank for AttnPool LoRA:", attn_rank)
    attn_targets = getattr(
        cfg.TRAINER[cfg.TRAINER.NAME],
        "RN_ATTNPOOL_LINEAR",
        ["c_proj"],
    )

    for name in attn_targets:
        submodule = getattr(attnpool, name, None)
        if isinstance(submodule, nn.Linear):
            new_linear_lora = LinearLoRA(
                submodule,
                r=attn_rank,
                lora_alpha=cfg.TRAINER[cfg.TRAINER.NAME].ALPHA,
                dropout_rate=cfg.TRAINER[cfg.TRAINER.NAME].DROPOUT_RATE,
            ).to(device=submodule.weight.device, dtype=submodule.weight.dtype)
            setattr(attnpool, name, new_linear_lora)
            list_lora_layers.append(new_linear_lora)
# lora_adapters.py

def _apply_resnet_lora(cfg, model, list_lora_layers):
    """
    专门为 CLIP-style ResNet 的 AttentionPool2d 模块和卷积 stage 注入 LoRA。
    """
    vision_encoder = getattr(model, 'image_encoder', None)
    if vision_encoder is None:
        raise AttributeError("Model lacks image_encoder required for RN backbones.")

    #_apply_stem_lora(cfg, model, list_lora_layers)

    _apply_stage_lora(cfg, model, list_lora_layers)

    _apply_attnpool_lora(cfg, vision_encoder, list_lora_layers)

def apply_lora_rn(cfg, model, encoder='both'):
    list_lora_layers = []
    if encoder in ('both', 'text'):
        _apply_text_lora(cfg, model, list_lora_layers)
    if encoder in ('both', 'vision', 'image'):
        _apply_resnet_lora(cfg, model, list_lora_layers)
    return list_lora_layers

def _compute_rank(block_idx, cfg):
    if not cfg.TRAINER[cfg.TRAINER.NAME].get("RANK_RAMP", False):
        return cfg.TRAINER[cfg.TRAINER.NAME].R

    ramp = cfg.TRAINER[cfg.TRAINER.NAME].RANK_RAMP 
    
    for i, ramp_start_index in reversed(list(enumerate(ramp))):
        if block_idx >= ramp_start_index:
            rank_multiplier = 2**(i + 1)  
            return cfg.TRAINER[cfg.TRAINER.NAME].R * rank_multiplier
    return cfg.TRAINER[cfg.TRAINER.NAME].R

def _apply_text_lora(cfg, model, list_lora_layers):
    indices = INDEX_POSITIONS_TEXT.get(cfg.TRAINER[cfg.TRAINER.NAME].POSITION)
    if indices is None:
        raise KeyError(f"Unknown text position: {cfg.TRAINER[cfg.TRAINER.NAME].POSITION}")
    text_encoder = model.text_encoder.transformer
    for i, block in enumerate(text_encoder.resblocks):
        if i not in indices:
            continue
        rank = _compute_rank(i, cfg)
        for name, submodule in block.named_children():
            if isinstance(submodule, nn.MultiheadAttention):
                new_multi_head_lora = PlainMultiheadAttentionLoRA(
                    submodule,
                    enable_lora={*cfg.TRAINER[cfg.TRAINER.NAME].PARAMS},
                    r=rank,
                    lora_alpha=cfg.TRAINER[cfg.TRAINER.NAME].ALPHA,
                    dropout_rate=cfg.TRAINER[cfg.TRAINER.NAME].DROPOUT_RATE
                )
                setattr(block, name, new_multi_head_lora)
                list_lora_layers.append(new_multi_head_lora)


def _apply_vit_lora(cfg, model, list_lora_layers):
    backbone = cfg.MODEL.BACKBONE.NAME
    positions = INDEX_POSITIONS_VISION.get(backbone, {})
    indices = positions.get(cfg.TRAINER[cfg.TRAINER.NAME].POSITION)
    if indices is None:
        raise KeyError(f"Unknown vision position '{cfg.TRAINER[cfg.TRAINER.NAME].POSITION}' for backbone '{backbone}'")
    vision_encoder = model.image_encoder.transformer
    for i, block in enumerate(vision_encoder.resblocks):
        if i not in indices:
            continue
        rank = _compute_rank(i, cfg)
        for name, submodule in block.named_children():
            if isinstance(submodule, nn.MultiheadAttention):
                new_multi_head_lora = PlainMultiheadAttentionLoRA(
                    submodule,
                    enable_lora={*cfg.TRAINER[cfg.TRAINER.NAME].PARAMS},
                    r=rank,
                    lora_alpha=cfg.TRAINER[cfg.TRAINER.NAME].ALPHA,
                    dropout_rate=cfg.TRAINER[cfg.TRAINER.NAME].DROPOUT_RATE
                )
                setattr(block, name, new_multi_head_lora)
                list_lora_layers.append(new_multi_head_lora)

def save_lora(cfg, list_lora_layers, save_dir, filename):
    trainer_cfg = cfg.TRAINER[cfg.TRAINER.NAME]
    weights = {}
    for i, layer in enumerate(list_lora_layers):
        layer_weights = {}
        for name, param in layer.state_dict().items():
            if 'lora' in name:
                layer_weights[name] = param.detach().cpu()
        weights[f'layer_{i}'] = layer_weights

    metadata = {
        'r': trainer_cfg.R,
        'alpha': trainer_cfg.ALPHA,
        'encoder': 'both',
        'params': trainer_cfg.PARAMS,
        'position': 'all'
    }

    save_data = {
        'weights': weights,
        'metadata': metadata
    }
    save_path = f'{save_dir}/{filename}.pt'
    torch.save(save_data, save_path)
    print(f'LoRA weights saved to {save_path}')


def load_lora(cfg, list_lora_layers, save_dir, filename):
    trainer_cfg = cfg.TRAINER[cfg.TRAINER.NAME]
    load_path = f'{save_dir}/{filename}.pt'

    if not os.path.exists(load_path):
        raise FileNotFoundError(f'File {load_path} does not exist.')

    loaded_data = torch.load(load_path)

    metadata = loaded_data['metadata']
    if metadata['r'] != trainer_cfg.R:
        raise ValueError(
            f"r mismatch: expected {trainer_cfg.R}, found {metadata['r']}")
    if metadata['alpha'] != trainer_cfg.ALPHA:
        raise ValueError(
            f"alpha mismatch: expected {trainer_cfg.ALPHA}, found {metadata['alpha']}")
    if metadata['encoder'] != 'both':
        raise ValueError(
            f"Encoder mismatch: expected {'both'}, found {metadata['encoder']}")
    if metadata['params'] != trainer_cfg.PARAMS:
        raise ValueError(
            f"Params mismatch: expected {trainer_cfg.PARAMS}, found {metadata['params']}")
    if metadata['position'] != 'all':
        raise ValueError(
            f"Position mismatch: expected {'all'}, found {metadata['position']}")

    weights = loaded_data['weights']
    for i, layer in enumerate(list_lora_layers):
        layer_weights = weights.get(f'layer_{i}', {})
        named_params = dict(layer.named_parameters())
        for name, tensor in layer_weights.items():
            if name in named_params:
                named_params[name].data.copy_(tensor.to(named_params[name].device))

    print(f'LoRA weights loaded from {load_path}')
