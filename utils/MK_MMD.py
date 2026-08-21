import torch


def _gaussian_kernel_matrix(x, y, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = x.size(0) + y.size(0)
    total = torch.cat([x, y], dim=0)
    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    l2_distance = ((total0 - total1) ** 2).sum(2)

    if fix_sigma is not None:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(l2_distance.detach()) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

    kernel_val = [torch.exp(-l2_distance / bw) for bw in bandwidth_list]
    return sum(kernel_val)


def MK_MMD(source_features, target_features, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """Multi-kernel Maximum Mean Discrepancy between two feature sets.

    source_features: (N, d) tensor
    target_features: (M, d) tensor
    """
    n_source = source_features.size(0)
    kernels = _gaussian_kernel_matrix(source_features, target_features, kernel_mul, kernel_num, fix_sigma)

    XX = kernels[:n_source, :n_source]
    YY = kernels[n_source:, n_source:]
    XY = kernels[:n_source, n_source:]
    YX = kernels[n_source:, :n_source]

    return torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
