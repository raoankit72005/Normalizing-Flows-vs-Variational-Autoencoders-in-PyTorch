import math

import torch


def sample_mixture(num_samples, centers, component_std, generator):
    components = torch.randint(8, (num_samples,), generator=generator)
    noise = torch.randn(num_samples, 2, generator=generator)
    return centers[components] + component_std * noise


def generate_data(seed=42):
    generator = torch.Generator().manual_seed(seed)
    angles = torch.arange(8) * (2 * math.pi / 8)
    centers = 2 * torch.stack([angles.cos(), angles.sin()], dim=1)
    component_std = 0.15
    train_data = sample_mixture(10000, centers, component_std, generator)
    test_data = sample_mixture(2000, centers, component_std, generator)
    mean = train_data.mean(dim=0)
    std = train_data.std(dim=0, unbiased=False)
    train_data = (train_data - mean) / std
    test_data = (test_data - mean) / std
    centers = (centers - mean) / std
    component_std = component_std / std
    return train_data, test_data, centers, component_std


def true_log_prob(data, centers, component_std):
    standardized = (data[:, None, :] - centers[None, :, :]) / component_std
    log_normalizer = torch.log(2 * math.pi * component_std.square()).sum()
    component_log_prob = -0.5 * (standardized.square().sum(dim=-1) + log_normalizer)
    return torch.logsumexp(component_log_prob, dim=1) - math.log(8)
