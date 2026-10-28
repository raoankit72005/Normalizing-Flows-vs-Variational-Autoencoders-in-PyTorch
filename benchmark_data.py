"""Independent synthetic splits; normalization is fitted on training data only."""
import math
import torch
from data import sample_mixture, true_log_prob


def make_splits(dataset, seed, n_train=10000, n_val=2000, n_test=2000):
    if min(n_train, n_val, n_test) < 2:
        raise ValueError('Each split needs at least two observations')
    angles = torch.arange(8) * (2 * math.pi / 8)
    centers = 2 * torch.stack([angles.cos(), angles.sin()], dim=1)

    def draw(n, split):
        g = torch.Generator().manual_seed(seed + split * 1000003)
        if dataset == 'mixture':
            return sample_mixture(n, centers, 0.15, g)
        if dataset == 'moons':
            theta = torch.rand(n, generator=g) * math.pi
            side = torch.randint(2, (n,), generator=g).bool()
            x = torch.where(side, 1 - theta.cos(), theta.cos())
            y = torch.where(side, 0.5 - theta.sin(), theta.sin())
            return torch.stack([x, y], 1) + 0.08 * torch.randn(n, 2, generator=g)
        if dataset == 'checkerboard':
            # Uniform over eight alternating unit cells in [-2, 2]^2.
            cells = torch.tensor([(i, j) for i in range(-2, 2)
                                  for j in range(-2, 2) if (i + j) % 2 == 0])
            return cells[torch.randint(8, (n,), generator=g)] + torch.rand(n, 2, generator=g)
        raise ValueError(f'Unknown dataset: {dataset}')

    raw_train, raw_val, raw_test = [draw(n, i) for i, n in enumerate((n_train, n_val, n_test))]
    mean = raw_train.mean(0)
    std = raw_train.std(0, unbiased=False).clamp_min(1e-6)
    meta = {'mean': mean, 'std': std, 'centers': (centers - mean) / std,
            'component_std': 0.15 / std}
    return *((x - mean) / std for x in (raw_train, raw_val, raw_test)), meta


def reference_log_prob(data, dataset, meta):
    if dataset == 'mixture':
        return true_log_prob(data, meta['centers'], meta['component_std'])
    if dataset == 'checkerboard':
        raw = data * meta['std'] + meta['mean']
        inside = ((raw >= -2) & (raw < 2)).all(-1)
        occupied = (raw.floor().sum(-1).long() % 2 == 0)
        value = -math.log(8) + meta['std'].log().sum()
        return torch.where(inside & occupied, value, -torch.inf)
    return None


def mode_metrics(samples, meta, radius=3.0, minimum_fraction=0.01):
    """A mode is covered if >=1% of all draws fall within its 3-sigma ellipse."""
    distances = (((samples[:, None] - meta['centers']) / meta['component_std']) ** 2).sum(-1).sqrt()
    distance, nearest = distances.min(-1)
    valid = distance <= radius
    counts = torch.bincount(nearest[valid], minlength=8)
    return {'covered_modes': int((counts.float() / len(samples) >= minimum_fraction).sum()),
            'in_mode_fraction': valid.float().mean().item(), 'mode_counts': counts.tolist()}
