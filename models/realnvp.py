import math

import torch
from torch import nn


class AffineCoupling(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.register_buffer("mask", torch.tensor(mask, dtype=torch.float32))
        self.network = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def coupling_parameters(self, data):
        masked_data = data * self.mask
        scale, shift = self.network(masked_data).chunk(2, dim=-1)
        scale = torch.tanh(scale) * (1 - self.mask)
        shift = shift * (1 - self.mask)
        return scale, shift

    def forward(self, data):
        scale, shift = self.coupling_parameters(data)
        latent = data * self.mask + (1 - self.mask) * (data * scale.exp() + shift)
        return latent, scale.sum(dim=-1)

    def inverse(self, latent):
        scale, shift = self.coupling_parameters(latent)
        return latent * self.mask + (1 - self.mask) * ((latent - shift) * (-scale).exp())


class RealNVP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            AffineCoupling([index % 2, 1 - index % 2])
            for index in range(6)
        ])

    def forward(self, data):
        latent = data
        log_determinant = data.new_zeros(data.shape[0])
        for layer in self.layers:
            latent, layer_log_determinant = layer(latent)
            log_determinant = log_determinant + layer_log_determinant
        return latent, log_determinant

    def inverse(self, latent):
        data = latent
        for layer in reversed(self.layers):
            data = layer.inverse(data)
        return data

    def log_prob(self, data):
        latent, log_determinant = self(data)
        base_log_prob = -0.5 * (latent.square() + math.log(2 * math.pi)).sum(dim=-1)
        return base_log_prob + log_determinant

    def loss(self, data):
        return -self.log_prob(data).mean()

    @torch.no_grad()
    def sample(self, num_samples):
        mask = self.layers[0].mask
        latent = torch.randn(num_samples, 2, device=mask.device, dtype=mask.dtype)
        return self.inverse(latent)
