import math

import torch
from torch import nn


def kl_divergence(mean, log_variance):
    return 0.5 * (mean.square() + log_variance.exp() - 1 - log_variance).sum(dim=-1)


class VAE(nn.Module):
    def __init__(self, latent_dim=2, hidden_dim=64, observation_variance=0.01):
        super().__init__()
        if latent_dim < 1 or hidden_dim < 1 or observation_variance <= 0:
            raise ValueError("Model dimensions and observation variance must be positive")
        self.latent_dim = latent_dim
        self.observation_variance = observation_variance
        self.encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.log_variance_layer = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def encode(self, data):
        hidden = self.encoder(data)
        return self.mean_layer(hidden), self.log_variance_layer(hidden)

    def reparameterize(self, mean, log_variance):
        noise = torch.randn_like(mean)
        return mean + (0.5 * log_variance).exp() * noise

    def decode(self, latent):
        return self.decoder(latent)

    def forward(self, data):
        mean, log_variance = self.encode(data)
        latent = self.reparameterize(mean, log_variance)
        return self.decode(latent), mean, log_variance

    def reconstruction_nll(self, data, reconstruction_mean):
        squared_error = (data - reconstruction_mean).square()
        log_normalizer = math.log(2 * math.pi * self.observation_variance)
        return 0.5 * (squared_error / self.observation_variance + log_normalizer).sum(dim=-1)

    def loss(self, data):
        reconstruction_mean, mean, log_variance = self(data)
        reconstruction_loss = self.reconstruction_nll(data, reconstruction_mean)
        kl_loss = kl_divergence(mean, log_variance)
        return (reconstruction_loss + kl_loss).mean()

    @torch.no_grad()
    def sample(self, num_samples):
        weight = self.mean_layer.weight
        latent = torch.randn(num_samples, self.latent_dim, device=weight.device, dtype=weight.dtype)
        mean = self.decode(latent)
        return mean + math.sqrt(self.observation_variance) * torch.randn_like(mean)

