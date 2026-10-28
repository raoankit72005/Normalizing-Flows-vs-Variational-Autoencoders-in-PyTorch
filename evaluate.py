import math

import torch


@torch.no_grad()
def vae_negative_elbo(model, data, batch_size=256):
    model.eval()
    device = next(model.parameters()).device
    total_loss = 0.0
    for batch in data.split(batch_size):
        batch = batch.to(device)
        total_loss += model.loss(batch).item() * len(batch)
    return total_loss / len(data)


@torch.no_grad()
def vae_log_prob(model, data, num_samples=100, batch_size=128):
    if num_samples < 1 or batch_size < 1 or len(data) == 0:
        raise ValueError("Importance samples, batch size and data length must be positive")
    model.eval()
    device = next(model.parameters()).device
    log_probabilities = []
    for batch in data.split(batch_size):
        batch = batch.to(device)
        mean, log_variance = model.encode(batch)
        noise = torch.randn(num_samples, len(batch), mean.shape[-1], device=device, dtype=mean.dtype)
        latent = mean + (0.5 * log_variance).exp() * noise
        reconstruction_mean = model.decode(latent)
        log_likelihood = -model.reconstruction_nll(batch, reconstruction_mean)
        log_prior = -0.5 * (latent.square() + math.log(2 * math.pi)).sum(dim=-1)
        standardized = (latent - mean).square() / log_variance.exp()
        log_posterior = -0.5 * (standardized + log_variance + math.log(2 * math.pi)).sum(dim=-1)
        log_weights = log_likelihood + log_prior - log_posterior
        log_prob = torch.logsumexp(log_weights, dim=0) - math.log(num_samples)
        log_probabilities.append(log_prob.cpu())
    return torch.cat(log_probabilities)


@torch.no_grad()
def flow_log_prob(model, data, batch_size=256):
    model.eval()
    device = next(model.parameters()).device
    log_probabilities = []
    for batch in data.split(batch_size):
        log_probabilities.append(model.log_prob(batch.to(device)).cpu())
    return torch.cat(log_probabilities)


def evaluate_models(vae, realnvp, test_data):
    return {
        "vae_negative_elbo": vae_negative_elbo(vae, test_data),
        "vae_estimated_nll": -vae_log_prob(vae, test_data, num_samples=100).mean().item(),
        "realnvp_nll": -flow_log_prob(realnvp, test_data).mean().item(),
    }


def print_results(results):
    print("\nTest results: nats per 2D point, normalized coordinates")
    print(f"{'Model':<10} {'Metric':<41} {'Value':>10}")
    print("-" * 63)
    print(f"{'VAE':<10} {'Negative ELBO (separate objective)':<41} {results['vae_negative_elbo']:>10.4f}")
    print(f"{'VAE':<10} {'Estimated NLL (100 importance samples)':<41} {results['vae_estimated_nll']:>10.4f}")
    print(f"{'RealNVP':<10} {'Exact NLL':<41} {results['realnvp_nll']:>10.4f}")
    print("\nVAE negative ELBO is not exact negative log-likelihood.")
    print("Importance sampling estimates VAE likelihood; it does not make it exact.")
    print("RealNVP likelihood is exact under its model, not necessarily the true density.")
    print("Both models use the same Gaussian mixture training and test data.")

