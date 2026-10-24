import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from data import generate_data, true_log_prob
from evaluate import evaluate_models, flow_log_prob, print_results, vae_log_prob
from models.realnvp import RealNVP
from models.vae import VAE
from plot_results import plot_densities, plot_likelihoods, plot_losses, plot_samples
from train import train_realnvp, train_vae


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    seed = 42
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    train_data, test_data, centers, component_std = generate_data(seed)

    set_seed(seed)
    vae = VAE().to(device)
    vae_history = train_vae(vae, train_data, test_data, seed=seed)

    set_seed(seed)
    realnvp = RealNVP().to(device)
    flow_history = train_realnvp(realnvp, train_data, test_data, seed=seed)

    set_seed(seed + 1)
    results = evaluate_models(vae, realnvp, test_data)
    vae_samples = vae.sample(2000).cpu()
    flow_samples = realnvp.sample(2000).cpu()
    limit = max(2.5, test_data.abs().max().item(), vae_samples.abs().max().item(), flow_samples.abs().max().item()) + 0.2
    coordinates = torch.linspace(-limit, limit, 160)
    grid_y, grid_x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    true_density = true_log_prob(grid, centers, component_std).exp().reshape(160, 160)
    vae_density = vae_log_prob(vae, grid, num_samples=100).exp().reshape(160, 160)
    flow_density = flow_log_prob(realnvp, grid).exp().reshape(160, 160)

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(exist_ok=True)
    plot_samples(test_data, vae_samples, flow_samples, limit, results_dir)
    plot_losses(vae_history, flow_history, results_dir)
    plot_densities(grid_x, grid_y, true_density, vae_density, flow_density, results_dir)
    plot_likelihoods(results, results_dir)
    print_results(results)
    print(f"\nGraphs saved in {results_dir}")
    plt.show()


if __name__ == "__main__":
    main()
