import matplotlib.pyplot as plt


def plot_samples(test_data, vae_samples, flow_samples, limit, results_dir):
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    datasets = [test_data, vae_samples, flow_samples]
    titles = ["Original test data", "VAE samples", "RealNVP samples"]
    for axis, data, title in zip(axes, datasets, titles):
        data = data.cpu().numpy()
        axis.scatter(data[:, 0], data[:, 1], s=5, alpha=0.45, color="#2867a5", edgecolors="none")
        axis.set(title=title, xlabel="Normalized x₁", ylabel="Normalized x₂")
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.set_aspect("equal")
    figure.suptitle("Original vs generated samples · 2,000 points per panel")
    figure.savefig(results_dir / "samples.png", dpi=160)


def plot_losses(vae_history, flow_history, results_dir):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    titles = ["VAE: negative ELBO", "RealNVP: exact negative log-likelihood"]
    for axis, history, title in zip(axes, [vae_history, flow_history], titles):
        epochs = range(1, len(history["train"]) + 1)
        axis.plot(epochs, history["train"], label="Training", color="#2867a5")
        axis.plot(epochs, history["test"], label="Test", color="#d16a36")
        axis.set(title=title, xlabel="Epoch", ylabel="Nats per 2D point")
        axis.legend()
        axis.grid(alpha=0.2)
    figure.suptitle("Training and test losses · different objectives")
    figure.savefig(results_dir / "losses.png", dpi=160)


def plot_densities(grid_x, grid_y, true_density, vae_density, flow_density, results_dir):
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    densities = [true_density, vae_density, flow_density]
    titles = ["True Gaussian mixture", "VAE marginal: IS estimate (K=100)", "RealNVP: exact model density"]
    maximum = max(density.max().item() for density in densities)
    for axis, density, title in zip(axes, densities, titles):
        heatmap = axis.pcolormesh(
            grid_x.numpy(), grid_y.numpy(), density.numpy(),
            shading="auto", cmap="viridis", vmin=0, vmax=maximum,
        )
        axis.set(title=title, xlabel="Normalized x₁", ylabel="Normalized x₂")
        axis.set_aspect("equal")
    figure.colorbar(heatmap, ax=axes, label="Probability density", shrink=0.8)
    figure.suptitle("Density in normalized coordinates · shared color scale")
    figure.savefig(results_dir / "densities.png", dpi=160)


def plot_likelihoods(results, results_dir):
    figure, axis = plt.subplots(figsize=(7, 4.5), layout="constrained")
    values = [results["realnvp_nll"], results["vae_estimated_nll"]]
    labels = ["RealNVP\nExact model NLL", "VAE\nEstimated NLL (K=100)"]
    bars = axis.bar(labels, values, color=["#2867a5", "#d16a36"], width=0.55)
    axis.bar_label(bars, fmt="%.4f", padding=5)
    axis.margins(y=0.25)
    axis.axhline(0, color="black", linewidth=0.7)
    axis.set(title="Test likelihood comparison · lower is better", ylabel="Nats per 2D point")
    figure.suptitle(f"VAE negative ELBO: {results['vae_negative_elbo']:.4f} (reported separately)")
    figure.savefig(results_dir / "likelihood_comparison.png", dpi=160)
