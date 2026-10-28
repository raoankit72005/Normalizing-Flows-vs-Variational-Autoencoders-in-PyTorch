"""Run reproducible, paired VAE/RealNVP comparisons without changing the original demo."""
import argparse
import hashlib
import csv
import json
import platform
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from benchmark_data import make_splits, reference_log_prob, mode_metrics
from evaluate import vae_log_prob, flow_log_prob, vae_negative_elbo
from models.vae import VAE
from models.realnvp import RealNVP


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit(model, train, val, args, seed):
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    device = next(model.parameters()).device
    history = []
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for batch in loader:
            optimizer.zero_grad()
            loss = model.loss(batch.to(device))
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss')
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        # Validation randomness must not alter subsequent training draws.
        devices = [device.index or 0] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed + 10000 + epoch)
            validation = (vae_negative_elbo(model, val) if isinstance(model, VAE)
                          else -flow_log_prob(model, val).mean().item())
        history.append({'epoch': epoch + 1, 'train': total / len(train), 'validation': validation})
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return history, time.perf_counter() - start


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def figures(models, samples, test, histories, dataset, meta, folder, args):
    # Identical axes; include all generated points, even extreme outliers.
    limit = max(3.0, *(x.abs().max().item() for x in [test, *samples.values()])) + 0.1
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    fig.subplots_adjust(left=.06, right=.98, bottom=.12, top=.88, wspace=.25)
    for ax, (name, points) in zip(axes, [('Test data', test), *samples.items()]):
        ax.scatter(*points.T.numpy(), s=3, alpha=.4)
        ax.set(title=name, xlim=(-limit, limit), ylim=(-limit, limit), aspect='equal')
    fig.savefig(folder / 'samples.png', dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, (name, history) in zip(axes, histories.items()):
        for key in ['train', 'validation']:
            ax.plot([x['epoch'] for x in history], [x[key] for x in history], label=key)
        ax.set(title=name + (' negative ELBO' if name == 'VAE' else ' exact NLL'), xlabel='Epoch', ylabel='Nats / point')
        ax.legend()
    fig.savefig(folder / 'losses.png', dpi=130)
    plt.close(fig)
    # Density grid uses data extent rather than potentially extreme generated outliers.
    extent = max(3., test.abs().max().item() + .2)
    axis = torch.linspace(-extent, extent, args.grid_size)
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    grid = torch.stack([xx.flatten(), yy.flatten()], -1)
    seed_all(args.plot_seed)
    densities = {'VAE estimated (K=%d)' % args.plot_k: vae_log_prob(models['VAE'], grid, args.plot_k).exp(),
                 'RealNVP exact': flow_log_prob(models['RealNVP'], grid).exp()}
    reference = reference_log_prob(grid, dataset, meta)
    if reference is not None:
        densities = {'True density': reference.exp(), **densities}
    fig, axes = plt.subplots(1, len(densities), figsize=(4 * len(densities), 4), constrained_layout=True)
    vmax = max(x.max().item() for x in densities.values())
    for ax, (name, density) in zip(axes, densities.items()):
        mesh = ax.pcolormesh(xx, yy, density.reshape(xx.shape), vmin=0, vmax=vmax, shading='auto')
        ax.set(title=name, aspect='equal')
    fig.colorbar(mesh, ax=list(axes), label='Density in normalized coordinates')
    fig.savefig(folder / 'densities.png', dpi=130)
    plt.close(fig)


def summarize(rows, output):
    groups = {}
    for row in rows:
        key = (row['dataset'], row['model'], row['k'])
        groups.setdefault(key, []).append(row)
    summary = []
    for (dataset, model, k), values in groups.items():
        record = {'dataset': dataset, 'model': model, 'k': k, 'n_seeds': len(values)}
        for metric in ['test_nll', 'training_seconds', 'parameters', 'covered_modes', 'in_mode_fraction']:
            numbers = [x[metric] for x in values if metric in x]
            if numbers:
                record[metric + '_mean'] = float(np.mean(numbers))
                record[metric + '_std'] = float(np.std(numbers, ddof=1)) if len(numbers) > 1 else None
        summary.append(record)
    write_json(output / 'summary.json', summary)
    keys = sorted(set().union(*(x.keys() for x in rows)))
    with (output / 'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Measured benchmark results', '', 'NLL is in nats per normalized 2D point. VAE NLL is estimated.',
             'SD is across paired data/model seeds; it is not a confidence interval. NA means one seed.', '',
             '| Dataset | Model | K | Seeds | Test NLL mean | SD |', '|---|---|---:|---:|---:|---:|']
    for row in summary:
        sd = row['test_nll_std']
        lines.append(f"| {row['dataset']} | {row['model']} | {row['k']} | {row['n_seeds']} | {row['test_nll_mean']:.4f} | {format(sd, '.4f') if sd is not None else 'NA'} |")
    (output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    datasets = list(dict.fromkeys(x['dataset'] for x in summary))
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), squeeze=False, constrained_layout=True)
    for ax, dataset in zip(axes[0], datasets):
        vae = sorted([x for x in summary if x['dataset'] == dataset and x['model'] == 'VAE'], key=lambda x: x['k'])
        flow = next(x for x in summary if x['dataset'] == dataset and x['model'] == 'RealNVP')
        ax.errorbar([x['k'] for x in vae], [x['test_nll_mean'] for x in vae],
                    yerr=[x['test_nll_std'] or 0 for x in vae], marker='o', capsize=3, label='VAE estimate: mean +/- SD')
        ax.axhline(flow['test_nll_mean'], color='tab:orange', label='RealNVP exact: mean +/- SD')
        if flow['test_nll_std'] is not None:
            ax.axhspan(flow['test_nll_mean'] - flow['test_nll_std'], flow['test_nll_mean'] + flow['test_nll_std'], color='tab:orange', alpha=.15)
        refs = [x['reference_nll'] for x in rows if x['dataset'] == dataset and x['model'] == 'RealNVP' and 'reference_nll' in x]
        if refs:
            ax.axhline(float(np.mean(refs)), color='gray', linestyle=':', label='True distribution: mean NLL')
        ax.set(xscale='log', xlabel='Importance samples K', ylabel='Test NLL (nats / point)', title=dataset)
        ax.legend(fontsize=7)
    fig.savefig(output / 'likelihood_summary.png', dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=['mixture', 'moons', 'checkerboard'], default=['mixture', 'moons', 'checkerboard'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--ks', nargs='+', type=int, default=[10, 100, 1000])
    for name, default in [('epochs', 100), ('batch-size', 256), ('n-train', 10000), ('n-val', 2000), ('n-test', 2000), ('latent-dim', 2), ('flow-layers', 6), ('hidden-dim', 64), ('samples', 2000), ('grid-size', 80), ('plot-k', 100), ('plot-seed', 12345), ('threads', 1)]:
        parser.add_argument('--' + name, type=int, default=default)
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--observation-variance', type=float, default=.01)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--output', type=Path, default=Path('benchmark_results/default'))
    args = parser.parse_args()
    positive = ['epochs', 'batch_size', 'n_train', 'n_val', 'n_test', 'latent_dim', 'flow_layers', 'hidden_dim', 'samples', 'grid_size', 'plot_k', 'threads', 'lr', 'observation_variance']
    if any(getattr(args, x) <= 0 for x in positive) or min(args.ks) < 1:
        parser.error('Sizes, budgets, K, learning rate and variance must be positive')
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error('Seeds must be distinct and nonnegative')
    if len(set(args.datasets)) != len(args.datasets) or len(set(args.ks)) != len(args.ks):
        parser.error('Datasets and K values must be distinct')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    config = {**vars(args), 'output': str(args.output)}
    write_json(args.output / 'config.json', config)
    source_root = Path(__file__).resolve().parent
    write_json(args.output / 'source_hashes.json', {str(p.relative_to(source_root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_root.rglob('*.py')) if 'benchmark_results' not in p.parts})
    write_json(args.output / 'environment.json', {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__, 'matplotlib': matplotlib.__version__, 'device': args.device, 'platform': platform.platform()})
    rows = []
    for dataset in args.datasets:
        for seed in args.seeds:
            folder = args.output / dataset / f'seed_{seed}'
            folder.mkdir(parents=True)
            train, val, test, meta = make_splits(dataset, seed, args.n_train, args.n_val, args.n_test)
            torch.save({'train': train, 'validation': val, 'test': test, **meta}, folder / 'data.pt')
            models, samples, histories = {}, {}, {}
            reference = reference_log_prob(test, dataset, meta)
            for name in ['VAE', 'RealNVP']:
                seed_all(seed)
                model = (VAE(args.latent_dim, args.hidden_dim, args.observation_variance) if name == 'VAE'
                         else RealNVP(args.flow_layers, args.hidden_dim)).to(args.device)
                history, seconds = fit(model, train, val, args, seed)
                models[name], histories[name] = model, history
                seed_all(seed + 20000)
                samples[name] = model.sample(args.samples).cpu()
                torch.save(model.state_dict(), folder / f'{name}.pt')
                torch.save(samples[name], folder / f'{name}_samples.pt')
                base = {'dataset': dataset, 'seed': seed, 'model': name, 'parameters': sum(p.numel() for p in model.parameters()), 'training_seconds': seconds}
                if reference is not None:
                    base['reference_nll'] = -reference.mean().item()
                if dataset == 'mixture':
                    diagnostics = mode_metrics(samples[name], meta)
                    write_json(folder / f'{name}_modes.json', diagnostics)
                    base.update({k: v for k, v in diagnostics.items() if k != 'mode_counts'})
                for k in args.ks if name == 'VAE' else [0]:
                    seed_all(seed + 30000)
                    logp = vae_log_prob(model, test, k) if name == 'VAE' else flow_log_prob(model, test)
                    nll = -logp.mean().item()
                    if not np.isfinite(nll):
                        raise RuntimeError('Non-finite test NLL')
                    rows.append({**base, 'k': k, 'test_nll': nll})
                    torch.save(logp, folder / f'{name}_test_logp_k{k}.pt')
                print(f'{dataset} seed={seed} {name}: completed', flush=True)
            write_json(folder / 'history.json', histories)
            figures(models, samples, test, histories, dataset, meta, folder, args)
            summarize(rows, args.output)
    print(f'Results: {args.output / "REPORT.md"}')


if __name__ == '__main__':
    main()
