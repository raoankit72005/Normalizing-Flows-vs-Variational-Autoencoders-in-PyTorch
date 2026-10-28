# Reproducible density-estimation benchmark

Question: how do VAE and RealNVP density estimates and generated samples change across distributions and seeds?

The original `python main.py` demonstration and historical `results/` figures remain unchanged. This is a separate entry point; do not mix its results with the original seed-42 table.

## Run locally or on Colab

```bash
python -m pip install -r requirements.txt
python -m unittest -v test_correctness
python benchmark.py --output benchmark_results/baseline
```

For the exact locally verified package versions, use `python -m pip install -r requirements-tested.txt`. The unpinned requirements retain compatibility with Colab.

Defaults: both models, 100 epochs, mixture/moons/checkerboard, seeds 0–4, and CPU with one thread. Use `--device cuda` for GPU. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before starting Python for deterministic CUDA operation. Hardware/package versions can still affect results. Existing output directories are rejected to avoid mixing runs.

Quick pipeline check, not research evidence:

```bash
python benchmark.py --epochs 2 --seeds 0 1 --n-train 128 --n-val 32 --n-test 32 --samples 100 --grid-size 12 --plot-k 10 --output benchmark_results/smoke
```

In Colab, clone the repository and install dependencies as shown in README, then run these commands prefixed with `!`. Download the output folder before the session ends. No external dataset is needed.

## Protocol and interpretation

- Independent train/validation/test draws contain 10,000/2,000/2,000 points. Separate local generators isolate splits; normalization is fitted on training only.
- Models share exact splits and minibatch order within each seed. Seeds change both data and initialization, so SD combines their variability.
- Adam uses learning rate 0.001, batch size 256, fixed epochs, no early stopping. Validation is monitored with RNG isolated from training; test data never enters training. Evaluate final-epoch weights.
- Equal updates and widths are not equal capacity or FLOPs. Parameter counts and training time are recorded. Timing includes validation, excludes final evaluation/plotting. No hyperparameter search is performed.
- VAE NLL is importance-estimated at K=10,100,1000. RealNVP model NLL is exact; K=0 in its CSV row means not applicable. Negative ELBO is a separate objective in history plots.
- Finite-K log estimates are downward biased in expectation, hence estimated NLL is upward biased. Individual realizations need not improve monotonically with K. K sensitivity is not proof of convergence; independent evaluation repeats would isolate estimator variance.
- Mixture matches the original raw distribution. Moons have equal expected component weights, uniform angles, Gaussian noise SD 0.08. Checkerboard is uniform over eight alternating unit squares in [-2,2]^2. Each dataset is normalized separately.
- True normalized density/NLL is provided for mixture and checkerboard, not moons. Compare models within each dataset; NLL across different datasets is not a difficulty ranking.

## Saved outputs

`config.json`, `environment.json`, `metrics.csv`, `summary.json`, `REPORT.md` describe the configuration and measurements. Summary uses sample SD across seeds (`ddof=1`), not standard error or a confidence interval. One-seed SD is null/NA. Do not aggregate different K values or architectures.

Every dataset/seed folder includes exact normalized splits and preprocessing in `data.pt`, final state dictionaries `VAE.pt`/`RealNVP.pt`, generated samples, per-point test log probabilities, history, and sample/loss/density plots. Reconstruct models from config before loading state dictionaries.

Sample panels share axes and include outliers. Density grids use test-data extent, common color scales, and no finite-grid renormalization. VAE plots use `--plot-k` (100), which can differ from evaluation K. True-density panels are omitted for moons.

For mixture, a draw is in-mode if its nearest center has standardized distance <=3, using known component SD. A mode is covered if at least 1% of all draws fall within its ellipse. Saved diagnostics include covered modes (0–8), in-mode fraction, and counts. These thresholds are operational, not universal quality metrics. VAE samples include decoder noise.

## Controlled ablations

Change one factor and keep all other arguments/seeds fixed:

```bash
python benchmark.py --flow-layers 2 --output benchmark_results/flow2
python benchmark.py --flow-layers 4 --output benchmark_results/flow4
python benchmark.py --flow-layers 8 --output benchmark_results/flow8
python benchmark.py --latent-dim 1 --output benchmark_results/latent1
python benchmark.py --latent-dim 4 --output benchmark_results/latent4
```

Baseline: six flow layers, latent dimension two. The unchanged comparator is rerun for simplicity. Depth and latent dimension also change capacity; do not describe these as capacity-matched comparisons. `--hidden-dim`, `--observation-variance`, sizes, budgets, and learning rate are configurable too.

Use validation to guide future settings. If you select a setting after viewing exploratory test results, obtain a fresh test draw for a confirmatory claim. Never choose the best seed.

## Report and interview discussion

Explain ELBO versus likelihood; why K matters; train-only normalization; agreement across seeds; missed modes and density between clusters; limitations from capacity, decoder variance, compute and estimator error. Full numerical claims require an actually completed run. These experiments cannot establish universal superiority of either model.
