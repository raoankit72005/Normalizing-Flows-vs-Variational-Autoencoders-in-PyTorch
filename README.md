# Normalizing Flows vs Variational Autoencoders: A Comparative Study of Density Estimation

A small PyTorch experiment for learning how a VAE and a RealNVP flow model the same eight-mode, two-dimensional distribution. Both models train from scratch. The only third-party dependencies are PyTorch, NumPy, and Matplotlib.

## Multi-seed benchmark

A separate `benchmark.py` runner adds three datasets, independent validation data, five seeds, K-sensitivity checks, configurable architectures, checkpoints, and measured summaries. See [BENCHMARK.md](BENCHMARK.md) for commands, fairness caveats, and ablations, and the [measured five-seed report](reports/baseline/README.md) for results. Run `python -m unittest -v test_correctness` for mathematical checks. The original demo and historical results below are retained unchanged.

## Run locally

Use Python 3.10 or newer:

```bash
git clone https://github.com/raoankit72005/Normalizing-Flows-vs-Variational-Autoencoders-in-PyTorch.git normalizing-flows-vs-vae
cd normalizing-flows-vs-vae
python -m pip install -r requirements.txt
python main.py
```

CUDA is selected automatically when available; otherwise the experiment runs on CPU. Each model trains for 100 epochs. Progress prints every 10 epochs, followed by a test-results table. Four figures are displayed and saved in `results/` beside `main.py`. On a terminal without a graphical display, open the saved PNGs.

## Run on Google Colab

Optionally select **Runtime → Change runtime type → GPU**, then run these notebook cells in order:

```python
!git clone https://github.com/raoankit72005/Normalizing-Flows-vs-Variational-Autoencoders-in-PyTorch.git normalizing-flows-vs-vae
%cd normalizing-flows-vs-vae
%pip install -r requirements.txt
```

```python
%run main.py
```

Using `%run` displays the figures inside the notebook. The PNGs are also available in Colab's Files panel under `normalizing-flows-vs-vae/results/`.

## Read the code in this order

```text
normalizing-flows-vs-vae/
├── models/
│   ├── vae.py
│   └── realnvp.py
├── data.py
├── train.py
├── evaluate.py
├── plot_results.py
├── main.py
├── requirements.txt
└── README.md
```

Start with `data.py`, then the two models, `train.py`, `evaluate.py`, `plot_results.py`, and finally `main.py`. The original demonstration is explained below; the benchmark includes short implementation docstrings. The four experiment output files are PNGs in `results/`.

## Dataset and training

Eight equally weighted Gaussian components have centers on a circle of radius 2 and isotropic standard deviation 0.15 before normalization. PyTorch generates 10,000 training points and 2,000 independent test points with seed 42. Both models receive the exact same tensors.

Each coordinate is normalized using only the training mean and population standard deviation. The true density uses the correspondingly transformed component means and standard deviations. If raw data are `u` and normalized data are `x = (u - mean) / std`, then:

$$p_X(x) = p_U(\mathrm{mean} + \mathrm{std} \odot x)\prod_j \mathrm{std}_j.$$

All reported densities and losses use normalized coordinates. No plot is separately rescaled to integrate to one on its finite grid.

Both models use Adam with learning rate 0.001, batches of 256, and 100 epochs. The seed is reset to 42 before initializing each model. Separate data-loader generators with the same seed produce the same minibatch ordering. Evaluation and sampling use seed 43. Training losses are sample-weighted averages of the losses encountered during each epoch; test losses use that epoch's final parameters. The test set never contributes gradients or selects a model.

## VAE

The encoder has two width-64 ReLU layers and separate two-dimensional mean and log-variance outputs. It approximates the posterior with a diagonal Gaussian:

$$q_\phi(z\mid x)=\mathcal N(\mu_\phi(x),\operatorname{diag}(\exp(\ell_\phi(x)))).$$

Reparameterization makes the sampled latent variable differentiable with respect to the encoder outputs:

$$z=\mu+\exp(\ell/2)\odot\epsilon,\qquad \epsilon\sim\mathcal N(0,I).$$

The decoder has two width-64 ReLU layers and outputs a two-dimensional Gaussian mean. Its fixed observation variance is 0.01 **in normalized coordinates**:

$$p(z)=\mathcal N(0,I),\qquad p_\theta(x\mid z)=\mathcal N(f_\theta(z),0.01I).$$

For each point, the Gaussian reconstruction NLL and analytic KL are:

$$-\log p_\theta(x\mid z)=\frac12\sum_{j=1}^{2}\left[\frac{(x_j-f_\theta(z)_j)^2}{0.01}+\log(2\pi\cdot0.01)\right],$$

$$D_{KL}(q_\phi(z\mid x)\Vert p(z))=\frac12\sum_{j=1}^{2}\left[\mu_j^2+\exp(\ell_j)-1-\ell_j\right].$$

`loss` averages their sum over points: the negative ELBO, with no beta coefficient. Reconstruction uses one posterior sample per point during training and epoch-wise test evaluation. Gaussian normalizing constants are included. Generated observations use both a prior draw and decoder observation noise, `x = decoder(z) + 0.1 * noise`; decoder means alone would have a different distribution.

## RealNVP

Six affine coupling layers alternate masks `[0, 1]` and `[1, 0]`. Each has a network with two width-64 ReLU hidden layers that predicts scale and shift. `tanh` bounds the log scale, and the final linear layer starts at zero so each coupling initially equals the identity.

For an unchanged coordinate `a` and transformed coordinate `b`, this implementation's forward direction is **data to latent**:

$$z_a=x_a,\qquad z_b=x_b\exp(s(x_a))+t(x_a),\qquad \log|\det(dz/dx)|=s(x_a).$$

The inverse recovers `x_b = (z_b - t(z_a)) * exp(-s(z_a))`. Inverse layers run in reverse order. With a standard Gaussian base:

$$\log p_\theta(x)=\log p_0(z)+\sum_{k=1}^{6}\log|\det J_k|.$$

Training minimizes the mean exact model NLL. Sampling draws from the base Gaussian and applies the inverse flow. "Exact" describes likelihood computation under the fitted model; it does not mean the model exactly matches the data distribution.

## Likelihood versus ELBO

The VAE marginal integrates over latent variables, which is generally intractable. Its encoder provides approximate inference. The identity

$$\log p_\theta(x)=\operatorname{ELBO}(x)+D_{KL}(q_\phi(z\mid x)\Vert p_\theta(z\mid x))$$

shows why the expected negative ELBO upper-bounds the VAE's NLL. A single Monte Carlo ELBO observation can fluctuate. RealNVP uses an invertible transformation and a tractable Jacobian, so it evaluates its model likelihood directly.

Only for evaluation, the VAE uses 100 independent posterior draws **per test point**:

$$z_k\sim q_\phi(z\mid x),\qquad \widehat{\log p_\theta(x)}=\operatorname{logsumexp}_{k=1}^{K}\left[\log p_\theta(x\mid z_k)+\log p(z_k)-\log q_\phi(z_k\mid x)\right]-\log K.$$

The mean importance weight estimates density. Taking its logarithm introduces downward bias in expected log likelihood at finite K, so the reported estimated NLL can overstate the true VAE NLL. An individual realization has no guaranteed ordering. A poor posterior proposal can produce high variance. This estimate is not exact likelihood, and it is never the training loss.

The terminal reports VAE negative ELBO separately from VAE estimated NLL and RealNVP exact NLL. All values are **nats per two-dimensional point**. Lower NLL is better; negative continuous-data NLL values are possible because a probability density can exceed one.

## The four graphs

| File | Interpretation |
| --- | --- |
| `results/samples.png` | Original test data and 2,000 observations from each model, with identical limits and equal aspect ratios. Look for all eight clusters and excess points between them. |
| `results/losses.png` | Separate VAE negative-ELBO and RealNVP exact-NLL panels. Compare training behavior within each panel; their vertical values are different objectives. |
| `results/densities.png` | True density, importance-estimated VAE marginal density, and exact flow density on a 160 × 160 grid. Every panel shares coordinates, a linear color scale, and one colorbar. VAE evaluation uses K=100 in batches of 128 grid points. |
| `results/likelihood_comparison.png` | RealNVP exact test NLL versus VAE importance-estimated test NLL. The VAE estimate is labeled, and negative ELBO appears separately above the chart. |

The density image covers a finite plotting region, while each model defines a density over the whole plane. The VAE density image can contain importance-sampling noise, especially away from training data.

## Measured experiment

The full 100-epoch experiment was executed on CPU with seed 42, PyTorch 2.14.1+cpu, NumPy 2.3.5, and Matplotlib 3.10.8. The run used one CPU thread and took about 23 seconds on the verification machine; other machines can differ substantially. These are measured results, not expected targets:

| Model | Test metric | Nats per 2D point |
| --- | --- | ---: |
| RealNVP | Exact model NLL | 0.8325 |
| VAE | Importance-estimated NLL, K=100 | 1.0447 |
| VAE | Negative ELBO, reported separately | 1.5818 |

1. **Eight modes:** both models generated samples around every mode. RealNVP formed tighter clusters, although its density still shows distorted shapes and connections between clusters.
2. **Sample realism:** RealNVP more closely matched the original clusters in this run. Of the 2,000 plotted observations, 86.05% of RealNVP samples and 64.50% of VAE samples fell within standardized distance 3 of a true component center; the test data scored 98.80%. Here distance means `sqrt(sum(((point - center) / component_std) ** 2))`, taking the nearest center. This is a simple diagnostic, not a universal quality metric.
3. **Test likelihood:** RealNVP's exact NLL was lower than the VAE's K=100 estimate by about 0.2122 nats per point. Because the VAE estimate has finite-sample error and bias, this alone does not establish the ordering of their exact likelihoods.
4. **Likelihood versus ELBO:** the VAE's estimated NLL of 1.0447 and negative ELBO of 1.5818 answer different questions. The ELBO includes the penalty for approximate posterior inference; their observed difference is not an exact posterior-KL measurement.
5. **Training behavior:** both objectives improved sharply early on, then more gradually with fluctuations. VAE test negative ELBO decreased from 2.4676 at epoch 10 to 1.5999 at epoch 100. RealNVP test NLL decreased from 1.1660 to 0.8325. Their objective values cannot be compared as convergence speeds. The final VAE negative ELBO differs slightly from the last epoch value because evaluation uses fresh latent draws.
6. **Architectural limits:** the VAE left more mass between modes; its diagonal approximate posterior and fixed observation variance restrict the model/inference pair. RealNVP also produced bridges and some outliers despite exact model likelihood. It has 27,672 parameters versus the VAE's 9,094, so this is not a capacity-matched comparison.

Verification checked the VAE reparameterization and gradients, Gaussian normalizing constants, analytic KL, and an importance-sampling case with known exact likelihood. Non-identity and trained flow transformations passed forward/inverse and full autograd-Jacobian checks. The trained float32 flow round-trip error was below `6e-7`. Numerical integrals over the plotted grid were approximately 1.0000 for the true density, 0.9926 for the VAE estimate, and 0.9985 for RealNVP. These are finite-grid checks, not artificial density renormalization or a proof that the importance estimate is accurate everywhere. All four PNGs were generated and visually inspected.

## Limitations

This is one seed, one small synthetic distribution, and one fixed architecture per method. It cannot establish that either model is universally better. Equal hidden widths and epochs do not imply equal parameter counts or equal computation. Results can differ across PyTorch versions, hardware, and CUDA kernels despite fixed seeds.

The VAE uses a restricted diagonal posterior and fixed observation variance, which can limit inference and the shapes or widths it learns. Its likelihood estimate has finite-sample bias and variance. The six-layer RealNVP has limited affine-coupling capacity, must preserve dimensionality, and uses bounded scales. A connected Gaussian base transformed continuously can leave low-density connections between modes. Exact likelihood does not eliminate model misspecification or optimization difficulties.

Test curves are provided for understanding this predetermined run. If you later tune architectures or settings, use a separate validation set and reserve a fresh test set for final reporting. Repeated seeds and uncertainty estimates would be needed for stronger research claims.

## References

- [Kingma and Welling: Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- [Dinh, Sohl-Dickstein, and Bengio: Density estimation using Real NVP](https://arxiv.org/abs/1605.08803)
- [Burda, Grosse, and Salakhutdinov: Importance Weighted Autoencoders](https://arxiv.org/abs/1509.00519) — background for the evaluation estimator; this project trains the VAE with the ordinary ELBO.

