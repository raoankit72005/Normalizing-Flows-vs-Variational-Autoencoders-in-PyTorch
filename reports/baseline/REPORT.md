# Measured benchmark results

NLL is in nats per normalized 2D point. VAE NLL is estimated.
SD is across paired data/model seeds; it is not a confidence interval. NA means one seed.

| Dataset | Model | K | Seeds | Test NLL mean | SD |
|---|---|---:|---:|---:|---:|
| mixture | VAE | 10 | 5 | 1.1948 | 0.0450 |
| mixture | VAE | 100 | 5 | 1.1019 | 0.0236 |
| mixture | VAE | 1000 | 5 | 1.0807 | 0.0203 |
| mixture | RealNVP | 0 | 5 | 0.7907 | 0.0687 |
| moons | VAE | 10 | 5 | 1.9517 | 0.0208 |
| moons | VAE | 100 | 5 | 1.9186 | 0.0184 |
| moons | VAE | 1000 | 5 | 1.9126 | 0.0172 |
| moons | RealNVP | 0 | 5 | 1.6659 | 0.0221 |
| checkerboard | VAE | 10 | 5 | 2.3574 | 0.0169 |
| checkerboard | VAE | 100 | 5 | 2.3343 | 0.0140 |
| checkerboard | VAE | 1000 | 5 | 2.3307 | 0.0130 |
| checkerboard | RealNVP | 0 | 5 | 2.0371 | 0.0271 |
