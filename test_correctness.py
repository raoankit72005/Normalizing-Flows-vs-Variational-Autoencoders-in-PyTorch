"""Mathematical checks, independent of the benchmark's measured outcomes."""
import math
import unittest
import torch
from benchmark_data import make_splits, mode_metrics
from evaluate import vae_log_prob
from models.vae import VAE, kl_divergence
from models.realnvp import RealNVP


class CorrectnessTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(1)

    def test_nonidentity_flow_inverse_and_jacobian(self):
        model = RealNVP(num_layers=4, hidden_dim=12).double()
        for layer in model.layers:
            torch.nn.init.normal_(layer.network[-1].weight, std=.08)
            torch.nn.init.normal_(layer.network[-1].bias, std=.08)
        x = torch.randn(6, 2, dtype=torch.float64)
        z, logdet = model(x)
        torch.testing.assert_close(model.inverse(z), x, atol=1e-9, rtol=1e-9)
        jac = torch.autograd.functional.jacobian(lambda a: model(a[None])[0][0], x[0])
        torch.testing.assert_close(torch.linalg.slogdet(jac).logabsdet, logdet[0])

    def test_vae_dimensions_and_gradients(self):
        for dim in [1, 2, 4]:
            model = VAE(latent_dim=dim, hidden_dim=12)
            x = torch.randn(5, 2)
            loss = model.loss(x)
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertEqual(tuple(model.sample(7).shape), (7, 2))
            self.assertEqual(tuple(vae_log_prob(model, x, 3).shape), (5,))
        torch.testing.assert_close(kl_divergence(torch.zeros(3, 4), torch.zeros(3, 4)), torch.zeros(3))

    def test_importance_estimator_known_density(self):
        # q(z|x)=p(z), decoder mean=0: every importance weight equals p(x).
        model = VAE(latent_dim=3)
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
        x = torch.randn(9, 2) * .1
        exact = -.5 * (x.square() / .01 + math.log(2 * math.pi * .01)).sum(-1)
        for k in [1, 10, 100]:
            torch.testing.assert_close(vae_log_prob(model, x, k), exact)

    def test_splits_and_normalization(self):
        for dataset in ['mixture', 'moons', 'checkerboard']:
            a = make_splits(dataset, 11, 200, 40, 50)
            b = make_splits(dataset, 11, 200, 40, 90)
            for index in [0, 1]:
                torch.testing.assert_close(a[index], b[index])
            torch.testing.assert_close(a[0].mean(0), torch.zeros(2), atol=1e-6, rtol=0)
            torch.testing.assert_close(a[0].std(0, unbiased=False), torch.ones(2))
            self.assertFalse(torch.equal(a[0][:40], a[1]))

    def test_mode_diagnostic(self):
        *_, meta = make_splits('mixture', 0)
        metrics = mode_metrics(meta['centers'].repeat(20, 1), meta)
        self.assertEqual(metrics['covered_modes'], 8)
        self.assertEqual(metrics['in_mode_fraction'], 1.)
        self.assertEqual(mode_metrics(torch.ones(20, 2) * 100, meta)['covered_modes'], 0)


if __name__ == '__main__':
    unittest.main()
