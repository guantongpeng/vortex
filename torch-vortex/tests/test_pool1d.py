import torch
import torch.nn.functional as F


def test_pool1d_and_indices_match_cpu(backend):
    torch.manual_seed(83)
    x = torch.randn(2, 3, 9)
    dx = x.to("vortex")
    for fn in (lambda t: F.max_pool1d(t, 3, 2, 1),
               lambda t: F.avg_pool1d(t, 3, 2, 1)):
        torch.testing.assert_close(fn(dx).cpu(), fn(x), rtol=1e-6, atol=1e-6)
    got, indices = F.max_pool1d(dx, 3, 2, 1, return_indices=True)
    want, want_indices = F.max_pool1d(x, 3, 2, 1, return_indices=True)
    torch.testing.assert_close(got.cpu(), want)
    torch.testing.assert_close(indices.cpu(), want_indices)


def test_adaptive_avg_pool1d_global_matches_cpu(backend):
    x = torch.randn(2, 2, 11)
    torch.testing.assert_close(
        F.adaptive_avg_pool1d(x.to("vortex"), 1).cpu(),
        F.adaptive_avg_pool1d(x, 1), rtol=1e-6, atol=1e-6)
