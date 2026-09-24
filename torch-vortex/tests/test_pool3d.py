import torch
import torch.nn.functional as F


def test_pool3d_and_indices_match_cpu(backend):
    torch.manual_seed(85)
    x = torch.randn(1, 2, 4, 5, 6)
    dx = x.to("vortex")
    cases = [
        lambda t: F.max_pool3d(t, (2, 3, 2), (2, 2, 2), (1, 1, 0)),
        lambda t: F.avg_pool3d(t, (2, 3, 2), (2, 2, 2), (1, 1, 0)),
    ]
    for fn in cases:
        torch.testing.assert_close(fn(dx).cpu(), fn(x), rtol=1e-6, atol=1e-6)
    got, indices = F.max_pool3d(dx, 2, 2, 1, return_indices=True)
    want, want_indices = F.max_pool3d(x, 2, 2, 1, return_indices=True)
    torch.testing.assert_close(got.cpu(), want)
    torch.testing.assert_close(indices.cpu(), want_indices)
