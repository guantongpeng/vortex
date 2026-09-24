import math

import torch


def test_rope_swiglu_and_kv_append_use_device_kernels(backend):
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    got = torch.ops.vortex.rope(x.to("vortex")).cpu()
    want = x.clone()
    for pos in range(x.size(0)):
        theta = 1.0 / (10000.0 ** (0.0 / x.size(1)))
        cs, sn = math.cos(pos * theta), math.sin(pos * theta)
        want[pos, 0], want[pos, 1] = x[pos, 0] * cs - x[pos, 1] * sn, x[pos, 0] * sn + x[pos, 1] * cs
        theta = 1.0 / (10000.0 ** (2.0 / x.size(1)))
        cs, sn = math.cos(pos * theta), math.sin(pos * theta)
        want[pos, 2], want[pos, 3] = x[pos, 2] * cs - x[pos, 3] * sn, x[pos, 2] * sn + x[pos, 3] * cs
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)

    a = torch.randn(17).to("vortex")
    b = torch.randn(17).to("vortex")
    torch.testing.assert_close(torch.ops.vortex.swiglu(a, b).cpu(),
                               a.cpu() * torch.sigmoid(a.cpu()) * b.cpu())

    cache = torch.zeros(6, 4).to("vortex")
    values = torch.arange(8, dtype=torch.float32).reshape(2, 4).to("vortex")
    torch.ops.vortex.kv_append_(cache, values, 3)
    want_cache = torch.zeros(6, 4)
    want_cache[3:5] = values.cpu()
    torch.testing.assert_close(cache.cpu(), want_cache)


def test_mamba_selective_scan_returns_final_state(backend):
    batch, channels, seqlen, dstate = 1, 2, 4, 3
    a = torch.tensor([-0.2, -0.3])
    dt = torch.full((batch, channels, seqlen), 0.1)
    b = torch.arange(seqlen * dstate, dtype=torch.float32).reshape(seqlen, dstate) / 10
    c = torch.flip(b, (1,))
    x = torch.arange(batch * channels * seqlen, dtype=torch.float32).reshape(batch, channels, seqlen) / 7
    state = torch.zeros(batch, channels, dstate)
    want = torch.empty_like(x)
    flat_x, flat_dt, flat_a = x.reshape(-1, seqlen), dt.reshape(-1, seqlen), a
    for bc in range(batch * channels):
        for t in range(seqlen):
            state.view(-1, dstate)[bc] = (
                torch.exp(flat_a[bc] * flat_dt[bc, t]) * state.view(-1, dstate)[bc]
                + b[t] * flat_x[bc, t])
            want.view(-1, seqlen)[bc, t] = (c[t] * state.view(-1, dstate)[bc]).sum()
    got, final = torch.ops.vortex.selective_scan(
        a.to("vortex"), dt.to("vortex"), b.to("vortex"), c.to("vortex"), x.to("vortex"))
    torch.testing.assert_close(got.cpu(), want, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(final.cpu(), state, rtol=1e-5, atol=1e-5)
