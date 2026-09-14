# Reference numerics through the Triton interpreter (plan P4-01): the
# kernels below are the official-tutorial shapes (vector add, softmax,
# layernorm); once codegen lands, the same functions run through
# VortexBackend and must match these results.
import os
import sys

import numpy as np
import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + offs, mask=mask)
    y = tl.load(y_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, x + y, mask=mask)


@triton.jit
def softmax_kernel(in_ptr, out_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n_cols
    x = tl.load(in_ptr + row * n_cols + offs, mask=mask, other=float("-inf"))
    m = tl.max(x, 0)
    e = tl.exp(x - m)
    s = tl.sum(e, 0)
    tl.store(out_ptr + row * n_cols + offs, e / s, mask=mask)


@triton.jit
def layernorm_kernel(x_ptr, g_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < n
    x = tl.load(x_ptr + row * n + offs, mask=mask, other=0.0)
    mean = tl.sum(x, 0) / n
    d = tl.where(mask, x - mean, 0.0)
    var = tl.sum(d * d, 0) / n
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    g = tl.load(g_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + row * n + offs, d * rstd * g + b, mask=mask)


def test_interpreter_vecadd():
    n = 100
    x = torch.arange(n, dtype=torch.float32)
    y = torch.arange(n, dtype=torch.float32) * 2
    out = torch.empty_like(x)
    add_kernel[(triton.cdiv(n, 16),)](x, y, out, n, BLOCK=16)
    torch.testing.assert_close(out, x + y)


def test_interpreter_softmax():
    rows, cols = 3, 37
    x = torch.randn(rows, cols)
    out = torch.empty_like(x)
    softmax_kernel[(rows,)](x, out, cols, BLOCK=64)
    torch.testing.assert_close(out, torch.softmax(x, dim=-1), rtol=1e-4,
                               atol=1e-5)


def test_interpreter_layernorm():
    rows, cols = 4, 33
    x = torch.randn(rows, cols)
    g = torch.randn(cols)
    b = torch.randn(cols)
    out = torch.empty_like(x)
    layernorm_kernel[(rows,)](x, g, b, out, cols, BLOCK=64)
    ref = torch.nn.functional.layer_norm(x, (cols,), g, b, 1e-5)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
