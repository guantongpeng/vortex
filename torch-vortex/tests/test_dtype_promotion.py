import torch


def test_float_and_integer_device_promotion(backend):
    f = torch.tensor([1.0, 2.0, 3.0]).to("vortex")
    i = torch.tensor([2, 4, 6], dtype=torch.int32).to("vortex")
    torch.testing.assert_close((f + i).cpu(), torch.tensor([3.0, 6.0, 9.0]))
    torch.testing.assert_close((i + f).cpu(), torch.tensor([3.0, 6.0, 9.0]))


def test_float64_promotion_keeps_output_dtype(backend):
    f32 = torch.tensor([1.0, 2.0]).to("vortex")
    f64 = torch.tensor([3.0, 4.0], dtype=torch.float64).to("vortex")
    got = f32 + f64
    assert got.dtype == torch.float64
    torch.testing.assert_close(got.cpu(), torch.tensor([4.0, 6.0], dtype=torch.float64))


def test_integer_binary_promotion(backend):
    a = torch.tensor([1, 2, 7], dtype=torch.int32)
    b = torch.tensor([3, 5, 2], dtype=torch.int64)
    got = a.to("vortex") + b.to("vortex")
    want = a + b
    assert got.dtype == want.dtype
    torch.testing.assert_close(got.cpu(), want)
