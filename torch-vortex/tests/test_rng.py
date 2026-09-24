import torch

import torch_vortex


def test_philox_uniform_is_reproducible(backend):
    torch_vortex.manual_seed(1234)
    first = torch_vortex.rand(9).cpu()
    torch_vortex.manual_seed(1234)
    second = torch_vortex.rand((9,)).cpu()
    assert torch.equal(first, second)
    assert torch.all((first >= 0) & (first < 1))


def test_philox_stream_advances_by_counter_blocks(backend):
    torch_vortex.manual_seed(77)
    joined = torch.cat((torch_vortex.rand(4), torch_vortex.rand(4))).cpu()
    torch_vortex.manual_seed(77)
    whole = torch_vortex.rand(8).cpu()
    assert torch.equal(joined, whole)


def test_philox_state_round_trip(backend):
    torch_vortex.manual_seed(91)
    torch_vortex.rand(4)
    state = torch_vortex.get_rng_state()
    expected = torch_vortex.rand(4).cpu()
    torch_vortex.set_rng_state(state)
    assert torch.equal(expected, torch_vortex.rand(4).cpu())
