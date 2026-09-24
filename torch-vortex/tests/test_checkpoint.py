import io

import torch


def test_state_dict_checkpoint_round_trip(backend):
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.ReLU()).eval()
    state = {key: value.to("vortex") for key, value in model.state_dict().items()}
    payload = {key: value.cpu() for key, value in state.items()}
    stream = io.BytesIO()
    torch.save(payload, stream)
    stream.seek(0)
    restored = torch.load(stream, map_location="cpu", weights_only=True)
    loaded = {key: value.to("vortex") for key, value in restored.items()}
    for key in state:
        assert torch.equal(loaded[key].cpu(), state[key].cpu())
