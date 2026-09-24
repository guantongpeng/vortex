import torch
import torch.nn as nn


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.q = nn.Linear(4, 4)
        self.k = nn.Linear(4, 4)
        self.v = nn.Linear(4, 4)
        self.proj = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.ff = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 4))

    def forward(self, tokens):
        x = self.embedding(tokens)
        q, k, v = self.q(x), self.k(x), self.v(x)
        scores = (q @ k.transpose(-2, -1).contiguous()) / 2
        context = torch.softmax(scores, dim=-1) @ v
        x = self.norm(x + self.proj(context))
        return x + self.ff(x)


def test_tiny_transformer_components_match_cpu(backend):
    torch.manual_seed(20260924)
    cpu_model = TinyDecoder().eval()
    vortex_model = TinyDecoder().eval()
    vortex_model.load_state_dict(cpu_model.state_dict())
    vortex_model.to("vortex")
    tokens = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
    with torch.inference_mode():
        want = cpu_model(tokens)
        got = vortex_model(tokens.to("vortex")).cpu()
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)
