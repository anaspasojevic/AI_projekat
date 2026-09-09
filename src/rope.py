import torch


def primijeni_rope(x, baza=10000):
    # x oblika B T D, D mora biti paran
    B, T, D = x.shape
    assert D % 2 == 0

    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    pozicije = torch.arange(T, device=x.device).float()
    frekvencije = 1.0 / (baza ** (torch.arange(0, D, 2, device=x.device).float() / D))
    uglovi = pozicije[:, None] * frekvencije[None, :]

    sin = torch.sin(uglovi)[None, :, :]
    cos = torch.cos(uglovi)[None, :, :]

    r1 = x1 * cos - x2 * sin
    r2 = x1 * sin + x2 * cos

    izlaz = torch.stack((r1, r2), dim=-1)
    return izlaz.flatten(-2)