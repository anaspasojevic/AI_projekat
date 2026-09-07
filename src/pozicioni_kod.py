import torch
from torch import nn


class PozicioniKod(nn.Module):
    def primijeni(self, x):
        raise NotImplementedError


class BezPozicije(PozicioniKod):
    # ne dodaje nikakvu informaciju o poziciji

    def primijeni(self, x):
        return x


class NauceniApsolutniEmbeding(PozicioniKod):
    # svaka pozicija ima svoj vektor koji se uci i sabira sa token embedingom

    def __init__(self, duzina_konteksta, d_model):
        super().__init__()
        self.embeding_pozicije = nn.Embedding(duzina_konteksta, d_model)

    def primijeni(self, x):
        # x oblika B T d_model
        _, T, _ = x.shape
        pozicije = torch.arange(T, device=x.device)
        return x + self.embeding_pozicije(pozicije)