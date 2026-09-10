import math
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


class SinusoidnoKodiranje(PozicioniKod):
    # fiksno kodiranje, ne uci se tokom treninga, racuna se unaprijed po formuli

    def __init__(self, duzina_konteksta, d_model, maks_duzina=None):
        super().__init__()
        maks_duzina = maks_duzina or duzina_konteksta

        pe = torch.zeros(maks_duzina, d_model)
        pozicije = torch.arange(0, maks_duzina, dtype=torch.float).unsqueeze(1)
        djelilac = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pozicije * djelilac)
        pe[:, 1::2] = torch.cos(pozicije * djelilac)

        self.register_buffer("pe", pe)

    def primijeni(self, x):
        _, T, _ = x.shape
        return x + self.pe[:T]


def napravi_pozicioni_kod(naziv, duzina_konteksta, d_model):
    # zajednicka funkcija, koriste je i trening i generisanje
    if naziv == "bez_pozicije":
        return BezPozicije()
    elif naziv == "naucena_apsolutna":
        return NauceniApsolutniEmbeding(duzina_konteksta, d_model)
    elif naziv == "sinusoidno":
        return SinusoidnoKodiranje(duzina_konteksta, d_model)
    elif naziv == "rope":
        # kod rope se pozicija ne dodaje ovdje nego u attention sloju
        return BezPozicije()
    else:
        raise ValueError(f"Nepoznat pozicioni kod: {naziv}")