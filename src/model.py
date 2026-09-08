import torch
from torch import nn


class SamoPaznja(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.upit = nn.Linear(d_model, d_model)
        self.kljuc = nn.Linear(d_model, d_model)
        self.vrijednost = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape

        Q = self.upit(x)
        K = self.kljuc(x)
        V = self.vrijednost(x)

        skor = Q @ K.transpose(-2, -1) / (D ** 0.5)

        # token ne smije da vidi tokene poslije sebe
        maska = torch.tril(torch.ones(T, T, device=x.device))
        skor = skor.masked_fill(maska == 0, float("-inf"))

        paznja = torch.softmax(skor, dim=-1)
        izlaz = paznja @ V
        return izlaz


class Blok(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norma1 = nn.LayerNorm(d_model)
        self.paznja = SamoPaznja(d_model)
        self.norma2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.paznja(self.norma1(x))
        x = x + self.mlp(self.norma2(x))
        return x


class MiniTransformer(nn.Module):
    def __init__(self, velicina_vokabulara, duzina_konteksta, d_model, pozicioni_kod):
        super().__init__()
        self.token_embeding = nn.Embedding(velicina_vokabulara, d_model)
        self.pozicioni_kod = pozicioni_kod

        self.blok = Blok(d_model)
        self.zavrsna_norma = nn.LayerNorm(d_model)
        self.izlazni_sloj = nn.Linear(d_model, velicina_vokabulara)

    def forward(self, x):
        x = self.token_embeding(x)
        x = self.pozicioni_kod.primijeni(x)
        x = self.blok(x)
        x = self.zavrsna_norma(x)
        logiti = self.izlazni_sloj(x)
        return logiti