import torch
from torch import nn
from src.rope import primijeni_rope


class SelfAttention(nn.Module):
    def __init__(self, d_model, koristi_rope=False):
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.koristi_rope = koristi_rope

    def forward(self, x):
        B, T, D = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        if self.koristi_rope:
            Q = primijeni_rope(Q)
            K = primijeni_rope(K)

        skor = Q @ K.transpose(-2, -1) / (D ** 0.5)

        # token ne smije da vidi tokene poslije sebe
        maska = torch.tril(torch.ones(T, T, device=x.device))
        skor = skor.masked_fill(maska == 0, float("-inf"))

        paznja = torch.softmax(skor, dim=-1)
        izlaz = paznja @ V
        return izlaz


class Blok(nn.Module):
    def __init__(self, d_model, koristi_rope=False):
        super().__init__()
        self.norma1 = nn.LayerNorm(d_model)
        self.paznja = SelfAttention(d_model, koristi_rope=koristi_rope)
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
    def __init__(self, velicina_vokabulara, duzina_konteksta, d_model, pozicioni_kod, koristi_rope=False):
        super().__init__()
        self.token_embeding = nn.Embedding(velicina_vokabulara, d_model)
        self.pozicioni_kod = pozicioni_kod

        self.blok = Blok(d_model, koristi_rope=koristi_rope)
        self.zavrsna_norma = nn.LayerNorm(d_model)
        self.izlazni_sloj = nn.Linear(d_model, velicina_vokabulara)

    def forward(self, x):
        x = self.token_embeding(x)
        x = self.pozicioni_kod.primijeni(x)
        x = self.blok(x)
        x = self.zavrsna_norma(x)
        logiti = self.izlazni_sloj(x)
        return logiti