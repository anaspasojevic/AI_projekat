import os
import sys
import torch
import sentencepiece as spm

KORIJEN_PROJEKTA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, KORIJEN_PROJEKTA)

from src.pozicioni_kod import BezPozicije
from src.model import MiniTransformer

DUZINA_KONTEKSTA = 32
D_MODEL = 64


def generisi(model, sp, pocetak, broj_tokena=80):
    tokeni = sp.encode(pocetak, out_type=int)
    for _ in range(broj_tokena):
        ulaz = torch.tensor(tokeni[-DUZINA_KONTEKSTA:]).unsqueeze(0)
        with torch.no_grad():
            logiti = model(ulaz)
        sledeci = torch.argmax(logiti[0, -1]).item()
        tokeni.append(sledeci)
    return sp.decode(tokeni)


if __name__ == "__main__":
    putanja_tokenizatora = os.path.join(KORIJEN_PROJEKTA, "tokenizator", "spm_veliki.model")
    sp = spm.SentencePieceProcessor(model_file=putanja_tokenizatora)

    model = MiniTransformer(
        velicina_vokabulara=sp.get_piece_size(),
        duzina_konteksta=DUZINA_KONTEKSTA,
        d_model=D_MODEL,
        pozicioni_kod=BezPozicije(),
        koristi_rope=True,
    )
    putanja_modela = os.path.join(KORIJEN_PROJEKTA, "rezultati", "model_rope_veliki.pt")
    model.load_state_dict(torch.load(putanja_modela, map_location="cpu"))
    model.eval()

    pocetak = input("Unesi pocetak recenice: ")
    tekst = generisi(model, sp, pocetak)
    print(tekst)