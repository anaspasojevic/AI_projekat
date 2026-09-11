import os
import sys
import torch

KORIJEN_PROJEKTA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KORIJEN_PROJEKTA)

from tokenizator.treniraj_tokenizator import ucitaj_tokenizator
from src.pozicioni_kod import napravi_pozicioni_kod
from src.model import MiniTransformer

NAZIV_POZICIONOG_KODA = "rope"
#NAZIV_POZICIONOG_KODA = "sinusoidno"
#NAZIV_POZICIONOG_KODA = "naucena_apsolutna"
#NAZIV_POZICIONOG_KODA = "bez_pozicije"


DUZINA_KONTEKSTA = 32 # br. tokena
D_MODEL = 64 # dimenzija vektora kojima su postavljeni tokeni


def generisi(model, sp, pocetak, broj_tokena=45):
    tokeni = sp.encode(pocetak, out_type=int)
    for _ in range(broj_tokena):
        ulaz = torch.tensor(tokeni[-DUZINA_KONTEKSTA:]).unsqueeze(0)
        with torch.no_grad():
            logiti = model(ulaz)
        sledeci = torch.argmax(logiti[0, -1]).item()
        tokeni.append(sledeci)
    tekst = sp.decode(tokeni)

    # ako model pocne sljedece pitanje, prikazujemo samo odgovor na nase
    kraj = tekst.find("Pitanje:", 1)
    if kraj != -1:
        tekst = tekst[:kraj].strip()
    return tekst


if __name__ == "__main__":
    sp = ucitaj_tokenizator()
    pozicioni_kod = napravi_pozicioni_kod(NAZIV_POZICIONOG_KODA, DUZINA_KONTEKSTA, D_MODEL)
    model = MiniTransformer(
        velicina_vokabulara=sp.get_piece_size(),
        duzina_konteksta=DUZINA_KONTEKSTA,
        d_model=D_MODEL,
        pozicioni_kod=pozicioni_kod,
        koristi_rope=(NAZIV_POZICIONOG_KODA == "rope"),
    )
    putanja_modela = os.path.join(KORIJEN_PROJEKTA, "rezultati", f"model_{NAZIV_POZICIONOG_KODA}.pt")
    model.load_state_dict(torch.load(putanja_modela, map_location="cpu"))
    model.eval()

    pocetak = input("Unesi pocetak recenice: ")
    tekst = generisi(model, sp, pocetak)
    print(tekst)