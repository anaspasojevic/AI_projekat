import os
import sys
import torch

KORIJEN_PROJEKTA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KORIJEN_PROJEKTA)

from tokenizator.treniraj_tokenizator import ucitaj_tokenizator
from src.skup_podataka import napravi_ucitavac
from src.pozicioni_kod import napravi_pozicioni_kod
from src.model import MiniTransformer


#NAZIV_POZICIONOG_KODA = "naucena_apsolutna"
#NAZIV_POZICIONOG_KODA = "sinusoidno"
NAZIV_POZICIONOG_KODA = "rope"

DUZINA_KONTEKSTA = 32
VELICINA_PAKETA = 8
D_MODEL = 64
BROJ_EPOHA = 30
STOPA_UCENJA = 3e-4


def treniraj():
    uredjaj = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Koristi se uredjaj: {uredjaj}")

    sp = ucitaj_tokenizator()
    putanja_korpusa = os.path.join(KORIJEN_PROJEKTA, "data", "trening.txt")
    tekst = open(putanja_korpusa, encoding="utf-8").read()
    tokeni = sp.encode(tekst, out_type=int)
    print(f"Broj tokena u korpusu: {len(tokeni)}")

    ucitavac = napravi_ucitavac(tokeni, DUZINA_KONTEKSTA, VELICINA_PAKETA)

    pozicioni_kod = napravi_pozicioni_kod(NAZIV_POZICIONOG_KODA, DUZINA_KONTEKSTA, D_MODEL)
    model = MiniTransformer(
        velicina_vokabulara=sp.get_piece_size(),
        duzina_konteksta=DUZINA_KONTEKSTA,
        d_model=D_MODEL,
        pozicioni_kod=pozicioni_kod,
        koristi_rope=(NAZIV_POZICIONOG_KODA == "rope"),
    ).to(uredjaj)

    optimizator = torch.optim.AdamW(model.parameters(), lr=STOPA_UCENJA)

    print(f"Pocinje trening sa pozicionim kodom: '{NAZIV_POZICIONOG_KODA}'")
    for epoha in range(BROJ_EPOHA):
        for ulaz, cilj in ucitavac:
            ulaz, cilj = ulaz.to(uredjaj), cilj.to(uredjaj)

            logiti = model(ulaz)
            gubitak = torch.nn.functional.cross_entropy(
                logiti.reshape(-1, logiti.size(-1)),
                cilj.reshape(-1),
            )

            optimizator.zero_grad()
            gubitak.backward()
            optimizator.step()

        print(f"Epoha {epoha + 1}/{BROJ_EPOHA}, gubitak = {gubitak.item():.4f}")

    putanja_modela = os.path.join(
        KORIJEN_PROJEKTA, "rezultati", f"model_{NAZIV_POZICIONOG_KODA}.pt"
    )
    torch.save(model.state_dict(), putanja_modela)
    print(f"Model sacuvan: {putanja_modela}")


if __name__ == "__main__":
    treniraj()