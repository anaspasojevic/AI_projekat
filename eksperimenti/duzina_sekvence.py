import os
import sys
import torch
import matplotlib.pyplot as plt

KORIJEN_PROJEKTA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KORIJEN_PROJEKTA)

from tokenizator.treniraj_tokenizator import ucitaj_tokenizator
from src.pozicioni_kod import BezPozicije, NauceniApsolutniEmbeding, SinusoidnoKodiranje
from src.model import MiniTransformer

DUZINA_TRENINGA = 32
DUZINE_TESTA = [32, 40, 48, 56, 64, 80]
D_MODEL = 64


def ucitaj_model(naziv, pozicioni_kod, koristi_rope, velicina_vokabulara):
    model = MiniTransformer(
        velicina_vokabulara=velicina_vokabulara,
        duzina_konteksta=DUZINA_TRENINGA,
        d_model=D_MODEL,
        pozicioni_kod=pozicioni_kod,
        koristi_rope=koristi_rope,
    )
    putanja = os.path.join(KORIJEN_PROJEKTA, "rezultati", f"model_{naziv}.pt")
    model.load_state_dict(torch.load(putanja, map_location="cpu"))
    model.eval()
    return model


def izracunaj_gubitak(model, tokeni, duzina):
    ulaz = torch.tensor(tokeni[:duzina]).unsqueeze(0)
    cilj = torch.tensor(tokeni[1:duzina + 1]).unsqueeze(0)
    with torch.no_grad():
        logiti = model(ulaz)
        gubitak = torch.nn.functional.cross_entropy(
            logiti.reshape(-1, logiti.size(-1)), cilj.reshape(-1)
        )
    return gubitak.item()


if __name__ == "__main__":
    sp = ucitaj_tokenizator()
    putanja_korpusa = os.path.join(KORIJEN_PROJEKTA, "data", "trening.txt")
    tekst = open(putanja_korpusa, encoding="utf-8").read()
    tokeni = sp.encode(tekst, out_type=int)
    velicina_vokabulara = sp.get_piece_size()

    print(f"Model treniran na duzini {DUZINA_TRENINGA}, testiramo na: {DUZINE_TESTA}\n")

    rezultati = {"naucena_apsolutna": [], "sinusoidno": [], "rope": []}

    model_naucena = ucitaj_model(
        "naucena_apsolutna", NauceniApsolutniEmbeding(DUZINA_TRENINGA, D_MODEL), False, velicina_vokabulara
    )
    for duzina in DUZINE_TESTA:
        try:
            gubitak = izracunaj_gubitak(model_naucena, tokeni, duzina)
            rezultati["naucena_apsolutna"].append(gubitak)
            print(f"naucena_apsolutna, duzina {duzina}: gubitak = {gubitak:.4f}")
        except Exception:
            rezultati["naucena_apsolutna"].append(None)
            print(f"naucena_apsolutna, duzina {duzina}: greska, tabela pozicija ne pokriva ovu duzinu")

    # sinusoidno se moze unaprijed izracunati za bilo koju duzinu bez treninga
    maks_duzina_testa = max(DUZINE_TESTA)
    model_sinusoidno = ucitaj_model(
        "sinusoidno", SinusoidnoKodiranje(DUZINA_TRENINGA, D_MODEL), False, velicina_vokabulara
    )
    model_sinusoidno.pozicioni_kod = SinusoidnoKodiranje(DUZINA_TRENINGA, D_MODEL, maks_duzina=maks_duzina_testa)
    for duzina in DUZINE_TESTA:
        gubitak = izracunaj_gubitak(model_sinusoidno, tokeni, duzina)
        rezultati["sinusoidno"].append(gubitak)
        print(f"sinusoidno, duzina {duzina}: gubitak = {gubitak:.4f}")

    model_rope = ucitaj_model("rope", BezPozicije(), True, velicina_vokabulara)
    for duzina in DUZINE_TESTA:
        gubitak = izracunaj_gubitak(model_rope, tokeni, duzina)
        rezultati["rope"].append(gubitak)
        print(f"rope, duzina {duzina}: gubitak = {gubitak:.4f}")

    plt.figure(figsize=(7, 5))
    for naziv, vrijednosti in rezultati.items():
        x = [d for d, v in zip(DUZINE_TESTA, vrijednosti) if v is not None]
        y = [v for v in vrijednosti if v is not None]
        plt.plot(x, y, marker="o", label=naziv)
    plt.axvline(DUZINA_TRENINGA, color="gray", linestyle="--", label="duzina treninga")
    plt.xlabel("duzina sekvence")
    plt.ylabel("gubitak")
    plt.title("Ponasanje mehanizama na razlicitim duzinama sekvence")
    plt.legend()
    plt.tight_layout()

    izlazni_folder = os.path.dirname(os.path.abspath(__file__))
    putanja_slike = os.path.join(izlazni_folder, "duzina_sekvence.png")
    plt.savefig(putanja_slike)
    print(f"\nSacuvano: {putanja_slike}")