import os
import sys
import time

import torch
import sentencepiece as spm

KORIJEN_PROJEKTA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, KORIJEN_PROJEKTA)

from src.skup_podataka import napravi_ucitavac
from src.pozicioni_kod import BezPozicije
from src.model import MiniTransformer

# koliko MB iz svakog jezika uzimamo za stvarni trening
MB_PO_JEZIKU = 1.0
DUZINA_KONTEKSTA = 32
VELICINA_PAKETA = 16
D_MODEL = 64
BROJ_EPOHA = 2
STOPA_UCENJA = 3e-4
VELICINA_VOKABULARA = 2000


def napravi_trening_podskup():
    putanja_sr = os.path.join(KORIJEN_PROJEKTA, "data", "veliki_korpus_sr.txt")
    putanja_en = os.path.join(KORIJEN_PROJEKTA, "data", "veliki_korpus_en.txt")
    putanja_izlaza = os.path.join(KORIJEN_PROJEKTA, "data", "veliki_korpus_trening.txt")

    granica = int(MB_PO_JEZIKU * 1_000_000)

    with open(putanja_izlaza, "w", encoding="utf-8") as izlaz:
        for putanja in [putanja_sr, putanja_en]:
            with open(putanja, encoding="utf-8") as ulaz:
                sadrzaj = ulaz.read(granica)
                izlaz.write(sadrzaj)
                izlaz.write("\n\n")

    print(f"Napravljen trening podskup: {putanja_izlaza} ({os.path.getsize(putanja_izlaza)/1_000_000:.2f} MB)")
    return putanja_izlaza


def istreniraj_tokenizator_za_veliki(putanja_korpusa):
    prefiks = os.path.join(KORIJEN_PROJEKTA, "tokenizator", "spm_veliki")
    spm.SentencePieceTrainer.train(
        input=putanja_korpusa,
        model_prefix=prefiks,
        vocab_size=VELICINA_VOKABULARA,
        model_type="bpe",
        character_coverage=0.9995,
    )
    return spm.SentencePieceProcessor(model_file=prefiks + ".model")


def treniraj():
    putanja_korpusa = napravi_trening_podskup()
    sp = istreniraj_tokenizator_za_veliki(putanja_korpusa)

    tekst = open(putanja_korpusa, encoding="utf-8").read()
    tokeni = sp.encode(tekst, out_type=int)
    print(f"Broj tokena u velikom korpusu: {len(tokeni)}")

    ucitavac = napravi_ucitavac(tokeni, DUZINA_KONTEKSTA, VELICINA_PAKETA)

    model = MiniTransformer(
        velicina_vokabulara=sp.get_piece_size(),
        duzina_konteksta=DUZINA_KONTEKSTA,
        d_model=D_MODEL,
        pozicioni_kod=BezPozicije(),
        koristi_rope=True,
    )

    optimizator = torch.optim.AdamW(model.parameters(), lr=STOPA_UCENJA)

    pocetak = time.time()
    print("Pocinje trening na velikom korpusu (rope)")
    for epoha in range(BROJ_EPOHA):
        for i, (ulaz, cilj) in enumerate(ucitavac):
            logiti = model(ulaz)
            gubitak = torch.nn.functional.cross_entropy(
                logiti.reshape(-1, logiti.size(-1)),
                cilj.reshape(-1),
            )
            optimizator.zero_grad()
            gubitak.backward()
            optimizator.step()

            if i % 200 == 0:
                proteklo = time.time() - pocetak
                print(f"Epoha {epoha + 1}/{BROJ_EPOHA}, paket {i}, gubitak = {gubitak.item():.4f}, proteklo {proteklo/60:.1f} min")

        print(f"Zavrsena epoha {epoha + 1}/{BROJ_EPOHA}, gubitak = {gubitak.item():.4f}")

    putanja_modela = os.path.join(KORIJEN_PROJEKTA, "rezultati", "model_rope_veliki.pt")
    torch.save(model.state_dict(), putanja_modela)
    print(f"Model sacuvan: {putanja_modela}")


if __name__ == "__main__":
    treniraj()