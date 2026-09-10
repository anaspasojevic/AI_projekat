import os
import sys
import torch
import matplotlib.pyplot as plt

KORIJEN_PROJEKTA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KORIJEN_PROJEKTA)

from src.rope import primijeni_rope

D_MODEL = 64
BROJ_POZICIJA = 40

if __name__ == "__main__":
    torch.manual_seed(0)
    vektor = torch.randn(1, D_MODEL)

    # isti vektor ponovljen na svakoj poziciji, pa rotiran po poziciji
    ulaz = vektor.repeat(BROJ_POZICIJA, 1).unsqueeze(0)
    rotirano = primijeni_rope(ulaz)[0]

    # koliko je rotirani vektor na poziciji 0 slican rotiranom vektoru na poziciji i
    referenca = rotirano[0]
    slicnost = [torch.dot(referenca, rotirano[i]).item() for i in range(BROJ_POZICIJA)]

    plt.figure(figsize=(7, 5))
    plt.plot(range(BROJ_POZICIJA), slicnost, marker="o")
    plt.xlabel("relativna razlika pozicija")
    plt.ylabel("dot produkt sa pozicijom 0")
    plt.title("RoPE: slicnost zavisi samo od relativne pozicije")
    plt.tight_layout()

    izlazni_folder = os.path.dirname(os.path.abspath(__file__))
    putanja_slike = os.path.join(izlazni_folder, "rope_rotacija.png")
    plt.savefig(putanja_slike)
    print(f"Sacuvano: {putanja_slike}")