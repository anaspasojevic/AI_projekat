import os
import sys
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

KORIJEN_PROJEKTA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, KORIJEN_PROJEKTA)

from src.pozicioni_kod import SinusoidnoKodiranje

DUZINA_KONTEKSTA = 32
D_MODEL = 64


def ucitaj_naucene_pozicije():
    putanja = os.path.join(KORIJEN_PROJEKTA, "rezultati", "model_naucena_apsolutna.pt")
    stanje = torch.load(putanja, map_location="cpu")
    return stanje["pozicioni_kod.embeding_pozicije.weight"]


def nacrtaj_pca(vektori, naslov, putanja_slike):
    pca = PCA(n_components=2)
    tacke = pca.fit_transform(vektori.detach().numpy())

    plt.figure(figsize=(6, 5))
    plt.scatter(tacke[:, 0], tacke[:, 1], c=range(len(tacke)), cmap="viridis")
    for i in range(0, len(tacke), 4):
        plt.annotate(str(i), (tacke[i, 0], tacke[i, 1]))
    plt.colorbar(label="pozicija")
    plt.title(naslov)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig(putanja_slike)
    plt.close()
    print(f"Sacuvano: {putanja_slike}")


def nacrtaj_heatmap(matrica, naslov, putanja_slike):
    plt.figure(figsize=(8, 5))
    plt.imshow(matrica.detach().numpy(), aspect="auto", cmap="coolwarm")
    plt.colorbar(label="vrijednost")
    plt.title(naslov)
    plt.xlabel("dimenzija")
    plt.ylabel("pozicija")
    plt.tight_layout()
    plt.savefig(putanja_slike)
    plt.close()
    print(f"Sacuvano: {putanja_slike}")


if __name__ == "__main__":
    izlazni_folder = os.path.dirname(os.path.abspath(__file__))

    naucene = ucitaj_naucene_pozicije()
    nacrtaj_pca(naucene, "Naucena apsolutna embedovanja (PCA)", os.path.join(izlazni_folder, "pca_naucena.png"))
    nacrtaj_heatmap(naucene, "Naucena apsolutna embedovanja", os.path.join(izlazni_folder, "heatmap_naucena.png"))

    sinusoidno = SinusoidnoKodiranje(DUZINA_KONTEKSTA, D_MODEL)
    nacrtaj_pca(sinusoidno.pe, "Sinusoidno kodiranje (PCA)", os.path.join(izlazni_folder, "pca_sinusoidno.png"))
    nacrtaj_heatmap(sinusoidno.pe, "Sinusoidno kodiranje", os.path.join(izlazni_folder, "heatmap_sinusoidno.png"))