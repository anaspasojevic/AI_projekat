import torch
from torch.utils.data import Dataset, DataLoader


class TekstualniSkupPodataka(Dataset):
    def __init__(self, tokeni, duzina_konteksta):
        self.tokeni = tokeni
        self.duzina_konteksta = duzina_konteksta

    def __len__(self):
        return len(self.tokeni) - self.duzina_konteksta

    def __getitem__(self, indeks):
        ulaz = torch.tensor(
            self.tokeni[indeks: indeks + self.duzina_konteksta],
            dtype=torch.long,
        )
        cilj = torch.tensor(
            self.tokeni[indeks + 1: indeks + self.duzina_konteksta + 1],
            dtype=torch.long,
        )
        return ulaz, cilj


def napravi_ucitavac(tokeni, duzina_konteksta, velicina_paketa, promijesaj=True):
    skup_podataka = TekstualniSkupPodataka(tokeni, duzina_konteksta)
    return DataLoader(skup_podataka, batch_size=velicina_paketa, shuffle=promijesaj)