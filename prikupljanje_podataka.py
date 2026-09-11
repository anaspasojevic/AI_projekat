import os
import sys
import time
import requests

KORIJEN_PROJEKTA = os.path.dirname(os.path.abspath(__file__))

# koliko bajtova teksta zelimo po jeziku
# 50_000_000 je otprilike 50 MB, 100_000_000 je otprilike 100 MB
CILJANA_VELICINA_PO_JEZIKU = 50_000_000

JEZICI = {
    "sr": "https://sr.wikipedia.org/w/api.php",
    "en": "https://en.wikipedia.org/w/api.php",
}

HEADERS = {
    "User-Agent": "AI-projekat"
}


def preuzmi_paket_clanaka(api_url, broj_clanaka=50):
    parametri = {
        "action": "query",
        "generator": "random",
        "grnnamespace": 0,
        "grnlimit": broj_clanaka,
        "prop": "extracts",
        "explaintext": 1,
        "exlimit": "max",
        "format": "json",
        "formatversion": 2,
    }
    odgovor = requests.get(api_url, params=parametri, headers=HEADERS, timeout=30)
    odgovor.raise_for_status()
    podaci = odgovor.json()
    stranice = podaci.get("query", {}).get("pages", [])
    tekstovi = []
    for stranica in stranice:
        tekst = stranica.get("extract", "").strip()
        if len(tekst) > 200:
            tekstovi.append(tekst)
    return tekstovi

def prikupi_korpus(jezik, api_url, ciljana_velicina):
    putanja_izlaza = os.path.join(KORIJEN_PROJEKTA, "data", f"veliki_korpus_{jezik}.txt")
    trenutna_velicina = 0
    if os.path.exists(putanja_izlaza):
        trenutna_velicina = os.path.getsize(putanja_izlaza)

    print(f"[{jezik}] pocinje prikupljanje, trenutno vec ima {trenutna_velicina/1_000_000:.1f} MB")

    pauza_kod_greske = 5

    with open(putanja_izlaza, "a", encoding="utf-8") as fajl:
        while trenutna_velicina < ciljana_velicina:
            try:
                tekstovi = preuzmi_paket_clanaka(api_url, broj_clanaka=20)
                pauza_kod_greske = 5
            except requests.exceptions.RequestException as greska:
                print(f"[{jezik}] greska, cekam {pauza_kod_greske} sekundi: {greska}")
                time.sleep(pauza_kod_greske)
                pauza_kod_greske = min(pauza_kod_greske * 2, 60)
                continue

            for tekst in tekstovi:
                fajl.write(tekst)
                fajl.write("\n\n")
                trenutna_velicina += len(tekst.encode("utf-8"))

            fajl.flush()
            print(f"[{jezik}] {trenutna_velicina/1_000_000:.1f} / {ciljana_velicina/1_000_000:.1f} MB")

            # malo duza pauza izmedju uspjesnih zahtjeva da se ne izazove ogranicenje
            time.sleep(3)

    print(f"[{jezik}] zavrseno, sacuvano u {putanja_izlaza}")

if __name__ == "__main__":
    zeljeni_jezici = sys.argv[1:] if len(sys.argv) > 1 else list(JEZICI.keys())
    for jezik in zeljeni_jezici:
        if jezik not in JEZICI:
            print(f"Nepoznat jezik: {jezik}, dostupno je: {list(JEZICI.keys())}")
            continue
        prikupi_korpus(jezik, JEZICI[jezik], CILJANA_VELICINA_PO_JEZIKU)