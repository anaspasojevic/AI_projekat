import os
import sentencepiece as spm

KORIJEN_PROJEKTA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUTANJA_KORPUSA = os.path.join(KORIJEN_PROJEKTA, "data", "trening.txt")
PREFIKS_MODELA = os.path.join(os.path.dirname(__file__), "spm")
VELICINA_VOKABULARA = 500


def treniraj_tokenizator():
    spm.SentencePieceTrainer.train(
        input=PUTANJA_KORPUSA,
        model_prefix=PREFIKS_MODELA,
        vocab_size=VELICINA_VOKABULARA,
        model_type="bpe",
        character_coverage=0.9995,
    )
    print(f"Tokenizator istreniran. Fajlovi sacuvani pored: {PREFIKS_MODELA}.model / .vocab")


def ucitaj_tokenizator():
    putanja_modela = PREFIKS_MODELA + ".model"
    if not os.path.exists(putanja_modela):
        raise FileNotFoundError("Tokenizator nije pronadjen. Prvo pokreni: python tokenizator/treniraj_tokenizator.py")
    return spm.SentencePieceProcessor(model_file=putanja_modela)


if __name__ == "__main__":
    treniraj_tokenizator()
    sp = ucitaj_tokenizator()
    primjer = "Ko je bio Nikola Tesla?"
    id_tokeni = sp.encode(primjer, out_type=int)
    print(f"\nProvjera na primjeru: '{primjer}'")
    print("ID tokeni:", id_tokeni)
    print("Dekodirano nazad:", sp.decode(id_tokeni))