import os
import json
import re
from openai import OpenAI
from dotenv import load_dotenv

# --- KONFIGURACIJA ---
VHODNA_DATOTEKA_TXT = "koledar_odpadkov_surovo.txt"
IZHODNA_DATOTEKA_JSONL = "koledar_odpadkov_2025.jsonl"
VIR_INFORMACIJE = "Koledar Odvoza Odpadkov 2025"
# ---------------------

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def pretvori_koledar_v_jsonl():
    """
    Prebere surovo besedilo iz koledarja odpadkov, ga s pomočjo AI strukturira
    in shrani v JSONL format, primeren za bazo znanja.
    """
    print(f"Začenjam pretvorbo datoteke: {VHODNA_DATOTEKA_TXT}")
    
    try:
        with open(VHODNA_DATOTEKA_TXT, 'r', encoding='utf-8') as f:
            vsebina = f.read()
    except FileNotFoundError:
        print(f"NAPAKA: Vhodne datoteke '{VHODNA_DATOTEKA_TXT}' ni mogoče najti.")
        return

    # S pomočjo AI izluščimo strukturirane podatke
    try:
        print("-> Pošiljam vsebino v obdelavo AI modelu. To lahko traja minuto ali dve...")
        prompt = f"""
        Iz spodnjega besedila, ki predstavlja koledar odvoza odpadkov za leto 2025, izlušči podatke za vsako območje posebej.
        Za vsako območje navedi seznam naselij in nato za vsak tip odpadka (MEŠANI KOMUNALNI ODPADKI, ODPADNA EMBALAŽA, PAPIR IN KARTON, STEKLENA EMBALAŽA, BIOLOŠKI ODPADKI) navedi TOČNE datume odvoza v formatu 'DD.MM.'.
        
        Primer izhoda za eno območje:
        OBMOČJE: Brezula, Podova, Rače...
        MEŠANI KOMUNALNI ODPADKI: 10.01., 24.01., 07.02., 21.02., ...
        ODPADNA EMBALAŽA: 10.01., 24.01., 07.02., 21.02., ...
        PAPIR IN KARTON: 15.01., 12.03., ...
        STEKLENA EMBALAŽA: 20.02., 17.04., ...
        BIOLOŠKI ODPADKI: vsak teden od 1.3. do 30.11., sicer vsaka dva tedna.

        Besedilo:
        "{vsebina}"

        Strukturirani podatki:
        """

        response = client.chat.completions.create(
            model="gpt-4o", # Uporabimo najzmogljivejši model za to kompleksno nalogo
            messages=[{"role": "user", "content": prompt}]
        )
        
        strukturirano_besedilo = response.choices[0].message.content.strip()
        print("-> AI model je uspešno obdelal besedilo.")

    except Exception as e:
        print(f"NAPAKA pri komunikaciji z OpenAI: {e}")
        return

    # Pretvorimo strukturirano besedilo v JSONL
    print("-> Pretvarjam strukturirano besedilo v JSONL format...")
    with open(IZHODNA_DATOTEKA_JSONL, 'w', encoding='utf-8') as f_out:
        trenutno_obmocje = "Neznano območje"
        for vrstica in strukturirano_besedilo.split('\n'):
            vrstica = vrstica.strip()
            if not vrstica:
                continue
            
            if vrstica.startswith("OBMOČJE:"):
                trenutno_obmocje = vrstica.replace("OBMOČJE:", "").strip()
            else:
                deli = vrstica.split(':', 1)
                if len(deli) == 2:
                    tip_odpadka = deli[0].strip()
                    datumi = deli[1].strip()
                    
                    zapis = {
                        "text": f"Za območje, ki vključuje '{trenutno_obmocje}', je odvoz za '{tip_odpadka}' v letu 2025 predviden na naslednje datume: {datumi}.",
                        "metadata": {"source": VIR_INFORMACIJE, "kategorija": "Odvoz odpadkov"}
                    }
                    f_out.write(json.dumps(zapis, ensure_ascii=False) + '\n')
    
    print(f"\nPretvorba uspešno končana!")
    print(f"Podatki so shranjeni v datoteko '{IZHODNA_DATOTEKA_JSONL}'.")


if __name__ == "__main__":
    pretvori_koledar_v_jsonl()