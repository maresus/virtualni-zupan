import os
import json
from openai import OpenAI
from dotenv import load_dotenv

# --- KONFIGURACIJA ---
# Spremenite samo ti dve vrstici glede na vaše potrebe
VHODNA_DATOTEKA_TXT = "koledar_odpadkov_2025.txt"  # Ime datoteke z golim besedilom
IZHODNA_DATOTEKA_JSONL = "koledar_odpadkov_2025.jsonl" # Ime datoteke, ki bo ustvarjena
VIR_INFORMACIJE = "Spletna stran občine" # Vir, ki bo zapisan v metapodatke
# ---------------------

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def pretvori_besedilo_v_jsonl():
    """
    Prebere surovo besedilo, ga s pomočjo AI razdeli na 'atomska dejstva'
    in jih shrani v strukturirano JSONL datoteko.
    """
    print(f"Začenjam pretvorbo datoteke: {VHODNA_DATOTEKA_TXT}")
    
    try:
        with open(VHODNA_DATOTEKA_TXT, 'r', encoding='utf-8') as f:
            vsebina = f.read()
    except FileNotFoundError:
        print(f"Napaka: Vhodne datoteke '{VHODNA_DATOTEKA_TXT}' ni mogoče najti.")
        return

    # Besedilo razdelimo na odstavke
    odstavki = [p.strip() for p in vsebina.split('\n\n') if p.strip()]
    print(f"Najdenih {len(odstavki)} odstavkov za obdelavo.")

    vsa_dejstva = []
    
    for i, odstavek in enumerate(odstavki):
        print(f"  -> Obdelujem odstavek {i+1}/{len(odstavki)}...")
        
        # Prekratek odstavek preskočimo
        if len(odstavek) < 50:
            continue
            
        try:
            # Pripravimo navodila za AI model
            prompt = f"""
            Iz naslednjega besedila izlušči ključna dejstva v obliki kratkih, jedrnatih stavkov. Vsako dejstvo naj bo samostojno in razumljivo. 
            Vsako dejstvo zapiši v svojo vrstico. Ne dodajaj alinej ali oštevilčenja.

            Besedilo:
            "{odstavek}"

            Izluščena dejstva:
            """

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            
            izluscena_dejstva = response.choices[0].message.content.strip().split('\n')
            vsa_dejstva.extend(izluscena_dejstva)

        except Exception as e:
            print(f"    Napaka pri obdelavi odstavka: {e}")

    # Shranimo rezultate v JSONL datoteko
    with open(IZHODNA_DATOTEKA_JSONL, 'w', encoding='utf-8') as f_out:
        for dejstvo in vsa_dejstva:
            if dejstvo.strip():
                zapis = {
                    "text": dejstvo.strip(),
                    "metadata": {"source": VIR_INFORMACIJE}
                }
                f_out.write(json.dumps(zapis, ensure_ascii=False) + '\n')
    
    print(f"\nPretvorba uspešno končana!")
    print(f"{len(vsa_dejstva)} dejstev je bilo shranjenih v datoteko '{IZHODNA_DATOTEKA_JSONL}'.")

if __name__ == "__main__":
    pretvori_besedilo_v_jsonl()