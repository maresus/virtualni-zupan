import os
from PyPDF2 import PdfReader

# --- KONFIGURACIJA ---
# Ime vaše vhodne PDF datoteke
VHODNI_PDF = "koledar_odpadkov_2025.pdf" 

# Ime izhodne TXT datoteke, ki bo ustvarjena
IZHODNI_TXT = "koledar_odpadkov_2025.txt"

def ekstrahiraj_besedilo_iz_pdf():
    """
    Odpre PDF datoteko, prebere besedilo z vseh strani in ga shrani
    v eno samo, surovo TXT datoteko.
    """
    print(f"Začenjam ekstrakcijo besedila iz datoteke: {VHODNI_PDF}")
    
    # Preverimo, ali datoteka obstaja v isti mapi kot skripta
    if not os.path.exists(VHODNI_PDF):
        print(f"NAPAKA: Datoteke '{VHODNI_PDF}' ni mogoče najti. Prepričajte se, da je v isti mapi kot ta skripta.")
        return

    try:
        reader = PdfReader(VHODNI_PDF)
        celotno_besedilo = ""
        for i, page in enumerate(reader.pages):
            print(f"  -> Berem stran {i+1}/{len(reader.pages)}")
            besedilo_strani = page.extract_text()
            if besedilo_strani:
                celotno_besedilo += besedilo_strani + "\n--- NOVA STRAN ---\n"
        
        with open(IZHODNI_TXT, 'w', encoding='utf-8') as f:
            f.write(celotno_besedilo)
            
        print(f"\nUSPEH! Vsebina je shranjena v datoteko '{IZHODNI_TXT}'.")
        print("Naslednji korak je, da to TXT datoteko uporabite s pretvornikom v JSONL format.")

    except Exception as e:
        print(f"Prišlo je do napake med obdelavo PDF-ja: {e}")

if __name__ == "__main__":
    ekstrahiraj_besedilo_iz_pdf()