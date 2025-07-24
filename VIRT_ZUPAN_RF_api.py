import os
import json
import chromadb
import requests
import traceback
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from chromadb.utils import embedding_functions

# --- KONFIGURACIJA (ostaja enaka) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))
CHROMA_DB_PATH = "/data/chroma_db_prod" # Pot za Render
LOG_FILE_PATH = "/data/zupan_pogovori.jsonl"

COLLECTION_NAME = "obcina_race_fram_prod" 
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
GENERATOR_MODEL_NAME = "gpt-4o-mini"
NAP_TOKEN_URL = "https://b2b.nap.si/uc/user/token"
NAP_DATA_URL = "https://b2b.nap.si/data/b2b.roadworks.geojson.sl_SI"
LOKACIJE_ZA_FILTER = ["Rače", "Fram", "Slivnica"]
NAP_USERNAME = os.getenv("NAP_USERNAME")
NAP_PASSWORD = os.getenv("NAP_PASSWORD")

class VirtualniZupan:
    def __init__(self):
        print("Pripravljam virtualnega župana (verzija 5.2 - s spominom in povezavami)...")
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.collection = None
        self.nap_access_token = None
        # --- DODANO: Slovar za shranjevanje zgodovine pogovorov po uporabnikih ---
        self.zgodovina_seje = {}
        # Baza se naloži šele ob prvi poizvedbi znotraj `odgovori` funkcije.

    def nalozi_bazo(self):
        """Naloži bazo znanja samo, ko je to res potrebno."""
        if self.collection is None:
            try:
                print(f"Poskušam naložiti bazo znanja iz poti: {CHROMA_DB_PATH}")
                openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"), model_name=EMBEDDING_MODEL_NAME)
                chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
                self.collection = chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=openai_ef)
                print(f"USPEH: Povezan z bazo znanja. V bazi je {self.collection.count()} dokumentov.")
            except Exception as e:
                print(f"KRITIČNA NAPAKA: Baze znanja ni mogoče naložiti. Razlog: {e}")

    def belezi_pogovor(self, session_id, vprasanje, odgovor):
        # Ta funkcija ostaja enaka
        try:
            zapis = {"timestamp": datetime.now().isoformat(), "session_id": session_id, "vprasanje": vprasanje, "odgovor": odgovor}
            with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
                f.write(json.dumps(zapis, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Napaka pri beleženju pogovora: {e}")
    
    def preveri_zapore_cest(self):
        # Ta funkcija ostaja enaka
        if not NAP_USERNAME or not NAP_PASSWORD:
            return "Dostop do prometnih informacij ni mogoč, ker niso vpisani prijavni podatki za NAP API."
        # ... (ostanek kode za NAP API)
        return "Funkcija za preverjanje zapor cest še ni v celoti implementirana."


    def odgovori(self, uporabnikovo_vprasanje: str, session_id: str):
        self.nalozi_bazo() # Zagotovimo, da je baza naložena
        if not self.collection or self.collection.count() == 0:
            odgovor = "Oprostite, zdi se, da moja baza znanja ni na voljo ali pa je prazna."
            self.belezi_pogovor(session_id, uporabnikovo_vprasanje, odgovor)
            return odgovor

        # Priprava zgodovine pogovora za kontekst
        if session_id not in self.zgodovina_seje:
            self.zgodovina_seje[session_id] = []
        zgodovina_za_prompt = "\n".join([f"Uporabnik: {q}\nŽupan: {a}" for q, a in self.zgodovina_seje[session_id]])

        print(f"1. Iščem informacije za vprašanje: '{uporabnikovo_vprasanje}'")
        rezultati_iskanja = self.collection.query(
            query_texts=[uporabnikovo_vprasanje],
            n_results=7,
            include=["documents", "metadatas"] # Pridobimo tudi metapodatke (za URL-je)
        )
        
        # Sestavimo kontekst iz dokumentov in metapodatkov
        kontekst_baza = ""
        if rezultati_iskanja and rezultati_iskanja['documents']:
            for i, doc in enumerate(rezultati_iskanja['documents'][0]):
                meta = rezultati_iskanja['metadatas'][0][i]
                kontekst_baza += f"VIR: {meta.get('source', 'Neznan')}\n"
                if meta and meta.get('source_url'): # Preverimo, ali 'meta' ni None
                    kontekst_baza += f"POVEZAVA: {meta.get('source_url')}\n"
                kontekst_baza += f"VSEBINA: {doc}\n---\n"

        if not kontekst_baza.strip():
            odgovor = "Žal o tem nimam nobenih informacij."
            self.belezi_pogovor(session_id, uporabnikovo_vprasanje, odgovor)
            return odgovor

        print("2. Pripravljam odgovor z upoštevanjem konteksta...")
        trenutno_leto = datetime.now().year
        
        # --- POSODOBLJEN PROMPT Z OBEMA NOVIMA PRAVILOMA ---
        prompt_za_llm = f"""
        Ti si 'Virtualni župan občine Rače-Fram'. Bodi kratek, jedrnat in izjemno koristen.

        PRAVILO #1 (SLEDENJE POGOVORU): Upoštevaj pretekli pogovor za kontekst. Če se zadnje vprašanje ("{uporabnikovo_vprasanje}") nanaša na prejšnje teme, nadaljuj pogovor.

        PRAVILO #2 (DATUMI): Danes je leto {trenutno_leto}. Vedno preveri datume v priloženih informacijah. Če je podatek iz preteklega leta, to JASNO OMENI v odgovoru.

        PRAVILO #3 (POVEZAVE): Če v priloženih informacijah najdeš spletno povezavo (URL), ki se nanaša na vprašanje, jo VEDNO vključi v odgovor v obliki [Ime povezave](URL). Ne odgovarjaj samo 'na spletni strani', ampak TAKOJ navedi točen link.

        --- ZGODOVINA POGOVORA ---
        {zgodovina_za_prompt}
        ---

        --- INFORMACIJE IZ BAZE ZNANJA (uporabi jih za odgovor na zadnje vprašanje) ---
        {kontekst_baza}
        ---

        ZADNJE VPRAŠANJE UPORABNIKA: "{uporabnikovo_vprasanje}"
        
        TVOJ ODGOVOR:
        """
        
        response = self.openai_client.chat.completions.create(model=GENERATOR_MODEL_NAME, messages=[{"role": "user", "content": prompt_za_llm}], temperature=0.1)
        koncni_odgovor = response.choices[0].message.content
        
        self.zgodovina_seje[session_id].append((uporabnikovo_vprasanje, koncni_odgovor))
        if len(self.zgodovina_seje[session_id]) > 3:
            self.zgodovina_seje[session_id].pop(0)
        
        self.belezi_pogovor(session_id, uporabnikovo_vprasanje, koncni_odgovor)
        return koncni_odgovor

# ... (glavna zanka `if __name__ == "__main__":` ostane enaka) ...