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
CHROMA_DB_PATH = "/data/chroma_db"
LOG_FILE_PATH = "/data/zupan_pogovori.jsonl"

COLLECTION_NAME = "obcina_race_fram_prod" 
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
GENERATOR_MODEL_NAME = "gpt-4o-mini"
# ... (NAP API KONFIGURACIJA ostane enaka) ...

class VirtualniZupan:
    def __init__(self):
        # Ta del ostaja enak - baza se ne naloži takoj
        print("Inicializacija razreda VirtualniZupan (brez nalaganja baze)...")
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.collection = None
        self.nap_access_token = None
        # --- DODANO: Slovar za shranjevanje zgodovine pogovorov po uporabnikih ---
        self.zgodovina_seje = {}

    def nalozi_bazo(self):
        # Ta funkcija ostaja popolnoma enaka
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
        # Ta funkcija ostaja popolnoma enaka
        try:
            zapis = {
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "vprasanje": vprasanje,
                "odgovor": odgovor
            }
            with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
                f.write(json.dumps(zapis, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Napaka pri beleženju pogovora: {e}")

    def odgovori(self, uporabnikovo_vprasanje: str, session_id: str):
        # Ta del ostaja enak - najprej poskrbimo za bazo
        self.nalozi_bazo()
        if not self.collection or self.collection.count() == 0:
            odgovor = "Oprostite, zdi se, da moja baza znanja ni na voljo ali pa je prazna."
            self.belezi_pogovor(session_id, uporabnikovo_vprasanje, odgovor)
            return odgovor

        # --- DODANO: Priprava zgodovine pogovora za kontekst ---
        if session_id not in self.zgodovina_seje:
            self.zgodovina_seje[session_id] = []
        
        zgodovina_za_prompt = "\n".join([f"Uporabnik: {q}\nŽupan: {a}" for q, a in self.zgodovina_seje[session_id]])
        # --- KONEC DODATKA ---

        print(f"1. Iščem informacije za vprašanje: '{uporabnikovo_vprasanje}'")
        rezultati_iskanja = self.collection.query(query_texts=[uporabnikovo_vprasanje], n_results=7, include=["documents"])
        kontekst_baza = "\n\n---\n\n".join(rezultati_iskanja['documents'][0]) if rezultati_iskanja and rezultati_iskanja['documents'] else ""

        if not kontekst_baza:
            odgovor = "Žal o tem nimam nobenih informacij."
            self.belezi_pogovor(session_id, uporabnikovo_vprasanje, odgovor)
            return odgovor

        print("2. Pripravljam odgovor z upoštevanjem konteksta...")
        trenutno_leto = datetime.now().year
        
        # --- POSODOBLJEN PROMPT, KI VKLJUČUJE ZGODOVINO ---
        prompt_za_llm = f"""
        Ti si 'Virtualni župan občine Rače-Fram'. Tvoja naloga je, da natančno in vljudno odgovoriš na vprašanja.
        ZELO POMEMBNO: Upoštevaj pretekli pogovor za kontekst. Če se zadnje vprašanje nanaša na prejšnji odgovor, nadaljuj pogovor.

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
        
        # --- DODANO: Shranimo novo izmenjavo v spomin ---
        self.zgodovina_seje[session_id].append((uporabnikovo_vprasanje, koncni_odgovor))
        if len(self.zgodovina_seje[session_id]) > 3: # Omejimo spomin na zadnje 3 izmenjave
            self.zgodovina_seje[session_id].pop(0)
        # --- KONEC DODATKA ---
        
        self.belezi_pogovor(session_id, uporabnikovo_vprasanje, koncni_odgovor)
        return koncni_odgovor