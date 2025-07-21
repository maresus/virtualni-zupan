import os
import json
import chromadb
import requests
import traceback
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from chromadb.utils import embedding_functions

# --- KONFIGURACIJA ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))
CHROMA_DB_PATH = "/data/chroma_db"

COLLECTION_NAME = "obcina_race_fram_prod" 
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
GENERATOR_MODEL_NAME = "gpt-4o-mini"
NAP_TOKEN_URL = "https://b2b.nap.si/uc/user/token"
NAP_DATA_URL = "https://b2b.nap.si/data/b2b.roadworks_si.json" 
LOKACIJE_ZA_FILTER = ["Rače", "Fram", "Slivnica"]
NAP_USERNAME = os.getenv("NAP_USERNAME")
NAP_PASSWORD = os.getenv("NAP_PASSWORD")

class VirtualniZupan:
    def __init__(self):
        print("Inicializacija razreda VirtualniZupan (brez nalaganja baze)...")
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.collection = None
        self.nap_access_token = None
        self.zgodovina_pogovora = []

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
                traceback.print_exc()

    def odgovori(self, uporabnikovo_vprasanje: str):
        self.nalozi_bazo() # Zagotovimo, da je baza naložena
        if not self.collection or self.collection.count() == 0:
            return "Oprostite, zdi se, da moja baza znanja ni na voljo ali pa je prazna."

        # ... (preostanek funkcije odgovori ostane enak)
        print(f"1. Iščem informacije z vprašanjem: '{uporabnikovo_vprasanje}'")
        rezultati_iskanja = self.collection.query(query_texts=[uporabnikovo_vprasanje], n_results=7, include=["documents"])
        kontekst_baza = "\n\n---\n\n".join(rezultati_iskanja['documents'][0]) if rezultati_iskanja and rezultati_iskanja['documents'] else ""
        if not kontekst_baza:
            return "Žal o tem nimam nobenih informacij."

        print("2. Pripravljam odgovor...")
        trenutno_leto = datetime.now().year
        prompt_za_llm = f"""
        Ti si 'Virtualni župan občine Rače-Fram'. Bodi natančen in časovno ozaveščen.
        ZELO POMEMBNO PRAVILO: Danes je leto {trenutno_leto}. Vedno preveri datume v priloženih informacijah.
        Če najdeš podatek iz preteklega leta (npr. {trenutno_leto - 1}), to JASNO OMENI v odgovoru.
        Nikoli ne predstavljaj starih podatkov kot aktualne.

        --- INFORMACIJE ---
        {kontekst_baza}
        ---
        VPRAŠANJE: "{uporabnikovo_vprasanje}"
        ODGOVOR:
        """
        response = self.openai_client.chat.completions.create(model=GENERATOR_MODEL_NAME, messages=[{"role": "user", "content": prompt_za_llm}], temperature=0.1)
        return response.choices[0].message.content