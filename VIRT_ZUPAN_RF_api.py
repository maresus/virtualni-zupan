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
# Pot do TRAJNEGA DISKA na Renderju
CHROMA_DB_PATH = "/data/chroma_db_prod"

COLLECTION_NAME = "obcina_race_fram_prod" 
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
GENERATOR_MODEL_NAME = "gpt-4o-mini"

# NAP API KONFIGURACIJA
NAP_TOKEN_URL = "https://b2b.nap.si/uc/user/token"
NAP_DATA_URL = "https://b2b.nap.si/data/b2b.roadworks_si.json" 
LOKACIJE_ZA_FILTER = ["Rače", "Fram", "Slivnica"]
NAP_USERNAME = os.getenv("NAP_USERNAME")
NAP_PASSWORD = os.getenv("NAP_PASSWORD")

class VirtualniZupan:
    def __init__(self):
        print("Pripravljam virtualnega župana (verzija 5.0 - z ogrevanjem)...")
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.collection = None
        self.nap_access_token = None
        try:
            openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"), model_name=EMBEDDING_MODEL_NAME)
            chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            self.collection = chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=openai_ef)
            print(f"USPEH: Povezan z bazo znanja. V bazi je {self.collection.count()} dokumentov.")

            print("-> Začenjam z ogrevanjem modela...")
            self.collection.query(query_texts=["test"], n_results=1)
            print("-> Ogrevanje modela je končano. Sistem je pripravljen.")

        except Exception as e:
            print(f"KRITIČNA NAPAKA: Baze znanja ni mogoče naložiti. {e}")
            traceback.print_exc()
        
        self.zgodovina_pogovora = []
        if self.collection:
            print("\nVirtualni župan je pripravljen. Pozdravljeni! Kako vam lahko pomagam?")
            print('Za konec pogovora vpišite "adijo" ali "konec".')

    def pridobi_nap_zeton(self):
        # ... (koda ostane enaka)
        pass

    def preveri_zapore_cest(self):
        # ... (koda ostane enaka)
        pass

    def odgovori(self, uporabnikovo_vprasanje: str):
        # ... (koda ostane enaka)
        pass

if __name__ == "__main__":
    zupan = VirtualniZupan()
    if zupan.collection:
        while True:
            vprasanje = input("\nVi: ")
            if vprasanje.lower() in ["konec", "adijo", "exit"]:
                print("Hvala za pogovor. Nasvidenje!")
                break
            if not vprasanje.strip():
                print("Župan: Prosim, vnesite vprašanje.")
                continue
            
            odgovor = zupan.odgovori(vprasanje)
            print(f"Župan: {odgovor}")