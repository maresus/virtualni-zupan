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

# --- DODANO: Manjkajoča NAP API KONFIGURACIJA ---
NAP_TOKEN_URL = "https://b2b.nap.si/uc/user/token"
NAP_DATA_URL = "https://b2b.nap.si/data/b2b.roadworks.geojson.sl_SI" # PRAVILEN URL
LOKACIJE_ZA_FILTER = ["Rače", "Fram", "Slivnica"]
NAP_USERNAME = os.getenv("NAP_USERNAME")
NAP_PASSWORD = os.getenv("NAP_PASSWORD")

class VirtualniZupan:
    def __init__(self):
        # Ta del ostaja enak - baza se ne naloži takoj
        print("Inicializacija razreda VirtualniZupan (brez nalaganja baze)...")
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.collection = None
        self.nap_access_token = None
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

    # --- DODANO: Manjkajoče funkcije za delo z NAP API ---
    def pridobi_nap_zeton(self):
        if not NAP_USERNAME or not NAP_PASSWORD:
            return "Dostop do prometnih informacij ni mogoč, ker niso vpisani prijavni podatki za NAP API v .env datoteki."
        print("-> Pridobivam nov žeton za dostop do NAP...")
        payload = {'grant_type': 'password', 'username': NAP_USERNAME, 'password': NAP_PASSWORD}
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        try:
            response = requests.post(NAP_TOKEN_URL, data=payload, headers=headers, timeout=15)
            response.raise_for_status()
            self.nap_access_token = response.json().get('access_token')
            if not self.nap_access_token:
                return "Dostop do prometnih informacij ni uspel (ni bilo mogoče pridobiti žetona)."
            print("-> Žeton uspešno pridobljen.")
            return True
        except requests.exceptions.RequestException as e:
            return f"Napaka pri pridobivanju žetona: {e}"

    def preveri_zapore_cest(self):
        if not self.nap_access_token:
            rezultat_pridobivanja_zetona = self.pridobi_nap_zeton()
            if rezultat_pridobivanja_zetona is not True:
                return rezultat_pridobivanja_zetona

        print("-> Pridobivam podatke o zaporah cest...")
        headers = {'Authorization': f'Bearer {self.nap_access_token}'}
        try:
            data_response = requests.get(NAP_DATA_URL, headers=headers, timeout=15)
            data_response.raise_for_status()
            vsi_dogodki = data_response.json()
            relevantne_zapore = [
                dogodek['properties']
                for dogodek in vsi_dogodki.get('features', [])
                if any(lok.lower() in dogodek.get('properties', {}).get('description', '').lower() for lok in LOKACIJE_ZA_FILTER)
            ]
            if not relevantne_zapore:
                return "Na območju občine Rače-Fram trenutno ni zabeleženih del na cesti s strani Nacionalne točke dostopa."
            
            porocilo = "Našel sem naslednje aktualne informacije o delih na cesti:\n\n"
            for z in relevantne_zapore:
                porocilo += f"- Lokacija: {z.get('locationDescription', 'Ni podatka')}\n"
                porocilo += f"  Opis: {z.get('description', 'Ni podatka')}\n\n"
            return porocilo
        except requests.exceptions.RequestException as e:
            return f"Žal mi neposreden vpogled v stanje na cestah trenutno ne deluje. Za najnovejše informacije obiščite https://www.promet.si. Tehnični razlog: {e}"
    # --- KONEC DODANIH FUNKCIJ ---

    def odgovori(self, uporabnikovo_vprasanje: str, session_id: str):
        self.nalozi_bazo()
        if not self.collection or self.collection.count() == 0:
            odgovor = "Oprostite, zdi se, da moja baza znanja ni na voljo ali pa je prazna."
            self.belezi_pogovor(session_id, uporabnikovo_vprasanje, odgovor)
            return odgovor

        # --- DODANO: Klicanje funkcije za preverjanje prometa ---
        spletni_kontekst = ""
        kljucne_besede_promet = ["zapora", "ceste", "promet", "stanje na cestah", "dela na cesti"]
        if any(beseda in uporabnikovo_vprasanje.lower() for beseda in kljucne_besede_promet):
            spletni_kontekst = self.preveri_zapore_cest()
            if spletni_kontekst:
                self.belezi_pogovor(session_id, uporabnikovo_vprasanje, spletni_kontekst)
                return spletni_kontekst
        # --- KONEC DODATKA ---

        if session_id not in self.zgodovina_seje:
            self.zgodovina_seje[session_id] = []
        
        zgodovina_za_prompt = "\n".join([f"Uporabnik: {q}\nŽupan: {a}" for q, a in self.zgodovina_seje[session_id]])

        print(f"1. Iščem informacije za vprašanje: '{uporabnikovo_vprasanje}'")
        rezultati_iskanja = self.collection.query(query_texts=[uporabnikovo_vprasanje], n_results=7, include=["documents"])
        kontekst_baza = "\n\n---\n\n".join(rezultati_iskanja['documents'][0]) if rezultati_iskanja and rezultati_iskanja['documents'] else ""

        if not kontekst_baza:
            odgovor = "Žal o tem nimam nobenih informacij."
            self.belezi_pogovor(session_id, uporabnikovo_vprasanje, odgovor)
            return odgovor

        print("2. Pripravljam odgovor z upoštevanjem konteksta...")
        
        prompt_za_llm = f"""
        Ti si 'Virtualni župan občine Rače-Fram'. Upoštevaj pretekli pogovor za kontekst.
        --- ZGODOVINA POGOVORA ---
        {zgodovina_za_prompt}
        ---
        --- INFORMACIJE IZ BAZE ZNANJA ---
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