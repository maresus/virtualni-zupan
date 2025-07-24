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
LOG_FILE_PATH = "/data/zupan_pogovori.jsonl"

COLLECTION_NAME = "obcina_race_fram_prod" 
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
GENERATOR_MODEL_NAME = "gpt-4o-mini"

NAP_TOKEN_URL = "https://b2b.nap.si/uc/user/token"
NAP_DATA_URL = "https://b2b.nap.si/data/b2b.roadworks.geojson.sl_SI"
LOKACIJE_ZA_FILTER = ["Rače", "Fram", "Slivnica", "letališče", "avtocesta"]
NAP_USERNAME = os.getenv("NAP_USERNAME")
NAP_PASSWORD = os.getenv("NAP_PASSWORD")

class VirtualniZupan:
    def __init__(self):
        print("Inicializacija razreda VirtualniZupan (z lenim nalaganjem in spominom)...")
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.collection = None
        self.nap_access_token = None
        self.zgodovina_seje = {}

    def nalozi_bazo(self):
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
        try:
            zapis = {"timestamp": datetime.now().isoformat(), "session_id": session_id, "vprasanje": vprasanje, "odgovor": odgovor}
            with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
                f.write(json.dumps(zapis, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Napaka pri beleženju pogovora: {e}")

    def pridobi_nap_zeton(self):
        if not NAP_USERNAME or not NAP_PASSWORD: return "Dostop do prometnih informacij ni mogoč (manjkajo prijavni podatki)."
        print("-> Pridobivam nov žeton za dostop do NAP...")
        payload = {'grant_type': 'password', 'username': NAP_USERNAME, 'password': NAP_PASSWORD}
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        try:
            response = requests.post(NAP_TOKEN_URL, data=payload, headers=headers, timeout=15)
            response.raise_for_status()
            self.nap_access_token = response.json().get('access_token')
            return True if self.nap_access_token else "Dostop do prometnih informacij ni uspel (ni bilo mogoče pridobiti žetona)."
        except requests.exceptions.RequestException as e:
            return f"Napaka pri pridobivanju žetona: {e}"

    def preveri_zapore_cest(self):
        if not self.nap_access_token:
            rezultat_pridobivanja_zetona = self.pridobi_nap_zeton()
            if rezultat_pridobivanja_zetona is not True: return rezultat_pridobivanja_zetona

        print("-> Pridobivam podatke o zaporah cest...")
        headers = {'Authorization': f'Bearer {self.nap_access_token}'}
        try:
            data_response = requests.get(NAP_DATA_URL, headers=headers, timeout=15)
            data_response.raise_for_status()
            vsi_dogodki = data_response.json()
            relevantne_zapore = [
                dogodek['properties']
                for dogodek in vsi_dogodki.get('features', [])
                if any(lok.lower() in dogodek.get('properties', {}).get('opis', '').lower() for lok in LOKACIJE_ZA_FILTER)
            ]
            if not relevantne_zapore:
                return "Na območju občine Rače-Fram po podatkih NAP trenutno ni zabeleženih del na cesti."
            
            porocilo = "Našel sem naslednje **trenutne** informacije o delih na cesti (vir: NAP API):\n\n"
            for z in relevantne_zapore:
                porocilo += f"- **Cesta:** {z.get('cesta', 'Ni podatka')}\n"
                porocilo += f"  **Opis:** {z.get('opis', 'Ni podatka')}\n\n"
            return porocilo
        except requests.exceptions.RequestException as e:
            if e.response and (e.response.status_code == 401 or e.response.status_code == 403):
                self.nap_access_token = None
                return "Žal mi neposreden vpogled v stanje na cestah trenutno ne deluje (težava z avtorizacijo). Prosim, preverite uradni portal promet.si."
            return f"Žal mi neposreden vpogled v stanje na cestah trenutno ne deluje. Za najnovejše informacije obiščite https://www.promet.si. Tehnični razlog: {e}"

    def odgovori(self, uporabnikovo_vprasanje: str, session_id: str):
        self.nalozi_bazo()
        if not self.collection: return "Oprostite, moja baza znanja trenutno ni na voljo."

        if session_id not in self.zgodovina_seje:
            self.zgodovina_seje[session_id] = []
        zgodovina_za_prompt = "\n".join([f"Uporabnik: {q}\nŽupan: {a}" for q, a in self.zgodovina_seje[session_id]])

        # Iskanje po bazi poteka vedno, da imamo kontekst
        print(f"1. Iščem informacije za vprašanje: '{uporabnikovo_vprasanje}'")
        rezultati_iskanja = self.collection.query(query_texts=[uporabnikovo_vprasanje], n_results=7, include=["documents", "metadatas"])
        kontekst_baza = "\n\n---\n\n".join([f"VIR: {m.get('source', 'Neznan')}\nPOVEZAVA: {m.get('source_url', 'Brez')}\nVSEBINA: {d}" for d, m in zip(rezultati_iskanja['documents'][0], rezultati_iskanja['metadatas'][0])]) if rezultati_iskanja and rezultati_iskanja['documents'] else ""

        # Preverimo, ali gre za vprašanje o prometu
        spletni_kontekst = ""
        kljucne_besede_promet = ["zapora", "ceste", "promet", "stanje na cestah", "dela na cesti"]
        if any(beseda in uporabnikovo_vprasanje.lower() for beseda in kljucne_besede_promet):
            spletni_kontekst = self.preveri_zapore_cest()

        if not kontekst_baza and not spletni_kontekst:
            odgovor = "Žal o tem nimam nobenih informacij."
            self.belezi_pogovor(session_id, uporabnikovo_vprasanje, odgovor)
            return odgovor

        print("2. Pripravljam odgovor z upoštevanjem vseh virov...")
        trenutno_leto = datetime.now().year
        
        prompt_za_llm = f"""
        Ti si 'Virtualni župan občine Rače-Fram'. Bodi kratek, jedrnat in izjemno koristen.

        PRAVILO #1 (SLEDENJE POGOVORU): Upoštevaj pretekli pogovor za kontekst. Če se zadnje vprašanje ("{uporabnikovo_vprasanje}") nanaša na prejšnje teme (npr. "kaj pa njegova telefonska številka?"), moraš pravilno povezati kontekst.

        PRAVILO #2 (DATUMI): Danes je leto {trenutno_leto}. Vedno preveri datume v priloženih informacijah. Če najdeš podatek iz preteklega leta, MORAŠ JASNO povedati, iz katerega leta je informacija (npr. "Zadnja informacija, ki jo imam, je iz leta 2024..."). Nikoli ne predstavljaj starih podatkov kot aktualne.

        PRAVILO #3 (POVEZAVE): Če v informacijah najdeš spletno povezavo (URL), ki se nanaša na vprašanje, jo VEDNO vključi v odgovor v obliki [Ime povezave](URL). Povezave ponudi samo, če so neposredno relevantne za odgovor.

        PRAVILO #4 (PROMET): Če so na voljo SVEŽE INFORMACIJE O PROMETU, jih vedno predstavi najprej, nato pa lahko dodaš še informacije o načrtovanih delih iz interne baze.

        --- ZGODOVINA POGOVORA ---
        {zgodovina_za_prompt}
        ---
        --- SVEŽE INFORMACIJE O PROMETU (iz NAP API) ---
        {spletni_kontekst if spletni_kontekst else "Ni relevantnih svežih podatkov o prometu za to vprašanje."}
        ---
        --- INFORMACIJE IZ INTERNE BAZE ZNANJA ---
        {kontekst_baza}
        ---
        
        ZADNJE VPRAŠANJE UPORABNIKA: "{uporabnikovo_vprasanje}"
        
        TVOJ ODGOVOR:
        """
        
        response = self.openai_client.chat.completions.create(model=GENERATOR_MODEL_NAME, messages=[{"role": "user", "content": prompt_za_llm}], temperature=0.0)
        koncni_odgovor = response.choices[0].message.content
        
        self.zgodovina_seje[session_id].append((uporabnikovo_vprasanje, koncni_odgovor))
        if len(self.zgodovina_seje[session_id]) > 4:
            self.zgodovina_seje[session_id].pop(0)
        
        self.belezi_pogovor(session_id, uporabnikovo_vprasanje, koncni_odgovor)
        return koncni_odgovor

# Glavna zanka za lokalno testiranje ostane enaka
if __name__ == "__main__":
    zupan = VirtualniZupan()
    if zupan.collection:
        session_id_za_test = "terminal_pogovor_123"
        while True:
            vprasanje = input("\nVi: ")
            if vprasanje.lower() in ["konec", "adijo", "exit"]:
                print("Hvala za pogovor. Nasvidenje!")
                break
            if not vprasanje.strip():
                print("Župan: Prosim, vnesite vprašanje.")
                continue
            odgovor = zupan.odgovori(vprasanje, session_id=session_id_za_test)
            print(f"Župan: {odgovor}")