import os
import json
import chromadb
import requests
import re
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
LOKACIJE_ZA_FILTER = ["rače", "fram", "slivnica", "letališče", "avtocesta", "brunšvik", "podova"]
NAP_USERNAME = os.getenv("NAP_USERNAME")
NAP_PASSWORD = os.getenv("NAP_PASSWORD")

class VirtualniZupan:
    def __init__(self):
        print("Inicializacija razreda VirtualniZupan (Verzija 12.0 - Stabilna)...")
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

    def preveri_zapore_cest(self):
        if not NAP_USERNAME or not NAP_PASSWORD: return "Dostop do prometnih informacij ni mogoč."
        print("-> Pridobivam podatke o zaporah cest...")
        try:
            payload = {'grant_type': 'password', 'username': NAP_USERNAME, 'password': NAP_PASSWORD}
            headers = {'Content-Type': 'application/x-www-form-urlencoded'}
            response = requests.post(NAP_TOKEN_URL, data=payload, headers=headers, timeout=10)
            response.raise_for_status()
            token = response.json().get('access_token')
            if not token: return "Dostop do prometnih informacij ni uspel (ni bilo mogoče pridobiti žetona)."

            headers_data = {'Authorization': f'Bearer {token}'}
            data_response = requests.get(NAP_DATA_URL, headers=headers_data, timeout=10)
            data_response.raise_for_status()
            vsi_dogodki = data_response.json().get('features', [])
            
            relevantne_zapore = [
                d['properties'] for d in vsi_dogodki
                if any(lok in d.get('properties', {}).get('opis', '').lower() for lok in LOKACIJE_ZA_FILTER)
            ]
            if not relevantne_zapore:
                return "Po podatkih portala promet.si na območju občine Rače-Fram trenutno ni zabeleženih del na cesti ali zapor."
            
            porocilo = "Našel sem naslednje **trenutne** informacije o delih na cesti (vir: promet.si):\n\n"
            for z in relevantne_zapore:
                porocilo += f"- **Cesta:** {z.get('cesta', 'Ni podatka')}\n  **Opis:** {z.get('opis', 'Ni podatka')}\n\n"
            return porocilo
        except requests.exceptions.RequestException as e:
            return f"Žal mi neposreden vpogled v stanje na cestah trenutno ne deluje. Poskusite kasneje ali obiščite promet.si. Tehnični razlog: {e}"

    def obravnavaj_odvoz_odpadkov(self, uporabnikovo_vprasanje, session_id):
        stanje = self.zgodovina_seje[session_id].get('stanje', {})
        naslov_iz_vprasanja = uporabnikovo_vprasanje
        
        if stanje.get('caka_na') == 'naslov':
            naslov_iz_vprasanja = stanje.get('izvirno_vprasanje', '') + " " + uporabnikovo_vprasanje
        
        vsi_urniki = self.collection.get(where={"kategorija": "Odvoz odpadkov"})
        iskani_deli = [beseda for beseda in re.split(r'[\s,-]', naslov_iz_vprasanja.lower()) if len(beseda) > 2 and beseda not in ["odvoz", "odpadkov", "smeti", "na", "v"]]
        
        if not iskani_deli:
            stanje.update({'caka_na': 'naslov', 'namen': 'odpadki', 'izvirno_vprasanje': uporabnikovo_vprasanje})
            return "Seveda. Da vam lahko podam točen urnik, mi prosim poveste vašo ulico in kraj."

        najdeni_urniki = []
        for i in range(len(vsi_urniki['ids'])):
            meta = vsi_urniki['metadatas'][i]
            podatki_o_naseljih = meta.get('naselja', '').lower()
            if all(del_besede in podatki_o_naseljih for del_besede in iskani_deli):
                najdeni_urniki.append(vsi_urniki['documents'][i])
        
        najdeni_urniki = sorted(list(set(najdeni_urniki)))

        if not najdeni_urniki:
            stanje.clear()
            return f"Oprostite, za lokacijo '{' '.join(iskani_deli)}' ne najdem specifičnega urnika. Prosim, poskusite znova z bolj natančnim naslovom."

        stanje.clear()
        return "\n\n".join(najdeni_urniki)

    def odgovori(self, uporabnikovo_vprasanje: str, session_id: str):
        self.nalozi_bazo()
        if not self.collection: return "Oprostite, moja baza znanja trenutno ni na voljo."

        if session_id not in self.zgodovina_seje:
            self.zgodovina_seje[session_id] = {'zgodovina': [], 'stanje': {}}
        
        stanje = self.zgodovina_seje[session_id]['stanje']
        zgodovina = self.zgodovina_seje[session_id]['zgodovina']
        vprasanje_lower = uporabnikovo_vprasanje.lower()

        if stanje.get('caka_na') == 'naslov':
            odgovor = self.obravnavaj_odvoz_odpadkov(uporabnikovo_vprasanje, session_id)
            zgodovina.append((uporabnikovo_vprasanje, odgovor))
            return odgovor

        if any(k in vprasanje_lower for k in ["smeti", "odpadki", "odvoz", "komunala"]):
            odgovor = self.obravnavaj_odvoz_odpadkov(uporabnikovo_vprasanje, session_id)
            zgodovina.append((uporabnikovo_vprasanje, odgovor))
            return odgovor

        if any(k in vprasanje_lower for k in ["ceste", "promet", "dela", "zapora"]):
            odgovor = self.preveri_zapore_cest()
            zgodovina.append((uporabnikovo_vprasanje, odgovor))
            return odgovor

        # Splošno iskanje (RAG) za vse ostalo
        stanje.clear()
        zgodovina_za_prompt = "\n".join([f"Uporabnik: {q}\nŽupan: {a}" for q, a in zgodovina])
        ocisceno_vprasanje = re.sub(r'[^\w\s]', '', vprasanje_lower)
        
        rezultati_iskanja = self.collection.query(query_texts=[ocisceno_vprasanje], n_results=5)
        kontekst_baza = "\n\n---\n\n".join(rezultati_iskanja['documents'][0]) if rezultati_iskanja['documents'] else ""
        
        if not kontekst_baza:
            return "Žal o tem nimam nobenih informacij."

        now = datetime.now()
        meseci = ["januar", "februar", "marec", "april", "maj", "junij", "julij", "avgust", "september", "oktober", "november", "december"]
        poln_datum = f"{now.day}. {meseci[now.month - 1]} {now.year}"
        
        prompt_za_llm = f"""
        Ti si 'Virtualni župan občine Rače-Fram'. Bodi kratek, jedrnat in izjemno natančen.
        Današnji datum je: {poln_datum}.
        
        PRAVILA:
        1.  **NATANČNOST:** Odgovori samo na podlagi priloženih informacij iz baze znanja. Ne ugibaj.
        2.  **POVEZAVE:** Povezave (URL) vključi v odgovor samo in izključno, če so eksplicitno navedene v priloženih informacijah in direktno odgovarjajo na vprašanje. NIKOLI ne ponujaj splošne spletne strani občine, če ne najdeš odgovora, raje reci, da informacije nimaš.

        --- INFORMACIJE IZ BAZE ZNANJA ---
        {kontekst_baza}
        ---
        
        ZADNJE VPRAŠANJE UPORABNIKA: "{uporabnikovo_vprasanje}"
        
        TVOJ ODGOVOR:
        """
        
        response = self.openai_client.chat.completions.create(model=GENERATOR_MODEL_NAME, messages=[{"role": "user", "content": prompt_za_llm}], temperature=0.0)
        koncni_odgovor = response.choices[0].message.content
        
        zgodovina.append((uporabnikovo_vprasanje, koncni_odgovor))
        if len(zgodovina) > 4:
            zgodovina.pop(0)
        
        self.belezi_pogovor(session_id, uporabnikovo_vprasanje, koncni_odgovor)
        return koncni_odgovor