import os
import json
import chromadb
import requests
import traceback
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
LOKACIJE_ZA_FILTER = ["rače", "fram", "slivnica", "letališče", "avtocesta"]
NAP_USERNAME = os.getenv("NAP_USERNAME")
NAP_PASSWORD = os.getenv("NAP_PASSWORD")

class VirtualniZupan:
    def __init__(self):
        print("Inicializacija razreda VirtualniZupan (Verzija 10.0 - Kontekstualni Spomin)...")
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

    def preveri_zapore_cest(self, vprasanje):
        if not NAP_USERNAME or not NAP_PASSWORD: return "Dostop do prometnih informacij ni mogoč."
        print("-> Pridobivam podatke o zaporah cest...")
        
        # Logika za pridobivanje tokena
        payload = {'grant_type': 'password', 'username': NAP_USERNAME, 'password': NAP_PASSWORD}
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        try:
            response = requests.post(NAP_TOKEN_URL, data=payload, headers=headers, timeout=15)
            response.raise_for_status()
            self.nap_access_token = response.json().get('access_token')
        except requests.exceptions.RequestException:
            return "Napaka pri pridobivanju žetona za dostop do prometnih informacij."

        headers_data = {'Authorization': f'Bearer {self.nap_access_token}'}
        data_response = requests.get(NAP_DATA_URL, headers=headers_data, timeout=15)
        vsi_dogodki = data_response.json().get('features', [])
        
        # Poskusimo najti specifično lokacijo v vprašanju
        besede = vprasanje.lower().split()
        filter_locations = [b for b in besede if b in LOKACIJE_ZA_FILTER]
        if not filter_locations: filter_locations = LOKACIJE_ZA_FILTER

        relevantne_zapore = [
            d['properties'] for d in vsi_dogodki
            if any(lok in d.get('properties', {}).get('opis', '').lower() for lok in filter_locations)
        ]
        
        if not relevantne_zapore:
            return f"Na območju '{' '.join(filter_locations)}' po podatkih NAP trenutno ni zabeleženih del na cesti."
        
        porocilo = "Našel sem naslednje **trenutne** informacije o delih na cesti (vir: NAP API):\n\n"
        for z in relevantne_zapore:
            porocilo += f"- **Cesta:** {z.get('cesta', 'Ni podatka')}\n  **Opis:** {z.get('opis', 'Ni podatka')}\n\n"
        return porocilo

    def obravnavaj_odvoz_odpadkov(self, uporabnikovo_vprasanje, session_id):
        print("-> Zaznan namen: Odvoz odpadkov")
        stanje = self.zgodovina_seje[session_id]['stanje']
        naslov = stanje.get('naslov_v_obravnavi')

        if not naslov:
            # Izboljšana logika za iskanje naslova
            m = re.search(r'(na|v)\s+((?:[A-ZČŠŽ][a-zčšž]+\s*)+)', uporabnikovo_vprasanje)
            if m:
                naslov = m.group(2).strip()

        if not naslov or len(naslov) < 4:
            stanje.update({'caka_na': 'naslov', 'namen': 'odpadki'})
            return "Seveda. Da vam lahko podam točen urnik, mi prosim poveste vašo ulico in kraj."

        vsi_urniki = self.collection.get(where={"kategorija": "Odvoz odpadkov"})
        pravi_urniki_info = [
            {'doc': vsi_urniki['documents'][i], 'meta': vsi_urniki['metadatas'][i]}
            for i in range(len(vsi_urniki['ids']))
            if naslov.lower() in vsi_urniki['metadatas'][i].get('naselja', '').lower()
        ]

        if not pravi_urniki_info:
            stanje.clear()
            return f"Oprostite, za naslov '{naslov}' ne najdem specifičnega urnika. Poskusite znova s polnim imenom ulice."

        stanje.clear() # Našli smo urnik, počistimo stanje
        return "\n\n".join([u['doc'] for u in pravi_urniki_info])

    def odgovori(self, uporabnikovo_vprasanje: str, session_id: str):
        self.nalozi_bazo()
        if not self.collection: return "Oprostite, moja baza znanja trenutno ni na voljo."

        if session_id not in self.zgodovina_seje:
            self.zgodovina_seje[session_id] = {'zgodovina': [], 'stanje': {}}
        
        stanje = self.zgodovina_seje[session_id]['stanje']
        zgodovina = self.zgodovina_seje[session_id]['zgodovina']

        # Preverimo, ali odgovarja na prejšnje vprašanje
        if stanje.get('caka_na') == 'naslov' and stanje.get('namen') == 'odpadki':
            stanje['naslov_v_obravnavi'] = uporabnikovo_vprasanje
            odgovor = self.obravnavaj_odvoz_odpadkov(uporabnikovo_vprasanje, session_id)
            zgodovina.append((uporabnikovo_vprasanje, odgovor))
            return odgovor

        # Prepoznavanje namena
        namen = stanje.get('namen')
        vprasanje_lower = uporabnikovo_vprasanje.lower()
        if not namen:
            if any(k in vprasanje_lower for k in ["smeti", "odpadki", "odvoz"]): namen = "odpadki"
            elif any(k in vprasanje_lower for k in ["ceste", "promet", "dela", "zapora"]): namen = "promet"
        
        if namen == "odpadki":
            stanje['namen'] = "odpadki"
            odgovor = self.obravnavaj_odvoz_odpadkov(uporabnikovo_vprasanje, session_id)
            zgodovina.append((uporabnikovo_vprasanje, odgovor))
            return odgovor
        
        if namen == "promet":
            stanje['namen'] = "promet"
            odgovor = self.preveri_zapore_cest(uporabnikovo_vprasanje)
            zgodovina.append((uporabnikovo_vprasanje, odgovor))
            return odgovor

        # Če ni posebnega namena, izvedemo splošno iskanje (RAG)
        stanje.clear() # Počistimo namen, če preidemo na splošno temo
        zgodovina_za_prompt = "\n".join([f"Uporabnik: {q}\nŽupan: {a}" for q, a in zgodovina])
        ocisceno_vprasanje = re.sub(r'[^\w\s]', '', vprasanje_lower)
        
        rezultati_iskanja = self.collection.query(query_texts=[ocisceno_vprasanje], n_results=7, include=["documents", "metadatas"])
        kontekst_baza = "\n\n---\n\n".join([f"VIR: {m.get('source', 'Neznan')}\n{d}" for d, m in zip(rezultati_iskanja['documents'][0], rezultati_iskanja['metadatas'][0])])
        
        if not kontekst_baza:
            return "Žal o tem nimam nobenih informacij."

        now = datetime.now()
        meseci = ["januar", "februar", "marec", "april", "maj", "junij", "julij", "avgust", "september", "oktober", "november", "december"]
        poln_datum = f"{now.day}. {meseci[now.month - 1]} {now.year}"
        
        prompt_za_llm = f"""
        Ti si 'Virtualni župan občine Rače-Fram'. Bodi kratek, jedrnat in izjemno natančen.
        Današnji datum je: {poln_datum}.
        
        PRAVILA:
        1.  **SLEDENJE POGOVORU:** Upoštevaj pretekli pogovor za kontekst: "{zgodovina_za_prompt}".
        2.  **NATANČNOST:** Odgovori samo na podlagi priloženih informacij iz baze znanja. Ne ugibaj.
        3.  **POVEZAVE:** Povezave (URL) vključi v odgovor samo, če so eksplicitno navedene v priloženih informacijah in direktno odgovarjajo na vprašanje. NIKOLI ne ponujaj splošne spletne strani občine, če ne najdeš odgovora.
        4.  **ŠTEVILKE:** Pri zneskih navedi najprej skupno vsoto (če je na voljo), nato postavke. Ne seštevaj sam.

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