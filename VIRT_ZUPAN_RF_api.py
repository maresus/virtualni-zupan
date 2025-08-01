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
        print("Inicializacija razreda VirtualniZupan (Verzija 20.0 - Generalni Remont)...")
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.collection = None
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
        print("-> Kličem specialista za promet (NAP API)...")
        try:
            payload = {'grant_type': 'password', 'username': NAP_USERNAME, 'password': NAP_PASSWORD}
            headers = {'Content-Type': 'application/x-www-form-urlencoded'}
            response = requests.post(NAP_TOKEN_URL, data=payload, headers=headers, timeout=10)
            response.raise_for_status()
            token = response.json().get('access_token')
            if not token: return "Dostop do prometnih informacij ni uspel."
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
            return f"Žal mi neposreden vpogled v stanje na cestah trenutno ne deluje. Poskusite kasneje."

    def obravnavaj_odvoz_odpadkov(self, uporabnikovo_vprasanje, session_id):
        print("-> Kličem specialista za odpadke...")
        stanje = self.zgodovina_seje[session_id].get('stanje', {})
        vprasanje_za_iskanje = (stanje.get('izvirno_vprasanje', '') + " " + uporabnikovo_vprasanje).strip()
        
        besede_vprasanja = re.split(r'[\s,-]', vprasanje_za_iskanje.lower())
        kljucne_besede_lokacija = [b for b in besede_vprasanja if len(b) > 2 and b not in ["odvoz", "odpadkov", "smeti", "na", "v", "je", "kdaj", "naslednji", "mešanih", "embalažo", "papir", "steklo", "bioloških", "zanima", "me", "za"]]
        
        if not kljucne_besede_lokacija and 'caka_na' not in stanje:
            stanje.update({'caka_na': 'naslov', 'namen': 'odpadki', 'izvirno_vprasanje': uporabnikovo_vprasanje})
            return "Seveda. Da vam lahko podam točen urnik, mi prosim poveste vašo ulico in kraj."

        vsi_urniki = self.collection.get(where={"kategorija": "Odvoz odpadkov"})
        najdeni_urniki = []
        for i in range(len(vsi_urniki['ids'])):
            meta = vsi_urniki['metadatas'][i]
            podatki_o_naseljih = meta.get('naselja', '').lower()
            if all(beseda in podatki_o_naseljih for beseda in kljucne_besede_lokacija):
                najdeni_urniki.append(vsi_urniki['documents'][i])
        
        if not najdeni_urniki:
            stanje.clear()
            return f"Oprostite, za lokacijo '{' '.join(kljucne_besede_lokacija)}' ne najdem specifičnega urnika. Preverite ime ulice."

        stanje.clear()
        return "\n\n".join(sorted(list(set(najdeni_urniki))))

    def preoblikuj_vprasanje_s_kontekstom(self, zgodovina_pogovora, zadnje_vprasanje):
        if not zgodovina_pogovora: return zadnje_vprasanje
        
        print("-> Kličem specialista za spomin (preoblikovanje vprašanja)...")
        zgodovina_str = "\n".join([f"Uporabnik: {q}\nAsistent: {a}" for q, a in zgodovina_pogovora])
        prompt = f"""Tvoja naloga je, da glede na zgodovino pogovora preoblikuješ novo vprašanje v samostojno vprašanje, ki ga je mogoče razumeti brez zgodovine.

Primer:
Zgodovina:
Uporabnik: Ali imate poletni kamp?
Asistent: Da, imamo poletni kamp.
Novo vprašanje: "kakšna je cena?"
Samostojno vprašanje: "kakšna je cena za poletni kamp?"

Zdaj uporabi to logiko na spodnjih podatkih.

Zgodovina:
{zgodovina_str}

Novo vprašanje: "{zadnje_vprasanje}"

Samostojno vprašanje:
"""
        try:
            response = self.openai_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}], temperature=0.0, max_tokens=100)
            preoblikovano_vprasanje = response.choices[0].message.content.strip().replace('"', '')
            print(f"Originalno vprašanje: '{zadnje_vprasanje}' -> Preoblikovano: '{preoblikovano_vprasanje}'")
            return preoblikovano_vprasanje
        except Exception as e:
            print(f"Napaka pri preoblikovanju vprašanja: {e}")
            return zadnje_vprasanje

    def odgovori(self, uporabnikovo_vprasanje: str, session_id: str):
        self.nalozi_bazo()
        if not self.collection: return "Oprostite, moja baza znanja trenutno ni na voljo."

        if session_id not in self.zgodovina_seje:
            self.zgodovina_seje[session_id] = {'zgodovina': [], 'stanje': {}}
        
        stanje = self.zgodovina_seje[session_id]['stanje']
        zgodovina = self.zgodovina_seje[session_id]['zgodovina']
        
        pametno_vprasanje = self.preoblikuj_vprasanje_s_kontekstom(zgodovina, uporabnikovo_vprasanje)

        prompt_odlocanja = f"""Ti si pametni usmerjevalnik. Ugotovi namen uporabnikovega vprašanja. Možnosti so: `ODGOVORI_SPLOŠNO`, `POKLIČI_PROMET_API`, `ZAČNI_POGOVOR_ODPADKI`.
Uporabnikovo vprašanje: "{pametno_vprasanje}"
Odgovori samo z eno izmed treh možnosti."""
        
        response_odlocitev = self.openai_client.chat.completions.create(model=GENERATOR_MODEL_NAME, messages=[{"role": "user", "content": prompt_odlocanja}], temperature=0.0, max_tokens=20)
        odlocitev = response_odlocitev.choices[0].message.content.strip()
        print(f"Odločitev pametnega direktorja: {odlocitev}")

        if "POKLIČI_PROMET_API" in odlocitev:
            odgovor = self.preveri_zapore_cest()
        elif "ZAČNI_POGOVOR_ODPADKI" in odlocitev or stanje.get('caka_na') == 'naslov':
            odgovor = self.obravnavaj_odvoz_odpadkov(uporabnikovo_vprasanje, session_id)
        else:
            ocisceno_vprasanje = re.sub(r'[^\w\s]', '', pametno_vprasanje.lower())
            rezultati_iskanja = self.collection.query(query_texts=[ocisceno_vprasanje], n_results=5, include=["documents", "metadatas"])
            kontekst_baza = ""
            if rezultati_iskanja.get('documents'):
                for doc, meta in zip(rezultati_iskanja['documents'][0], rezultati_iskanja['metadatas'][0]):
                    kontekst_baza += f"--- VIR: {meta.get('source', 'Neznan')}\nPOVEZAVA: {meta.get('source_url', 'Brez')}\nVSEBINA: {doc}\n\n"
            
            if not kontekst_baza: return "Žal o tem nimam nobenih informacij."

            now = datetime.now()
            prompt_za_llm = f"""Ti si 'Virtualni župan občine Rače-Fram'.
DIREKTIVA #1 (VAROVALKA ZA DATUME): Tvoja najpomembnejša naloga je preveriti datume. Današnji datum je {now.strftime('%d.%m.%Y')}. Če je katerikoli podatek v priloženem kontekstu iz leta, ki je manjše od {now.year}, ga moraš brezpogojno IGNORIRATI. Povej, da nimaš svežih informacij o tej temi. To pravilo je pomembnejše od vsega drugega.
PRAVILA:
1. Oblikuj odgovor berljivo.
2. Če najdeš URL, ga vključi.
3. Odgovori samo iz priloženega konteksta.
--- KONTEKST ---
{kontekst_baza}---
VPRAŠANJE: "{uporabnikovo_vprasanje}"
ODGOVOR:"""
            response_odgovor = self.openai_client.chat.completions.create(model=GENERATOR_MODEL_NAME, messages=[{"role": "user", "content": prompt_za_llm}], temperature=0.0)
            odgovor = response_odgovor.choices[0].message.content

        zgodovina.append((uporabnikovo_vprasanje, odgovor))
        if len(zgodovina) > 4: zgodovina.pop(0)
        
        self.belezi_pogovor(session_id, uporabnikovo_vprasanje, odgovor)
        return odgovor