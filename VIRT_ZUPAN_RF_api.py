import os
import json
import chromadb
import requests
import traceback
import re 
from datetime import datetime, timedelta
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
NAP_TOKEN_URL = "https://b2b.nap.si/uc/user/token"
NAP_DATA_URL = "https://b2b.nap.si/data/b2b.roadworks.geojson.sl_SI"
LOKACIJE_ZA_FILTER = ["Rače", "Fram", "Slivnica", "letališče", "avtocesta"]
NAP_USERNAME = os.getenv("NAP_USERNAME")
NAP_PASSWORD = os.getenv("NAP_PASSWORD")

class VirtualniZupan:
    def __init__(self):
        print("Inicializacija razreda VirtualniZupan (Verzija 7.0 - Pogovorni Pomočnik)...")
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.collection = None
        self.nap_access_token = None
        # Nadgradnja zgodovine za shranjevanje "stanja" pogovora
        self.zgodovina_seje = {}

    # Funkcije nalozi_bazo, belezi_pogovor in preveri_zapore_cest ostanejo nespremenjene
    def nalozi_bazo(self):
        # ... (koda ostane enaka) ...

    def belezi_pogovor(self, session_id, vprasanje, odgovor, state=None):
        # ... (koda ostane enaka, le dodamo opcijski state) ...

    def preveri_zapore_cest(self):
        # ... (koda ostane enaka) ...

    # --- NADGRADNJA: Specializirani pomočniki ---
    def obravnavaj_odvoz_odpadkov(self, uporabnikovo_vprasanje, session_id):
        stanje_pogovora = self.zgodovina_seje[session_id].get('stanje', {})
        vprasanje_lower = uporabnikovo_vprasanje.lower()

        # Poskusimo najti naslov v vprašanju
        naslov = stanje_pogovora.get('naslov')
        if not naslov:
            # Preprosta logika za iskanje ulice - lahko se izboljša
            besede = uporabnikovo_vprasanje.split()
            for i, beseda in enumerate(besede):
                if beseda.lower() == 'ulica' and i + 1 < len(besede):
                    naslov = besede[i+1]
                    break
            if not naslov and len(besede) > 1: # Poskusimo z zadnjima dvema besedama
                naslov = f"{besede[-2]} {besede[-1]}"
            else:
                naslov = besede[-1]

        # Če še vedno nimamo naslova, vprašamo uporabnika
        if not naslov or len(naslov) < 4:
            self.zgodovina_seje[session_id]['stanje'] = {'namen': 'odpadki', 'caka_na': 'naslov'}
            return "Seveda. Da vam lahko podam točen urnik, mi prosim poveste vašo ulico in kraj."
        
        # Imamo naslov, poiščemo pravi urnik
        rezultati = self.collection.query(
            query_texts=[uporabnikovo_vprasanje],
            where={"kategorija": "Odvoz odpadkov"},
            n_results=10 # Dobimo več rezultatov, da najdemo pravi naslov
        )

        pravi_urniki = []
        if rezultati and rezultati['documents'][0]:
            for i, meta in enumerate(rezultati['metadatas'][0]):
                if naslov.lower() in meta.get('naselja', '').lower():
                    pravi_urniki.append(rezultati['documents'][0][i])
        
        if not pravi_urniki:
            return f"Oprostite, za naslov '{naslov}' ne najdem specifičnega urnika. Poskusite znova z imenom ulice."

        # Preverimo, ali uporabnik sprašuje za "naslednji" odvoz
        if "naslednji" in vprasanje_lower:
            danes = datetime.now()
            najblizji_datum = None
            najblizji_odpadek = None

            for urnik in pravi_urniki:
                try:
                    tip_odpadka = re.search(r"za '(.*?)'", urnik).group(1)
                    datumi_str = urnik.split('terminih:')[1].strip().replace('.', '').split(',')
                    
                    for datum_str in datumi_str:
                        datum_obj = datetime.strptime(f"{datum_str.strip()}.2025", "%d.%m.%Y")
                        if datum_obj > danes:
                            if najblizji_datum is None or datum_obj < najblizji_datum:
                                najblizji_datum = datum_obj
                                najblizji_odpadek = tip_odpadka
                            break
                except:
                    continue
            
            if najblizji_datum:
                self.zgodovina_seje[session_id]['stanje'] = {} # Počistimo stanje
                return f"Naslednji odvoz na vašem območju bo **{najblizji_datum.strftime('%d.%m.%Y')}**, ko se odvažajo **{najblizji_odpadek}**."
            else:
                return "V letošnjem letu ni več predvidenih odvozov."
        
        self.zgodovina_seje[session_id]['stanje'] = {} # Počistimo stanje
        return "\n".join(pravi_urniki)

    # --- GLAVNA FUNKCIJA Z NOVO LOGIKO ---
    def odgovori(self, uporabnikovo_vprasanje: str, session_id: str):
        self.nalozi_bazo()
        if not self.collection: return "Oprostite, moja baza znanja trenutno ni na voljo."

        if session_id not in self.zgodovina_seje:
            self.zgodovina_seje[session_id] = {'zgodovina': [], 'stanje': {}}
        
        # Preverimo, ali čakamo na odgovor od uporabnika
        stanje_pogovora = self.zgodovina_seje[session_id].get('stanje', {})
        if stanje_pogovora.get('caka_na') == 'naslov':
            odgovor = self.obravnavaj_odvoz_odpadkov(uporabnikovo_vprasanje, session_id)
            self.belezi_pogovor(session_id, uporabnikovo_vprasanje, odgovor)
            return odgovor

        # Prepoznavanje namena
        kljucne_besede_odpadki = ["smeti", "odpadki", "odvoz", "komunala"]
        if any(beseda in uporabnikovo_vprasanje.lower() for beseda in kljucne_besede_odpadki):
            odgovor = self.obravnavaj_odvoz_odpadkov(uporabnikovo_vprasanje, session_id)
            self.belezi_pogovor(session_id, uporabnikovo_vprasanje, odgovor)
            return odgovor

        # Če namen ni specifičen, nadaljujemo s splošnim iskanjem (vaša "vrhunska" skripta)
        zgodovina_za_prompt = "\n".join([f"Uporabnik: {q}\nŽupan: {a}" for q, a in self.zgodovina_seje[session_id]['zgodovina']])
        spletni_kontekst = ""
        kljucne_besede_promet = ["zapora", "ceste", "promet", "stanje na cestah", "dela na cesti"]
        if any(beseda in uporabnikovo_vprasanje.lower() for beseda in kljucne_besede_promet):
            spletni_kontekst = self.preveri_zapore_cest()

        ocisceno_vprasanje = re.sub(r'[^\w\s]', '', uporabnikovo_vprasanje).lower()
        print(f"1. Iščem informacije za normalizirano vprašanje: '{ocisceno_vprasanje}'")
        rezultati_iskanja = self.collection.query(query_texts=[ocisceno_vprasanje], n_results=7, include=["documents", "metadatas"])
        kontekst_baza = "\n\n---\n\n".join([f"VIR: {m.get('source', 'Neznan')}\nPOVEZAVA: {m.get('source_url', 'Brez')}\nVSEBINA: {d}" for d, m in zip(rezultati_iskanja['documents'][0], rezultati_iskanja['metadatas'][0])]) if rezultati_iskanja and rezultati_iskanja['documents'] else ""

        if not kontekst_baza and not spletni_kontekst:
            odgovor = "Žal o tem nimam nobenih informacij."
            self.belezi_pogovor(session_id, uporabnikovo_vprasanje, odgovor)
            return odgovor

        print("2. Pripravljam odgovor z upoštevanjem vseh virov...")
        now = datetime.now()
        meseci = ["januar", "februar", "marec", "april", "maj", "junij", "julij", "avgust", "september", "oktober", "november", "december"]
        poln_datum = f"{now.day}. {meseci[now.month - 1]} {now.year}"
        
        prompt_za_llm = f"""
        # ... (celoten prompt, ki vam je všeč, ostane nespremenjen) ...
        """
        
        response = self.openai_client.chat.completions.create(model=GENERATOR_MODEL_NAME, messages=[{"role": "user", "content": prompt_za_llm}], temperature=0.0)
        koncni_odgovor = response.choices[0].message.content
        
        self.zgodovina_seje[session_id]['zgodovina'].append((uporabnikovo_vprasanje, koncni_odgovor))
        if len(self.zgodovina_seje[session_id]['zgodovina']) > 4:
            self.zgodovina_seje[session_id]['zgodovina'].pop(0)
        
        self.belezi_pogovor(session_id, uporabnikovo_vprasanje, koncni_odgovor)
        return koncni_odgovor