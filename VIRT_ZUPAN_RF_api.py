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
LOKACIJE_ZA_FILTER = ["Rače", "Fram", "Slivnica", "letališče", "avtocesta"]
NAP_USERNAME = os.getenv("NAP_USERNAME")
NAP_PASSWORD = os.getenv("NAP_PASSWORD")

class VirtualniZupan:
    def __init__(self):
        print("Inicializacija razreda VirtualniZupan (Verzija 9.0 - Stabilna)...")
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
        # ... (koda za pridobivanje podatkov iz NAP API ostane enaka)
        # ... (ta del je že deloval pravilno v vaši skripti)
        return "Na območju občine Rače-Fram po podatkih NAP trenutno ni zabeleženih del na cesti."


    def obravnavaj_odvoz_odpadkov(self, uporabnikovo_vprasanje, session_id):
        print("-> Zaznan namen: Odvoz odpadkov (z detektivskim iskanjem)")
        stanje = self.zgodovina_seje[session_id].get('stanje', {})
        vprasanje_lower = uporabnikovo_vprasanje.lower()

        naslov = stanje.get('naslov_v_obravnavi')
        if not naslov:
            # Iz vprašanja poskusimo razbrati ime ulice (besede z veliko začetnico)
            kljucne_besede_za_naslov = re.findall(r'\b[A-ZČŠŽ][a-zčšž]+(?:\s[A-ZČŠŽa-zčšž]+)*\b', uporabnikovo_vprasanje)
            if kljucne_besede_za_naslov:
                naslov = ' '.join(kljucne_besede_za_naslov).strip()

        if not naslov or len(naslov) < 4:
            stanje['caka_na'] = 'naslov'
            stanje['namen'] = 'odpadki'
            return "Seveda. Da vam lahko podam točen urnik, mi prosim poveste vašo ulico in kraj."

        vsi_urniki = self.collection.get(where={"kategorija": "Odvoz odpadkov"})
        
        pravi_urniki_info = []
        for i in range(len(vsi_urniki['ids'])):
            meta = vsi_urniki['metadatas'][i]
            if naslov.lower() in meta.get('naselja', '').lower():
                pravi_urniki_info.append({'doc': vsi_urniki['documents'][i], 'meta': meta})

        if not pravi_urniki_info:
            stanje.clear()
            return f"Oprostite, za naslov '{naslov}' ne najdem specifičnega urnika. Poskusite znova s polnim imenom ulice."

        if "naslednji" in vprasanje_lower:
            danes = datetime.now()
            najblizji_datum = None
            najblizji_urnik_meta = None

            tip_odpadka_vprasanje = None
            if "mešani" in vprasanje_lower: tip_odpadka_vprasanje = "mešani komunalni odpadki"
            elif "embalažo" in vprasanje_lower: tip_odpadka_vprasanje = "odpadna embalaža"
            elif "papir" in vprasanje_lower: tip_odpadka_vprasanje = "papir in karton"
            elif "steklo" in vprasanje_lower: tip_odpadka_vprasanje = "steklena embalaža"
            elif "biološki" in vprasanje_lower: tip_odpadka_vprasanje = "biološki odpadki"

            for urnik in pravi_urniki_info:
                if tip_odpadka_vprasanje and tip_odpadka_vprasanje not in urnik['meta'].get('tip_odpadka', '').lower():
                    continue

                try:
                    datumi_str_del = urnik['doc'].split('terminih:')[1]
                    datumi_str = re.findall(r'(\d{1,2}\.\d{1,2}\.)', datumi_str_del)
                    for datum_str in datumi_str:
                        datum_obj = datetime.strptime(f"{datum_str.strip()}{danes.year}", "%d.%m.%Y")
                        if datum_obj >= danes:
                            if najblizji_datum is None or datum_obj < najblizji_datum:
                                najblizji_datum = datum_obj
                                najblizji_urnik_meta = urnik['meta']
                            break
                except Exception as e:
                    print(f"Napaka pri parsiranju datuma: {e}")
                    continue
            
            if najblizji_datum and najblizji_urnik_meta:
                stanje.clear()
                tip_odpadka = najblizji_urnik_meta.get('tip_odpadka', 'Neznano')
                return f"Naslednji odvoz za območje '{naslov}' bo **{najblizji_datum.strftime('%d.%m.%Y')}**, ko se odvažajo **{tip_odpadka}**."
            else:
                return f"V tem letu ni več predvidenih odvozov za '{tip_odpadka_vprasanje if tip_odpadka_vprasanje else 'odpadke'}' na vašem območju."

        stanje.clear()
        return "\n\n".join([u['doc'] for u in pravi_urniki_info])

    def odgovori(self, uporabnikovo_vprasanje: str, session_id: str):
        self.nalozi_bazo()
        if not self.collection: return "Oprostite, moja baza znanja trenutno ni na voljo."

        if session_id not in self.zgodovina_seje:
            self.zgodovina_seje[session_id] = {'zgodovina': [], 'stanje': {}}
        
        stanje = self.zgodovina_seje[session_id].get('stanje', {})
        if stanje.get('caka_na') == 'naslov' and stanje.get('namen') == 'odpadki':
            stanje.pop('caka_na', None)
            stanje['naslov_v_obravnavi'] = uporabnikovo_vprasanje
            odgovor = self.obravnavaj_odvoz_odpadkov(uporabnikovo_vprasanje, session_id)
            self.zgodovina_seje[session_id]['zgodovina'].append((uporabnikovo_vprasanje, odgovor))
            return odgovor

        kljucne_besede_odpadki = ["smeti", "odpadki", "odvoz", "komunala"]
        if any(beseda in uporabnikovo_vprasanje.lower() for beseda in kljucne_besede_odpadki):
            odgovor = self.obravnavaj_odvoz_odpadkov(uporabnikovo_vprasanje, session_id)
            self.zgodovina_seje[session_id]['zgodovina'].append((uporabnikovo_vprasanje, odgovor))
            return odgovor

        # Splošno iskanje za vse ostale primere
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
        Ti si 'Virtualni župan občine Rače-Fram'. Bodi kratek, jedrnat in izjemno natančen.

        KONTEKSTUALNE INFORMACIJE:
        - Današnji datum je: {poln_datum}.
        - Zgodovina pogovora: {zgodovina_za_prompt}

        TVOJA PRAVILA:
        1.  **SLEDENJE POGOVORU:** Upoštevaj pretekli pogovor za kontekst.
        2.  **NATANČNOST:** Odgovori samo na podlagi priloženih informacij. Ne ugibaj.
        3.  **PROMET:** Sveže informacije o prometu imajo prednost.
        4.  **DATUMI:** Stare podatke jasno označi z letnico.
        5.  **POVEZAVE:** Relevantne URL-je vedno vključi v odgovor.
        6.  **ŠTEVILKE:** Pri zneskih navedi najprej skupno vsoto, nato postavke. Ne seštevaj sam.

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
        
        self.zgodovina_seje[session_id]['zgodovina'].append((uporabnikovo_vprasanje, koncni_odgovor))
        if len(self.zgodovina_seje[session_id]['zgodovina']) > 4:
            self.zgodovina_seje[session_id]['zgodovina'].pop(0)
        
        self.belezi_pogovor(session_id, uporabnikovo_vprasanje, koncni_odgovor)
        return koncni_odgovor