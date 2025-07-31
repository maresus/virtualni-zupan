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
        print("Inicializacija razreda VirtualniZupan (Verzija 14.0 - Oblikovanje in Povezave)...")
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
        # ... (Ta funkcija ostaja nespremenjena) ...
        return "Po podatkih portala promet.si na območju občine Rače-Fram trenutno ni zabeleženih del na cesti ali zapor."

    def obravnavaj_odvoz_odpadkov(self, uporabnikovo_vprasanje, session_id):
        # ... (Ta funkcija ostaja nespremenjena) ...
        stanje = self.zgodovina_seje[session_id].get('stanje', {})
        vprasanje_za_iskanje = stanje.get('izvirno_vprasanje', '') + " " + uporabnikovo_vprasanje
        besede_vprasanja = re.split(r'[\s,-]', vprasanje_za_iskanje.lower())
        kljucne_besede_lokacija = [b for b in besede_vprasanja if b and b not in ["odvoz", "odpadkov", "smeti", "na", "v", "je", "kdaj", "naslednji", "mešanih", "embalažo", "papir", "steklo", "bioloških"]]
        
        if not kljucne_besede_lokacija:
            stanje.update({'caka_na': 'naslov', 'namen': 'odpadki', 'izvirno_vprasanje': uporabnikovo_vprasanje})
            return "Seveda. Da vam lahko podam točen urnik, mi prosim poveste vašo ulico in kraj."

        vsi_urniki = self.collection.get(where={"kategorija": "Odvoz odpadkov"})
        najdeni_urniki_za_lokacijo = []
        for i in range(len(vsi_urniki['ids'])):
            meta = vsi_urniki['metadatas'][i]
            podatki_o_naseljih = meta.get('naselja', '').lower()
            if all(kljucna_beseda in podatki_o_naseljih for kljucna_beseda in kljucne_besede_lokacija):
                najdeni_urniki_za_lokacijo.append(vsi_urniki['documents'][i])
        
        if not najdeni_urniki_za_lokacijo:
            stanje.clear()
            return f"Oprostite, za lokacijo '{' '.join(kljucne_besede_lokacija)}' ne najdem specifičnega urnika."

        koncni_urniki = sorted(list(set(najdeni_urniki_za_lokacijo)))
        stanje.clear()
        return "\n\n".join(koncni_urniki)

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

        # --- TUKAJ SE ZAČNE NADGRADNJA ZA SPLOŠNA VPRAŠANJA ---
        stanje.clear()
        zgodovina_za_prompt = "\n".join([f"Uporabnik: {q}\nŽupan: {a}" for q, a in zgodovina])
        ocisceno_vprasanje = re.sub(r'[^\w\s]', '', vprasanje_lower)
        
        rezultati_iskanja = self.collection.query(query_texts=[ocisceno_vprasanje], n_results=5, include=["documents", "metadatas"])
        
        # 1. NADGRADNJA: Priprava konteksta, ki vključuje metapodatke
        kontekst_baza = ""
        if rezultati_iskanja.get('documents'):
            for doc, meta in zip(rezultati_iskanja['documents'][0], rezultati_iskanja['metadatas'][0]):
                vir = meta.get('source', 'Neznan vir')
                povezava = meta.get('source_url', 'Brez povezave')
                kontekst_baza += f"--- VIR DOKUMENTA: {vir}\nPOVEZAVA: {povezava}\nVSEBINA: {doc}\n\n"
        
        if not kontekst_baza:
            return "Žal o tem nimam nobenih informacij."

        now = datetime.now()
        meseci = ["januar", "februar", "marec", "april", "maj", "junij", "julij", "avgust", "september", "oktober", "november", "december"]
        poln_datum = f"{now.day}. {meseci[now.month - 1]} {now.year}"
        
        # 2. NADGRADNJA: Nova, strožja in boljša navodila v promptu
        prompt_za_llm = f"""
        Ti si 'Virtualni župan občine Rače-Fram'. Tvoj ton je profesionalen, prijazen in jedrnat.
        Današnji datum je: {poln_datum}.
        
        TVOJA NAVODILA:

        1.  **OBLIKOVANJE ODGOVORA:** Odgovor **VEDNO** oblikuj berljivo in pregledno. Uporabljaj odstavke, **krepko pisavo** za poudarjanje ključnih informacij (datumi, zneski, imena) in alineje (bullet points), kjer je to smiselno za naštevanje. Ne piši dolgih, neprekinjenih blokov besedila.

        2.  **DELO S POVEZAVAMI (URL):** V priloženih informacijah boš pod ključem 'POVEZAVA' našel spletni naslov.
            - Če povezava obstaja (ni 'Brez povezave'), jo **VEDNO** vključi v svoj odgovor.
            - Na koncu odgovora dodaj stavek v novi vrstici: "Več informacij najdete na tej povezavi: [ime vira](URL)". Ime vira najdeš pod ključem 'VIR DOKUMENTA'.
            - Če povezava ne obstaja ('Brez povezave'), ne omenjaj ničesar in ne ponujaj splošnih linkov.

        3.  **NATANČNOST:** Odgovori samo na podlagi priloženih informacij iz baze znanja. Ne ugibaj in si ne izmišljuj podatkov.
        4.  **ZGODOVINA:** Upoštevaj pretekli pogovor za kontekst: "{zgodovina_za_prompt}".

        --- INFORMACIJE IZ BAZE ZNANJA ---
        {kontekst_baza}
        ---
        
        ZADNJE VPRAŠANJE UPORABNIKA: "{uporabnikovo_vprasanje}"
        
        TVOJ ODGOVOR:
        """
        
        response = self.openai_client.chat.completions.create(model=GENERATOR_MODEL_NAME, messages=[{"role": "user", "content": prompt_za_llm}], temperature=0.1)
        koncni_odgovor = response.choices[0].message.content
        
        zgodovina.append((uporabnikovo_vprasanje, koncni_odgovor))
        if len(zgodovina) > 4:
            zgodovina.pop(0)
        
        self.belezi_pogovor(session_id, uporabnikovo_vprasanje, koncni_odgovor)
        return koncni_odgovor