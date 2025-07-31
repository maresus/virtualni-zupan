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
        print("Inicializacija razreda VirtualniZupan (Verzija 8.1 - Združena)...")
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
        if not NAP_USERNAME or not NAP_PASSWORD: return "Dostop do prometnih informacij ni mogoč (manjkajo prijavni podatki)."
        print("-> Pridobivam nov žeton za dostop do NAP...")
        # ... (celotna koda za preveri_zapore_cest ostane enaka)
        payload = {'grant_type': 'password', 'username': NAP_USERNAME, 'password': NAP_PASSWORD}
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        try:
            response = requests.post(NAP_TOKEN_URL, data=payload, headers=headers, timeout=15)
            response.raise_for_status()
            self.nap_access_token = response.json().get('access_token')
            if not self.nap_access_token: return "Dostop do prometnih informacij ni uspel (ni bilo mogoče pridobiti žetona)."
        except requests.exceptions.RequestException as e:
            return f"Napaka pri pridobivanju žetona: {e}"

        print("-> Pridobivam podatke o zaporah cest...")
        headers = {'Authorization': f'Bearer {self.nap_access_token}'}
        try:
            data_response = requests.get(NAP_DATA_URL, headers=headers, timeout=15)
            data_response.raise_for_status()
            vsi_dogodki = data_response.json().get('features', [])
            
            relevantne_zapore = [
                dogodek['properties'] for dogodek in vsi_dogodki
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
            return f"Žal mi neposreden vpogled v stanje na cestah trenutno ne deluje. Tehnični razlog: {e}"

    def obravnavaj_odvoz_odpadkov(self, uporabnikovo_vprasanje, session_id):
        print("-> Zaznan namen: Odvoz odpadkov")
        stanje_pogovora = self.zgodovina_seje[session_id].get('stanje', {})
        vprasanje_lower = uporabnikovo_vprasanje.lower()

        naslov = stanje_pogovora.get('naslov')
        if not naslov:
            # Poskusimo razčleniti naslov iz trenutnega vprašanja
            # Ta del je preprost in ga je mogoče v prihodnosti izboljšati z boljšim prepoznavanjem entitet
            besede = re.split(r'[\s,]', uporabnikovo_vprasanje)
            potential_naslov = ' '.join(besede[-2:]) # Vzamemo zadnji dve besedi kot možen naslov
            if len(potential_naslov) > 4:
                naslov = potential_naslov

        # Če v nobenem primeru nimamo naslova, vprašamo uporabnika
        if not naslov:
            self.zgodovina_seje[session_id]['stanje'] = {'namen': 'odpadki', 'caka_na': 'naslov'}
            return "Seveda. Da vam lahko podam točen urnik, mi prosim poveste vašo ulico in kraj."

        # Imamo naslov, poiščemo pravi urnik
        rezultati = self.collection.query(
            query_texts=[naslov],
            where={"kategorija": "Odvoz odpadkov"},
            n_results=15 # Dobimo več rezultatov, da zagotovo najdemo pravi naslov
        )
        
        pravi_urniki = []
        if rezultati and rezultati['documents'][0]:
            for i, meta in enumerate(rezultati['metadatas'][0]):
                if naslov.lower() in meta.get('naselja', '').lower():
                    # Dodamo celoten dokument, saj vsebuje urnik za en tip odpadka
                    pravi_urniki.append({'doc': rezultati['documents'][0][i], 'meta': meta})
        
        if not pravi_urniki:
            self.zgodovina_seje[session_id]['stanje'] = {} # Počistimo stanje, da ne ostanemo v zanki
            return f"Oprostite, za naslov '{naslov}' ne najdem specifičnega urnika. Poskusite znova s polnim imenom ulice."

        # Preverimo, ali uporabnik sprašuje za "naslednji" odvoz
        if "naslednji" in vprasanje_lower:
            danes = datetime.now()
            najblizji_datum = None
            najblizji_urnik = None

            for urnik_info in pravi_urniki:
                try:
                    datumi_str = urnik_info['doc'].split('terminih:')[1].strip().replace('.', '').split(',')
                    for datum_str in datumi_str:
                        # Pravilno parsanje datuma
                        dan, mesec = map(int, datum_str.strip().split('/'))
                        datum_obj = datetime(danes.year, mesec, dan)
                        
                        if datum_obj > danes:
                            if najblizji_datum is None or datum_obj < najblizji_datum:
                                najblizji_datum = datum_obj
                                najblizji_urnik = urnik_info
                            break
                except Exception as e:
                    print(f"Napaka pri parsiranju datuma: {e}")
                    continue
            
            if najblizji_datum and najblizji_urnik:
                self.zgodovina_seje[session_id]['stanje'] = {} # Počistimo stanje
                tip_odpadka = najblizji_urnik['meta'].get('tip_odpadka', 'Neznano')
                return f"Naslednji odvoz za območje '{naslov}' bo **{najblizji_datum.strftime('%d.%m.%Y')}**, ko se odvažajo **{tip_odpadka}**."
            else:
                return "V letošnjem letu ni več predvidenih odvozov za vaše območje."
        
        self.zgodovina_seje[session_id]['stanje'] = {} # Počistimo stanje
        return "\n".join([u['doc'] for u in pravi_urniki])

    def odgovori(self, uporabnikovo_vprasanje: str, session_id: str):
        self.nalozi_bazo()
        if not self.collection: return "Oprostite, moja baza znanja trenutno ni na voljo."

        if session_id not in self.zgodovina_seje:
            self.zgodovina_seje[session_id] = {'zgodovina': [], 'stanje': {}}
        
        stanje_pogovora = self.zgodovina_seje[session_id].get('stanje', {})
        if stanje_pogovora.get('caka_na') == 'naslov' and stanje_pogovora.get('namen') == 'odpadki':
            odgovor = self.obravnavaj_odvoz_odpadkov(uporabnikovo_vprasanje, session_id)
            self.belezi_pogovor(session_id, uporabnikovo_vprasanje, odgovor)
            return odgovor

        kljucne_besede_odpadki = ["smeti", "odpadki", "odvoz", "komunala", "kos za smeti"]
        if any(beseda in uporabnikovo_vprasanje.lower() for beseda in kljucne_besede_odpadki):
            odgovor = self.obravnavaj_odvoz_odpadkov(uporabnikovo_vprasanje, session_id)
            self.belezi_pogovor(session_id, uporabnikovo_vprasanje, odgovor)
            return odgovor
            
        # Če namen ni specifičen, nadaljujemo s splošnim iskanjem
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
        - Današnji datum je: {poln_datum}. Vse odgovore podaj iz perspektive tega datuma.
        - Pogovarjaš se z uporabnikom, katerega zgodovina pogovora je spodaj.

        TVOJA PRAVILA:
        1.  **SLEDENJE POGOVORU:** Pretekli pogovor je ključen za kontekst. Če se zadnje vprašanje ("{uporabnikovo_vprasanje}") nanaša na osebo ali temo iz prejšnjega odgovora (npr. "kaj pa njegova telefonska?"), MORAŠ pravilno povezati kontekst.
        2.  **NATANČNOST:** Odgovori samo na podlagi priloženih informacij. Če te vprašajo za telefonsko številko Gregorja Ovnika, poišči podatek, ki je neposredno vezan na njegovo ime. Ne ugibaj in ne povezuj podatkov različnih oseb.
        3.  **PROMET:** Če so na voljo SVEŽE INFORMACIJE O PROMETU, jih vedno predstavi najprej, nato pa lahko dodaš še informacije o načrtovanih delih iz interne baze, a jasno loči med obojim.
        4.  **PREVERJANJE DATUMOV:** Vedno primerjaj datume v priloženih informacijah z današnjim datumom. Če je podatek iz preteklosti (npr. iz leta 2024), to JASNO OMENI v odgovoru. Nikoli ne predstavljaj starih podatkov kot aktualne.
        5.  **POVEZAVE:** Če v informacijah najdeš spletno povezavo (URL), ki se nanaša na vprašanje, jo VEDNO vključi v odgovor v obliki [Ime povezave](URL). Povezave ponudi samo, če so neposredno relevantne za odgovor (npr. pri vlogah in obrazcih).

        ZELO POMEMBNO PRAVILO #1 (DELO S ŠTEVILKAMI):
        Ko te uporabnik vpraša po zneskih ali proračunskih postavkah, moraš biti izjemno natančen.
        1.  Najprej navedi SKUPNI znesek za celotno področje, če je na voljo (npr. "Šport in prostočasne aktivnosti skupaj: 1.044.200 €").
        2.  Nato naštej VSE posamezne postavke znotraj tega področja, ki jih najdeš v priloženih informacijah, skupaj z njihovimi zneski.
        3.  Ne seštevaj postavk sam in si ne izmišljuj skupnih vsot. Samo navajaj podatke, kot so zapisani.
        4.  Ne ponujaj povezav, razen če te uporabnik izrecno prosi zanje.

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