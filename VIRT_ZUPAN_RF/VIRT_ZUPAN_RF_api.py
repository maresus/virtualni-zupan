import os
import sys
import json
import re
import unicodedata
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple

import requests
import chromadb
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
from chromadb.utils import embedding_functions

# ------------------------------------------------------------------------------
# 1. KONFIGURACIJA
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    ENV_TYPE: str = os.getenv('ENV_TYPE', 'development')
    BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR: str = "/data" if ENV_TYPE == 'production' else os.path.join(BASE_DIR, "data")
    
    CHROMA_DB_PATH: str = field(init=False)
    LOG_FILE_PATH: str = field(init=False)
    
    COLLECTION_NAME: str = "obcina_race_fram_prod"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    GENERATOR_MODEL: str = "gpt-4o-mini"
    
    NAP_TOKEN_URL: str = "https://b2b.nap.si/uc/user/token"
    NAP_DATA_URL: str = "https://b2b.nap.si/data/b2b.roadworks.geojson.sl_SI"
    NAP_USERNAME: Optional[str] = os.getenv("NAP_USERNAME")
    NAP_PASSWORD: Optional[str] = os.getenv("NAP_PASSWORD")
    
    REQUIRED_ENVS: Tuple[str, ...] = ("OPENAI_API_KEY", "NAP_USERNAME", "NAP_PASSWORD")
    
    KLJUCNE_ODPADKI: Tuple[str, ...] = ("smeti", "odpadki", "odvoz", "odpavkov", "komunala")
    KLJUCNE_PROMET: Tuple[str, ...] = ("cesta", "ceste", "cesti", "promet", "dela", "delo", "zapora", "zapore", "zaprta", "zastoj", "gneča", "kolona")
    PROMET_FILTER: Tuple[str, ...] = ("rače", "fram", "slivnica", "brunšvik", "podova", "morje", "hoče", "r2-430", "r3-711", "g1-2", "priključek slivnica", "razcep slivnica", "letališče maribor", "odcep za rače")
    
    WASTE_VARIANTS: Dict[str, List[str]] = field(default_factory=lambda: {
        "Biološki odpadki": ["bioloski odpadki", "bioloskih odpakov", "bio", "biološki odpadki"],
        "Mešani komunalni odpadki": ["mesani komunalni odpadki", "komunalni odpadki", "mešani"],
        "Odpadna embalaža": ["odpadna embalaza", "embalaža", "rumena kanta"],
        "Papir in karton": ["papir in karton", "papir", "karton"],
        "Steklena embalaža": ["steklena embalaza", "steklo"]
    })

    def __post_init__(self):
        object.__setattr__(self, 'CHROMA_DB_PATH', os.path.join(self.DATA_DIR, "chroma_db"))
        object.__setattr__(self, 'LOG_FILE_PATH', os.path.join(self.DATA_DIR, "zupan_pogovori.jsonl"))
        os.makedirs(self.DATA_DIR, exist_ok=True)

cfg = Config()

# Validacija okolja
missing = [e for e in cfg.REQUIRED_ENVS if not os.getenv(e)]
if missing:
    sys.stderr.write(f"[ERROR] Manjkajoče okoljske spremenljivke: {', '.join(missing)}\n")
    sys.exit(1)

# ------------------------------------------------------------------------------
# 2. LOGGING
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("VirtualniZupan")

# ------------------------------------------------------------------------------
# 3. POMOŽNE FUNKCIJE
# ------------------------------------------------------------------------------
def normalize_text(s: Optional[str]) -> str:
    if not s: return ""
    s = s.lower()
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8')
    s = re.sub(r'[^\w\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def get_canonical_waste(text: str) -> Optional[str]:
    norm = normalize_text(text)
    for canon, variants in cfg.WASTE_VARIANTS.items():
        if any(v in norm for v in variants):
            return canon
    return None

def extract_locations_from_naselja(field: str) -> List[str]:
    parts = set()
    clean = re.sub(r'\(h\. *št\..*?\)', '', field)
    segments = re.split(r'([A-ZČŠŽ][a-zčšž]+\s*:)', clean)
    prefix = ""
    for seg in segments:
        seg = seg.strip()
        if not seg: continue
        if seg.endswith(':'):
            prefix = normalize_text(seg[:-1])
            if prefix: parts.add(prefix)
        else:
            for sub in seg.split(','):
                n = normalize_text(sub)
                if n:
                    parts.add(n)
                    if prefix: parts.add(f"{n} {prefix}")
    return list(parts)

# ------------------------------------------------------------------------------
# 4. GLAVNI RAZRED
# ------------------------------------------------------------------------------
class VirtualniZupan:
    def __init__(self) -> None:
        prefix = "PRODUCTION" if cfg.ENV_TYPE == 'production' else "DEVELOPMENT"
        logger.info(f"[{prefix}] VirtualniŽupan v40.0 (Končna Sinteza) inicializiran.")
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.collection: Optional[chromadb.Collection] = None
        self.zgodovina_seje: Dict[str, Dict[str, Any]] = {}
        self._nap_access_token: Optional[str] = None
        self._nap_token_expiry: Optional[datetime] = None
        
        self._http = requests.Session()
        retries = Retry(total=3, backoff_factor=0.3, status_forcelist=[500, 502, 504])
        self._http.mount("https://", HTTPAdapter(max_retries=retries))

        self._promet_cache: Optional[List[Dict]] = None
        self._promet_cache_ts: Optional[datetime] = None

    def _call_llm(self, prompt: str, **kwargs) -> str:
        try:
            res = self.openai_client.chat.completions.create(
                model=cfg.GENERATOR_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                **kwargs
            )
            return res.choices[0].message.content.strip()
        except OpenAIError:
            logger.exception("LLM napaka")
            return "Oprostite, prišlo je do napake pri generiranju odgovora."

    def nalozi_bazo(self) -> None:
        if self.collection: return
        logger.info(f"Nalaganje baze znanja iz: {cfg.CHROMA_DB_PATH}")
        try:
            ef = embedding_functions.OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"), model_name=cfg.EMBEDDING_MODEL)
            client = chromadb.PersistentClient(path=cfg.CHROMA_DB_PATH)
            self.collection = client.get_collection(name=cfg.COLLECTION_NAME, embedding_function=ef)
            logger.info(f"Baza uspešno naložena. Število dokumentov: {self.collection.count()}")
        except Exception:
            logger.exception("KRITIČNA NAPAKA: Baze znanja ni mogoče naložiti.")
            self.collection = None

    def belezi_pogovor(self, session_id: str, vprasanje: str, odgovor: str) -> None:
        try:
            zapis = {"timestamp": datetime.now().isoformat(), "session_id": session_id, "vprasanje": vprasanje, "odgovor": odgovor}
            with open(cfg.LOG_FILE_PATH, 'a', encoding='utf-8') as f:
                f.write(json.dumps(zapis, ensure_ascii=False) + '\n')
        except Exception:
            logger.exception(f"Napaka pri beleženju pogovora za sejo {session_id}")

    def preoblikuj_vprasanje_s_kontekstom(self, zgodovina_pogovora: List[Tuple[str, str]], zadnje_vprasanje: str) -> str:
        if not zgodovina_pogovora: return zadnje_vprasanje
        
        logger.info("Kličem specialista za spomin...")
        zgodovina_str = "\n".join([f"Uporabnik: {q}\nAsistent: {a}" for q, a in zgodovina_pogovora])
        prompt = f"""Tvoja naloga je, da glede na zgodovino pogovora preoblikuješ novo vprašanje v samostojno vprašanje.
DIREKTIVA: Če je "Novo vprašanje" že smiselno in popolno samo po sebi (vsebuje aktivnost in lokacijo), ga vrni nespremenjenega.

Primer 1:
Zgodovina:
Uporabnik: kdaj je odvoz smeti na gortanovi ulici?
Novo vprašanje: "kaj pa papir?"
Samostojno vprašanje: "kdaj je odvoz za papir na gortanovi ulici?"

Primer 2:
Zgodovina:
Uporabnik: kaj je za malico 1.9?
Novo vprašanje: "kaj pa naslednji dan?"
Samostojno vprašanje: "kaj je za malico 2.9?"
---
Zgodovina:
{zgodovina_str}
Novo vprašanje: "{zadnje_vprasanje}"
Samostojno vprašanje:"""
        preoblikovano = self._call_llm(prompt, max_tokens=100)
        logger.info(f"Originalno: '{zadnje_vprasanje}' -> Preoblikovano: '{preoblikovano}'")
        return preoblikovano if "napake" not in preoblikovano else zadnje_vprasanje

    def _ensure_nap_token(self) -> Optional[str]:
        if self._nap_access_token and self._nap_token_expiry and datetime.utcnow() < self._nap_token_expiry - timedelta(seconds=60):
            return self._nap_access_token
        logger.info("Pridobivam/osvežujem NAP API žeton...")
        try:
            payload = {'grant_type': 'password', 'username': cfg.NAP_USERNAME, 'password': cfg.NAP_PASSWORD}
            response = self._http.post(cfg.NAP_TOKEN_URL, data=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            self._nap_access_token = data['access_token']
            self._nap_token_expiry = datetime.utcnow() + timedelta(seconds=data.get('expires_in', 3600))
            return self._nap_access_token
        except requests.RequestException:
            logger.exception("NAP API napaka pri pridobivanju žetona.")
            return None

    def preveri_zapore_cest(self) -> str:
        logger.info("Kličem specialista za promet (NAP API)...")
        now = datetime.utcnow()
        if self._promet_cache and self._promet_cache_ts and now - self._promet_cache_ts < timedelta(minutes=5):
            logger.info("Uporabljam predpomnjene (cached) podatke o prometu.")
            vsi_dogodki = self._promet_cache
        else:
            try:
                token = self._ensure_nap_token()
                if not token: return "Dostop do prometnih informacij trenutno ni mogoč."
                headers = {'Authorization': f'Bearer {token}'}
                data_response = self._http.get(cfg.NAP_DATA_URL, headers=headers, timeout=15)
                data_response.raise_for_status()
                vsi_dogodki = data_response.json().get('features', [])
                self._promet_cache = vsi_dogodki
                self._promet_cache_ts = now
            except requests.RequestException:
                logger.exception("NAP API napaka pri pridobivanju podatkov o prometu.")
                return "Žal mi neposreden vpogled v stanje na cestah trenutno ne deluje."
        
        relevantne_zapore = [d['properties'] for d in vsi_dogodki if any(kljucnik in " ".join(str(d.get('properties', {}).get(polje, '')).lower() for polje in ['cesta', 'opis']) for kljucnik in cfg.PROMET_FILTER)]
        if not relevantne_zapore:
            return "Po podatkih portala promet.si na območju občine Rače-Fram trenutno ni zabeleženih del na cesti, zapor ali zastojev."
        unique_zapore = [dict(t) for t in {tuple(d.items()) for d in relevantne_zapore}]
        porocilo = "Našel sem naslednje **trenutne** informacije o dogodkih na cesti (vir: promet.si):\n\n"
        for z in unique_zapore:
            porocilo += f"- **Cesta:** {z.get('cesta', 'Ni podatka')}\n  **Opis:** {z.get('opis', 'Ni podatka')}\n\n"
        return porocilo

    def _poisci_naslednji_datum(self, datumi_str: str, danes: datetime) -> Optional[str]:
        leto = danes.year
        datumi = []
        # Popravljen regex za branje datumov, ki so lahko ločeni s piko ali brez
        for del_str in re.findall(r'(\d{1,2}\.\d{1,2}\.?)', datumi_str):
            try:
                dan, mesec = map(int, del_str.replace('.', ' ').strip().split())
                datumi.append(datetime(leto, mesec, dan))
            except (ValueError, IndexError): continue
        for datum in sorted(datumi):
            if datum.date() >= danes.date():
                return datum.strftime('%d.%m.%Y')
        return None

    def obravnavaj_odvoz_odpadkov(self, uporabnikovo_vprasanje: str, session_id: str) -> str:
        logger.info("Kličem specialista za odpadke...")
        stanje = self.zgodovina_seje[session_id].get('stanje', {})
        if not self.collection: return "Baza urnikov ni na voljo."
        
        vprasanje_norm = normalize_text(uporabnikovo_vprasanje)
        iskani_tip = get_canonical_waste(vprasanje_norm)
        
        stop_words = {"kdaj", "je", "naslednji", "odvoz", "odpadkov", "smeti", "na", "v", "za", "kako", "poteka"}
        for variants in cfg.WASTE_VARIANTS.values():
            for v in variants:
                stop_words.add(v)
        
        iskane_besede_lokacije = [b for b in vprasanje_norm.split() if b not in stop_words and len(b) > 2]

        if not iskane_besede_lokacije and not stanje.get('caka_na'):
             stanje.update({'caka_na': 'lokacija', 'namen': 'odpadki'})
             return "Za katero lokacijo (naselje ali ulico) vas zanima urnik?"
        
        vsi_urniki = self.collection.get(where={"kategorija": "Odvoz odpadkov"})
        if not vsi_urniki or not vsi_urniki.get('ids'):
            return "V bazi znanja ni podatkov o urnikih."

        kandidati = []
        for i in range(len(vsi_urniki['ids'])):
            meta = vsi_urniki['metadatas'][i]
            vse_lokacije_meta = extract_locations_from_naselja(meta.get('naselja', ''))
            vse_lokacije_meta.append(normalize_text(meta.get('obmocje', '')))
            if any(beseda in lokacija for beseda in iskane_besede_lokacije for lokacija in vse_lokacije_meta):
                kandidati.append({'doc': vsi_urniki['documents'][i], 'meta': meta})
        
        if not kandidati:
            return f"Za lokacijo '{' '.join(iskane_besede_lokacije)}' žal nisem našel nobenega urnika."
            
        koncni_urniki_info = []
        if iskani_tip:
            for kandidat in kandidati:
                meta_tip_canon = get_canonical_waste(kandidat['meta'].get('tip_odpadka', ''))
                if meta_tip_canon == iskani_tip:
                    koncni_urniki_info.append(kandidat)
            if not koncni_urniki_info:
                 return f"Za lokacijo '{' '.join(iskane_besede_lokacije)}' sem našel urnike, a ne za tip '{iskani_tip}'."
        else:
            koncni_urniki_info = kandidati

        odgovori = []
        if "naslednji" in vprasanje_norm:
            for info in koncni_urniki_info:
                datumi_match = re.search(r':\s*([\d\.,\s]+)', info['doc'])
                if datumi_match:
                    naslednji_datum = self._poisci_naslednji_datum(datumi_match.group(1), datetime.now())
                    if naslednji_datum:
                        odgovori.append(f"Naslednji odvoz za **{info['meta'].get('tip_odpadka', '')}** v območju **{info['meta'].get('obmocje', '')}** je **{naslednji_datum}**.")
        else:
            odgovori = [info['doc'] for info in koncni_urniki_info]

        stanje.clear()
        return "\n\n".join(sorted(list(set(odgovori)))) if odgovori else "Žal mi ni uspelo najti ustreznega urnika."

    def _zgradi_rag_prompt(self, vprasanje: str, zgodovina: List[Tuple[str, str]]) -> Optional[str]:
        logger.info(f"Gradim RAG prompt za vprašanje: '{vprasanje}'")
        if not self.collection: return None
        
        results = self.collection.query(query_texts=[normalize_text(vprasanje)], n_results=5, include=["documents", "metadatas"])
        context = ""
        if results and results.get('documents') and results['documents'][0]:
            for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                context += f"--- VIR: {meta.get('source', '?')}\nPOVEZAVA: {meta.get('source_url', '')}\nVSEBINA: {doc}\n\n"
        
        if not context: return None

        now = datetime.now()
        zgodovina_str = "\n".join([f"U: {q}\nA: {a}" for q, a in zgodovina])

        return f"""Ti si 'Virtualni župan občine Rače-Fram'.
DIREKTIVA #1 (VAROVALKA ZA DATUME): Današnji datum je {now.strftime('%d.%m.%Y')}. Če je podatek iz leta, ki je manjše od {now.year}, ga IGNORIRAJ.
DIREKTIVA #2 (OBLIKOVANJE): Odgovor mora biti izjemno pregleden. Ključne informacije **poudari s krepko pisavo**. Kjer naštevaš, **obvezno uporabi alineje (-)**.
DIREKTIVA #3 (POVEZAVE): Če v kontekstu pod ključem 'POVEZAVA' najdeš URL, ki ni prazen, ga MORAŠ vključiti v klikljivi obliki: "[Ime vira](URL)".
DIREKTIVA #4 (SPECIFIČNOST): Če ne najdeš specifičnega podatka (npr. 'kontakt'), NE ponavljaj splošnih informacij. Raje reci: "Žal nimam specifičnega kontakta za to temo."

--- KONTEKST ---
{context}---
ZGODOVINA POGOVORA:
{zgodovina_str}
---
VPRAŠANJE UPORABNIKA: "{vprasanje}"
ODGOVOR:"""

    def odgovori(self, uporabnikovo_vprasanje: str, session_id: str) -> str:
        self.nalozi_bazo()
        if not self.collection:
            return "Oprostite, moja baza znanja trenutno ni na voljo."

        stanje = self.zgodovina_seje.setdefault(session_id, {'zgodovina': [], 'stanje': {}})
        zgodovina = stanje['zgodovina']
        
        pametno_vprasanje = self.preoblikuj_vprasanje_s_kontekstom(zgodovina, uporabnikovo_vprasanje)
        vprasanje_lower = pametno_vprasanje.lower()

        if any(re.search(r'\b' + re.escape(k) + r'\b', vprasanje_lower) for k in cfg.KLJUCNE_ODPADKI) or stanje.get('stanje', {}).get('namen') == 'odpadki':
            odgovor = self.obravnavaj_odvoz_odpadkov(pametno_vprasanje, session_id)
        elif any(re.search(r'\b' + re.escape(k) + r'\b', vprasanje_lower) for k in cfg.KLJUCNE_PROMET):
            odgovor = self.preveri_zapore_cest()
        else:
            prompt = self._zgradi_rag_prompt(pametno_vprasanje, zgodovina)
            if not prompt:
                odgovor = "Žal o tej temi nimam nobenih informacij."
            else:
                odgovor = self._call_llm(prompt)

        zgodovina.append((uporabnikovo_vprasanje, odgovor))
        if len(zgodovina) > 4: zgodovina.pop(0)
        
        self.belezi_pogovor(session_id, uporabnikovo_vprasanje, odgovor)
        return odgovor

# ------------------------------------------------------------------------------
# 5. CLI za hiter test
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    zupan = VirtualniZupan()
    zupan.nalozi_bazo()
    if not zupan.collection:
        logger.error("Kolekcija ni na voljo, zapiram.")
        sys.exit(1)
    
    session_id = "local_cli_test"
    print("Župan je pripravljen. Vprašajte ga ('konec' za izhod):")
    
    while True:
        try:
            q = input("> ").strip()
            if q.lower() in ['konec', 'exit', 'quit']:
                break
            if not q: continue
            
            print("\n--- ODGOVOR ŽUPANA ---\n")
            print(zupan.odgovori(q, session_id))
            print("\n---------------------\n")
        except KeyboardInterrupt:
            break
            
    print("\nNasvidenje!")