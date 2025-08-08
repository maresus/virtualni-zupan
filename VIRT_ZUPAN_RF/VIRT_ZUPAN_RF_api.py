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
from collections import Counter
from difflib import SequenceMatcher

# ------------------------------------------------------------------------------
# 0) ENV
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '..', '.env'))

# ------------------------------------------------------------------------------
# 1) KONFIG
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    ENV_TYPE: str = os.getenv('ENV_TYPE', 'development')
    BASE_DIR: str = BASE_DIR
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

    GENERIC_STREET_WORDS: Tuple[str, ...] = ("cesta", "ceste", "cesti", "ulica", "ulice", "ulici", "pot", "trg", "naselje", "območje")
    GENERIC_PREPS: Tuple[str, ...] = ("na", "v", "za", "ob", "pod", "pri", "nad", "do", "od", "k", "proti")

    WASTE_VARIANTS: Dict[str, List[str]] = field(default_factory=lambda: {
        "Biološki odpadki": ["bio", "bio odpadki", "bioloski odpadki", "biološki odpadki"],
        "Mešani komunalni odpadki": ["mešani komunalni odpadki", "mesani komunalni odpadki", "mešani", "mesani", "komunalni odpadki"],
        "Odpadna embalaža": ["odpadna embalaža", "odpadna embalaza", "embalaža", "rumena kanta", "rumene kante"],
        "Papir in karton": ["papir", "karton", "papir in karton"],
        "Steklena embalaža": ["steklo", "steklena embalaža", "steklena embalaza"]
    })

    FALLBACK_CONTACT: str = "Občina Rače-Fram: 02 609 60 10"

    def __post_init__(self):
        object.__setattr__(self, 'CHROMA_DB_PATH', os.path.join(self.DATA_DIR, "chroma_db"))
        object.__setattr__(self, 'LOG_FILE_PATH', os.path.join(self.DATA_DIR, "zupan_pogovori.jsonl"))
        os.makedirs(self.DATA_DIR, exist_ok=True)

cfg = Config()

missing = [e for e in cfg.REQUIRED_ENVS if not os.getenv(e)]
if missing:
    sys.stderr.write(f"[ERROR] Manjkajoče okoljske spremenljivke: {', '.join(missing)}\n")
    sys.exit(1)

# ------------------------------------------------------------------------------
# 2) LOGGING
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("VirtualniZupan")

# ------------------------------------------------------------------------------
# 3) UTIL: normalizacija, ujemanje, itd.
# ------------------------------------------------------------------------------
def normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.lower()
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8')
    s = re.sub(r'[^\w\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def sl_variant_eq(a: str, b: str) -> bool:
    a_n = normalize_text(a)
    b_n = normalize_text(b)
    if a_n == b_n:
        return True
    a_s = re.sub(r'(ski|ska|sko|skega|skem|skih)$', '', a_n)
    b_s = re.sub(r'(ski|ska|sko|skega|skem|skih)$', '', b_n)
    if a_s and a_s == b_s:
        return True
    if len(a_n) > 2 and len(b_n) > 2 and a_n[:-1] == b_n[:-1] and {a_n[-1], b_n[-1]} <= {"a","i","e","o","u"}:
        return True
    return False

def strip_generics(s: str) -> str:
    tokens = normalize_text(s).split()
    stop = set(normalize_text(" ".join(cfg.GENERIC_STREET_WORDS)).split()) | set(normalize_text(" ".join(cfg.GENERIC_PREPS)).split())
    kept = [t for t in tokens if t not in stop]
    return " ".join(kept)

def gen_street_keys(name: str) -> List[str]:
    base = strip_generics(name)
    keys = {base}
    if base.endswith("ska"):
        keys.add(base[:-3] + "ski")
    if base.endswith("ski"):
        keys.add(base[:-3] + "ska")
    if base.endswith("a"):
        keys.add(base[:-1] + "i")
    if base.endswith("i"):
        keys.add(base[:-1] + "a")
    return [k for k in keys if k]

def parse_dates_from_text(text: str) -> List[str]:
    dates = re.findall(r'(\d{1,2}\.\d{1,2}\.?)', text)
    seen = set()
    out = []
    for d in dates:
        dn = d if d.endswith('.') else d + '.'
        if dn not in seen:
            seen.add(dn)
            out.append(dn)
    return out

def get_canonical_waste(text: str) -> Optional[str]:
    norm = normalize_text(text)
    # -- SPECIFIČNO PRED GENERIČNIM --
    if "stekl" in norm:
        return "Steklena embalaža"
    if "papir" in norm or "karton" in norm:
        return "Papir in karton"
    if "bio" in norm or "biolo" in norm:
        return "Biološki odpadki"
    if ("komunaln" in norm and "odpadk" in norm) or re.search(r'\bmesan|\bmešani|\bme{s|š}an', norm):
        return "Mešani komunalni odpadki"
    # generično, samo če ni stekla
    if (("rumen" in norm and "kant" in norm) or "embala" in norm) and "stekl" not in norm:
        return "Odpadna embalaža"
    # vnaprej definirani sinonimi
    for canon, variants in cfg.WASTE_VARIANTS.items():
        for v in variants:
            if normalize_text(v) in norm:
                return canon
    # fuzzy
    for canon, variants in cfg.WASTE_VARIANTS.items():
        for v in variants:
            if _ratio(norm, normalize_text(v)) >= 0.90:
                return canon
    return None

def extract_locations_from_naselja(field: str) -> List[str]:
    parts = set()
    if not field:
        return []
    clean = re.sub(r'\(h\. *št\..*?\)', '', field)
    segments = re.split(r'([A-ZČŠŽ][a-zčšž]+\s*:)', clean)
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        if seg.endswith(':'):
            continue
        for sub in seg.split(','):
            n = normalize_text(sub)
            if n:
                parts.add(n)
    return list(parts)

def detect_intent_qna(q_norm: str) -> str:
    if re.search(r'\bkdo je\b', q_norm) and ('zupan' in q_norm or 'župan' in q_norm):
        return 'who_is_mayor'
    if 'kontakt' in q_norm or 'telefon' in q_norm or 'e-mail' in q_norm or 'stevilka' in q_norm or 'številka' in q_norm:
        return 'contact'
    if 'prevoz' in q_norm or 'vozni red' in q_norm or 'minibus' in q_norm or 'avtobus' in q_norm or 'kopivnik' in q_norm:
        return 'transport'
    if 'komunaln' in q_norm and 'prispev' in q_norm:
        return 'komunalni_prispevek'
    if 'poletn' in q_norm and ('tabor' in q_norm or 'kamp' in q_norm or 'varstvo' in q_norm):
        return 'camp'
    return 'general'

# ------------------------------------------------------------------------------
# 4) GLAVNI RAZRED
# ------------------------------------------------------------------------------
class VirtualniZupan:
    def __init__(self) -> None:
        prefix = "PRODUCTION" if cfg.ENV_TYPE == 'production' else "DEVELOPMENT"
        logger.info(f"[{prefix}] VirtualniŽupan v47.0 inicializiran.")
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

        # Indeksi za odpadke
        self._street_index: Dict[str, List[Dict[str, Any]]] = {}
        self._area_type_docs: Dict[Tuple[str, str], str] = {}
        self._street_keys_list: List[str] = []

    # ---- infra
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
        if self.collection:
            return
        logger.info(f"Nalaganje baze znanja iz: {cfg.CHROMA_DB_PATH}")
        try:
            ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name=cfg.EMBEDDING_MODEL
            )
            client = chromadb.PersistentClient(path=cfg.CHROMA_DB_PATH)
            self.collection = client.get_collection(name=cfg.COLLECTION_NAME, embedding_function=ef)
            logger.info(f"Baza uspešno naložena. Število dokumentov: {self.collection.count()}")
            self._build_waste_indexes()
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

    # ---- spomin
    def preoblikuj_vprasanje_s_kontekstom(self, zgodovina_pogovora: List[Tuple[str, str]], zadnje_vprasanje: str) -> str:
        if not zgodovina_pogovora:
            return zadnje_vprasanje
        logger.info("Kličem specialista za spomin...")
        zgodovina_str = "\n".join([f"Uporabnik: {q}\nAsistent: {a}" for q, a in zgodovina_pogovora])
        prompt = f"""Tvoja naloga je, da glede na zgodovino pogovora preoblikuješ novo vprašanje v samostojno vprašanje.
DIREKTIVA: Če je "Novo vprašanje" že smiselno in popolno samo po sebi (vsebuje aktivnost IN lokacijo), ga vrni nespremenjenega.
---
Zgodovina:
{zgodovina_str}
Novo vprašanje: "{zadnje_vprasanje}"
Samostojno vprašanje:"""
        preoblikovano = self._call_llm(prompt, max_tokens=80)
        if not preoblikovano:
            return zadnje_vprasanje
        return preoblikovano.replace('"','').strip()

    # ---- NAP / promet
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
            vsi_dogodki = self._promet_cache
        else:
            try:
                token = self._ensure_nap_token()
                if not token:
                    return "Dostop do prometnih informacij trenutno ni mogoč."
                headers = {'Authorization': f'Bearer {token}'}
                data_response = self._http.get(cfg.NAP_DATA_URL, headers=headers, timeout=15)
                data_response.raise_for_status()
                vsi_dogodki = data_response.json().get('features', [])
                self._promet_cache = vsi_dogodki
                self._promet_cache_ts = now
            except requests.RequestException:
                logger.exception("NAP API napaka pri pridobivanju podatkov o prometu.")
                return "Žal mi neposreden vpogled v stanje na cestah trenutno ne deluje."

        relevantne_zapore = []
        for d in vsi_dogodki:
            props = d.get('properties', {})
            cel = " ".join(str(props.get(polje, '')).lower() for polje in ['cesta', 'opis', 'imeDogodka'])
            if any(k in cel for k in cfg.PROMET_FILTER):
                relevantne_zapore.append(props)

        if not relevantne_zapore:
            return "Po podatkih portala promet.si na območju občine Rače-Fram trenutno ni zabeleženih del na cesti, zapor ali zastojev."

        unique = [dict(t) for t in {tuple(sorted(x.items())) for x in relevantne_zapore}]
        porocilo = "Našel sem naslednje **trenutne** informacije o dogodkih na cesti (vir: promet.si):\n\n"
        for z in unique:
            porocilo += f"- **Cesta:** {z.get('cesta', 'Ni podatka')}\n  **Opis:** {z.get('opis', 'Ni podatka')}\n\n"
        return porocilo.strip()

    # ---- Odpadki: indeks
    def _build_waste_indexes(self) -> None:
        logger.info("Gradim indekse za odpadke …")
        if not self.collection:
            return
        all_docs = self.collection.get(where={"kategorija": "Odvoz odpadkov"})
        if not all_docs or not all_docs.get('ids'):
            logger.warning("Ni dokumentov za odpadke.")
            return

        for i in range(len(all_docs['ids'])):
            meta = all_docs['metadatas'][i]
            doc_text = all_docs['documents'][i]
            tip = get_canonical_waste(meta.get('tip_odpadka', '') or '') or (meta.get('tip_odpadka', '') or '')
            area = normalize_text(meta.get('obmocje', '') or '')
            self._area_type_docs[(area, tip)] = doc_text

            streets = extract_locations_from_naselja(meta.get('naselja', '') or '')
            for s in streets:
                display = s  # normalizirano
                for key in gen_street_keys(s):
                    self._street_index.setdefault(key, []).append({
                        "area": area,
                        "tip": tip,
                        "doc": doc_text,
                        "display": display
                    })

        self._street_keys_list = list(self._street_index.keys())
        logger.info(f"Indeks: {len(self._street_keys_list)} uličnih ključev, {len(self._area_type_docs)} (območje,tip).")

    # ---- Odpadki: iskanje
    def _best_street_key_for_query(self, phrases: List[str]) -> Optional[str]:
        for ph in phrases:
            for key in gen_street_keys(ph):
                if key in self._street_index:
                    return key
        best = (None, 0.0)
        for ph in phrases:
            base = strip_generics(ph)
            if len(base) < 3:
                continue
            for key in self._street_keys_list:
                sc = _ratio(base, key)
                if sc > best[1]:
                    best = (key, sc)
        if best[0] and best[1] >= 0.88:  # malo znižano, da ujame "mlinski ulic"
            return best[0]
        return None

    def _build_location_phrases(self, vprasanje_norm: str) -> List[str]:
        waste_stop = set()
        for variants in cfg.WASTE_VARIANTS.values():
            for v in variants:
                waste_stop.add(normalize_text(v))
        extra_stop = {"kdaj","je","ja","naslednji","odvoz","odpadkov","smeti","urnik","urniki","termini","termine",
                      "kako","kateri","katera","kaj","koga","kje","kam"} \
                      | set(normalize_text(" ".join(cfg.GENERIC_STREET_WORDS)).split()) \
                      | set(normalize_text(" ".join(cfg.GENERIC_PREPS)).split())
        stop = waste_stop.union(extra_stop)

        toks = [t for t in re.split(r'[,\s]+', vprasanje_norm) if t and t not in stop]
        phrases = []
        for size in (3,2):
            for i in range(len(toks)-size+1):
                p = " ".join(toks[i:i+size]).strip()
                if p and p not in phrases:
                    phrases.append(p)
        for t in toks:
            if len(t) >= 4 and t not in phrases:
                phrases.append(t)
        logger.info(f"Lokacijske fraze: {phrases}")
        return phrases

    def _format_dates_for_tip(self, street_display: str, tip_odpadka: str, doc_text: str, only_next: bool) -> Optional[str]:
        dates = parse_dates_from_text(doc_text)
        if not dates:
            return None
        if only_next:
            today = datetime.now().date()
            cand = None
            for d in dates:
                try:
                    dd, mm = d.replace('.',' ').strip().split()[:2]
                    dt = datetime(datetime.now().year, int(mm), int(dd)).date()
                    if dt >= today:
                        cand = f"{dd}.{mm}."
                        break
                except Exception:
                    continue
            if cand:
                return f"Naslednji odvoz za **{tip_odpadka}** na **{street_display.title()}** je **{cand}**."
            else:
                return f"Za **{tip_odpadka}** na **{street_display.title()}** v tem letu ni več terminov."
        else:
            return f"Za **{street_display.title()}** je odvoz **{tip_odpadka}**: {', '.join(dates)}"

    def obravnavaj_odvoz_odpadkov(self, uporabnikovo_vprasanje: str, session_id: str) -> str:
        logger.info("Kličem specialista za odpadke…")
        ses = self.zgodovina_seje.setdefault(session_id, {'zgodovina': [], 'stanje': {}})
        stanje = ses['stanje']
        if not self.collection:
            return "Baza urnikov ni na voljo."

        vprasanje_norm = normalize_text(uporabnikovo_vprasanje)
        only_next = "naslednji" in vprasanje_norm

        wanted_tip = get_canonical_waste(vprasanje_norm)
        if not wanted_tip and only_next and stanje.get('zadnji_tip'):
            wanted_tip = stanje['zadnji_tip']

        phrases = self._build_location_phrases(vprasanje_norm)
        if not phrases and stanje.get('zadnja_lokacija_norm'):
            phrases = [stanje['zadnja_lokacija_norm']]

        if not phrases:
            stanje['namen'] = 'odpadki'
            stanje['caka_na'] = 'lokacija'
            return "Za katero ulico te zanima urnik? (npr. 'Bistriška cesta, Fram')"

        key = self._best_street_key_for_query(phrases)
        logger.info(f"Najboljši ulični ključ: {key}")

        if not key:
            if not wanted_tip:
                stanje['zadnja_lokacija_norm'] = phrases[0]
                stanje['namen'] = 'odpadki'
                stanje['caka_na'] = 'tip'
                return "Kateri tip odpadkov te zanima? (bio, mešani, embalaža, papir, steklo)"
            return "Za navedeno ulico žal nisem našel urnika za izbrani tip."

        entries = self._street_index.get(key, [])
        if not entries:
            return "Za navedeno ulico žal nimam podatkov."

        best_entry = None
        if wanted_tip:
            for e in entries:
                if get_canonical_waste(e["tip"]) == wanted_tip:
                    best_entry = e
                    break
            if not best_entry:
                area_counts = Counter([e["area"] for e in entries if e["area"]])
                if area_counts:
                    area_norm, _ = area_counts.most_common(1)[0]
                    doc = self._area_type_docs.get((area_norm, wanted_tip))
                    if doc:
                        street_disp = entries[0]["display"]
                        ans = self._format_dates_for_tip(street_disp, wanted_tip, doc, only_next)
                        if ans:
                            stanje['zadnja_lokacija_norm'] = strip_generics(street_disp)
                            stanje['zadnji_tip'] = wanted_tip
                            stanje['namen'] = 'odpadki'
                            stanje.pop('caka_na', None)
                            return ans
                return "Za navedeno ulico žal nisem našel urnika za izbrani tip."
        else:
            # brez tipa – izberi prvi vnos (a je zdaj resno težko, ker skoraj vedno zaznamo tip)
            best_entry = entries[0]

        tip_canon = get_canonical_waste(best_entry["tip"]) or best_entry["tip"]
        ans = self._format_dates_for_tip(best_entry["display"], tip_canon, best_entry["doc"], only_next)
        if not ans:
            return "Žal ne najdem datumov v urniku."

        stanje['zadnja_lokacija_norm'] = strip_generics(best_entry["display"])
        stanje['zadnji_tip'] = tip_canon
        stanje['namen'] = 'odpadki'
        stanje.pop('caka_na', None)
        return ans

    # ---- RAG
    def _strip_past_year_lines(self, text: str, this_year: int) -> str:
        lines = text.splitlines()
        kept = []
        for ln in lines:
            years = [int(y) for y in re.findall(r'\b(20\d{2})\b', ln)]
            if years and max(years) < this_year:
                continue
            kept.append(ln)
        return "\n".join(kept)

    def _filter_rag_results_by_year(self, docs: List[str], metas: List[Dict[str,Any]], intent: str) -> Tuple[List[str], List[Dict[str,Any]]]:
        this_year = datetime.now().year
        keep_docs, keep_metas = [], []
        for doc, meta in zip(docs, metas):
            combined = f"{doc}\n{json.dumps(meta, ensure_ascii=False)}"
            years = [int(y) for y in re.findall(r'\b(20\d{2})\b', combined)]
            if intent == 'camp':
                if years and max(years) < this_year:
                    continue
                doc = self._strip_past_year_lines(doc, this_year)
            else:
                if years and max(years) < this_year:
                    continue
            keep_docs.append(doc)
            keep_metas.append(meta)
        return keep_docs, keep_metas

    def _zgradi_rag_prompt(self, vprasanje: str, zgodovina: List[Tuple[str, str]]) -> Optional[str]:
        logger.info(f"Gradim RAG prompt za vprašanje: '{vprasanje}'")
        if not self.collection:
            return None

        q_norm = normalize_text(vprasanje)
        intent = detect_intent_qna(q_norm)

        results = self.collection.query(query_texts=[q_norm], n_results=8, include=["documents", "metadatas"])
        docs = results['documents'][0] if results and results.get('documents') else []
        metas = results['metadatas'][0] if results and results.get('metadatas') else []

        # 1) odstrani odpadke iz RAG, če vprašanje NI o odpadkih
        filtered_docs, filtered_metas = [], []
        for d, m in zip(docs, metas):
            if (m.get('kategorija','') or '').lower() == 'odvoz odpadkov':
                continue
            filtered_docs.append(d); filtered_metas.append(m)
        docs, metas = filtered_docs, filtered_metas

        # 2) letni filter (kamp ipd.)
        docs, metas = self._filter_rag_results_by_year(docs, metas, intent)

        context = ""
        for doc, meta in zip(docs, metas):
            context += f"--- VIR: {meta.get('source', '?')}\nPOVEZAVA: {meta.get('source_url', '')}\nVSEBINA: {doc}\n\n"
        if not context:
            return None

        now = datetime.now()
        zgodovina_str = "\n".join([f"U: {q}\nA: {a}" for q, a in zgodovina])

        extra = []
        if intent == 'who_is_mayor':
            extra.append("Če vprašanje sprašuje 'kdo je župan', odgovori v ENEM kratkem stavku samo z imenom in funkcijo, brez dodatkov.")
        if intent == 'contact':
            extra.append(f"Če uporabnik prosi za 'kontakt' in v kontekstu ni specifičnega, uporabi generični kontakt: '{cfg.FALLBACK_CONTACT}'.")
        if intent == 'transport':
            extra.append("Če je vprašanje o prevozu/voznem redu, navedi samo relacijo, ključne ure in kontakt – brez nepotrebnih podrobnosti.")
        if intent == 'komunalni_prispevek':
            extra.append("Za komunalni prispevek odgovori v 5–7 alinejah: (1) Kaj je, (2) Kdo je zavezanec, (3) Kdaj se plača, (4) Kam oddam vlogo, (5) Priloge/izračun, (6) Kontakt, (7) Povezava.")
        if intent == 'camp':
            extra.append("Za poletni kamp navedi samo aktualno leto: datume, lokacijo, ceno (če je), in kontakt/URL – brez omembe preteklih let.")

        directives = "\n".join(extra)

        return f"""Ti si 'Virtualni župan občine Rače-Fram'.
DIREKTIVA #1 (DATUMI): Današnji datum je {now.strftime('%d.%m.%Y')}. Ne navajaj informacij iz preteklih let.
DIREKTIVA #2 (OBLIKA): Odgovor naj bo kratek in pregleden. Ključne informacije **poudari**. Kjer naštevaš, **uporabi alineje (-)**.
DIREKTIVA #3 (POVEZAVE): Če v kontekstu pod ključem 'POVEZAVA' najdeš URL, ga MORAŠ vključiti kot '[Ime vira](URL)'.
DIREKTIVA #4 (SPECIFIČNOST): Če specifičnega podatka ni, povej 'Žal nimam natančnega podatka.' – brez balasta.
{directives}

--- KONTEKST ---
{context}---
ZGODOVINA POGOVORA:
{zgodovina_str}
---
VPRAŠANJE: "{vprasanje}"
ODGOVOR:"""

    # ---- glavni vmesnik
    def odgovori(self, uporabnikovo_vprasanje: str, session_id: str) -> str:
        self.nalozi_bazo()
        if not self.collection:
            return "Oprostite, moja baza znanja trenutno ni na voljo."

        ses = self.zgodovina_seje.setdefault(session_id, {'zgodovina': [], 'stanje': {}})
        zgodovina = ses['zgodovina']
        stanje = ses['stanje']

        q_norm = normalize_text(uporabnikovo_vprasanje)

        # Hard shortcut: župan v enem stavku
        if re.search(r'\bkdo je\b', q_norm) and ('zupan' in q_norm or 'župan' in q_norm):
            odgovor = "Župan občine Rače-Fram je **Samo Rajšp**."
            zgodovina.append((uporabnikovo_vprasanje, odgovor))
            if len(zgodovina) > 4: zgodovina.pop(0)
            self.belezi_pogovor(session_id, uporabnikovo_vprasanje, odgovor)
            if stanje.get('namen') == 'odpadki' and not stanje.get('caka_na'):
                stanje.pop('namen', None)
            return odgovor

        # preoblikovanje
        pametno_vprasanje = self.preoblikuj_vprasanje_s_kontekstom(zgodovina, uporabnikovo_vprasanje)
        vprasanje_lower = normalize_text(pametno_vprasanje)

        # domena
        is_waste_query = any(re.search(r'\b' + re.escape(k) + r'\b', vprasanje_lower) for k in cfg.KLJUCNE_ODPADKI) \
                         or (get_canonical_waste(vprasanje_lower) is not None) \
                         or ('naslednji' in vprasanje_lower)
        is_traffic_query = any(re.search(r'\b' + re.escape(k) + r'\b', vprasanje_lower) for k in cfg.KLJUCNE_PROMET)
        waste_followup_needed = (stanje.get('namen') == 'odpadki' and stanje.get('caka_na') in ('lokacija','tip'))

        if is_waste_query or waste_followup_needed:
            odgovor = self.obravnavaj_odvoz_odpadkov(pametno_vprasanje, session_id)
        elif is_traffic_query:
            if stanje.get('namen') == 'odpadki' and not waste_followup_needed:
                stanje.pop('namen', None); stanje.pop('caka_na', None)
            odgovor = self.preveri_zapore_cest()
        else:
            if stanje.get('namen') == 'odpadki' and not waste_followup_needed:
                stanje.pop('namen', None); stanje.pop('caka_na', None)
            prompt = self._zgradi_rag_prompt(pametno_vprasanje, zgodovina)
            if not prompt:
                odgovor = "Žal o tej temi nimam nobenih informacij."
            else:
                odgovor = self._call_llm(prompt)

        # beleženje
        zgodovina.append((uporabnikovo_vprasanje, odgovor))
        if len(zgodovina) > 4:
            zgodovina.pop(0)
        self.belezi_pogovor(session_id, uporabnikovo_vprasanje, odgovor)
        return odgovor

# ------------------------------------------------------------------------------
# 5) CLI
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
            if not q:
                continue
            print("\n--- ODGOVOR ŽUPANA ---\n")
            print(zupan.odgovori(q, session_id))
            print("\n---------------------\n")
        except KeyboardInterrupt:
            break
    print("\nNasvidenje!")
