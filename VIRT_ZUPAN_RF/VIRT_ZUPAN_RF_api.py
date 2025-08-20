# VIRT_ZUPAN_RF_api.py  (v56.0)

import os
import sys
import json
import re
import unicodedata
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple, Set
from collections import Counter
from difflib import SequenceMatcher
from urllib.parse import urlparse
import unittest

import requests
import chromadb
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
from chromadb.utils import embedding_functions

# ------------------------------------------------------------------------------
# 0) ENV
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

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

    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "obcina_race_fram_prod")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    GENERATOR_MODEL: str = os.getenv("GENERATOR_MODEL", "gpt-5")
    OPENAI_TIMEOUT_S: int = int(os.getenv("OPENAI_TIMEOUT_S", "12"))

    # HITER ODGOVOR: LLM/RAG privzeto izklopljen
    USE_LLM: bool = os.getenv("USE_LLM", "0") not in ("0","false","False","no","NO")
    RAG_TOPK: int = int(os.getenv("RAG_TOPK", "5"))

    # Videz izpisa: "pretty" (naslov + vrstice), "plain"
    RENDER_STYLE: str = os.getenv("RENDER_STYLE", "pretty")

    # NAP (promet)
    NAP_TOKEN_URL: str = os.getenv("NAP_TOKEN_URL", "https://b2b.nap.si/uc/user/token")
    NAP_DATA_URL: str = os.getenv("NAP_DATA_URL", "https://b2b.nap.si/data/b2b.roadworks.geojson.sl_SI")
    NAP_USERNAME: Optional[str] = os.getenv("NAP_USERNAME")
    NAP_PASSWORD: Optional[str] = os.getenv("NAP_PASSWORD")
    NAP_CACHE_MIN: int = int(os.getenv("NAP_CACHE_MIN", "5"))

    REQUIRED_ENVS: Tuple[str, ...] = ("OPENAI_API_KEY",)

    # Ključne besede
    KLJUCNE_ODPADKI: Tuple[str, ...] = ("smeti", "odpadki", "odvoz", "komunala", "urnik", "termin")
    KLJUCNE_PROMET: Tuple[str, ...] = ("cesta", "ceste", "cesti", "promet", "dela", "delo", "zapora", "zapore", "zaprta", "zastoj", "gneča", "gneca", "kolona")

    # Transport
    TRANSPORT_HINTS: Tuple[str, ...] = ("prevoz", "vozni red", "minibus", "avtobus", "ura", "odhod", "prihod")
    TRANSPORT_PLACES: Tuple[str, ...] = ("os fram","o s fram","fram","os race","o s race","race","kopivnik","morje","slivnica","brunsvik","brunšvik")
    TRANSPORT_CONTACT_URL: str = os.getenv("TRANSPORT_CONTACT_URL", "https://www.osfram.si/prevozi/")
    TRANSPORT_CONTACT_TEXT: str = "Kontakt in posodobitve: Prevozi OŠ Fram"

    # Geo okvir (za NAP)
    GEO_LAT_MIN: float = float(os.getenv("GEO_LAT_MIN", "46.38"))
    GEO_LAT_MAX: float = float(os.getenv("GEO_LAT_MAX", "46.56"))
    GEO_LON_MIN: float = float(os.getenv("GEO_LON_MIN", "15.54"))
    GEO_LON_MAX: float = float(os.getenv("GEO_LON_MAX", "15.75"))
    GEO_CENTER_LAT: float = float(os.getenv("GEO_CENTER_LAT", "46.46"))
    GEO_CENTER_LON: float = float(os.getenv("GEO_CENTER_LON", "15.64"))
    GEO_RADIUS_KM: float = float(os.getenv("GEO_RADIUS_KM", "2.0"))

    PROMET_FILTER: Tuple[str, ...] = (
        "rače", "race", "fram", "slivnica", "brunšvik", "brunsvik", "podova", "morje", "hoče", "hoce",
        "r2-430", "r3-711", "g1-2", "priključek slivnica", "razcep slivnica", "letališče maribor", "odcep za rače"
    )
    PROMET_BAN: Tuple[str, ...] = ("spuhlja", "ormož", "ormoz", "pesnica")

    GENERIC_STREET_WORDS: Tuple[str, ...] = ("cesta", "ceste", "cesti", "ulica", "ulice", "ulici", "pot", "trg", "naselje", "območje", "obmocje")
    GENERIC_PREPS: Tuple[str, ...] = ("na", "v", "za", "ob", "pod", "pri", "nad", "do", "od", "k", "proti")

    # Naselja/uporabni kraji (za “ulica + naselje”)
    AREA_WORDS: Tuple[str, ...] = (
        "fram","race","rače","podova","slivnica","brunšvik","brunsvik","pozeg","požeg","morje",
        "spodnja gorica","zgornja gorica","gorica","jesenca","jesenec"
    )

    WASTE_FUZZ: float = float(os.getenv("WASTE_FUZZ", "0.82"))  # malo bolj popustljivo
    LOG_MAX_MB: int = int(os.getenv("LOG_MAX_MB", "5"))

    WASTE_VARIANTS: Dict[str, List[str]] = field(default_factory=lambda: {
        "Biološki odpadki": ["bio", "bio odpadki", "bioloski odpadki", "biološki odpadki"],
        "Mešani komunalni odpadki": ["mešani komunalni odpadki", "mesani komunalni odpadki", "mešani", "mesani", "komunalni odpadki"],
        "Odpadna embalaža": ["odpadna embalaža", "odpadna embalaza", "embalaža", "rumena kanta", "rumene kante"],
        "Papir in karton": ["papir", "karton", "papir in karton"],
        "Steklena embalaža": ["steklo", "steklena embalaža", "steklena embalaza", "steklena"]
    })

    INTENT_URL_RULES: Dict[str, Dict[str, Set[str]]] = field(default_factory=lambda: {
        "komunalni_prispevek": {"must": {"komunalni", "prispevek"}, "ban": set()},
        "gradbeno":            {"must": {"gradben", "dovoljen"},   "ban": set()},
        "camp":                {"must": {"poletni", "tabor"} | {"varstvo", "pocitnisko", "počitnisko"}, "ban": set()},
        "fram_info":           {"must": {"fram"}, "ban": {"pgd", "gasil"}},
        "pgd":                 {"must": {"pgd", "gasil"}, "ban": set()},
        "awards":              {"must": {"nagrade", "priznanja", "petic"}, "ban": set()},
        "zapora_vloga":        {"must": {"zapora", "ceste", "vloga", "obrazec"}, "ban": {"promet", "roadworks", "geojson"}},
    })

    FALLBACK_CONTACT: str = os.getenv("FALLBACK_CONTACT", "Občina Rače-Fram: 02 609 60 10")
    ROAD_CLOSURE_FORM_URL: str = os.getenv("ROAD_CLOSURE_FORM_URL", "https://www.race-fram.si/objava/400297")
    EUPRAVA_GRADBENO_URL: str = os.getenv("EUPRAVA_GRADBENO_URL",
        "https://e-uprava.gov.si/si/podrocja/nepremicnine-in-okolje/nepremicnine-stavbe/gradbeno-dovoljenje.html?lang=si")

    POLNILNICE_NOTE: str = os.getenv("POLNILNICE_NOTE",
        "Da – polnilnici v Račah in Framu delujeta (nameščeni novi polnilnici).")

    def __post_init__(self):
        object.__setattr__(self, 'CHROMA_DB_PATH', os.path.join(self.DATA_DIR, "chroma_db"))
        object.__setattr__(self, 'LOG_FILE_PATH', os.path.join(self.DATA_DIR, "zupan_pogovori.jsonl"))
        os.makedirs(self.DATA_DIR, exist_ok=True)

cfg = Config()

missing = [e for e in cfg.REQUIRED_ENVS if not os.getenv(e)]
if missing:
    sys.stderr.write(f"[ERROR] Manjkajoče okoljske spremenljivke: {', '.join(missing)}\n")
    sys.exit(1)

if not (cfg.NAP_USERNAME and cfg.NAP_PASSWORD):
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("VirtualniZupan").warning("NAP poverilnice manjkajo – funkcija prometa bo omejena.")

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
# 3) UTIL
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

def strip_generics(s: str) -> str:
    tokens = normalize_text(s).split()
    stop = set(normalize_text(" ".join(cfg.GENERIC_STREET_WORDS)).split()) | set(normalize_text(" ".join(cfg.GENERIC_PREPS)).split())
    kept = [t for t in tokens if t not in stop]
    return " ".join(kept)

def gen_street_keys(name: str) -> List[str]:
    base = strip_generics(name)
    base = base.replace("bistriška", "bistriska")
    keys = {base}
    # -ska/-ski/-ske
    if base.endswith("ska"): keys.add(base[:-3] + "ski"); keys.add(base[:-3] + "ske")
    if base.endswith("ski"): keys.add(base[:-3] + "ska"); keys.add(base[:-3] + "ske")
    if base.endswith("ske"): keys.add(base[:-3] + "ska"); keys.add(base[:-3] + "ski")
    # -a/-e/-i
    if base.endswith("a"):
        keys.add(base[:-1] + "i"); keys.add(base[:-1] + "e")
    if base.endswith("i"):
        keys.add(base[:-1] + "a"); keys.add(base[:-1] + "e")
    if base.endswith("e"):
        keys.add(base[:-1] + "a"); keys.add(base[:-1] + "i")
    return [k for k in keys if k]

def parse_dates_from_text(text: str) -> List[str]:
    dates = re.findall(r'(\d{1,2}\.\d{1,2}\.?)', text)
    seen = set(); out = []
    for d in dates:
        dn = d if d.endswith('.') else d + '.'
        if dn not in seen:
            seen.add(dn); out.append(dn)
    return out

def get_canonical_waste(text: str) -> Optional[str]:
    norm = normalize_text(text)
    if re.search(r'stekl|stelk', norm): return "Steklena embalaža"
    if "papir" in norm or "karton" in norm: return "Papir in karton"
    if "bio" in norm or "biolo" in norm: return "Biološki odpadki"
    if re.search(r'\bme[sš]an(i|i\s+komunalni|i\s+odpadki)?\b', norm): return "Mešani komunalni odpadki"
    if (("rumen" in norm and "kant" in norm) or "embala" in norm) and not re.search(r'stekl|stelk', norm): return "Odpadna embalaža"
    for canon, variants in cfg.WASTE_VARIANTS.items():
        for v in variants:
            if normalize_text(v) in norm:
                return canon
    for canon, variants in cfg.WASTE_VARIANTS.items():
        for v in variants:
            if _ratio(norm, normalize_text(v)) >= 0.84:
                return canon
    return None

def tokens_from_text(s: str) -> set:
    return {t for t in re.split(r'[^a-z0-9]+', normalize_text(s)) if len(t) > 2}

# -------------------- Lepa oblika izpisa (brez alinej) ------------------------
def render_block(title: Optional[str], rows: List[Tuple[str, str]]) -> str:
    if cfg.RENDER_STYLE == "plain":
        head = f"{title}\n" if title else ""
        body = "\n".join(f"{k}: {v}" for k, v in rows)
        return (head + body).strip()
    # pretty (markdown-ish, a brez alinej)
    head = f"**{title}**\n" if title else ""
    body = "\n".join(f"**{k}:** {v}" for k, v in rows)
    return (head + body).strip()

# ------------------------------------------------------------------------------
# INTENTI
# ------------------------------------------------------------------------------
def _has_transport_place(q_norm: str) -> bool:
    return bool(re.search(r'\b(os\s*fram|os\s*rac?e|kopivnik|morje|slivnic\w*|brun[sš]?vik|fram|rac?e)\b', q_norm))

def _looks_like_transport_followup(q_norm: str) -> bool:
    return bool(
        re.search(r'\b(zjutraj|popoldne)\b', q_norm) or
        (_has_transport_place(q_norm) and re.search(r'\b(od|iz|do|v|na)\b', q_norm)) or
        (_has_transport_place(q_norm) and len(q_norm.split()) <= 3)
    )

def detect_intent_qna(q_norm: str, last_intent: Optional[str] = None) -> str:
    if any(k in q_norm for k in ['kontakt','kontaktna','stevilka','številka','telefon','e mail','email','mail','naslov','pisarna']): return 'contact'
    if re.search(r'\bpolnilnic\w*|\belektri\w*\s+polniln\w*|\bev\s*polniln\w*', q_norm): return 'ev_charging'
    if any(k in q_norm for k in cfg.KLJUCNE_ODPADKI) or get_canonical_waste(q_norm) or ('naslednji' in q_norm): return 'waste'
    if last_intent == 'waste' and (re.match(r'^\s*(kaj pa|pa)\b', q_norm) or re.search(r'\b(od|iz|v|na)\b', q_norm)): return 'waste'
    if re.search(r'\bzapor\w*', q_norm) and re.search(r'\bvlog\w*|\bobrazec\w*', q_norm): return "zapora_vloga"
    if (re.search(r'\b(zlat\w*\s+peti\w*|petic\w*|peti\b)', q_norm) or re.search(r'\bzupanov\w*\s+nagrad\w*', q_norm) or
        re.search(r'\bzupanov\w*\s+priznanj\w*', q_norm) or re.search(r'\bzupanova?\s+nagrada', q_norm) or 'nagrade' in q_norm or 'priznanj' in q_norm):
        if re.search(r'\b(19\d{2}|20\d{2})\b', q_norm): return 'awards'
    if re.search(r'\bkdo je\b', q_norm) and re.search(r'\bzupan\b', q_norm): return 'who_is_mayor'
    if any(k in q_norm for k in cfg.KLJUCNE_PROMET): return 'traffic'
    has_transport_hint = any(k in q_norm for k in cfg.TRANSPORT_HINTS)
    if has_transport_hint or re.search(r'\bod\s+(os\s*fram|os\s*rac?e|kopivnik|morje|slivnic\w*|brun[sš]?vik|fram|rac?e)\s+(do|v|na)\s+', q_norm): return 'transport'
    if last_intent == 'transport' and _looks_like_transport_followup(q_norm): return 'transport'
    if 'gradben' in q_norm and 'dovoljen' in q_norm: return 'gradbeno'
    if 'poletn' in q_norm and ('tabor' in q_norm or 'kamp' in q_norm or 'varstvo' in q_norm): return 'camp'
    if 'pgd' in q_norm or 'gasil' in q_norm: return 'pgd'
    if 'fram' in q_norm and not any(k in q_norm for k in ['odpad', 'promet', 'tabor', 'kamp', 'prispev', 'gradben', 'pgd', 'gasil', 'zapora', 'nagrad']): return 'fram_info'
    return 'general'

def map_award_alias(q_norm: str) -> str:
    if re.search(r'\bpeti\w*|zlat\w*\s+peti\w*', q_norm): return "zlata_petica"
    if (re.search(r'priznan\w*\s+zupan\w*', q_norm) or re.search(r'zupanov\w*\s+nagrad\w*', q_norm) or re.search(r'zupanova?\s+nagrada', q_norm)): return "priznanje_zupana"
    if "nagrad" in q_norm or "priznanj" in q_norm: return "all"
    return "any"

# ------------------------------------------------------------------------------
# 4) GLAVNI RAZRED
# ------------------------------------------------------------------------------
class VirtualniZupan:
    def __init__(self) -> None:
        prefix = "PRODUCTION" if cfg.ENV_TYPE == 'production' else "DEVELOPMENT"
        logger.info(f"[{prefix}] VirtualniŽupan v56.0 inicializiran.")
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.collection: Optional[chromadb.Collection] = None
        self.zgodovina_seje: Dict[str, Dict[str, Any]] = {}
        self._nap_access_token: Optional[str] = None
        self._nap_token_expiry: Optional[datetime] = None

        self._http = requests.Session()
        retries = Retry(total=2, backoff_factor=0.2, status_forcelist=[500, 502, 504], allowed_methods=frozenset(["GET","POST"]))
        self._http.mount("https://", HTTPAdapter(max_retries=retries))

        self._promet_cache: Optional[List[Dict]] = None
        self._promet_cache_ts: Optional[datetime] = None

        self._street_index: Dict[str, List[Dict[str, Any]]] = {}
        self._area_type_docs: Dict[Tuple[str, str], str] = {}
        self._street_keys_list: List[str] = []

        self._awards_by_year: Dict[int, Dict[str, List[str]]] = {}
        self._pgd_contacts: Dict[str, Dict[str, str]] = {}

    # ---- LLM (opcijsko)
    def _call_llm(self, prompt: str, **kwargs) -> str:
        if not cfg.USE_LLM:
            return "Žal o tej temi nimam dovolj podatkov."
        try:
            client = self.openai_client.with_options(timeout=cfg.OPENAI_TIMEOUT_S)
            model_lower = (cfg.GENERATOR_MODEL or "").lower()
            is_gpt5_like = "gpt-5" in model_lower or model_lower.startswith("gpt5") or "o4" in model_lower
            token_limit = kwargs.get("max_tokens", 350)
            create_kwargs = {"model": cfg.GENERATOR_MODEL, "messages": [{"role": "user", "content": prompt}]}
            if is_gpt5_like:
                create_kwargs["max_completion_tokens"] = token_limit
            else:
                create_kwargs["max_tokens"] = token_limit
                create_kwargs["temperature"] = 0.0
            res = client.chat.completions.create(**create_kwargs)
            content = (res.choices[0].message.content or "").strip()
            return content if content else "Žal o tej temi nimam dovolj podatkov."
        except (OpenAIError, Exception):
            return "Žal o tej temi nimam dovolj podatkov."

    # ---- Baza
    def nalozi_bazo(self) -> None:
        if self.collection:
            return
        try:
            ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name=cfg.EMBEDDING_MODEL
            )
            client = chromadb.PersistentClient(path=cfg.CHROMA_DB_PATH)
            self.collection = client.get_collection(name=cfg.COLLECTION_NAME, embedding_function=ef)
            self._build_waste_indexes()
            self._build_awards_index()
            self._build_pgd_contacts()
        except Exception:
            self.collection = None

    def _ensure_log_rotation(self):
        try:
            if os.path.exists(cfg.LOG_FILE_PATH) and os.path.getsize(cfg.LOG_FILE_PATH) > cfg.LOG_MAX_MB * 1_000_000:
                base, ext = os.path.splitext(cfg.LOG_FILE_PATH)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                os.rename(cfg.LOG_FILE_PATH, f"{base}_{ts}{ext}")
        except Exception:
            pass

    def belezi_pogovor(self, session_id: str, vprasanje: str, odgovor: str) -> None:
        try:
            self._ensure_log_rotation()
            zapis = {"timestamp": datetime.now().isoformat(), "session_id": session_id, "vprasanje": vprasanje, "odgovor": odgovor}
            with open(cfg.LOG_FILE_PATH, 'a', encoding='utf-8') as f:
                f.write(json.dumps(zapis, ensure_ascii=False) + '\n')
        except Exception:
            pass

    # ---------------------- TRANSPORT ----------------------
    @staticmethod
    def _canon_place(tok: str) -> Optional[str]:
        t = normalize_text(tok)
        if re.search(r'\bos\s*fram\w*\b', t): return "OŠ Fram"
        if re.search(r'\bos\s*rac?e\w*\b', t): return "OŠ Rače"
        if re.search(r'\bkopivnik\w*\b', t):   return "Kopivnik"
        if re.search(r'\bmorje\w*\b', t):      return "Morje"
        if re.search(r'\bslivnic\w*\b', t):    return "Slivnica"
        if re.search(r'\bbrun[sš]?vik\w*\b', t): return "Brunšvik"
        if re.search(r'\bfram\w*\b', t):       return "Fram"
        if re.search(r'\brac?e\w*\b', t):      return "Rače"
        return None

    def _extract_route(self, q_norm: str, stanje: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        origin = None; dest = None; time_hint = None
        if re.search(r'\bzjutraj\b', q_norm): time_hint = "zjutraj"
        if re.search(r'\bpopoldne\b', q_norm): time_hint = "popoldne"
        place_rx = r'(os\s*fram\w*|os\s*rac?e\w*|kopivnik\w*|morje\w*|slivnic\w*|brun[sš]?vik\w*|fram\w*|rac?e\w*)'
        m1 = re.search(rf'\bod\s+{place_rx}.*?\b(do|v|na)\s+{place_rx}\b', q_norm)
        if m1:
            origin = self._canon_place(m1.group(1)); dest = self._canon_place(m1.group(3))
        if not origin or not dest:
            m2o = re.search(rf'\b(iz|od)\s+{place_rx}\b', q_norm)
            m2d = re.search(rf'\b(do|v|na)\s+{place_rx}\b', q_norm)
            if m2o: origin = origin or self._canon_place(m2o.group(2))
            if m2d: dest   = dest   or self._canon_place(m2d.group(2))
        if not dest:
            single = self._canon_place(q_norm)
            if single:
                if self._canon_place(stanje.get('transport_last_origin') or ""):
                    dest = single
                else:
                    if single in ("OŠ Fram", "OŠ Rače"): origin = origin or single
                    else: dest = single
        if not origin: origin = self._canon_place(stanje.get('transport_last_origin') or "")
        if not dest:   dest   = self._canon_place(stanje.get('transport_last_dest') or "")
        return origin, dest, time_hint

    def _answer_transport(self, q_raw: str, q_norm: str, stanje: Dict[str, Any]) -> str:
        origin, dest, time_hint = self._extract_route(q_norm, stanje)
        if not origin and not dest:
            stanje['namen'] = 'transport'; stanje['transport_waiting'] = 'route'
            return "Relacija? primer: od OŠ Fram do Kopivnika"
        if origin and not dest:
            stanje['namen'] = 'transport'; stanje['transport_last_origin'] = origin; stanje['transport_waiting'] = 'dest'
            return "Kam želiš? primer: v Kopivnik ali v Morje"
        if dest and not origin:
            stanje['namen'] = 'transport'; stanje['transport_last_dest'] = dest; stanje['transport_waiting'] = 'origin'
            return "Iz kje? primer: iz OŠ Fram ali iz OŠ Rače"

        stanje['transport_last_origin'] = origin
        stanje['transport_last_dest']   = dest
        stanje['namen'] = 'transport'
        stanje.pop('transport_waiting', None)

        def is_pair(a, b, A, B):
            return (normalize_text(a) == normalize_text(A) and normalize_text(b) == normalize_text(B))

        if (is_pair(origin, dest, "OŠ Fram", "Kopivnik") or is_pair(origin, dest, "Kopivnik", "OŠ Fram")):
            rows = []
            if not time_hint or time_hint == "zjutraj": rows.append(("Zjutraj", "7.25"))
            if not time_hint or time_hint == "popoldne": rows.append(("Popoldne", "14.40"))
            rows.append((cfg.TRANSPORT_CONTACT_TEXT, cfg.TRANSPORT_CONTACT_URL))
            return render_block(f"Prevoz {origin} → {dest}", rows)

        # fallback iz baze (če imamo kaj uporabnega)
        if self.collection:
            q_texts = [f"prevoz {origin} {dest} vozni red odhod prihod", f"prevozi {origin} {dest} minibus avtobus"]
            res = self.collection.query(query_texts=q_texts, n_results=5, include=["documents","metadatas"])
            docs = (res.get("documents") or [[]])[0]
            if len(res.get("documents", [])) > 1: docs += (res.get("documents") or [[]])[1]
            pairs = []
            metas = (res.get("metadatas") or [[]])[0]
            if len(res.get("metadatas", [])) > 1: metas += (res.get("metadatas") or [[]])[1]
            for d, m in zip(docs, metas):
                low = normalize_text(d + " " + json.dumps(m, ensure_ascii=False))
                if any(k in low for k in ("prevoz","vozni red","prevozi","minibus","avtobus")):
                    pairs.append((d, m))
            if pairs:
                times = re.findall(r'\b(\d{1,2}[:\.]\d{2})\b', " \n ".join(p[0] for p in pairs))
                times = [t.replace(':','.') for t in times]
                uniq = []
                for t in times:
                    if t not in uniq: uniq.append(t)
                if uniq:
                    morn = [t for t in uniq if int(t.split('.')[0]) < 12]
                    aft  = [t for t in uniq if int(t.split('.')[0]) >= 12]
                    rows = []
                    if time_hint in (None, "zjutraj") and morn: rows.append(("Zjutraj", ", ".join(morn)))
                    if time_hint in (None, "popoldne") and aft: rows.append(("Popoldne", ", ".join(aft)))
                    rows.append((cfg.TRANSPORT_CONTACT_TEXT, cfg.TRANSPORT_CONTACT_URL))
                    return render_block(f"Prevoz {origin} → {dest}", rows)

        return render_block(f"Prevoz {origin} → {dest}", [("Opomba", "Nimam točnih ur. Preveri pri organizatorju."), (cfg.TRANSPORT_CONTACT_TEXT, cfg.TRANSPORT_CONTACT_URL)])

    # ---------------------- NAP / PROMET ----------------------
    def _ensure_nap_token(self) -> Optional[str]:
        if not (cfg.NAP_USERNAME and cfg.NAP_PASSWORD):
            return None
        if self._nap_access_token and self._nap_token_expiry and datetime.utcnow() < self._nap_token_expiry - timedelta(seconds=60):
            return self._nap_access_token
        try:
            payload = {'grant_type': 'password', 'username': cfg.NAP_USERNAME, 'password': cfg.NAP_PASSWORD}
            response = self._http.post(cfg.NAP_TOKEN_URL, data=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            self._nap_access_token = data['access_token']
            self._nap_token_expiry = datetime.utcnow() + timedelta(seconds=data.get('expires_in', 3600))
            return self._nap_access_token
        except requests.RequestException:
            return None

    def _iter_geo_coords(self, geometry: dict):
        if not geometry: return
        gtype = geometry.get("type"); coords = geometry.get("coordinates")
        if not coords: return
        def yx(p): return (p[1], p[0])
        if gtype == "Point": yield yx(coords)
        elif gtype == "MultiPoint":
            for p in coords: yield yx(p)
        elif gtype == "LineString":
            for p in coords: yield yx(p)
        elif gtype == "MultiLineString":
            for line in coords:
                for p in line: yield yx(p)
        elif gtype == "Polygon":
            for ring in coords:
                for p in ring: yield yx(p)
        elif gtype == "MultiPolygon":
            for poly in coords:
                for ring in poly:
                    for p in ring: yield yx(p)

    def _geometry_near_municipality(self, geometry: dict) -> bool:
        lat_min, lat_max = cfg.GEO_LAT_MIN, cfg.GEO_LAT_MAX
        lon_min, lon_max = cfg.GEO_LON_MIN, cfg.GEO_LON_MAX
        pts = list(self._iter_geo_coords(geometry))
        for (lat, lon) in pts:
            if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
                return True
        from math import radians, sin, cos, asin, sqrt
        R = 6371.0
        clat, clon = radians(cfg.GEO_CENTER_LAT), radians(cfg.GEO_CENTER_LON)
        for (lat, lon) in pts:
            la, lo = radians(lat), radians(lon)
            dlat = la - clat; dlon = lo - clon
            a = sin(dlat/2)**2 + cos(clat)*cos(la)*sin(dlon/2)**2
            c = 2 * asin(sqrt(a)); dist = R * c
            if dist <= cfg.GEO_RADIUS_KM: return True
        return False

    def preveri_zapore_cest(self) -> str:
        now = datetime.utcnow()
        if self._promet_cache and self._promet_cache_ts and now - self._promet_cache_ts < timedelta(minutes=cfg.NAP_CACHE_MIN):
            vsi_dogodki = self._promet_cache
        else:
            token = self._ensure_nap_token()
            if not token:
                return "Promet: dostop do podatkov trenutno ni mogoč."
            try:
                headers = {"Authorization": f"Bearer {token}"}
                resp = self._http.get(cfg.NAP_DATA_URL, headers=headers, timeout=12)
                resp.raise_for_status()
                payload = resp.json()
                vsi_dogodki = payload.get("features", []) if isinstance(payload, dict) else []
                self._promet_cache = vsi_dogodki
                self._promet_cache_ts = now
            except requests.RequestException:
                return "Promet: storitev ne deluje."

        relevantne = []
        for d in vsi_dogodki:
            props = d.get("properties", {}) or {}
            geom  = d.get("geometry", {}) or {}
            cel = " ".join(str(props.get(polje, "")).lower() for polje in ("cesta", "opis", "imeDogodka"))
            geo_ok = self._geometry_near_municipality(geom)
            has_anchor = any(k in cel for k in cfg.PROMET_FILTER)
            has_ban    = any(b in cel for b in cfg.PROMET_BAN)
            if (geo_ok or has_anchor) and not has_ban:
                relevantne.append(props)

        if not relevantne:
            return "Stanje na cestah: ni zabeleženih zapor ali zastojev (promet.si)."

        ts = datetime.now().strftime("%d.%m.%Y %H:%M")
        rows = [("Vir", f"NAP/promet.si, {ts}")]
        out = [f"Stanje na cestah ({ts})"]
        lines = []
        for z in relevantne[:10]:
            cesta = z.get("cesta") or "Ni podatka"
            opis  = z.get("opis")  or "Ni podatka"
            lines.append(f"Cesta: {cesta} | Opis: {opis}")
        return (render_block("Stanje na cestah", [("Vir", "NAP/promet.si")] ) + "\n" + "\n".join(lines)).strip()

    # ---------------------- ODPADKI ----------------------
    def _expand_street_variants(self, s_norm: str) -> Set[str]:
        out: Set[str] = set()
        base = strip_generics(s_norm)
        if not base: return out
        toks = base.split()

        # osnovne variante + “ulica + naselje”, prvi token, predpone
        for k in gen_street_keys(base):
            out.add(k)
            # predponsko ujemanje npr. bistrisk* → bistriska/bistriski/...
            parts = k.split()
            if parts:
                out.add(parts[0])
                if len(parts) >= 2:
                    out.add(f"{parts[0]} {parts[-1]}")

        return {k for k in out if k}

    def _street_prefix_match(self, needle: str, hay: str) -> bool:
        # predponsko ujemanje na nivoju tokenov (bistrisk ~ bistriska, bistriski)
        nt = needle.split()
        ht = hay.split()
        if not nt or not ht: return False
        # poskusi ujemati prvi token
        return ht[0].startswith(nt[0][:5])  # 5 znakov praga je ok za “bistr*”

    def _build_waste_indexes(self) -> None:
        if not self.collection: return
        all_docs = self.collection.get(where={"kategorija": "Odvoz odpadkov"})
        if not all_docs or not all_docs.get('ids'): return

        self._street_index.clear()
        self._area_type_docs.clear()

        for i in range(len(all_docs['ids'])):
            meta = all_docs['metadatas'][i]
            doc_text = all_docs['documents'][i]
            tip_meta = meta.get('tip_odpadka', '') or ''
            tip = get_canonical_waste(tip_meta) or tip_meta
            area = normalize_text(meta.get('obmocje', '') or '')
            self._area_type_docs[(area, tip)] = doc_text

            # “naselja” pogosto vsebuje “Bistriška cesta, Fram; …”
            raw = meta.get('naselja', '') or ''
            # razbij po ; ali \n, vejice pustimo za kombinacijo
            chunks = [c.strip() for c in re.split(r'[;\n]+', raw) if c.strip()]
            for ch in chunks:
                ch_norm = normalize_text(ch).replace("bistriška", "bistriska")
                display = ch_norm
                # variacije: polno, brez generičnih, prvi token, ulica + naselje
                # če je “x, fram” → naredi “x fram”
                if "," in ch_norm:
                    left, right = [p.strip() for p in ch_norm.split(",", 1)]
                    if right in cfg.AREA_WORDS:
                        ch_norm = f"{left} {right}"
                        display = ch_norm
                for key in self._expand_street_variants(ch_norm):
                    self._street_index.setdefault(key, []).append({
                        "area": area, "tip": tip, "doc": doc_text, "display": display
                    })

        self._street_keys_list = list(self._street_index.keys())

    def _best_street_key_for_query(self, phrases: List[str]) -> Optional[str]:
        # 1) točna varianta
        for ph in phrases:
            for key in gen_street_keys(ph):
                if key in self._street_index:
                    return key
        # 2) predponsko (bistrisk* → bistriska/bistriski…)
        for ph in phrases:
            base = strip_generics(ph)
            if len(base) < 4: continue
            for key in self._street_keys_list:
                if self._street_prefix_match(base, key):
                    return key
        # 3) “vsebuje” (npr. “bistriska fram” vs “bistriska”)
        for ph in phrases:
            base = strip_generics(ph)
            if len(base) < 4: continue
            if base in self._street_index: return base
            for key in self._street_keys_list:
                if f" {base} " in f" {key} ":
                    return key
        # 4) fuzzy
        best = (None, 0.0)
        for ph in phrases:
            base = strip_generics(ph)
            if len(base) < 3: continue
            for key in self._street_keys_list:
                sc = _ratio(base, key)
                if sc > best[1]: best = (key, sc)
        if best[0] and best[1] >= cfg.WASTE_FUZZ: return best[0]
        return None

    def _build_location_phrases(self, vprasanje_norm: str) -> List[str]:
        waste_stop = set()
        for variants in cfg.WASTE_VARIANTS.values():
            for v in variants: waste_stop.add(normalize_text(v))
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
                if p and p not in phrases: phrases.append(p)
        for t in toks:
            if len(t) >= 4 and t not in phrases: phrases.append(t)
        return phrases

    def _format_dates_for_tip(self, street_display: str, tip_odpadka: str, doc_text: str, only_next: bool) -> Optional[str]:
        dates = parse_dates_from_text(doc_text)
        if not dates: return None
        today = datetime.now().date()
        year = today.year
        next_date_fmt = None
        try:
            candidates = []
            for d in dates:
                dd, mm = d.replace('.',' ').strip().split()[:2]
                dt = datetime(year, int(mm), int(dd)).date()
                if dt < today:
                    dt = datetime(year + 1, int(mm), int(dd)).date()
                candidates.append((dt, f"{int(dd)}.{int(mm)}."))
            candidates.sort(key=lambda x: x[0])
            if candidates:
                next_date_fmt = candidates[0][1]
        except Exception:
            next_date_fmt = None

        if only_next and next_date_fmt:
            return render_block(f"Naslednji odvoz – {street_display.title()}", [(tip_odpadka, next_date_fmt)])

        rows = [("Tip", tip_odpadka)]
        if next_date_fmt: rows.append(("Naslednji odvoz", next_date_fmt))
        rows.append(("Vsi termini", ", ".join(dates)))
        return render_block(f"Odvoz odpadkov – {street_display.title()}", rows)

    def obravnavaj_odvoz_odpadkov(self, uporabnikovo_vprasanje: str, session_id: str) -> str:
        ses = self.zgodovina_seje.setdefault(session_id, {'zgodovina': [], 'stanje': {}})
        stanje = ses['stanje']
        if not self.collection: return "Baza urnikov ni na voljo."

        vprasanje_norm = normalize_text(uporabnikovo_vprasanje)
        only_next = "naslednji" in vprasanje_norm
        wanted_tip = get_canonical_waste(vprasanje_norm)
        if not wanted_tip and only_next and stanje.get('zadnji_tip'):
            wanted_tip = stanje['zadnji_tip']

        phrases = self._build_location_phrases(vprasanje_norm)
        if not phrases and stanje.get('zadnja_lokacija_norm'):
            phrases = [stanje['zadnja_lokacija_norm']]
        if not phrases:
            stanje['namen'] = 'odpadki'; stanje['caka_na'] = 'lokacija'
            return "Za katero ulico/naselje te zanima urnik? primer: Bistriška cesta, Fram"

        key = self._best_street_key_for_query(phrases)
        if not key:
            if not wanted_tip and stanje.get('zadnji_tip'):
                wanted_tip = stanje['zadnji_tip']
            all_areas = {e["area"] for v in self._street_index.values() for e in v if e.get("area")}
            area_hit = None
            for a in all_areas:
                if a and a in vprasanje_norm: area_hit = a; break
            if area_hit and wanted_tip:
                doc = self._area_type_docs.get((area_hit, wanted_tip))
                if doc:
                    ans = self._format_dates_for_tip(area_hit, wanted_tip, doc, only_next)
                    if ans:
                        stanje['zadnja_lokacija_norm'] = strip_generics(area_hit)
                        stanje['zadnji_tip'] = wanted_tip
                        stanje['namen'] = 'odpadki'; stanje.pop('caka_na', None)
                        return ans
            if not wanted_tip:
                stanje['zadnja_lokacija_norm'] = phrases[0]
                stanje['namen'] = 'odpadki'; stanje['caka_na'] = 'tip'
                return "Kateri tip odpadkov te zanima? izberi: bio, mešani, embalaža, papir, steklo"
            return "Za to ulico ali območje nimam najdenega urnika za ta tip."

        entries = self._street_index.get(key, [])
        if not entries:
            return "Za navedeno ulico nimam podatkov."

        best_entry = None
        if wanted_tip:
            for e in entries:
                if get_canonical_waste(e["tip"]) == wanted_tip:
                    best_entry = e; break
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
                            stanje['namen'] = 'odpadki'; stanje.pop('caka_na', None)
                            return ans
                return "Za to ulico nisem našel urnika za izbrani tip."
        else:
            best_entry = entries[0]

        tip_canon = get_canonical_waste(best_entry["tip"]) or best_entry["tip"]
        ans = self._format_dates_for_tip(best_entry["display"], tip_canon, best_entry["doc"], only_next)
        if not ans: return "Žal ne najdem datumov v urniku."

        stanje['zadnja_lokacija_norm'] = strip_generics(best_entry["display"])
        stanje['zadnji_tip'] = tip_canon
        stanje['namen'] = 'odpadki'; stanje.pop('caka_na', None)
        return ans

    # ---------------------- NAGRADE + PGD (okrajšano – enako kot prej) --------
    def _clean_bullet(self, line: str) -> str:
        ln = line.strip()
        ln = re.sub(r'^\s*[-–—\*•]+\s*', '', ln)
        ln = re.sub(r'^\s*\*\*\s*', '', ln)
        ln = re.sub(r'\*\*', '', ln)
        return ln.strip()

    def _split_names(self, chunk: str) -> List[str]:
        raw = [p.strip(" .;") for p in re.split(r'[;,]\s*', chunk) if p.strip()]
        return [re.sub(r'^\*+', '', r).strip() for r in raw if r and len(r) > 1]

    def _add_award(self, year: int, key: str, items: List[str]):
        if not year or not items: return
        y = self._awards_by_year.setdefault(year, {})
        y.setdefault(key, [])
        for it in items:
            if it and it not in y[key]: y[key].append(it)

    def _parse_zlata_petica_block(self, year: int, block: str):
        block = re.sub(r'\*\*(.*?)\*\*', r'\1', block)
        inline_matches_race = re.findall(r'^\s*[-–•]?\s*o[sš]\s*ra[cč]e\s*:?\s*(.+)$', block, flags=re.IGNORECASE | re.MULTILINE)
        inline_matches_fram = re.findall(r'^\s*[-–•]?\s*o[sš]\s*fram\s*:?\s*(.+)$', block, flags=re.IGNORECASE | re.MULTILINE)
        race_items, fram_items = [], []
        for m in inline_matches_race: race_items.extend(self._split_names(m))
        for m in inline_matches_fram: fram_items.extend(self._split_names(m))

        def collect_following_lines(school_regex: str) -> List[str]:
            items: List[str] = []
            pattern = re.compile(rf'^(?P<head>\s*o[sš]\s*{school_regex}\s*:?\s*)(?P<rest>.*)$', flags=re.IGNORECASE | re.MULTILINE)
            for m in pattern.finditer(block):
                head_end = m.end(); rest = m.group('rest').strip()
                if rest:
                    items.extend(self._split_names(rest)); continue
                tail = block[head_end:]
                for ln in tail.splitlines():
                    if not ln.strip(): break
                    if re.match(r'^\s*(o[sš]\s*ra[cč]e|o[sš]\s*fram)\s*:?\s*$', ln, flags=re.IGNORECASE): break
                    m_b = re.match(r'^\s*[-–•]\s*(.+)$', ln)
                    if m_b: items.extend(self._split_names(m_b.group(1)))
                    else: items.extend(self._split_names(ln))
                break
            return items

        if not race_items: race_items = collect_following_lines(r'ra[cč]e')
        if not fram_items: fram_items = collect_following_lines(r'fram')

        if race_items: self._add_award(year, "zlata_petica_os_race", race_items)
        if fram_items: self._add_award(year, "zlata_petica_os_fram", fram_items)
        if not race_items and not fram_items:
            lines = [self._clean_bullet(ln) for ln in block.splitlines()]; lines = [ln for ln in lines if ln]
            generic: List[str] = []
            for ln in lines:
                if re.match(r'^\s*o[sš]\s*(ra[cč]e|fram)\s*:?\s*$', ln, flags=re.IGNORECASE): continue
                generic.extend(self._split_names(ln))
            if generic: self._add_award(year, "zlata_petica", generic)

    def _build_awards_index(self) -> None:
        if not self.collection: return
        got = self.collection.get(where={"kategorija": "Nagrade in Priznanja"})
        if not got or not got.get("ids"): return
        for i in range(len(got["ids"])):
            doc = got["documents"][i]; text = doc
            years = [int(y) for y in re.findall(r'\b(19\d{2}|20\d{2})\b', text)]
            year = max(years) if years else None
            if not year: continue
            m_zp = re.search(r"(prejemniki\s+zlat\w*\s+petic\w*|zlat\w*\s+petic\w*)\s*:\s*(.*?)(?:\n\s*\n|$)", text, flags=re.IGNORECASE | re.DOTALL)
            if m_zp: self._parse_zlata_petica_block(year, m_zp.group(2).strip())
            m_pz = re.search(r"(priznanje\s+župana|priznanje\s+zupana|zupanova\s+nagrada|zupanove\s+nagrade)\s*:\s*(.*?)(?:\n\s*\n|$)", text, flags=re.IGNORECASE | re.DOTALL)
            if m_pz:
                names = []
                for ln in m_pz.group(2).splitlines():
                    ln0 = self._clean_bullet(ln)
                    if ln0: names.extend(self._split_names(ln0))
                self._add_award(year, "priznanje_zupana", names)

    def _answer_awards(self, q_norm: str) -> Optional[str]:
        years = [int(y) for y in re.findall(r'\b(19\d{2}|20\d{2})\b', q_norm)]
        year = years[0] if years else None
        if not year: return None
        data = self._awards_by_year.get(year) or {}
        if not data: return "Žal nimam podatkov za to leto."
        # kratko: samo glavne kategorije
        rows = []
        for k in ("zlata_petica_os_race","zlata_petica_os_fram","priznanje_zupana"):
            if data.get(k):
                rows.append((k.replace("_"," ").title(), ", ".join(data[k])))
        return render_block(f"Nagrade in priznanja {year}", rows) if rows else "Žal nimam podatkov za to leto."

    # ---------------------- PGD ----------------------
    def _build_pgd_contacts(self) -> None:
        if not self.collection: return
        queries = ["pgd gasilsko društvo kontakti rače fram email telefon", "prostovoljno gasilsko društvo Rače Fram Podova Gorica kontakt"]
        seen = set(); possible = []
        for q in queries:
            res = self.collection.query(query_texts=[q], n_results=10, include=["documents","metadatas"])
            docs = res.get("documents",[[]])[0]
            metas = res.get("metadatas",[[]])[0]
            for d,m in zip(docs,metas):
                key = (d, json.dumps(m, ensure_ascii=False))
                if key in seen: continue
                seen.add(key)
                if "pgd" in normalize_text(d) or "gasil" in normalize_text(d):
                    possible.append((d,m))
        names = ["PGD Rače","PGD Fram","PGD Podova"]
        for n in names:
            self._pgd_contacts.setdefault(n.lower(), {"ime": n})
        for d,m in possible:
            low = normalize_text(d)
            for key in list(self._pgd_contacts.keys()):
                if normalize_text(self._pgd_contacts[key]["ime"]) in low:
                    email = None; telefon = None
                    if isinstance(m, dict):
                        email = m.get("email") or m.get("e_posta")
                        telefon = m.get("telefon") or m.get("tel")
                    if not email:
                        m_email = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', d)
                        if m_email: email = m_email.group(0)
                    if not telefon:
                        m_tel = re.search(r'(\+?\d[\d\s/.-]{6,})', d)
                        if m_tel: telefon = m_tel.group(1).strip()
                    if email: self._pgd_contacts[key]["email"] = email
                    if telefon: self._pgd_contacts[key]["telefon"] = telefon

    def _answer_pgd_list(self) -> str:
        order = ["PGD Rače","PGD Fram","PGD Podova"]
        lines = []
        for n in order:
            key = n.lower()
            info = self._pgd_contacts.get(key, {"ime": n})
            line = info.get('ime', n)
            if info.get("telefon"): line += f" — tel.: {info['telefon']}"
            if info.get("email"):   line += f" — e-pošta: {info['email']}"
            lines.append(line)
        return "Gasilska društva v občini Rače-Fram\n" + "\n".join(lines)

    # ---------------------- CAMP (poletni tabor) ----------------------
    def _answer_camp(self) -> str:
        if not self.collection:
            return "Podatek o poletnem taboru ni na voljo."
        res = self.collection.query(query_texts=["poletni tabor občine rače-fram termini cena svizec"], n_results=5, include=["documents","metadatas"])
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        best_doc = ""; best_url = ""
        for d,m in zip(docs,metas):
            low = normalize_text(d)
            if "poletni tabor" in low and ("termin" in low or "cena" in low):
                best_doc = d; best_url = (m or {}).get("source_url",""); break
        if not best_doc and docs:
            best_doc = docs[0]; best_url = (metas[0] or {}).get("source_url","")
        if not best_doc:
            return "Trenutno nimam objave o poletnem taboru."
        # grobo izlušči termin, ure, cene
        term_line = re.search(r'(od\s+\d{1,2}\.\s*\w+.*?do\s+\d{1,2}\.\s*\w+|\d{1,2}\.\s*\d{1,2}\.\s*do\s*\d{1,2}\.\s*\d{1,2}\.)', best_doc, flags=re.IGNORECASE)
        hours = re.search(r'(\b\d{1,2}[:\.]\d{2}\b)\s*[-–]\s*(\b\d{1,2}[:\.]\d{2}\b)|od\s*(\d{1,2}\.?\d{2})\s*do\s*(\d{1,2}\.?\d{2})', best_doc, flags=re.IGNORECASE)
        price1 = re.search(r'cena\s*1\D*(\d{1,3}[.,]\d{2})', best_doc, flags=re.IGNORECASE)
        price2 = re.search(r'cena\s*2\D*(\d{1,3}[.,]\d{2})', best_doc, flags=re.IGNORECASE)
        rows = []
        if term_line: rows.append(("Termini", term_line.group(0)))
        if hours:
            hh = hours.groups()
            rows.append(("Ure", f"{hh[0] or hh[2]}–{hh[1] or hh[3]}".replace('.',':')))
        if price1: rows.append(("Cena 1 (RF otroci)", price1.group(1).replace(',', '.').replace('.', ',') + "€"))
        if price2: rows.append(("Cena 2 (ostali)", price2.group(1).replace(',', '.').replace('.', ',') + "€"))
        if best_url: rows.append(("Povezava", best_url))
        return render_block("Poletni tabor", rows or [("Info", "Najdena objava, podatki v povezavi." )])

    # ---------------------- RAG PROMPT (če USE_LLM=True) ----------------------
    def _zgradi_rag_prompt(self, vprasanje: str, zgodovina: List[Tuple[str, str]]) -> Optional[Tuple[str, str]]:
        if not self.collection: return None
        q_norm = normalize_text(vprasanje)
        intent = detect_intent_qna(q_norm)
        if intent in ("awards","pgd","zapora_vloga","waste","traffic","who_is_mayor","transport","contact","komunalni_prispevek","ev_charging","camp"):
            return None
        results = self.collection.query(query_texts=[q_norm], n_results=cfg.RAG_TOPK, include=["documents", "metadatas"])
        docs = results.get('documents', [[]])[0]; metas = results.get('metadatas', [[]])[0]
        filtered_docs, filtered_metas = [], []
        for d, m in zip(docs, metas):
            if (m.get('kategorija','') or '').lower() == 'odvoz odpadkov': continue
            filtered_docs.append(d); filtered_metas.append(m)
        if not filtered_docs: return None
        context = "\n---\n".join(filtered_docs[:3])[:6000]
        now = datetime.now()
        zgodovina_str = "\n".join([f"U: {q}\nA: {a}" for q, a in zgodovina[-3:]])
        prompt = f"""Ti si Virtualni župan Občine Rače-Fram.
DATUM: {now.strftime('%d.%m.%Y')}
Odgovori kratko, berljivo, v nekaj vrsticah, brez alinej.

KONTEKST:
{context}

ZGODOVINA:
{zgodovina_str}

VPRAŠANJE: {vprasanje}
ODGOVOR:"""
        return (prompt, "")

    # ---------------------- KONTAKTI / OSTALO ----------------------
    def _answer_contacts(self, q_norm: str) -> str:
        if "karmen" in q_norm and "kotnik" in q_norm:
            return render_block("Direktorica občinske uprave", [("Ime", "mag. Karmen Kotnik"), ("Kontakt", "prek tajništva Občine Rače-Fram: 02 609 60 10")])
        if re.search(r'\bos\s*fram\b', q_norm):
            return render_block("Kontakt OŠ Fram", [("Telefon", "02 603 56 00"), ("Splet", "osfram.si"), ("Prevozi", cfg.TRANSPORT_CONTACT_URL)])
        if re.search(r'\bos\s*rac?e\b', q_norm):
            return render_block("Kontakt OŠ Rače", [("Telefon", "02 609 71 00"), ("Splet", "os-race.si")])
        return "Kontakt občine: " + cfg.FALLBACK_CONTACT

    def _answer_komunalni(self, q_norm: str) -> str:
        rows = [
            ("Kaj", "Plačilo dela stroškov komunalne opreme"),
            ("Kdo", "Investitor/lastnik pri novogradnji ali spremembi namembnosti"),
            ("Kdaj", "Pred izdajo gradbenega dovoljenja"),
            ("Kje", "Vloga na občino, možen informativni izračun"),
            ("Kontakt", cfg.FALLBACK_CONTACT)
        ]
        return render_block("Komunalni prispevek – osnove", rows)

    def _answer_ev(self) -> str:
        return render_block("Polnilnice za EV", [("Lokacije", "Rače in Fram"), ("Opomba", cfg.POLNILNICE_NOTE)])

    # ---------------------- GLAVNI ODGOVOR ----------------------
    def odgovori(self, uporabnikovo_vprasanje: str, session_id: str) -> str:
        self.nalozi_bazo()
        if not self.collection:
            return "Baza znanja ni na voljo."

        ses = self.zgodovina_seje.setdefault(session_id, {'zgodovina': [], 'stanje': {}})
        zgodovina = ses['zgodovina']; stanje = ses['stanje']
        q_norm = normalize_text(uporabnikovo_vprasanje)

        if stanje.get('namen') == 'transport' and stanje.get('transport_waiting') and not _looks_like_transport_followup(q_norm):
            stanje.pop('namen', None); stanje.pop('transport_waiting', None)

        last_intent = stanje.get('last_intent')
        intent = detect_intent_qna(q_norm, last_intent)

        if intent == 'zapora_vloga':
            odgovor = render_block("Vloga za zaporo ceste", [("Povezava", cfg.ROAD_CLOSURE_FORM_URL), ("Opomba", "Priloži skico zapore in terminski plan")])
        elif intent == 'contact':
            odgovor = self._answer_contacts(q_norm)
        elif intent == 'ev_charging':
            odgovor = self._answer_ev()
        elif intent == 'komunalni_prispevek':
            odgovor = self._answer_komunalni(q_norm)
        elif intent == 'awards':
            odgovor = self._answer_awards(q_norm) or "Žal nimam podatkov."
        elif intent == 'who_is_mayor':
            odgovor = "Župan občine Rače-Fram je Samo Rajšp."
        elif intent == 'pgd':
            odgovor = self._answer_pgd_list()
        elif intent == 'camp':
            odgovor = self._answer_camp()
        elif intent == 'transport' or (stanje.get('namen') == 'transport' and stanje.get('transport_waiting') and _looks_like_transport_followup(q_norm)):
            odgovor = self._answer_transport(uporabnikovo_vprasanje, q_norm, stanje)
        elif intent == 'waste' or (stanje.get('namen') == 'odpadki' and stanje.get('caka_na') in ('lokacija','tip')):
            odgovor = self.obravnavaj_odvoz_odpadkov(uporabnikovo_vprasanje, session_id)
        elif intent == 'traffic':
            stanje.pop('namen', None); stanje.pop('caka_na', None); stanje.pop('transport_waiting', None)
            odgovor = self.preveri_zapore_cest()
        else:
            built = self._zgradi_rag_prompt(uporabnikovo_vprasanje, zgodovina) if cfg.USE_LLM else None
            if not built:
                if intent == 'gradbeno':
                    odgovor = render_block("Gradbeno dovoljenje – povzetek", [
                        ("1) Lokacijska informacija", "merila in pogoji"),
                        ("2) PGD", "projektna dokumentacija in soglasja"),
                        ("3) Vloga", "UE Maribor ali eUprava"),
                        ("Veljavnost", "2–3 leta"),
                        ("Povezava", cfg.EUPRAVA_GRADBENO_URL)
                    ])
                else:
                    odgovor = "Žal o tej temi nimam dovolj podatkov."
            else:
                prompt, _ = built
                odgovor = self._call_llm(prompt)

        zgodovina.append((uporabnikovo_vprasanje, odgovor))
        stanje['last_intent'] = intent
        if len(zgodovina) > 4: zgodovina.pop(0)
        self.belezi_pogovor(session_id, uporabnikovo_vprasanje, odgovor)
        return odgovor

    # ---------------------- DIAGNOSTIKA ----------------------
    def debug_street(self, q: str) -> str:
        self.nalozi_bazo()
        if not self._street_index:
            return "Indeks ulic je prazen."
        qn = normalize_text(q)
        phrases = self._build_location_phrases(qn)
        key = self._best_street_key_for_query(phrases)
        # najdi top 8 najbolj podobnih
        ranks = []
        base = strip_generics(" ".join(phrases)) or qn
        for k in self._street_keys_list:
            ranks.append((k, _ratio(base, k)))
        ranks.sort(key=lambda x: x[1], reverse=True)
        head = f'Query="{q}" | phrases={phrases} | key={key}'
        top = "\n".join([f"{i+1:02d}. {k} ({sc:.3f})" for i,(k,sc) in enumerate(ranks[:8])])
        return head + "\nTOP:\n" + top

# ------------------------------------------------------------------------------
# 5) CLI + SELFTEST
# ------------------------------------------------------------------------------
def _diag(zupan: VirtualniZupan):
    print("== DIAGNOSTIKA ==")
    zupan.nalozi_bazo()
    if not zupan.collection:
        print("Kolekcija NI na voljo.")
        return
    try:
        cnt = zupan.collection.count()
        print(f"Chroma kolekcija: {cfg.COLLECTION_NAME} | dokumentov: {cnt}")
        cats = {}
        got = zupan.collection.get(limit=2000)
        for m in got.get("metadatas", []):
            cat = (m.get("kategorija") or "?").lower()
            cats[cat] = cats.get(cat, 0) + 1
        print("Kategorije:", json.dumps(cats, ensure_ascii=False))
    except Exception as e:
        print("Napaka diagnostike:", e)

class _MiniTests(unittest.TestCase):
    def test_detect_waste_over_traffic_bistriska(self):
        q = normalize_text("KDAJ je odvoz stekla na Bistriški cesti?")
        self.assertEqual(detect_intent_qna(q), 'waste')
    def test_detect_camp(self):
        q = normalize_text("Ali imamo v občini poletni tabor?")
        self.assertEqual(detect_intent_qna(q), 'camp')
    def test_transport_extract(self):
        vz = VirtualniZupan()
        qn = normalize_text("od OŠ Fram do Kopivnika zjutraj")
        o, d, t = vz._extract_route(qn, {})
        self.assertEqual(o, "OŠ Fram"); self.assertEqual(d, "Kopivnik"); self.assertEqual(t, "zjutraj")
    def test_parse_dates(self):
        self.assertEqual(parse_dates_from_text("7.1., 4.3., 29.4."), ["7.1.", "4.3.", "29.4."])

if __name__ == "__main__":
    zupan = VirtualniZupan()
    args = sys.argv[1:]
    if "--diag" in args:
        _diag(zupan); sys.exit(0)
    if "--selftest" in args:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(_MiniTests)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    if "--debug-street" in args:
        try:
            ix = args.index("--debug-street")
            q = args[ix+1]
        except Exception:
            print("Uporaba: --debug-street \"Bistriška cesta\""); sys.exit(2)
        print(zupan.debug_street(q)); sys.exit(0)

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
