# VIRT_ZUPAN_RF_api.py  (v57.0)

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
from dotenv import load_dotenv
from chromadb.utils import embedding_functions

try:
    # OpenAI je neobvezen (FAST_MODE=1 ga sploh ne uporablja)
    from openai import OpenAI, OpenAIError
except Exception:  # pragma: no cover
    OpenAI = None
    OpenAIError = Exception

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
    OPENAI_TIMEOUT_S: int = int(os.getenv("OPENAI_TIMEOUT_S", "20"))
    FAST_MODE: int = int(os.getenv("FAST_MODE", "1"))  # 1=brez LLM (hitro), 0=uporabi LLM za splošna RAG vprašanja

    # NAP
    NAP_TOKEN_URL: str = os.getenv("NAP_TOKEN_URL", "https://b2b.nap.si/uc/user/token")
    NAP_DATA_URL: str = os.getenv("NAP_DATA_URL", "https://b2b.nap.si/data/b2b.roadworks.geojson.sl_SI")
    NAP_USERNAME: Optional[str] = os.getenv("NAP_USERNAME")
    NAP_PASSWORD: Optional[str] = os.getenv("NAP_PASSWORD")
    NAP_CACHE_MIN: int = int(os.getenv("NAP_CACHE_MIN", "5"))

    REQUIRED_ENVS: Tuple[str, ...] = tuple([e for e in ("OPENAI_API_KEY",) if int(os.getenv("FAST_MODE","1")) == 0])

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

    GENERIC_STREET_WORDS: Tuple[str, ...] = ("cesta", "ceste", "cesti", "ulica", "ulice", "ulici", "pot", "trg", "naselje", "območje", "obmocje", "naselju")
    GENERIC_PREPS: Tuple[str, ...] = ("na", "v", "za", "ob", "pod", "pri", "nad", "do", "od", "k", "proti")

    RAG_TOPK: int = int(os.getenv("RAG_TOPK", "8"))
    WASTE_FUZZ: float = float(os.getenv("WASTE_FUZZ", "0.86"))
    LOG_MAX_MB: int = int(os.getenv("LOG_MAX_MB", "5"))

    # znani ulični/vaški tokeni (pomaga pri splitu brez ločil)
    KNOWN_PLACE_HINTS: Tuple[str, ...] = (
        "bistriska","mlinska","pozeg","pozeg","pod","terasami","jesenca","fram","race","morje","brunsvik","brunšvik","kopivnik",
        "cvetlicna","cvetlična","soncna","sončna","sencna","senčna","krožna","krozna","ravna","strma","stara","gora","log","loka",
        "livadi","gozdu","potoku","gaj","priso(j)na","krozna","krožna","tajna","bukovec","brezovec","cestarska","ribniku","sadovnjaku",
        "pirkmajerjeva","turnerjeva","koropceva","koropčeva","eberlova","framska"
    )

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
        "Da – polnilnici v Račah in Framu ponovno delujeta (nameščeni novi polnilnici).")

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
    if base.endswith("ska"): keys.add(base[:-3] + "ski"); keys.add(base[:-3] + "ske")
    if base.endswith("ski"): keys.add(base[:-3] + "ska"); keys.add(base[:-3] + "ske")
    if base.endswith("ske"): keys.add(base[:-3] + "ska"); keys.add(base[:-3] + "ski")
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

def _token_is_noise(tok: str) -> bool:
    if not tok or len(tok) <= 1: return True
    if tok in {"h","st","st.","št","št.","st","hs","hs."}: return True
    if re.fullmatch(r'\d+[a-z]?|[a-z]?\d+[a-z]?|(\d+[a-z]?/[a-z]?\d+)', tok): return True
    return False

def extract_locations_from_naselja(field: str) -> List[str]:
    """
    Robustno razbijanje 'naselja':
    - Če vsebuje vejice/';'/novored, delimo po teh ločilih.
    - Če NIMA ločil in je dolg string, ga razbijemo v posamezne tokens
      in zadržimo samo smiselne kandidate za ulice/naselja.
    """
    if not field:
        return []
    cleaned = re.sub(r'\(h\. *št\..*?\)', '', field, flags=re.IGNORECASE).strip()
    norm = normalize_text(cleaned)
    # klasičen primer z ločili
    if re.search(r'[;,]|\n', cleaned):
        out = []
        for seg in re.split(r'[;,\n]+', cleaned):
            seg = seg.strip()
            if not seg: continue
            n = normalize_text(seg).replace("bistriška", "bistriska")
            if n: out.append(n)
        return out
    # brez ločil -> razbij po tokenih in obdrži kandidate
    toks = [t for t in norm.split() if not _token_is_noise(t)]
    keep = []
    for t in toks:
        # če je znan namig ali izgleda kot ulica/kraj (pridevniška oblika)
        if t in cfg.KNOWN_PLACE_HINTS or t.endswith(("ska","ski","ske")) or len(t) >= 5:
            keep.append(t)
    # deduplikacija
    res = []
    seen = set()
    for t in keep:
        if t not in seen:
            seen.add(t)
            res.append(t)
    return res

def tokens_from_text(s: str) -> set:
    return {t for t in re.split(r'[^a-z0-9]+', normalize_text(s)) if len(t) > 2}

def url_is_relevant(url: str, doc: str, q_tokens: set, intent: str) -> bool:
    if not url: return False
    try: u = urlparse(url)
    except Exception: return False
    path_tokens = tokens_from_text(u.path)
    rules = cfg.INTENT_URL_RULES.get(intent, {"must": set(), "ban": set()})
    if rules["ban"] & path_tokens: return False
    if rules["must"] and not (rules["must"] & path_tokens):
        doc_tokens = tokens_from_text(doc)
        if not (rules["must"] & doc_tokens): return False
    if not (q_tokens & path_tokens):
        doc_tokens = tokens_from_text(doc)
        if len(q_tokens & doc_tokens) < 1: return False
    return True

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
    if any(k in q_norm for k in ['kontakt','kontaktna','stevilka','številka','telefon','e mail','email','mail','naslov','pisarna']):
        return 'contact'
    if re.search(r'\bpolnilnic\w*|\belektri\w*\s+polniln\w*|\bev\s*polniln\w*', q_norm):
        return 'ev_charging'

    if any(k in q_norm for k in cfg.KLJUCNE_ODPADKI) or get_canonical_waste(q_norm) or ('naslednji' in q_norm):
        return 'waste'
    if last_intent == 'waste' and (re.match(r'^\s*(kaj pa|pa)\b', q_norm) or re.search(r'\b(od|iz|v|na)\b', q_norm)):
        return 'waste'

    if re.search(r'\bzapor\w*', q_norm) and re.search(r'\bvlog\w*|\bobrazec\w*', q_norm):
        return "zapora_vloga"

    if (re.search(r'\b(zlat\w*\s+peti\w*|petic\w*|peti\b)', q_norm) or
        re.search(r'\bzupanov\w*\s+nagrad\w*', q_norm) or
        re.search(r'\bzupanov\w*\s+priznanj\w*', q_norm) or
        re.search(r'\bzupanova?\s+nagrada', q_norm) or
        'nagrade' in q_norm or 'priznanj' in q_norm):
        if re.search(r'\b(19\d{2}|20\d{2})\b', q_norm):
            return 'awards'

    if re.search(r'\bkdo je\b.*\bdirektor\w*\b', q_norm):
        return 'who_is_director'
    if re.search(r'\bkdo je\b', q_norm) and re.search(r'\bzupan\b', q_norm):
        return 'who_is_mayor'

    if any(k in q_norm for k in cfg.KLJUCNE_PROMET):
        return 'traffic'

    has_transport_hint = any(k in q_norm for k in cfg.TRANSPORT_HINTS)
    if has_transport_hint or re.search(r'\bod\s+(os\s*fram|os\s*rac?e|kopivnik|morje|slivnic\w*|brun[sš]?vik|fram|rac?e)\s+(do|v|na)\s+', q_norm):
        return 'transport'
    if last_intent == 'transport' and _looks_like_transport_followup(q_norm):
        return 'transport'

    if 'gradben' in q_norm and 'dovoljen' in q_norm:
        return 'gradbeno'
    if 'poletn' in q_norm and ('tabor' in q_norm or 'kamp' in q_norm or 'varstvo' in q_norm):
        return 'camp'
    if 'pgd' in q_norm or 'gasil' in q_norm:
        return 'pgd'
    if 'fram' in q_norm and not any(k in q_norm for k in ['odpad', 'promet', 'tabor', 'kamp', 'prispev', 'gradben', 'pgd', 'gasil', 'zapora', 'nagrad']):
        return 'fram_info'
    return 'general'

def map_award_alias(q_norm: str) -> str:
    if re.search(r'\bpeti\w*|zlat\w*\s+peti\w*', q_norm):
        return "zlata_petica"
    if (re.search(r'priznan\w*\s+zupan\w*', q_norm) or
        re.search(r'zupanov\w*\s+nagrad\w*', q_norm) or
        re.search(r'zupanova?\s+nagrada', q_norm)):
        return "priznanje_zupana"
    if "nagrad" in q_norm or "priznanj" in q_norm:
        return "all"
    return "any"

# ------------------------------------------------------------------------------
# 4) GLAVNI RAZRED
# ------------------------------------------------------------------------------
class VirtualniZupan:
    def __init__(self) -> None:
        prefix = "PRODUCTION" if cfg.ENV_TYPE == 'production' else "DEVELOPMENT"
        logger.info(f"[{prefix}] VirtualniŽupan v57.0 inicializiran.")
        self.openai_client = None
        if cfg.FAST_MODE == 0 and OpenAI is not None:
            self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.collection: Optional[chromadb.Collection] = None
        self.zgodovina_seje: Dict[str, Dict[str, Any]] = {}
        self._nap_access_token: Optional[str] = None
        self._nap_token_expiry: Optional[datetime] = None

        self._http = requests.Session()
        retries = Retry(total=3, backoff_factor=0.3, status_forcelist=[500, 502, 504], allowed_methods=frozenset(["GET","POST"]))
        self._http.mount("https://", HTTPAdapter(max_retries=retries))

        self._promet_cache: Optional[List[Dict]] = None
        self._promet_cache_ts: Optional[datetime] = None

        # indeksi
        self._street_index: Dict[str, List[Dict[str, Any]]] = {}
        self._area_type_docs: Dict[Tuple[str, str], str] = {}
        self._street_keys_list: List[str] = []

        self._awards_by_year: Dict[int, Dict[str, List[str]]] = {}
        self._pgd_contacts: Dict[str, Dict[str, str]] = {}

        logger.info(f"FAST_MODE={cfg.FAST_MODE} | Model: {cfg.GENERATOR_MODEL}")

    # ---- infra
    def _call_llm(self, prompt: str, **kwargs) -> str:
        if cfg.FAST_MODE == 1 or not self.openai_client:
            return "Žal o tej temi nimam dovolj podatkov."
        try:
            client = self.openai_client.with_options(timeout=cfg.OPENAI_TIMEOUT_S)
            model_lower = (cfg.GENERATOR_MODEL or "").lower()
            is_gpt5_like = "gpt-5" in model_lower or model_lower.startswith("gpt5") or "o4" in model_lower
            token_limit = kwargs.get("max_tokens", 500)
            create_kwargs = {"model": cfg.GENERATOR_MODEL, "messages": [{"role": "user", "content": prompt}]}
            if is_gpt5_like:
                create_kwargs["max_completion_tokens"] = token_limit
            else:
                create_kwargs["max_tokens"] = token_limit
                create_kwargs["temperature"] = 0.0
            res = client.chat.completions.create(**create_kwargs)
            content = (res.choices[0].message.content or "").strip()
            return content if content else "Žal o tej temi nimam dovolj podatkov."
        except Exception:
            logger.exception("LLM napaka")
            return "Žal o tej temi nimam dovolj podatkov."

    def nalozi_bazo(self) -> None:
        if self.collection:
            return
        logger.info(f"Nalaganje baze znanja iz: {cfg.CHROMA_DB_PATH}")
        try:
            ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY", "dummy") if cfg.FAST_MODE == 0 else "dummy",
                model_name=cfg.EMBEDDING_MODEL
            )
            client = chromadb.PersistentClient(path=cfg.CHROMA_DB_PATH)
            self.collection = client.get_collection(name=cfg.COLLECTION_NAME, embedding_function=ef)
            logger.info(f"Baza uspešno naložena. Število dokumentov: {self.collection.count()}")
            self._build_waste_indexes()
            self._build_awards_index()
            self._build_pgd_contacts()
        except Exception:
            logger.exception("KRITIČNA NAPAKA: Baze znanja ni mogoče naložiti.")
            self.collection = None

    def _ensure_log_rotation(self):
        try:
            if os.path.exists(cfg.LOG_FILE_PATH) and os.path.getsize(cfg.LOG_FILE_PATH) > cfg.LOG_MAX_MB * 1_000_000:
                base, ext = os.path.splitext(cfg.LOG_FILE_PATH)
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                os.rename(cfg.LOG_FILE_PATH, f"{base}_{ts}{ext}")
        except Exception:
            logger.exception("Rotacija loga ni uspela.")

    def belezi_pogovor(self, session_id: str, vprasanje: str, odgovor: str) -> None:
        try:
            self._ensure_log_rotation()
            zapis = {"timestamp": datetime.now().isoformat(), "session_id": session_id, "vprasanje": vprasanje, "odgovor": odgovor}
            with open(cfg.LOG_FILE_PATH, 'a', encoding='utf-8') as f:
                f.write(json.dumps(zapis, ensure_ascii=False) + '\n')
        except Exception:
            logger.exception(f"Napaka pri beleženju pogovora za sejo {session_id}")

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
        if m1: origin = self._canon_place(m1.group(1)); dest = self._canon_place(m1.group(3))
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
            return "Katera relacija te zanima? (npr. 'od OŠ Fram do Kopivnika')"
        if origin and not dest:
            stanje['namen'] = 'transport'; stanje['transport_last_origin'] = origin; stanje['transport_waiting'] = 'dest'
            return "Kam pa želiš (npr. 'v Kopivnik', 'v Morje')?"
        if dest and not origin:
            stanje['namen'] = 'transport'; stanje['transport_last_dest'] = dest; stanje['transport_waiting'] = 'origin'
            return "Iz kje pa? (npr. 'iz OŠ Fram', 'iz OŠ Rače')"

        stanje['transport_last_origin'] = origin
        stanje['transport_last_dest']   = dest
        stanje['namen'] = 'transport'
        stanje.pop('transport_waiting', None)

        def is_pair(a, b, A, B):
            return (normalize_text(a) == normalize_text(A) and normalize_text(b) == normalize_text(B))

        if (is_pair(origin, dest, "OŠ Fram", "Kopivnik") or is_pair(origin, dest, "Kopivnik", "OŠ Fram")):
            parts = []
            if not time_hint or time_hint == "zjutraj":   parts.append("Zjutraj: 7.25")
            if not time_hint or time_hint == "popoldne":  parts.append("Popoldne: 14.40")
            times = " | ".join(parts) if parts else "Ure niso na voljo."
            return f"Prevoz {origin} → {dest}. {times}. {cfg.TRANSPORT_CONTACT_TEXT}: {cfg.TRANSPORT_CONTACT_URL}"

        return (f"Za relacijo {origin} → {dest} nimam točnih ur. "
                f"Preveri pri organizatorju. {cfg.TRANSPORT_CONTACT_TEXT}: {cfg.TRANSPORT_CONTACT_URL}")

    # ---------------------- NAP / PROMET ----------------------
    def _ensure_nap_token(self) -> Optional[str]:
        if not (cfg.NAP_USERNAME and cfg.NAP_PASSWORD):
            return None
        if self._nap_access_token and self._nap_token_expiry and datetime.utcnow() < self._nap_token_expiry - timedelta(seconds=60):
            return self._nap_access_token
        logger.info("Pridobivam/osvežujem NAP API žeton…")
        try:
            payload = {'grant_type': 'password', 'username': cfg.NAP_USERNAME, 'password': cfg.NAP_PASSWORD}
            response = self._http.post(cfg.NAP_TOKEN_URL, data=payload, timeout=10)
            response.raise_for_status()
            data = response.json()
            self._nap_access_token = data['access_token']
            self._nap_token_expiry = datetime.utcnow() + timedelta(seconds=data.get('expires_in', 3600))
            return self._nap_access_token
        except requests.RequestException:
            logger.exception("NAP API napaka pri pridobivanju podatkov.")
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
        logger.info("Kličem specialista za promet (NAP API)…")
        now = datetime.utcnow()
        if self._promet_cache and self._promet_cache_ts and now - self._promet_cache_ts < timedelta(minutes=cfg.NAP_CACHE_MIN):
            vsi_dogodki = self._promet_cache
        else:
            token = self._ensure_nap_token()
            if not token:
                return "Prometne informacije trenutno niso na voljo."
            try:
                headers = {"Authorization": f"Bearer {token}"}
                resp = self._http.get(cfg.NAP_DATA_URL, headers=headers, timeout=15)
                resp.raise_for_status()
                payload = resp.json()
                vsi_dogodki = payload.get("features", []) if isinstance(payload, dict) else []
                self._promet_cache = vsi_dogodki
                self._promet_cache_ts = now
            except requests.RequestException:
                logger.exception("NAP API napaka pri pridobivanju podatkov o prometu.")
                return "Prometna poročila trenutno niso dostopna."

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
            return "Na območju občine Rače-Fram trenutno ni zabeleženih del ali zapor."

        seen = set(); unique = []
        for z in relevantne:
            key = (str(z.get("cesta","")).strip(), str(z.get("opis","")).strip())
            if key in seen: continue
            seen.add(key); unique.append(z)
        ts = datetime.now().strftime("%d.%m.%Y %H:%M")
        lines = [f"Stanje na cestah (vir: NAP/promet.si, {ts})"]
        for z in unique:
            cesta = z.get("cesta") or "Ni podatka"
            opis  = z.get("opis")  or "Ni podatka"
            lines.append(f"- {cesta}: {opis}")
        return "\n".join(lines)

    # ---------------------- ODPADKI ----------------------
    def _build_waste_indexes(self) -> None:
        logger.info("Gradim indekse za odpadke …")
        if not self.collection: return
        all_docs = self.collection.get(where={"kategorija": "Odvoz odpadkov"})
        if not all_docs or not all_docs.get('ids'):
            logger.warning("Ni dokumentov za odpadke.")
            return
        for i in range(len(all_docs['ids'])):
            meta = all_docs['metadatas'][i]
            doc_text = all_docs['documents'][i]
            tip_meta = meta.get('tip_odpadka', '') or ''
            tip = get_canonical_waste(tip_meta) or tip_meta
            area = normalize_text(meta.get('obmocje', '') or '')
            # shranimo tudi po območju+tip
            self._area_type_docs[(area, tip)] = doc_text

            streets = extract_locations_from_naselja(meta.get('naselja', '') or '')
            # če razbitje vrne nič (zelo redek primer), indexiraj fallback kot celoto
            if not streets:
                s_norm = normalize_text(meta.get('naselja', '') or '')
                if s_norm: streets = s_norm.split()

            for s_norm in streets:
                display = s_norm
                # indeksiramo vse ključe za dano ulico
                for key in gen_street_keys(s_norm):
                    self._street_index.setdefault(key, []).append({
                        "area": area, "tip": tip, "doc": doc_text, "display": display
                    })

        self._street_keys_list = list(self._street_index.keys())
        logger.info(f"Indeks: {len(self._street_keys_list)} uličnih ključev, {len(self._area_type_docs)} (območje,tip).")

    def _best_street_key_for_query(self, phrases: List[str]) -> Optional[str]:
        # 1) direktna in fleks (gen_street_keys)
        for ph in phrases:
            for key in gen_street_keys(ph):
                if key in self._street_index:
                    return key
        # 2) substring v obstoječih ključih (pomaga proti 'mega-ključem', če obstajajo)
        for ph in phrases:
            base = strip_generics(ph)
            if not base or len(base) < 3: continue
            for key in self._street_keys_list:
                if base in key:
                    return key
        # 3) fuzzy
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
            return f"Naslednji odvoz – {street_display.title()}: {tip_odpadka}: {next_date_fmt}"
        return f"Odvoz – {street_display.title()}: {tip_odpadka}. Termini: {', '.join(dates)}"

    def obravnavaj_odvoz_odpadkov(self, uporabnikovo_vprasanje: str, session_id: str) -> str:
        logger.info("Kličem specialista za odpadke…")
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
            return "Za katero ulico te zanima urnik? (npr. 'Bistriška cesta, Fram')"

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
                return "Kateri tip odpadkov te zanima? (bio, mešani, embalaža, papir, steklo)"
            return "Za to ulico ali območje nimam najdenega urnika za ta tip."

        entries = self._street_index.get(key, [])
        if not entries:
            return "Za to ulico nimam podatkov."

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
                return "Za to ulico ali območje nimam najdenega urnika za ta tip."
        else:
            best_entry = entries[0]

        tip_canon = get_canonical_waste(best_entry["tip"]) or best_entry["tip"]
        ans = self._format_dates_for_tip(best_entry["display"], tip_canon, best_entry["doc"], only_next)
        if not ans: return "Ni najdenih datumov v urniku."
        stanje['zadnja_lokacija_norm'] = strip_generics(best_entry["display"])
        stanje['zadnji_tip'] = tip_canon
        stanje['namen'] = 'odpadki'; stanje.pop('caka_na', None)
        return ans

    # ---------------------- NAGRADE + PGD ----------------------
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

    def _build_awards_index(self) -> None:
        logger.info("Gradim indeks nagrad …")
        if not self.collection: return
        got = self.collection.get(where={"kategorija": "Nagrade in Priznanja"})
        if not got or not got.get("ids"):
            logger.warning("Ni dokumentov za nagrade."); return
        for i in range(len(got["ids"])):
            doc = got["documents"][i]; text = doc
            years = [int(y) for y in re.findall(r'\b(19\d{2}|20\d{2})\b', text)]
            year = max(years) if years else None
            if not year: continue
            m_zp = re.search(r"(prejemniki\s+zlat\w*\s+petic\w*|zlat\w*\s+petic\w*)\s*:\s*(.*?)(?:\n\s*\n|$)", text, flags=re.IGNORECASE | re.DOTALL)
            if m_zp:
                # v virih je pogosto urejeno v alinejah, parser zgoraj zadošča
                self._add_award(year, "zlata_petica", [self._clean_bullet(x) for x in m_zp.group(2).splitlines() if self._clean_bullet(x)])

    def _answer_awards(self, q_norm: str) -> Optional[str]:
        years = [int(y) for y in re.findall(r'\b(19\d{2}|20\d{2})\b', q_norm)]
        year = years[0] if years else None
        if not year: return None
        data = self._awards_by_year.get(year) or {}
        if not data: return "Za izbrano leto nimam nagrajencev."
        zl = data.get("zlata_petica", [])
        if zl: return f"Zlata petica {year}: " + "; ".join(zl)
        return "Za izbrano leto nimam zbranih podatkov."

    # ---------------------- PGD ----------------------
    def _build_pgd_contacts(self) -> None:
        # Minimalno – če imaš v Chroma pravilne kontakte, jih lahko kasneje izboljšaš
        self._pgd_contacts = {
            "pgd rače": {"ime": "PGD Rače"},
            "pgd fram": {"ime": "PGD Fram"},
            "pgd podova": {"ime": "PGD Podova"},
            "pgd spodnja in zgornja gorica": {"ime": "PGD Spodnja in Zgornja Gorica"},
        }

    def _answer_pgd_list(self) -> str:
        order = ["PGD Rače","PGD Fram","PGD Podova","PGD Spodnja in Zgornja Gorica"]
        names = ", ".join(order)
        return f"Gasilska društva v občini Rače-Fram: {names}."

    # ---------------------- RAG PROMPT BUILDER ----------------------
    def _zgradi_rag_prompt(self, vprasanje: str, zgodovina: List[Tuple[str, str]]) -> Optional[str]:
        if cfg.FAST_MODE == 1:
            return None
        if not self.collection: return None
        q_norm = normalize_text(vprasanje)
        intent = detect_intent_qna(q_norm)
        if intent in ("awards","pgd","zapora_vloga","waste","traffic","who_is_mayor","who_is_director","transport","contact","komunalni_prispevek","ev_charging"):
            return None
        results = self.collection.query(query_texts=[q_norm], n_results=cfg.RAG_TOPK, include=["documents", "metadatas"])
        docs = results.get('documents', [[]])[0]; metas = results.get('metadatas', [[]])[0]
        if not docs: return None
        context_parts = []
        for (doc, meta) in zip(docs, metas):
            doc_short = (doc or "")[:2000]
            url = meta.get('source_url','') or ''
            context_parts.append(f"VIR: {meta.get('source','?')}\nURL: {url}\n{doc_short}")
        now = datetime.now().strftime("%d.%m.%Y")
        prompt = f"Danes je {now}. Odgovori zelo kratko in pregledno.\nKontekst:\n" + "\n---\n".join(context_parts) + f"\n\nVprasanje: {vprasanje}\nOdgovor:"
        return prompt

    # ---------------------- GLAVNI VMESNIK ----------------------
    def _answer_contacts(self, q_norm: str) -> str:
        if "karmen" in q_norm and "kotnik" in q_norm:
            return "Direktorica občinske uprave: mag. Karmen Kotnik. Kontakt prek tajništva: 02 609 60 10."
        if re.search(r'\bnk\b|\bnogometn\w+\s+klub\b', q_norm) and "fram" in q_norm:
            return "Kontakt NK Fram: predlagam preverbo pri občini (02 609 60 10) ali na družbenih omrežjih kluba."
        if re.search(r'\bos\s*fram\b', q_norm):
            return f"OŠ Fram – tel.: 02 603 56 00. Prevozi: {cfg.TRANSPORT_CONTACT_URL}"
        if re.search(r'\bos\s*rac?e\b', q_norm):
            return "OŠ Rače – tel.: 02 609 71 00."
        return "Kontakt občine: Občina Rače-Fram, 02 609 60 10."

    def _answer_komunalni(self) -> str:
        return ("Komunalni prispevek: plačilo dela stroškov komunalne opreme pred izdajo gradbenega dovoljenja. "
                "Informativni izračun in navodila so objavljeni na občinski strani (VLOGA št. 3540 za izdajo odločbe o komunalnem prispevku). "
                "Za vprašanja pokliči občino: 02 609 60 10.")

    def _answer_ev(self) -> str:
        return f"Polnilnice za EV: Rače in Fram – {cfg.POLNILNICE_NOTE}"

    def odgovori(self, uporabnikovo_vprasanje: str, session_id: str) -> str:
        self.nalozi_bazo()
        if not self.collection:
            return "Moja baza znanja trenutno ni na voljo."

        ses = self.zgodovina_seje.setdefault(session_id, {'zgodovina': [], 'stanje': {}})
        zgodovina = ses['zgodovina']; stanje = ses['stanje']
        q_norm = normalize_text(uporabnikovo_vprasanje)
        last_intent = stanje.get('last_intent')
        intent = detect_intent_qna(q_norm, last_intent)

        if intent == 'zapora_vloga':
            odgovor = f"Vloga za zaporo ceste: {cfg.ROAD_CLOSURE_FORM_URL}."
        elif intent == 'contact':
            odgovor = self._answer_contacts(q_norm)
        elif intent == 'ev_charging':
            odgovor = self._answer_ev()
        elif intent == 'komunalni_prispevek':
            odgovor = self._answer_komunalni()
        elif intent == 'awards':
            odgovor = self._answer_awards(q_norm) or "Za izbrano leto nimam podatkov."
        elif intent == 'who_is_mayor':
            odgovor = "Župan občine Rače-Fram je Samo Rajšp."
        elif intent == 'who_is_director':
            odgovor = "Direktorica občinske uprave je mag. Karmen Kotnik (kontakt prek tajništva: 02 609 60 10)."
        elif intent == 'pgd':
            odgovor = self._answer_pgd_list()
        elif intent == 'transport' or (stanje.get('namen') == 'transport' and stanje.get('transport_waiting') and _looks_like_transport_followup(q_norm)):
            odgovor = self._answer_transport(uporabnikovo_vprasanje, q_norm, stanje)
        elif intent == 'waste' or (stanje.get('namen') == 'odpadki' and stanje.get('caka_na') in ('lokacija','tip')):
            odgovor = self.obravnavaj_odvoz_odpadkov(uporabnikovo_vprasanje, session_id)
        elif intent == 'traffic':
            stanje.pop('namen', None); stanje.pop('caka_na', None); stanje.pop('transport_waiting', None)
            odgovor = self.preveri_zapore_cest()
        else:
            built = self._zgradi_rag_prompt(uporabnikovo_vprasanje, zgodovina)
            odgovor = self._call_llm(built) if built else "Žal o tem nimam dovolj podatkov."

        zgodovina.append((uporabnikovo_vprasanje, odgovor))
        stanje['last_intent'] = intent
        if len(zgodovina) > 6: zgodovina.pop(0)
        self.belezi_pogovor(session_id, uporabnikovo_vprasanje, odgovor)
        return odgovor

# ------------------------------------------------------------------------------
# 5) CLI + DIAG + SELFTEST + DEBUG
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

    def test_detect_director(self):
        q = normalize_text("kdo je direktor občinske uprave")
        self.assertEqual(detect_intent_qna(q), 'who_is_director')

    def test_parse_dates(self):
        self.assertEqual(parse_dates_from_text("7.1., 4.3., 29.4."), ["7.1.", "4.3.", "29.4."])

def _debug_street(vz: VirtualniZupan, street: str):
    vz.nalozi_bazo()
    print("\n[1] gen_street_keys ->", gen_street_keys(street))
    base = strip_generics(street)
    print("\n[2] Ključi v indeksu, ki vsebujejo base:", base)
    hits = {k: len(v) for k, v in vz._street_index.items() if base in k}
    if not hits:
        print("  NI ZADETKOV – preveri, ali je 'naselja' pravilno strukturirano v Chroma.")
    else:
        for k, n in list(hits.items())[:10]:
            print(" ", k, "=>", n)
    print("\n[3] Odgovor (steklo @", street, "):")
    print(vz.obravnavaj_odvoz_odpadkov(f"Kdaj je odvoz stekla na {street}", "debug"))

if __name__ == "__main__":
    vz = VirtualniZupan()
    args = sys.argv[1:]
    if "--diag" in args:
        _diag(vz); sys.exit(0)
    if "--selftest" in args:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(_MiniTests)
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)
    if "--debug-street" in args:
        try:
            idx = args.index("--debug-street")
            street = args[idx+1]
        except Exception:
            print("Uporaba: --debug-street \"Bistriška cesta\"")
            sys.exit(1)
        _debug_street(vz, street)
        sys.exit(0)

    vz.nalozi_bazo()
    if not vz.collection:
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
            print(vz.odgovori(q, session_id))
            print("\n---------------------\n")
        except KeyboardInterrupt:
            break
    print("\nNasvidenje!")
