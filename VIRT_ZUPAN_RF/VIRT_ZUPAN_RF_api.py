# VIRT_ZUPAN_RF_api.py — Finalna Production Ready verzija
# Verzija 40.5 — Popravljeno filtriranje prometa + manjši popravki
# -----------------------------------------------------------------------------

import os
import sys
import re
import json
import unicodedata
import argparse
import time
from threading import Semaphore, Lock
from datetime import datetime, timedelta, date, timezone, UTC
from difflib import SequenceMatcher
from collections import defaultdict
from typing import Dict, Any, Optional, List, Tuple

import chromadb
from chromadb.utils import embedding_functions
import requests
from dotenv import load_dotenv
from openai import OpenAI

# Tiktoken za točno štetje tokenov (neobvezno)
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    print("⚠️ tiktoken ni nameščen. Namestite z: pip install tiktoken")
    TIKTOKEN_AVAILABLE = False

# -----------------------------------------------------------------------------
# KONFIGURACIJA
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '..', '.env'))

if os.getenv('ENV_TYPE') == 'production':
    DATA_DIR = "/data"
    print("Zaznano produkcijsko okolje (Render). Poti so nastavljene na /data.")
else:
    DATA_DIR = os.path.join(BASE_DIR, "data")
    print("Zaznano lokalno okolje. Poti so nastavljene relativno.")
    os.makedirs(DATA_DIR, exist_ok=True)

# LOG direktorij iz ENV ali relativna pot
LOCAL_LOG_DIR = os.getenv("LOCAL_LOG_DIR", os.path.join(BASE_DIR, "logs"))
os.makedirs(LOCAL_LOG_DIR, exist_ok=True)

CHROMA_DB_PATH = os.path.join(DATA_DIR, "chroma_db")
LOG_FILE_PATH = os.path.join(DATA_DIR, "zupan_pogovori.jsonl")

COLLECTION_NAME = "obcina_race_fram_prod"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"

# --- NAP ---
NAP_TOKEN_URL = "https://b2b.nap.si/uc/user/token"
NAP_DATA_URL = "https://b2b.nap.si/data/b2b.roadworks.geojson.sl_SI"
NAP_USERNAME = os.getenv("NAP_USERNAME")
NAP_PASSWORD = os.getenv("NAP_PASSWORD")

# --- LLM scaling / multimodel ENV nastavitve ---
ECONOMY_MODE = os.getenv("ECONOMY_MODE", "0") == "1"
FORCE_MODEL = os.getenv("FORCE_MODEL", "")

if ECONOMY_MODE:
    PRIMARY_GEN_MODEL = "gpt-4o-mini"
    ALT_GEN_MODELS = ["gpt-3.5-turbo"]
    print("[ECONOMY MODE] Uporabljam samo poceni modele!")
else:
    PRIMARY_GEN_MODEL = os.getenv("PRIMARY_GEN_MODEL", "gpt-4o-mini")
    ALT_GEN_MODELS = [m.strip() for m in os.getenv("ALT_GEN_MODELS", "gpt-3.5-turbo,gpt-4o").split(",") if m.strip()]

if FORCE_MODEL:
    PRIMARY_GEN_MODEL = FORCE_MODEL
    ALT_GEN_MODELS = []
    print(f"[FORCE MODE] Uporabljam samo: {FORCE_MODEL}")

LLM_MAX_CONCURRENCY = int(os.getenv("LLM_MAX_CONCURRENCY", "4"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_BASE_BACKOFF = float(os.getenv("LLM_BASE_BACKOFF", "0.6"))
LLM_MAX_BACKOFF = float(os.getenv("LLM_MAX_BACKOFF", "4.0"))
LLM_DEBUG = os.getenv("LLM_DEBUG", "0") == "1"

# Stroški (EUR na 1k tokenov)
MODEL_COSTS = {
    "gpt-4o": {"input": 0.0023, "output": 0.0092},
    "gpt-4o-mini": {"input": 0.00014, "output": 0.00055},
    "gpt-3.5-turbo": {"input": 0.00046, "output": 0.0014},
    "gpt-4-turbo": {"input": 0.0092, "output": 0.028}
}
DAILY_COST_ALERT_EUR = float(os.getenv("DAILY_COST_ALERT_EUR", "5.0"))

_llm_semaphore = Semaphore(LLM_MAX_CONCURRENCY)

# -----------------------------------------------------------------------------
# Trigerji in konstante
# -----------------------------------------------------------------------------
# Ožji prometni triggerji – brez generičnih "cesta/ulica/pot"
KLJUCNE_BESEDE_PROMET_TRDE = [
    "promet", "zapora", "zapore", "zaprta", "zaprt", "oviran",
    "zastoj", "zastoji", "gneca", "gneča", "kolona",
    "dela", "prekop", "semafor", "izmenicno", "izmenično", "enosmerno",
    "preusmeritev", "obvoz"
]

# NOVO: Seznam krajev v občini Rače-Fram za natančno filtriranje prometa
PROMET_FILTER_LOKACIJE = [
    "rače", "fram", "morje pri framu", "požeg", "spodnja gorica", "zgornja gorica",
    "brezula", "ranče", "kopivnik", "loka pri framu", "brunšvik", "podova", "ješenca"
]

# Odpadki – razširjene oblike (skloni)
KLJUCNE_BESEDE_ODPADKI = [
    "odvoz", "odvozi", "odvozov",
    "odpadki", "odpadkov", "odpadke", "smeti", "urnik",
    "embalaza", "embalaža", "rumena", "kanta", "kante",
    "papir", "papirja", "papirju", "karton", "kartona",
    "bio", "bioloski", "biološki", "mesani", "mešani", "komunalni",
    "steklo", "stekla", "steklena", "steklena embalaža"
]

GENERIC_WORDS = {
    "cesta", "cesti", "ulica", "ulici", "pot", "trg", "naselje", "obmocje", "območje",
    "pri", "na", "v", "pod", "nad", "k", "do", "od", "proti", "terasa", "terasami"
}

WASTE_TYPE_VARIANTS = {
    "Biološki odpadki": [
        "bioloski odpadki", "bioloskih odpakov", "bioloski", "bioloskih", "bio",
        "biološki odpadki", "bioloskih odpadkov", "bioloski odpadkov",
        "bioloških odpadkov", "bioloških odpadki", "bio odpadki"
    ],
    "Mešani komunalni odpadki": [
        "mesani komunalni odpadki", "mešani komunalni odpadki", "mesani", "mešani",
        "mešane odpadke", "mesane odpadke", "mešani odpadki",
        "mešane komunalne", "mesane komunalne",
        "mešane komunalne odpadke", "mesane komunalne odpadke",
        "komunalni odpadki", "komunalnih odpadkov", "komunalne odpadke"
    ],
    "Odpadna embalaža": [
        "odpadna embalaza", "odpadna embalaža", "embalaza", "embalaža",
        "embalaže", "rumena kanta", "rumene kante"
    ],
    "Papir in karton": [
        "papir in karton", "papir", "karton", "papirja", "kartona", "papir in kartona"
    ],
    "Steklena embalaža": [
        "steklena embalaza", "steklena embalaža", "steklo", "stekla",
        "stekle", "stekleno", "stekleni", "steklen"
    ],
}

# -----------------------------------------------------------------------------
# Text utils (funkcije na modulu)
# -----------------------------------------------------------------------------
def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8')
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()

def fuzzy_match(a: str, b: str, threshold: float = 0.8) -> bool:
    if not a or not b:
        return False
    a_n = normalize_text(a)
    b_n = normalize_text(b)
    if a_n in b_n or b_n in a_n:
        return True
    return fuzzy_ratio(a_n, b_n) >= threshold

def slovenian_variant_equivalent(a: str, b: str) -> bool:
    a_n = normalize_text(a)
    b_n = normalize_text(b)
    if a_n == b_n:
        return True
    if len(a_n) > 1 and len(b_n) > 1:
        if a_n[:-1] == b_n[:-1] and {a_n[-1], b_n[-1]} <= {"a", "i", "e", "o", "u"}:
            return True
    return False

def root_token(w: str) -> str:
    w = normalize_text(w)
    if not w or w in GENERIC_WORDS:
        return ""
    w = re.sub(r'(ska|ski|sko|ške|ški|ško)$', 'sk', w)
    w = re.sub(r'(a|e|i|o|u)+$', '', w)
    w = re.sub(r'linu$', 'lin', w)
    w = re.sub(r'(ji|mi|ni|li|vi|ti)$', '', w)
    return w

def roots_of_phrase(text: str) -> set:
    toks = [t for t in normalize_text(text).split() if t]
    roots = {root_token(t) for t in toks}
    return {r for r in roots if r and r not in GENERIC_WORDS}

def street_phrase_matches(query_phrase: str, street_tok: str, threshold: float = 0.86) -> bool:
    qp = normalize_text(query_phrase)
    st = normalize_text(street_tok)

    if slovenian_variant_equivalent(qp, st):
        return True

    q_words = [w for w in qp.split() if w not in GENERIC_WORDS]
    street_words = [w for w in st.split() if w not in GENERIC_WORDS]

    qr = {root_token(w) for w in q_words}
    sr = {root_token(w) for w in street_words}
    qr.discard("")
    sr.discard("")
    if qr and sr and qr.isdisjoint(sr):
        return False

    if not q_words:
        return fuzzy_match(qp, st, threshold)

    for qw in q_words:
        if slovenian_variant_equivalent(qw, st):
            return True
        if fuzzy_ratio(qw, st) >= threshold or (qw in st or st in qw):
            return True
        for sw in street_words:
            if slovenian_variant_equivalent(qw, sw):
                return True
            if fuzzy_ratio(qw, sw) >= threshold or (qw in sw or sw in qw):
                return True
    return False

def extract_locations_from_naselja(naselja_field: str) -> List[str]:
    """Ekstrahira lokacije iz naselja polja"""
    if not naselja_field:
        return []
    text = re.sub(r'\(h\.?\s*št\.?.*?\)', '', naselja_field, flags=re.IGNORECASE)

    if re.search(r'[;,\n]', text):
        parts = []
        for chunk in re.split(r'[;,\n]+', text):
            chunk = chunk.strip()
            if chunk:
                parts.append(normalize_text(chunk))
        out, seen = [], set()
        for p in parts:
            if p and p not in seen:
                seen.add(p)
                out.append(p)
        return out

    toks = [t.strip() for t in text.split() if t.strip()]
    out, seen = [], set()
    buf = []
    for tok in toks:
        n = normalize_text(tok)
        if not n:
            continue
        buf.append(n)
    phrase = " ".join(buf).strip()
    if phrase:
        for p in re.split(r'\s*/\s*', phrase):
            p = p.strip()
            if p and p not in seen:
                seen.add(p)
                out.append(p)
    return out

def get_canonical_waste_type(text: str) -> Optional[str]:
    """Prepozna kanonični tip odpadka"""
    norm = normalize_text(text)
    if ("rumen" in norm or "rumena" in norm) and ("kanta" in norm or "kante" in norm):
        return "Odpadna embalaža"
    if "komunaln" in norm and "odpadk" in norm:
        return "Mešani komunalni odpadki"
    if (("bio" in norm or "biolos" in norm) and "odpadk" in norm) or "bioloski" in norm:
        return "Biološki odpadki"
    if "stekl" in norm:
        return "Steklena embalaža"
    if "papir" in norm or "karton" in norm:
        return "Papir in karton"
    if "embal" in norm:
        return "Odpadna embalaža"

    for canonical, variants in WASTE_TYPE_VARIANTS.items():
        if normalize_text(canonical) in norm:
            return canonical
        for v in variants:
            if normalize_text(v) in norm:
                return canonical
    for canonical, variants in WASTE_TYPE_VARIANTS.items():
        for v in variants:
            if fuzzy_ratio(norm, normalize_text(v)) >= 0.85:
                return canonical
    return None

# --- Router helperji ---
FALSE_POSITIVE_WASTE_PATTERNS = [
    r"\bkomunaln\w*\s+prispevek\w*\b",
    r"\bkomunaln\w*\s+taks\w*\b",
    r"\bkomunaln\w*\s+podjetj\w*\b",
    r"\bkomunaln\w*\s+storitv\w*\b",
    r"\bvodovod\w*\b",
    r"\bkanalizac\w*\b",
]

def _has_waste_intent(text_norm: str) -> bool:
    # Izloči lažne pozitive
    for pat in FALSE_POSITIVE_WASTE_PATTERNS:
        if re.search(pat, text_norm):
            return False
    # Preveri tip odpadka
    if get_canonical_waste_type(text_norm):
        return True
    # Splošne waste ključne besede
    keys = [
        "odvoz", "odvozi", "odvozov", "odpadki", "odpadkov", "odpadke",
        "smeti", "urnik", "embalaza", "embalaža", "papir", "papirja",
        "karton", "bio", "steklo", "steklena"
    ]
    return any(re.search(r'\b' + re.escape(k) + r'\b', text_norm) for k in keys)

def _has_traffic_intent(text_norm: str) -> bool:
    return any(re.search(r'\b' + re.escape(k) + r'\b', text_norm) for k in KLJUCNE_BESEDE_PROMET_TRDE)

# -----------------------------------------------------------------------------
# Thread-safe Cache
# -----------------------------------------------------------------------------
class ThreadSafeCache:
    def __init__(self, max_entries: int = 1000):
        self._cache = {}
        self._lock = Lock()
        self._hits = defaultdict(int)
        self._max_entries = max_entries
        self._access_times = {}

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if time.time() < entry['expires']:
                    self._hits[key] += 1
                    self._access_times[key] = time.time()
                    return entry['data']
                else:
                    del self._cache[key]
                    self._access_times.pop(key, None)
            return None

    def set(self, key: str, data: Any, ttl_seconds: int = 7200):
        with self._lock:
            if len(self._cache) >= self._max_entries:
                self._cleanup_old_entries()
            self._cache[key] = {'data': data, 'expires': time.time() + ttl_seconds}
            self._access_times[key] = time.time()

    def _cleanup_old_entries(self):
        if not self._access_times:
            return
        sorted_keys = sorted(self._access_times.keys(), key=lambda k: self._access_times[k])
        remove_count = max(1, len(sorted_keys) // 5)
        for key in sorted_keys[:remove_count]:
            self._cache.pop(key, None)
            self._access_times.pop(key, None)
            self._hits.pop(key, None)

    def stats(self) -> Dict:
        with self._lock:
            active = sum(1 for v in self._cache.values() if time.time() < v['expires'])
            return {"active_entries": active, "total_entries": len(self._cache), "hits": dict(self._hits)}

_global_cache = ThreadSafeCache()

# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------
class SystemLogger:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self._lock = Lock()

    def log_interaction(self, session_id: str, question: str, answer: str,
                        model_used: str = None, response_time: float = None,
                        was_fallback: bool = False, error: str = None):
        today = date.today().isoformat()
        log_file = os.path.join(self.log_dir, f"zupan_full_log_{today}.jsonl")
        entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "question": question,
            "answer": (answer[:500] if isinstance(answer, str) else None),
            "model_used": model_used,
            "response_time": round(response_time, 2) if response_time else None,
            "was_fallback": was_fallback,
            "error": error
        }
        with self._lock:
            try:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"[LOG ERROR] Ne morem pisati v {log_file}: {e}")

# -----------------------------------------------------------------------------
# LLM Model Router
# -----------------------------------------------------------------------------
class ModelRouter:
    def __init__(self, openai_client):
        self.client = openai_client
        self._stats_lock = Lock()
        self.model_stats = defaultdict(lambda: {
            "calls": 0, "total_time": 0, "failures": 0,
            "total_input_tokens": 0, "total_output_tokens": 0, "total_cost_eur": 0
        })
        self.daily_costs = defaultdict(float)
        self._encoders = {}

    def _count_tokens(self, text: str, model: str) -> int:
        if not TIKTOKEN_AVAILABLE:
            return max(1, len(text) // 4)
        try:
            if model not in self._encoders:
                try:
                    self._encoders[model] = tiktoken.encoding_for_model(model)
                except Exception:
                    self._encoders[model] = tiktoken.get_encoding("cl100k_base")
            return len(self._encoders[model].encode(text))
        except Exception as e:
            print(f"[TOKEN COUNT ERROR] {e}")
            return max(1, len(text) // 4)

    def _calculate_cost_eur(self, model: str, input_tokens: int, output_tokens: int) -> float:
        costs = MODEL_COSTS.get(model, {"input": 0.001, "output": 0.002})
        return (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1000

    def _candidates_for(self, task: str, input_len: int) -> List[str]:
        if FORCE_MODEL:
            return [FORCE_MODEL]
        base = [PRIMARY_GEN_MODEL] + [m for m in ALT_GEN_MODELS if m != PRIMARY_GEN_MODEL]
        if task in ["waste", "traffic", "rewrite"]:
            ordered = [m for m in base if ("mini" in m or "3.5" in m)] + [m for m in base if ("mini" not in m and "3.5" not in m)]
        elif task == "rag_answer":
            ordered = [m for m in base if ("mini" in m or "3.5" in m)] if ECONOMY_MODE else base
        else:
            if input_len < 800:
                ordered = [m for m in base if ("mini" in m or "3.5" in m)] + [m for m in base if ("mini" not in m and "3.5" not in m)]
            else:
                ordered = base
        dedup, seen = [], set()
        for m in ordered:
            if m and m not in seen:
                seen.add(m)
                dedup.append(m)
        return dedup

    def chat(self, messages, task="generic", temperature=0.0, max_tokens=None,
             request_name="") -> Tuple[str, str]:
        start_time = time.time()
        input_text = " ".join(m.get("content", "") for m in messages if isinstance(m, dict))
        candidates = self._candidates_for(task, len(input_text))

        successful_model, last_err = None, None
        for model in candidates:
            delay = LLM_BASE_BACKOFF
            for attempt in range(1, LLM_MAX_RETRIES + 1):
                try:
                    if LLM_DEBUG:
                        print(f"[LLM] {request_name or task} -> model={model} attempt={attempt}/{LLM_MAX_RETRIES}")
                    with _llm_semaphore:
                        resp = self.client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=temperature,
                            **({"max_tokens": max_tokens} if max_tokens else {})
                        )
                    output_text = (resp.choices[0].message.content or "").strip()
                    elapsed = time.time() - start_time
                    successful_model = model

                    in_toks = self._count_tokens(input_text, model)
                    out_toks = self._count_tokens(output_text, model)
                    cost = self._calculate_cost_eur(model, in_toks, out_toks)

                    with self._stats_lock:
                        st = self.model_stats[model]
                        st["calls"] += 1
                        st["total_time"] += elapsed
                        st["total_input_tokens"] += in_toks
                        st["total_output_tokens"] += out_toks
                        st["total_cost_eur"] += cost
                        today = date.today().isoformat()
                        self.daily_costs[today] += cost
                        if self.daily_costs[today] > DAILY_COST_ALERT_EUR:
                            print(f"⚠️ [COST ALERT] Današnji stroški: {self.daily_costs[today]:.2f} EUR")

                    if LLM_DEBUG:
                        print(f"[LLM] OK ({model}) in={in_toks} out={out_toks} cost={cost:.4f} EUR")
                    return output_text, successful_model

                except Exception as e:
                    last_err = e
                    with self._stats_lock:
                        self.model_stats[model]["failures"] += 1
                    if LLM_DEBUG:
                        print(f"[LLM] FAIL ({model}) attempt={attempt}: {e}")
                    if attempt == LLM_MAX_RETRIES:
                        break
                    time.sleep(min(delay, LLM_MAX_BACKOFF))
                    delay *= 1.8
        if LLM_DEBUG:
            print(f"[LLM] All candidates failed. Last error: {last_err}")
        raise last_err or RuntimeError("LLM call failed")

    def get_stats(self) -> Dict:
        with self._stats_lock:
            result = {}
            for model, st in self.model_stats.items():
                avg_time = st["total_time"] / st["calls"] if st["calls"] else 0.0
                result[model] = {
                    "calls": st["calls"],
                    "avg_time": round(avg_time, 2),
                    "failures": st["failures"],
                    "total_cost_eur": round(st["total_cost_eur"], 4),
                    "success_rate": round((1 - st["failures"] / max(st["calls"], 1)) * 100, 1)
                }
            return result

# -----------------------------------------------------------------------------
# Knowledge Base Manager
# -----------------------------------------------------------------------------
class KnowledgeBaseManager:
    def __init__(self):
        self.collection = None
        self._all_docs_cache = None
        self.chroma_path = CHROMA_DB_PATH
        self.collection_name = COLLECTION_NAME

    def load(self) -> bool:
        try:
            print(f"Poskušam naložiti bazo znanja iz: {self.chroma_path}")
            openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name=EMBEDDING_MODEL_NAME
            )
            chroma_client = chromadb.PersistentClient(path=self.chroma_path)
            self.collection = chroma_client.get_collection(
                name=self.collection_name,
                embedding_function=openai_ef
            )
            print(f"Povezano. Število dokumentov: {self.collection.count()}")
            return True
        except Exception as e:
            print(f"KRITIČNA NAPAKA: Baze znanja ni mogoče naložiti. Razlog: {e}")
            self.collection = None
            return False

    def _chroma_get_safe(self, **kwargs):
        include = kwargs.get("include")
        if include:
            include = [i for i in include if i in ("documents", "embeddings", "metadatas", "distances", "uris", "data")]
            kwargs["include"] = include
        try:
            return self.collection.get(**kwargs)
        except Exception:
            kwargs.pop("include", None)
            return self.collection.get(**kwargs)

    def search(self, query: str, n_results: int = 5) -> Dict:
        if not self.collection:
            return {}
        return self.collection.query(
            query_texts=[query],
            n_results=n_results,
            include=["documents", "metadatas"]
        )

    def get_waste_documents(self) -> Dict:
        if not self.collection:
            return {}
        return self._chroma_get_safe(
            where={"kategorija": "Odvoz odpadkov"},
            include=["documents", "metadatas"],
            limit=5000
        )

    def get_all_documents_cached(self) -> List[Dict]:
        cached = _global_cache.get("all_docs")
        if cached is not None:
            return cached
        if self._all_docs_cache is not None:
            return self._all_docs_cache
        try:
            res = self._chroma_get_safe(include=["documents", "metadatas"], limit=5000)
            docs = []
            if res and res.get("documents"):
                for i in range(len(res["documents"])):
                    docs.append({
                        "text": res["documents"][i],
                        "meta": res["metadatas"][i] if res.get("metadatas") else {}
                    })
            self._all_docs_cache = docs
            _global_cache.set("all_docs", docs, ttl_seconds=3600)
            return docs
        except Exception:
            return []

# -----------------------------------------------------------------------------
# Traffic Service (NAP) — POPRAVLJENO z lokalnim filtriranjem
# -----------------------------------------------------------------------------
# POPRAVEK: Celoten razred je posodobljen za delo z lokacijami
class TrafficService:
    def __init__(self, cache: ThreadSafeCache, location_keywords: List[str]):
        """Inicializira servis z lokacijskimi ključnimi besedami za filtriranje."""
        self.cache = cache
        self.location_keywords = location_keywords # Uporabljamo lokacije za filter
        self._token = None
        self._token_expiry = None

    def _ensure_token(self):
        if self._token and datetime.now(UTC) < self._token_expiry - timedelta(seconds=60):
            return self._token
        if not NAP_USERNAME or not NAP_PASSWORD:
            raise RuntimeError("NAP poverilnice niso nastavljene.")
        payload = {'grant_type': 'password', 'username': NAP_USERNAME, 'password': NAP_PASSWORD}
        resp = requests.post(NAP_TOKEN_URL, data=payload, headers={'Content-Type': 'application/x-www-form-urlencoded'}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        self._token, self._token_expiry = data['access_token'], datetime.now(UTC) + timedelta(seconds=data.get('expires_in', 3600))
        return self._token

    def check_traffic(self) -> str:
        """Pridobi in filtrira prometne dogodke za relevantne lokacije."""
        cached = self.cache.get("nap_traffic")
        if cached:
            return cached + "\n*(iz predpomnilnika)*"
        
        try:
            token = self._ensure_token()
            resp = requests.get(NAP_DATA_URL, headers={'Authorization': f'Bearer {token}'}, timeout=15)
            resp.raise_for_status()
            events = []
            
            for f in resp.json().get('features', []):
                props = f.get('properties', {})
                opis_celoten = props.get('opis', '') + props.get('cesta', '') + props.get('imeDogodka', '')
                normalized_opis = normalize_text(opis_celoten)
                
                # POPRAVEK: Preverjamo, ali opis vsebuje katero od NAŠIH lokacij
                if any(loc in normalized_opis for loc in self.location_keywords):
                    events.append(props)

            if not events:
                result = "Po podatkih portala promet.si na območju občine Rače-Fram trenutno ni zabeleženih posebnosti v prometu."
            else:
                ts = datetime.now().strftime("%d.%m.%Y %H:%M")
                report = [f"**Stanje na cestah (vir: NAP/promet.si, {ts})**\n*Prikazani so samo dogodki, relevantni za občino Rače-Fram.*"]
                
                merged = {e['opis']: e for e in events}.values()
                for e in merged:
                    report.append(f"- **Kje:** {e.get('cesta', 'N/A')}\n  **Opis:** {e.get('opis', 'N/A')}")
                result = "\n".join(report)
            
            self.cache.set("nap_traffic", result, ttl_seconds=900) # Cache za 15 min
            return result
        
        except Exception as e:
            print(f"NAP API napaka: {e}")
            return "Žal mi neposreden vpogled v stanje na cestah trenutno ne deluje."

# -----------------------------------------------------------------------------
# Waste Service - POPOLN z vsemi metodami
# -----------------------------------------------------------------------------
class WasteService:
   def __init__(self, kb_manager: KnowledgeBaseManager):
       self.kb = kb_manager

   def _extract_datumi(self, doc_text: str) -> List[str]:
       """Ekstrahira datume iz teksta"""
       ds = []
       for m in re.findall(r'(\d{1,2})\.(\d{1,2})\.', doc_text):
           try:
               d, mo = int(m[0]), int(m[1])
               ds.append((d, mo))
           except Exception:
               continue
       out, seen = [], set()
       for d, mo in ds:
           key = (d, mo)
           if key not in seen:
               seen.add(key)
               out.append(f"{d}.{mo}.")
       return out

   def _find_matches(self, urniki, location_phrases, waste_type):
       """Najde ujemanja za odpadke"""
       phrase_roots = set()
       for p in location_phrases:
           phrase_roots |= roots_of_phrase(p)

       street_indicators = {"cesta", "ulica", "pot", "trg", "naslov"}
       is_explicit_location = (any(len(p.split()) > 1 for p in location_phrases) or
                               any(ind in normalize_text(" ".join(location_phrases)) for ind in street_indicators))

       exact_street_matches, fuzzy_street_matches, area_matches = [], [], []

       docs = urniki['documents']
       metas = urniki.get('metadatas') or [{}] * len(docs)

       for i in range(len(docs)):
           meta = metas[i] or {}
           doc_text = docs[i] or ""

           meta_tip_raw = meta.get('tip_odpadka', '') or ''
           meta_tip_canon = get_canonical_waste_type(meta_tip_raw) or normalize_text(meta_tip_raw)
           if waste_type and meta_tip_canon != waste_type:
               continue

           lokacije = extract_locations_from_naselja(meta.get('naselja', '') or '')
           obm = normalize_text(meta.get('obmocje', '') or '')

           matched_for_this_doc = False

           # Exact matching
           for phrase in location_phrases:
               pr = roots_of_phrase(phrase)
               for street_tok in lokacije:
                   s_roots = roots_of_phrase(street_tok)
                   if pr and s_roots and pr.isdisjoint(s_roots):
                       continue
                   if normalize_text(phrase) == normalize_text(street_tok) or slovenian_variant_equivalent(phrase, street_tok):
                       exact_street_matches.append({
                           'doc': doc_text, 'meta': meta, 'tip_canon': meta_tip_canon,
                           'matched_street': street_tok, 'matched_phrase': phrase,
                           'score': 1.0,
                           'datumi': self._extract_datumi(doc_text)
                       })
                       matched_for_this_doc = True
                       break
               if matched_for_this_doc:
                   break

           # Fuzzy matching
           if not matched_for_this_doc:
               for phrase in location_phrases:
                   pr = roots_of_phrase(phrase)
                   for street_tok in lokacije:
                       thr = 0.88 if len(phrase.split()) == 1 else 0.83
                       s_roots = roots_of_phrase(street_tok)
                       if pr and s_roots and pr.isdisjoint(s_roots):
                           continue
                       if street_phrase_matches(phrase, street_tok, threshold=thr):
                           base_score = max(
                               fuzzy_ratio(phrase, street_tok),
                               max((fuzzy_ratio(phrase, w) for w in street_tok.split() if w not in GENERIC_WORDS), default=0)
                           )
                           if len(phrase.split()) > 1:
                               base_score += 0.04
                           if pr & s_roots:
                               base_score += 0.03
                           fuzzy_street_matches.append({
                               'doc': doc_text, 'meta': meta, 'tip_canon': meta_tip_canon,
                               'matched_street': street_tok, 'matched_phrase': phrase,
                               'score': min(base_score, 1.0),
                               'datumi': self._extract_datumi(doc_text)
                           })
                           matched_for_this_doc = True
                           break
                   if matched_for_this_doc:
                       break

           # Area matching
           if not matched_for_this_doc and obm and phrase_roots:
               for phrase in location_phrases:
                   if phrase_roots.isdisjoint(roots_of_phrase(obm)):
                       continue
                   if fuzzy_match(phrase, obm, threshold=0.82):
                       area_matches.append({
                           'doc': doc_text, 'meta': meta, 'tip_canon': meta_tip_canon,
                           'matched_area': obm, 'matched_phrase': phrase,
                           'score': fuzzy_ratio(phrase, obm),
                           'datumi': self._extract_datumi(doc_text)
                       })
                       break

       if exact_street_matches:
           kandidati = exact_street_matches
       elif fuzzy_street_matches:
           fuzzy_street_matches.sort(key=lambda x: x.get('score', 0), reverse=True)
           kandidati = fuzzy_street_matches
       else:
           kandidati = area_matches

       if kandidati and is_explicit_location:
           primary = None
           for size in (3, 2, 1):
               cands = [p for p in location_phrases if len(p.split()) == size]
               if cands:
                   primary = cands[0]
                   break
           if primary:
               pr = roots_of_phrase(primary)
               filt = [c for c in kandidati if pr & roots_of_phrase(c.get('matched_street', '') or c.get('matched_area', ''))]
               if filt:
                   filt.sort(key=lambda x: x.get('score', 0), reverse=True)
                   kandidati = filt

       return kandidati

   def process_waste_query(self, question: str, session_state: Dict) -> str:
       """Glavna metoda za procesiranje odpadkov"""
       vprasanje_norm = normalize_text(question)

       vsi_urniki = self.kb.get_waste_documents()
       if not vsi_urniki or not vsi_urniki.get('documents'):
           return "V bazi znanja ni podatkov o urnikih."

       iskani_tip = get_canonical_waste_type(vprasanje_norm)
       contains_naslednji = "naslednji" in vprasanje_norm

       # Ekstrakcija lokacij
       waste_type_stopwords = {normalize_text(k) for k in WASTE_TYPE_VARIANTS.keys()}
       for variants in WASTE_TYPE_VARIANTS.values():
           for v in variants:
               waste_type_stopwords.add(normalize_text(v))
       extra_stop = {
           "kdaj", "je", "naslednji", "odvoz", "odpadkov", "odpadke", "smeti", "na", "v",
           "za", "kako", "kateri", "katera", "kaj", "kje", "rumene", "rumena", "kanta",
           "kante", "ulici", "cesti"
       }
       odstrani = waste_type_stopwords.union(extra_stop)

       raw_tokens = [t for t in re.split(r'[,\s]+', vprasanje_norm) if t and t not in odstrani]

       location_phrases = []
       for size in (3, 2, 1):
           for i in range(len(raw_tokens) - size + 1):
               phrase = " ".join(raw_tokens[i:i + size])
               if phrase and phrase not in location_phrases:
                   location_phrases.append(phrase)

       if not location_phrases and iskani_tip:
           return "Dodaj prosim ulico ali območje, npr. 'Bistriška cesta, Fram'."

       kandidati = self._find_matches(vsi_urniki, location_phrases, iskani_tip)

       if not kandidati:
           if not iskani_tip:
               return "Kateri tip odpadka te zanima? (bio, mešani komunalni, embalaža, papir, steklo)"
           return "Za navedeno kombinacijo tipa in lokacije žal nimam urnika."

       if contains_naslednji:
           return self._format_next_collection(kandidati)
       else:
           return self._format_schedule(kandidati[0])

   def _format_next_collection(self, candidates):
       """Formatira odgovor za naslednji odvoz"""
       today = datetime.now().date()
       best_dt, best_tip, best_loc = None, None, None

       for c in candidates:
           tip = c['meta'].get('tip_odpadka', c.get('tip_canon', ''))
           loc = c.get('matched_street') or c.get('matched_area') or c['meta'].get('obmocje', '')

           for m in re.findall(r'(\d{1,2})\.(\d{1,2})\.', c.get('doc', '')):
               try:
                   dd, mm = int(m[0]), int(m[1])
                   year = today.year
                   dt = datetime(year, mm, dd).date()
                   if dt < today:
                       dt = datetime(year + 1, mm, dd).date()
                   if (best_dt is None) or (dt < best_dt):
                       best_dt, best_tip, best_loc = dt, tip, loc
               except Exception:
                   continue

       if best_dt:
           return f"Naslednji odvoz za **{best_tip}** na **{(best_loc or '').title()}** je **{best_dt.strftime('%d.%m.%Y')}**."
       return "Za iskani tip in lokacijo ni prihodnjih terminov."

   def _format_schedule(self, candidate):
       """Formatira urnik odpadkov"""
       tip = candidate['meta'].get('tip_odpadka', candidate.get('tip_canon', ''))
       loc = (candidate.get('matched_street') or candidate.get('matched_area') or candidate['meta'].get('obmocje', '')).title()
       datumi = candidate.get('datumi') or self._extract_datumi(candidate.get('doc', ''))
       datumi_str = ", ".join(datumi) if datumi else "ni zabeleženih terminov"
       return f"Odvoz – {loc}: **{tip}**. Termini: {datumi_str}"

# -----------------------------------------------------------------------------
# Rule-based Service
# -----------------------------------------------------------------------------
class RuleBasedService:
   def __init__(self, kb_manager: KnowledgeBaseManager):
       self.kb = kb_manager

   def check_rules(self, question_norm: str) -> Optional[str]:
       docs = self.kb.get_all_documents_cached()
       if not docs:
           return None

       # Direktor(ica) OU
       if ("direktor" in question_norm and "obcins" in question_norm) or \
          ("direktorica" in question_norm and "obcins" in question_norm):
           ime, tel, mail = self._find_director_info(docs)
           if ime or tel or mail:
               parts = [f"**Direktorica občinske uprave** je **{ime or 'mag. Karmen Kotnik'}**."]
               if tel: parts.append(f"**Telefon:** {tel}")
               if mail: parts.append(f"**E-pošta:** {mail}")
               return " ".join(parts)

       # Turizem
       if "turizem" in question_norm or "turistic" in question_norm:
           t = self._find_tourism_info(docs)
           if t: return t

       # E-pošta občine
       if ("e-post" in question_norm or "email" in question_norm or "e mail" in question_norm) and \
          ("obcine" in question_norm or "obcina" in question_norm):
           e = self._find_municipality_email(docs)
           if e: return e

       # EV polnilnice
       if ("polnilnic" in question_norm or "polnilnice" in question_norm or "ev" in question_norm) and \
          ("obcin" in question_norm or "race" in question_norm or "fram" in question_norm):
           p = self._find_ev_stations(docs)
           if p: return p

       return None

   def _find_director_info(self, docs):
       ime, tel, mail = None, None, None
       for d in docs:
           t = d["text"]
           n = normalize_text(t)
           if "direktorica obcinske uprave" in n or "direktor obcinske uprave" in n or "karmen kotnik" in n:
               if "karmen kotnik" in n:
                   ime = "mag. Karmen Kotnik"
               m = re.search(r'[\w\.-]+@race-fram\.si', t, re.IGNORECASE)
               if m: mail = m.group(0)
               m2 = re.search(r'\b02\s?609\s?60\s?\d{2}\b', t)
               if m2: tel = m2.group(0)
       return ime, tel, mail

   def _find_tourism_info(self, docs):
       for d in docs:
           n = normalize_text(d["text"])
           if "tanja kosi" in n and ("turizem" in n or "turistic" in n):
               return "Za področje **turizma** je zadolžena **Tanja Kosi** (višja svetovalka za okolje, kmetijstvo, turizem in CZ)."
       return None

   def _find_municipality_email(self, docs):
       emails = set()
       for d in docs:
           for m in re.findall(r'[\w\.-]+@race-fram\.si', d["text"], re.IGNORECASE):
               if m.lower() in ("obcina@race-fram.si", "info@race-fram.si"):
                   emails.add(m)
       if emails:
           return "**E-poštni naslov občine Rače-Fram**: " + " ali ".join(sorted(emails))
       return None

   def _find_ev_stations(self, docs):
       for d in docs:
           n = normalize_text(d["text"])
           if "polnilnic" in n or "polnilnice" in n:
               if ("grad race" in n or "dtv partizan fram" in n):
                   return "**Električne polnilnice** v občini: Grad Rače in DTV Partizan Fram. Cena: 0,25 EUR/kWh"
       return "**Električne polnilnice** v občini - preverite na občinski upravi."

# -----------------------------------------------------------------------------
# Glavni orchestrator - Virtualni Župan
# -----------------------------------------------------------------------------
class VirtualniZupan:
   def __init__(self, kb_manager=None, traffic_service=None, waste_service=None,
                rule_service=None, model_router=None, logger=None):
       print("Inicializacija razreda VirtualniZupan (Verzija 40.5 — Traffic Fix)...")

       # Čisti dependency injection
       self.kb_manager = kb_manager
       self.model_router = model_router
       self.traffic_service = traffic_service
       self.waste_service = waste_service
       self.rule_service = rule_service
       self.logger = logger

       self.zgodovina_seje = {}
       self.total_questions = 0

   def preoblikuj_vprasanje_s_kontekstom(self, zgodovina_pogovora, zadnje_vprasanje):
       if not zgodovina_pogovora:
           return zadnje_vprasanje
       try:
           zgodovina_str = "\n".join([f"Uporabnik: {q}\nAsistent: {a}" for q, a in zgodovina_pogovora])
           prompt = f"""Preoblikuj novo vprašanje v samostojno vprašanje glede na zgodovino.
Zgodovina:
{zgodovina_str}
Novo vprašanje: "{zadnje_vprasanje}"
Samostojno vprašanje:"""
           out, _ = self.model_router.chat(
               messages=[{"role": "user", "content": prompt}],
               task="rewrite", temperature=0.0, max_tokens=120, request_name="rewrite_with_context"
           )
           return out.replace('"', '') or zadnje_vprasanje
       except Exception:
           return zadnje_vprasanje

   def odgovori(self, uporabnikovo_vprasanje: str, session_id: str):
       start_time = time.time()
       self.total_questions += 1
       model_used, was_fallback = None, False

       self.zgodovina_seje.setdefault(session_id, {'zgodovina': [], 'stanje': {}})
       zgodovina = self.zgodovina_seje[session_id]['zgodovina']

       pametno_vprasanje = self.preoblikuj_vprasanje_s_kontekstom(zgodovina, uporabnikovo_vprasanje)
       vprasanje_lower = normalize_text(pametno_vprasanje)

       if LLM_DEBUG:
           print(f"[ROUTER] Original: '{uporabnikovo_vprasanje[:50]}...'")
           print(f"[ROUTER] Rewritten: '{pametno_vprasanje[:50]}...'")

       try:
           is_waste = _has_waste_intent(vprasanje_lower)
           is_traffic = _has_traffic_intent(vprasanje_lower)

           if LLM_DEBUG:
               print(f"[ROUTER] waste={is_waste}, traffic={is_traffic}")

           # 1) Waste ima absolutno prednost
           if is_waste:
               odgovor = self.waste_service.process_waste_query(
                   pametno_vprasanje, self.zgodovina_seje[session_id]['stanje']
               )
               model_used = "rule-based-waste"

           # 2) Promet samo če ni waste
           elif is_traffic:
               odgovor = self.traffic_service.check_traffic()
               model_used = "nap-api"

           # 3) Rule-based
           else:
               rb = self.rule_service.check_rules(vprasanje_lower)
               if rb:
                   odgovor = rb
                   model_used = "rule-based"
               else:
                   # 4) RAG + LLM
                   odgovor, model_used = self._process_with_rag(uporabnikovo_vprasanje, vprasanje_lower)

       except Exception as e:
           print(f"[ERROR] Kritična napaka: {e}")
           odgovor = "Prišlo je do napake. Poskusite ponovno."
           was_fallback = True
           self.logger.log_interaction(session_id, uporabnikovo_vprasanje, odgovor,
                                       response_time=time.time()-start_time, was_fallback=True, error=str(e))

       # Logging & zgodovina
       response_time = time.time() - start_time
       zgodovina.append((uporabnikovo_vprasanje, odgovor))
       if len(zgodovina) > 4:
           zgodovina.pop(0)

       self.logger.log_interaction(session_id, uporabnikovo_vprasanje, odgovor,
                                   model_used=model_used, response_time=response_time, was_fallback=was_fallback)
       return odgovor

   def _process_with_rag(self, original_question: str, normalized_question: str) -> Tuple[str, str]:
       rezultati = self.kb_manager.search(normalized_question, n_results=5)
       kontekst = ""
       if rezultati.get('documents'):
           for doc, meta in zip(rezultati['documents'][0], rezultati['metadatas'][0]):
               kontekst += f"--- VIR: {meta.get('source', 'Neznan')}\nVSEBINA: {doc}\n\n"

       if not kontekst:
           return "Žal o tem nimam informacij.", "no-context"

       now = datetime.now()
       prompt = f"""Ti si 'Virtualni župan občine Rače-Fram'.
DIREKTIVA #1: Današnji datum je {now.strftime('%d.%m.%Y')}. Če je podatek iz leta manjšega od {now.year}, ga IGNORIRAJ.
DIREKTIVA #2: Ključne informacije **poudari**. Uporabi alineje (-).
DIREKTIVA #3: Če najdeš URL, ga vključi kot [Ime](URL).

KONTEKST:
{kontekst}
VPRAŠANJE: "{original_question}"
ODGOVOR:"""

       try:
           odgovor, model = self.model_router.chat(
               messages=[{"role": "user", "content": prompt}],
               task="rag_answer", temperature=0.0, request_name="rag_answer"
           )
           return odgovor, model
       except Exception:
           if kontekst:
               return f"Našel sem informacije, vendar imam tehnične težave:\n{kontekst[:500]}", "fallback"
           return "Tehnične težave. Poskusite ponovno.", "fallback"

   def get_system_stats(self) -> Dict:
       model_stats = self.model_router.get_stats()
       today = date.today().isoformat()
       today_cost = self.model_router.daily_costs.get(today, 0.0)
       return {
           "timestamp": datetime.now().isoformat(),
           "total_questions": self.total_questions,
           "active_sessions": len(self.zgodovina_seje),
           "model_stats": model_stats,
           "cache_stats": _global_cache.stats(),
           "today_cost_eur": round(today_cost, 4),
           "cost_alert_triggered": today_cost > DAILY_COST_ALERT_EUR,
           "db_documents": self.kb_manager.collection.count() if self.kb_manager.collection else 0,
           "economy_mode": ECONOMY_MODE,
           "force_model": FORCE_MODEL or None
       }

   def print_stats(self):
       stats = self.get_system_stats()
       print("\n" + "="*60)
       print("VIRTUALNI ŽUPAN v40.5 - STATISTIKA")
       print("="*60)
       print(f"Skupno vprašanj: {stats['total_questions']}")
       print(f"Današnji stroški: {stats['today_cost_eur']:.4f} EUR")
       if stats['model_stats']:
           print("\nMODELI:")
           for model, data in stats['model_stats'].items():
               print(f"  {model}: {data['calls']} klicev, {data['total_cost_eur']:.4f} EUR (fail: {data.get('failures', 0)})")
       print("\nCACHE:")
       cache_info = stats['cache_stats']
       print(f"  Aktivni vnosi: {cache_info.get('active_entries', 0)} / {cache_info.get('total_entries', 0)}")
       if cache_info.get('hits'):
           print(f"  Zadetki: {cache_info['hits']}")
       print("="*60)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def _cmd_repl(zupan: VirtualniZupan):
   print("Župan je pripravljen. ('konec' za izhod, 'stats' za statistiko)")
   session_id = "local_cli"
   while True:
       try:
           q = input("> ").strip()
           if q.lower() in ("konec", "exit", "quit"):
               break
           if q.lower() == "stats":
               zupan.print_stats()
               continue
           if not q:
               continue
           print("\n--- ODGOVOR ---")
           print(zupan.odgovori(q, session_id))
           print("--------------\n")
       except KeyboardInterrupt:
           break
   zupan.print_stats()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Virtualni župan v40.5 - Z lokalnim prometom")
    parser.add_argument("--stats", action="store_true", help="Samo statistika")
    parser.add_argument("--test", action="store_true", help="Hitri testi")
    args = parser.parse_args()

    try:
        # Sestavljanje aplikacije (Composition Root), kjer ustvarimo vse objekte
        
        # 1. Osnovni servisi, ki jih drugi potrebujejo
        cache = ThreadSafeCache()
        logger = SystemLogger(LOCAL_LOG_DIR)
        
        # KnowledgeBaseManager si poti nastavi sam iz globalnih konstant
        kb = KnowledgeBaseManager()
        kb.load()
        
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        router = ModelRouter(openai_client)
        
        # 2. Servisi, ki so odvisni od osnovnih
        # POPRAVEK: TrafficService inicializiramo s seznamom LOKACIJ namesto splošnih besed
        traffic = TrafficService(cache=cache, location_keywords=PROMET_FILTER_LOKACIJE)
        waste = WasteService(kb)
        rules = RuleBasedService(kb)
        
        # 3. Glavni objekt, ki uporablja vse servise
        zupan = VirtualniZupan(
            kb_manager=kb, 
            traffic_service=traffic, 
            waste_service=waste, 
            rule_service=rules, 
            model_router=router, 
            logger=logger
        )

    except Exception as e:
        print(f"Napaka pri inicializaciji! Preverite .env datoteko in poti. Napaka: {e}")
        sys.exit(1)

    # Obdelava argumentov iz ukazne vrstice
    if args.stats:
        zupan.print_stats()
        sys.exit(0)

    if args.test:
        print("TEST MODE - osnovne funkcionalnosti:")
        tests = [
            "kdaj je odvoz stekla na Bistriški cesti?",
            "kdaj je odvoz papirja na Mlinski ulici?",
            "kakšno je stanje na cestah?",
            "kdo je direktor občinske uprave?",
            "kaj je komunalni prispevek in kako ga uredim?"
        ]
        for q in tests:
            print(f"\nQ: {q}")
            print("A:", zupan.odgovori(q, "test_session"))
        sys.exit(0)

    # Zagon interaktivne zanke (CLI)
    _cmd_repl(zupan)