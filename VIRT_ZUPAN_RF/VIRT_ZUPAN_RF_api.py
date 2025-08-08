# VIRT_ZUPAN_core_v50.py
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
from urllib.parse import urlparse

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
    KLJUCNE_PROMET: Tuple[str, ...]  = ("cesta", "ceste", "cesti", "promet", "zastoj", "gneča", "kolona", "zapora", "zapore")
    PROMET_FILTER: Tuple[str, ...]   = ("rače", "fram", "slivnica", "brunšvik", "podova", "morje", "hoče", "r2-430", "r3-711", "g1-2", "priključek slivnica", "razcep slivnica", "letališče maribor", "odcep za rače")

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

    # Fallbacki za povezave
    VLOGA_ZAPORA_FALLBACK_URL: str = "https://www.race-fram.si/objava/400297"
    GRADBENO_OFFICIAL_URL: str     = "https://e-uprava.gov.si/si/podrocja/nepremicnine-in-okolje/nepremicnine-stavbe/gradbeno-dovoljenje.html?lang=si"

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
# 3) UTIL: normalizacija, ujemanje, URL filtri, entitete
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
    keys = {base}
    if base.endswith("ska"): keys.add(base[:-3] + "ski")
    if base.endswith("ski"): keys.add(base[:-3] + "ska")
    if base.endswith("a"):   keys.add(base[:-1] + "i")
    if base.endswith("i"):   keys.add(base[:-1] + "a")
    return [k for k in keys if k]

def parse_dates_from_text(text: str) -> List[str]:
    dates = re.findall(r'(\d{1,2}\.\d{1,2}\.?)', text or "")
    seen = set(); out = []
    for d in dates:
        dn = d if d.endswith('.') else d + '.'
        if dn not in seen:
            seen.add(dn); out.append(dn)
    return out

def get_canonical_waste(text: str) -> Optional[str]:
    norm = normalize_text(text)
    if "stekl" in norm: return "Steklena embalaža"
    if "papir" in norm or "karton" in norm: return "Papir in karton"
    if "bio" in norm or "biolo" in norm: return "Biološki odpadki"
    if ("komunaln" in norm and "odpadk" in norm) or re.search(r'\bmesan|\bmešani|\bme{s|š}an', norm):
        return "Mešani komunalni odpadki"
    if (("rumen" in norm and "kant" in norm) or "embala" in norm) and "stekl" not in norm:
        return "Odpadna embalaža"
    for canon, variants in cfg.WASTE_VARIANTS.items():
        for v in variants:
            if normalize_text(v) in norm:
                return canon
    for canon, variants in cfg.WASTE_VARIANTS.items():
        for v in variants:
            if _ratio(norm, normalize_text(v)) >= 0.90:
                return canon
    return None

def extract_locations_from_naselja(field: str) -> List[str]:
    parts = set()
    if not field: return []
    clean = re.sub(r'\(h\. *št\..*?\)', '', field)
    segments = re.split(r'([A-ZČŠŽ][a-zčšž]+\s*:)', clean)
    for seg in segments:
        seg = seg.strip()
        if not seg or seg.endswith(':'): continue
        for sub in seg.split(','):
            n = normalize_text(sub)
            if n: parts.add(n)
    return list(parts)

def tokens_from_text(s: str) -> set:
    return {t for t in re.split(r'[^a-z0-9]+', normalize_text(s or "")) if len(t) > 2}

def extract_year_from_query(q_norm: str) -> Optional[int]:
    yrs = re.findall(r'\b(20\d{2})\b', q_norm)
    return int(yrs[0]) if yrs else None

def url_is_relevant(url: str, doc: str, q_tokens: set, must:set=None, ban:set=None) -> bool:
    if not url: return False
    try:
        u = urlparse(url)
    except Exception:
        return False
    path_tokens = tokens_from_text(u.path)
    must = must or set(); ban = ban or set()
    if ban & path_tokens: return False
    if must and not (must & path_tokens):
        doc_tokens = tokens_from_text(doc)
        if not (must & doc_tokens):
            return False
    if not (q_tokens & path_tokens):
        doc_tokens = tokens_from_text(doc)
        if len(q_tokens & doc_tokens) < 2:
            return False
    return True

# ------------------------------------------------------------------------------
# 4) DEKLARATIVNE VEŠČINE (router-friendly)
# ------------------------------------------------------------------------------
SKILLS = [
    {
        "name": "waste",
        "label": "Odvoz odpadkov po ulici in tipu (bio, mešani, embalaža, papir, steklo)",
        "must_words": ["odvoz", "odpad"],
        "ban_words": ["zapora", "promet", "vloga", "obrazec"],
        "entities": ["street", "waste_type", "next_only"],
        "examples": ["kdaj je odvoz stekla na bistriški cesti", "naslednji odvoz papirja pod terasami"]
    },
    {
        "name": "traffic_status",
        "label": "Stanje na cestah / zapore / dela",
        "must_words": ["cest", "promet", "zapora", "zapore", "zastoj"],
        "ban_words": ["vloga", "obrazec", "dovoljen"],
        "entities": [],
        "examples": ["ali v občini potekajo kakšna dela na cesti"]
    },
    {
        "name": "traffic_form",
        "label": "Vloga/obrazec za zaporo ceste",
        "must_words": ["vloga", "obrazec", "zaporo", "zapora", "ceste"],
        "ban_words": ["zastoj", "dela", "promet"],
        "entities": [],
        "examples": ["kje najdem vlogo za zaporo ceste", "obrazec za zaporo ceste"]
    },
    {
        "name": "awards",
        "label": "Občinske nagrade/petice po letih in dobitniki",
        "must_words": ["nagrada", "petica", "nagrajen", "dobitnik", "prejemnik"],
        "ban_words": [],
        "entities": ["year", "award_type"],
        "examples": ["kdo so občinski nagrajenci za leto 2012", "kdo so dobitniki županove petice za 2012"]
    },
    {
        "name": "fire_brigades",
        "label": "Seznam gasilskih društev PGD v občini + osnovni kontakti",
        "must_words": ["gasil", "pgd", "društv"],
        "ban_words": ["povelj"],
        "entities": [],
        "examples": ["katera gasilska društva imamo v občini"]
    },
    {
        "name": "fire_command",
        "label": "Poveljniški podatki (poveljnik/namestnik/predsednik) gasilskih društev",
        "must_words": ["gasil", "pgd", "povelj"],
        "ban_words": [],
        "entities": [],
        "examples": ["daj mi poveljniške podatke PGD"]
    },
    {
        "name": "mun_tax",
        "label": "Komunalni prispevek (osnovno, kdo, kdaj, kam, priloge, izračun, kontakt, povezava)",
        "must_words": ["komunaln", "prispev"],
        "ban_words": [],
        "entities": [],
        "examples": ["kje najdem informacije o komunalnem prispevku"]
    },
    {
        "name": "building_permit",
        "label": "Gradbeno dovoljenje (koraki, UE, veljavnost, link eUprava)",
        "must_words": ["gradben", "dovoljen"],
        "ban_words": [],
        "entities": [],
        "examples": ["kako dobim gradbeno dovoljenje"]
    },
    {
        "name": "summer_camp",
        "label": "Poletni tabor/varstvo – samo aktualno leto, razen če je v vprašanju letos navedeno",
        "must_words": ["poletn", "tabor", "kamp", "varstvo"],
        "ban_words": [],
        "entities": ["year"],
        "examples": ["ali imamo v občini poletni kamp"]
    },
    {
        "name": "mayor",
        "label": "Kdo je župan",
        "must_words": ["kdo", "župan", "zupan"],
        "ban_words": [],
        "entities": [],
        "examples": ["kdo je župan občine"]
    },
    {
        "name": "general",
        "label": "Splošne informacije (fallback RAG z varovalkami)",
        "must_words": [],
        "ban_words": [],
        "entities": [],
        "examples": ["povej mi nekaj informacij o framu"]
    },
]

# ------------------------------------------------------------------------------
# 5) GLAVNI RAZRED
# ------------------------------------------------------------------------------
class VirtualniZupan:
    def __init__(self) -> None:
        prefix = "PRODUCTION" if cfg.ENV_TYPE == 'production' else "DEVELOPMENT"
        logger.info(f"[{prefix}] VirtualniŽupan v50.0 (semantični router) inicializiran.")
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
            return ""

    def _call_llm_json(self, system_desc: str, user_prompt: str, max_tokens: int = 300) -> Dict[str, Any]:
        try:
            res = self.openai_client.chat.completions.create(
                model=cfg.GENERATOR_MODEL,
                response_format={"type":"json_object"},
                messages=[
                    {"role":"system","content":system_desc},
                    {"role":"user","content":user_prompt}
                ],
                temperature=0.0,
                max_tokens=max_tokens
            )
            txt = res.choices[0].message.content.strip()
            return json.loads(txt)
        except Exception:
            logger.exception("LLM JSON klasifikacija ni uspela")
            return {}

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

    # ---- semantični router (LLM JSON + pravila)
    def _route(self, q: str) -> Dict[str, Any]:
        q_norm = normalize_text(q)
        sys_desc = (
            "Razvrsti uporabniško vprašanje v ENO izmed veščin in izlušči entitete.\n"
            "Vrni JSON s ključi: intent (string iz {"+",".join(s['name'] for s in SKILLS)+"}), "
            "confidence (0-1), entities (object s ključi med: year(int), award_type(str), street(str), waste_type(str), next_only(bool))."
        )
        skill_desc = []
        for sk in SKILLS:
            skill_desc.append({
                "name": sk["name"],
                "label": sk["label"],
                "must_words": sk["must_words"],
                "ban_words": sk["ban_words"],
                "entities": sk["entities"],
                "examples": sk["examples"],
            })
        user_p = json.dumps({"question": q, "skills": skill_desc}, ensure_ascii=False)
        out = self._call_llm_json(sys_desc, user_p, max_tokens=300) or {}

        # Fallback pravila & entitete
        entities = out.get("entities", {}) if isinstance(out.get("entities"), dict) else {}
        # leto
        year = extract_year_from_query(q_norm)
        if year and not entities.get("year"):
            entities["year"] = year
        # award_type
        if "petic" in q_norm and not entities.get("award_type"):
            entities["award_type"] = "petica"
        elif any(w in q_norm for w in ["nagrad", "županov", "zupanov"]) and not entities.get("award_type"):
            entities["award_type"] = "nagrada"
        # waste next_only
        if "naslednji" in q_norm and entities.get("next_only") is None:
            entities["next_only"] = True

        # Dodatni pravilo-router: če res očitno
        def has_any(words): return any(w in q_norm for w in words)
        if has_any(["vloga","obrazec"]) and has_any(["zapora","zaporo"]) and not has_any(["zastoj","promet"]):
            out["intent"] = "traffic_form"; out["confidence"] = max(out.get("confidence",0), 0.9)
        if has_any(["kdo je"]) and has_any(["župan","zupan"]):
            out["intent"] = "mayor"; out["confidence"] = 0.99

        out["entities"] = entities
        if "intent" not in out or out.get("intent") not in {s["name"] for s in SKILLS}:
            out["intent"] = "general"
            out["confidence"] = 0.3
        return out

    # ---- spomin
    def preoblikuj_vprasanje_s_kontekstom(self, zgodovina_pogovora: List[Tuple[str, str]], zadnje_vprasanje: str) -> str:
        if not zgodovina_pogovora:
            return zadnje_vprasanje
        logger.info("Kličem specialista za spomin…")
        zgodovina_str = "\n".join([f"Uporabnik: {q}\nAsistent: {a}" for q, a in zgodovina_pogovora])
        prompt = f"""Tvoja naloga je: na podlagi zgodovine pogovora preoblikuj novo vprašanje v samostojno vprašanje.
Če je že popolno, ga pusti.
---
Zgodovina:
{zgodovina_str}
Novo vprašanje: "{zadnje_vprasanje}"
Samostojno vprašanje:"""
        preoblikovano = self._call_llm(prompt, max_tokens=80) or ""
        return preoblikovano.replace('"','').strip() or zadnje_vprasanje

    # ---- NAP / promet
    def _ensure_nap_token(self) -> Optional[str]:
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
            logger.exception("NAP API napaka pri pridobivanju žetona.")
            return None

    def preveri_zapore_cest(self) -> str:
        logger.info("Promet (NAP API)…")
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

    # ---- Odpadki: indeks (tvoja obstoječa napredna logika)
    def _build_waste_indexes(self) -> None:
        logger.info("Gradim indekse za odpadke …")
        if not self.collection:
            return
        all_docs = self.collection.get(where={"kategorija": "Odvoz odpadkov"})
        if not all_docs or not all_docs.get('ids'):
            logger.warning("Ni dokumentov za odpadke.")
            return

        self._area_type_docs.clear()
        self._street_index.clear()

        for i in range(len(all_docs['ids'])):
            meta = all_docs['metadatas'][i]
            doc_text = all_docs['documents'][i]
            tip = get_canonical_waste(meta.get('tip_odpadka', '') or '') or (meta.get('tip_odpadka', '') or '')
            area = normalize_text(meta.get('obmocje', '') or '')
            self._area_type_docs[(area, tip)] = doc_text

            streets = extract_locations_from_naselja(meta.get('naselja', '') or '')
            for s in streets:
                display = s
                for key in gen_street_keys(s):
                    self._street_index.setdefault(key, []).append({
                        "area": area,
                        "tip": tip,
                        "doc": doc_text,
                        "display": display
                    })

        self._street_keys_list = list(self._street_index.keys())
        logger.info(f"Indeks: {len(self._street_keys_list)} uličnih ključev, {len(self._area_type_docs)} (območje,tip).")

    def _best_street_key_for_query(self, phrases: List[str]) -> Optional[str]:
        for ph in phrases:
            for key in gen_street_keys(ph):
                if key in self._street_index:
                    return key
        best = (None, 0.0)
        for ph in phrases:
            base = strip_generics(ph)
            if len(base) < 3: continue
            for key in self._street_keys_list:
                sc = _ratio(base, key)
                if sc > best[1]:
                    best = (key, sc)
        if best[0] and best[1] >= 0.88:
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
        return phrases

    def _format_dates_for_tip(self, street_display: str, tip_odpadka: str, doc_text: str, only_next: bool) -> Optional[str]:
        dates = parse_dates_from_text(doc_text)
        if not dates:
            return None
        if only_next:
            today = datetime.now().date()
            for d in dates:
                try:
                    dd, mm = d.replace('.',' ').strip().split()[:2]
                    dt = datetime(datetime.now().year, int(mm), int(dd)).date()
                    if dt >= today:
                        return f"Naslednji odvoz za **{tip_odpadka}** na **{street_display.title()}** je **{dd}.{mm}.**."
                except Exception:
                    continue
            return f"Za **{tip_odpadka}** na **{street_display.title()}** v tem letu ni več terminov."
        else:
            return f"Za **{street_display.title()}** je odvoz **{tip_odpadka}**: {', '.join(dates)}"

    def obravnavaj_odvoz_odpadkov(self, uporabnikovo_vprasanje: str, session_id: str, next_only: bool=False) -> str:
        ses = self.zgodovina_seje.setdefault(session_id, {'zgodovina': [], 'stanje': {}})
        stanje = ses['stanje']
        if not self.collection:
            return "Baza urnikov ni na voljo."

        vprasanje_norm = normalize_text(uporabnikovo_vprasanje)
        only_next = next_only or ("naslednji" in vprasanje_norm)

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
                            stanje['namen'] = 'odpadki'
                            stanje.pop('caka_na', None)
                            return ans
                return "Za navedeno ulico žal nisem našel urnika za izbrani tip."
        else:
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

    # ---- GASILCI: seznam + poveljstvo
    def obravnavaj_gasilci(self) -> str:
        if not self.collection:
            return "Podatkov trenutno nimam."
        query = "PGD gasilsko društvo Rače Fram Podova Spodnja Zgornja Gorica kontakt telefon email naslov"
        res = self.collection.query(query_texts=[query], n_results=30, include=["documents", "metadatas"])
        docs = res.get('documents', [[]])[0] if res else []

        name_pat = re.compile(r'\b(?:pgd|prostovoljno\s+gasilsko\s+druš[tv]o)\s+([a-z0-9\s\-čšžćđ]+)', re.IGNORECASE)
        phone_pat = re.compile(r'(?:\+?\s*386|0)\s*\d(?:[\s/\-]?\d){5,}', re.IGNORECASE)
        email_pat = re.compile(r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}')
        addr_pat = re.compile(r'(?:naslov|sede[žz])\s*:\s*(.+)', re.IGNORECASE)

        def canon_name(raw: str) -> str:
            raw = re.sub(r'\s+', ' ', raw.strip())
            disp = re.split(r'[,\n]', raw)[0].title()
            if not disp.lower().startswith("pgd "): disp = "PGD " + disp
            return disp

        entries: Dict[str, Dict[str, Any]] = {}
        for d in docs:
            if not d: continue
            current = None
            for ln in d.splitlines():
                mname = name_pat.search(ln)
                if mname:
                    nm = canon_name(mname.group(1))
                    current = entries.setdefault(nm, {"phone": set(), "email": set(), "address": set()})
                if current:
                    for m in phone_pat.findall(ln):  current["phone"].add(re.sub(r'\s+', ' ', m.strip()))
                    for m in email_pat.findall(ln):  current["email"].add(m.strip())
                    for m in addr_pat.findall(ln):   current["address"].add(m.strip())

        # zagotovimo VSA društva
        must_have = ["PGD Rače", "PGD Fram", "PGD Podova", "PGD Spodnja In Zgornja Gorica"]
        for nm in must_have:
            entries.setdefault(nm, {"phone": set(), "email": set(), "address": set()})

        out = ["**Gasilska društva v občini Rače-Fram** (osnovni kontakti):"]
        for nm in sorted(entries.keys()):
            e = entries[nm]
            det = []
            if e["phone"]:  det.append("telefon: " + ", ".join(sorted(e["phone"])))
            if e["email"]:  det.append("e-pošta: " + ", ".join(sorted(e["email"])))
            if e["address"]:det.append("naslov: " + ", ".join(sorted(e["address"])))
            line = f"- **{nm}**"
            if det: line += " — " + "; ".join(det)
            out.append(line)
        out.append("\nČe želiš, lahko za posamezno društvo poiščem **poveljniške podatke**.")
        return "\n".join(out)

    def obravnavaj_gasilci_poveljstvo(self) -> str:
        if not self.collection:
            return "Podatkov trenutno nimam."
        query = "PGD poveljnik namestnik predsednik kontakt e-pošta telefon Rače Fram Podova Spodnja Zgornja Gorica"
        res = self.collection.query(query_texts=[query], n_results=30, include=["documents", "metadatas"])
        docs = res.get('documents', [[]])[0] if res else []

        name_pat = re.compile(r'\b(?:pgd|prostovoljno\s+gasilsko\s+druš[tv]o)\s+([a-z0-9\s\-čšžćđ]+)', re.IGNORECASE)
        poveljnik_pat  = re.compile(r'\bpoveljnik\b\s*:\s*(.+)', re.IGNORECASE)
        namestnik_pat  = re.compile(r'\bnamestnik\b\s*:\s*(.+)', re.IGNORECASE)
        predsednik_pat = re.compile(r'\bpredsednik\b\s*:\s*(.+)', re.IGNORECASE)
        phone_pat = re.compile(r'(?:\+?\s*386|0)\s*\d(?:[\s/\-]?\d){5,}', re.IGNORECASE)
        email_pat = re.compile(r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}')

        def canon_name(raw: str) -> str:
            raw = re.sub(r'\s+', ' ', raw.strip())
            disp = re.split(r'[,\n]', raw)[0].title()
            if not disp.lower().startswith("pgd "): disp = "PGD " + disp
            return disp

        data: Dict[str, Dict[str, Any]] = {}
        for d in docs:
            if not d: continue
            current = None
            for ln in d.splitlines():
                mname = name_pat.search(ln)
                if mname:
                    nm = canon_name(mname.group(1))
                    current = data.setdefault(nm, {"poveljnik": set(), "namestnik": set(), "predsednik": set(), "telefon": set(), "email": set()})
                if current:
                    for m in poveljnik_pat.findall(ln):  current["poveljnik"].add(m.strip())
                    for m in namestnik_pat.findall(ln):  current["namestnik"].add(m.strip())
                    for m in predsednik_pat.findall(ln): current["predsednik"].add(m.strip())
                    for m in phone_pat.findall(ln):      current["telefon"].add(re.sub(r'\s+', ' ', m.strip()))
                    for m in email_pat.findall(ln):      current["email"].add(m.strip())

        must_have = ["PGD Rače", "PGD Fram", "PGD Podova", "PGD Spodnja In Zgornja Gorica"]
        for nm in must_have:
            data.setdefault(nm, {"poveljnik": set(), "namestnik": set(), "predsednik": set(), "telefon": set(), "email": set()})

        lines = ["**Poveljniški podatki gasilskih društev**:"]
        for nm in sorted(data.keys()):
            e = data[nm]
            parts = []
            if e["poveljnik"]: parts.append("poveljnik: " + ", ".join(sorted(e["poveljnik"])))
            if e["namestnik"]: parts.append("namestnik: " + ", ".join(sorted(e["namestnik"])))
            if e["predsednik"]: parts.append("predsednik: " + ", ".join(sorted(e["predsednik"])))
            if e["telefon"]:   parts.append("telefon: " + ", ".join(sorted(e["telefon"])))
            if e["email"]:     parts.append("e-pošta: " + ", ".join(sorted(e["email"])))
            line = f"- **{nm}**"
            if parts: line += " — " + "; ".join(parts)
            lines.append(line)
        return "\n".join(lines)

    # ---- VLOGA ZA ZAPORO CESTE (nikoli NAP)
    def obravnavaj_vloga_zapora(self, q_norm: str) -> str:
        if not self.collection:
            return f"**Vloga za zaporo ceste**: {cfg.VLOGA_ZAPORA_FALLBACK_URL}\n\nZa pomoč: {cfg.FALLBACK_CONTACT}"
        q_tokens = tokens_from_text(q_norm)
        res = self.collection.query(
            query_texts=["vloga obrazec zapora ceste dovoljenje prijava zapore"],
            n_results=10,
            include=["documents", "metadatas"]
        )
        docs = res.get('documents', [[]])[0] if res else []
        metas = res.get('metadatas', [[]])[0] if res else []

        chosen_url = None
        must = {"zapora","cest","vloga","obrazec","dovoljen","prijav"}
        ban  = {"promet","geojson","nap","roadworks","b2b"}
        for doc, meta in zip(docs, metas):
            url = (meta.get('source_url') or "").strip()
            if url and url_is_relevant(url, doc, q_tokens, must, ban):
                chosen_url = url; break
        if not chosen_url: chosen_url = cfg.VLOGA_ZAPORA_FALLBACK_URL

        return (
            f"**Vloga za zaporo ceste**: {chosen_url}\n\n"
            f"- Na povezavi so **obrazec** in **navodila** za oddajo.\n"
            f"- Dodatna pomoč: **{cfg.FALLBACK_CONTACT}**."
        )

    # ---- NAGRADE / NAGRAJENCI
    def obravnavaj_nagrade(self, q_norm: str, year: Optional[int], award_type: Optional[str]) -> str:
        if not self.collection:
            return "Podatkov o nagradah trenutno nimam."
        tokens = ["občina rače-fram","nagrajenci","dobitnik","prejemnik"]
        if year: tokens.append(str(year))
        if award_type: tokens.append(award_type)
        res = self.collection.query(query_texts=[" ".join(tokens)], n_results=25, include=["documents", "metadatas"])
        docs = res.get('documents', [[]])[0] if res else []
        metas = res.get('metadatas', [[]])[0] if res else []

        names = []
        # zgrabi vrstice, ki imajo leto in besede o prejemnikih
        year_pat = re.compile(fr'\b{year}\b') if year else None
        award_hint = ("petica" if award_type and "petica" in award_type else ("nagrada" if award_type else None))
        dobitnik_pat = re.compile(r'(?:dobitnik|prejemnik)\w*\s+.*?(?:je|so)\s+([A-ZŠŽČ][^\n,\.]+)', re.IGNORECASE)
        bullet_pat = re.compile(r'^\s*(?:[-•–]\s*)?([A-ZŠŽČ][^,\n;]+?)(?:\s*[-–:]\s*.*)?$')

        for d in docs:
            for ln in d.splitlines():
                if year_pat and not year_pat.search(ln): continue
                if award_hint and award_hint not in normalize_text(ln): 
                    # če je tip eksplicitno naveden, filtriraj po njem
                    continue
                ln_clean = ln.strip()
                m = dobitnik_pat.search(ln_clean) or bullet_pat.search(ln_clean)
                if m:
                    cand = m.group(1).strip()
                    if len(cand.split()) >= 2 and not any(w in cand.lower() for w in ["nagrada","petica","županov","zupanov","občina","obcine","rače","fram"]):
                        names.append(cand)

        if names:
            names_uniq = sorted({n.strip(" -–:") for n in names})
            title = "Županova petica" if (award_type and "petica" in award_type) else ("Županova nagrada" if (award_type and "nagrada" in award_type) else "Nagrade")
            header = f"**{title} — {year}:**" if year else f"**{title}:**"
            body = "\n".join(f"- {n}" for n in names_uniq)
            return f"{header}\n{body}"

        # poskusi vsaj link
        q_tokens = tokens_from_text(" ".join(tokens))
        for doc, meta in zip(docs, metas):
            url = (meta.get('source_url') or "").strip()
            if url and url_is_relevant(url, doc, q_tokens, must={"nagrada","petica"}, ban=set()):
                kind = "petica" if (award_type and "petica" in award_type) else "nagrada"
                ytxt = f" ({year})" if year else ""
                return f"Zapis o **županovi {kind}{ytxt}**: {url}"
        return "Žal iz baze ne uspem zanesljivo izluščiti nagrajencev za to vprašanje."

    # ---- RAG (splošno)
    def _strip_past_year_lines(self, text: str, this_year: int) -> str:
        lines = text.splitlines()
        kept = []
        for ln in lines:
            years = [int(y) for y in re.findall(r'\b(20\d{2})\b', ln)]
            if years and max(years) < this_year:
                continue
            kept.append(ln)
        return "\n".join(kept)

    def _filter_rag_results_by_year(self, docs: List[str], metas: List[Dict[str,Any]], allow_past=False) -> Tuple[List[str], List[Dict[str,Any]]]:
        this_year = datetime.now().year
        keep_docs, keep_metas = [], []
        for doc, meta in zip(docs, metas):
            combined = f"{doc}\n{json.dumps(meta, ensure_ascii=False)}"
            years = [int(y) for y in re.findall(r'\b(20\d{2})\b', combined)]
            if not allow_past and years and max(years) < this_year:
                continue
            keep_docs.append(doc); keep_metas.append(meta)
        return keep_docs, keep_metas

    def _rag_answer(self, vprasanje: str, intent_hint: str = "general", allow_past=False) -> Optional[str]:
        if not self.collection: return None
        q_norm = normalize_text(vprasanje)
        q_tokens = tokens_from_text(q_norm)
        results = self.collection.query(query_texts=[q_norm], n_results=8, include=["documents","metadatas"])
        docs = results.get('documents', [[]])[0] if results else []
        metas = results.get('metadatas', [[]])[0] if results else []
        # izloči koledarje odpadkov
        fd, fm = [], []
        for d, m in zip(docs, metas):
            if (m.get('kategorija','') or '').lower() == 'odvoz odpadkov': continue
            fd.append(d); fm.append(m)
        docs, metas = self._filter_rag_results_by_year(fd, fm, allow_past=allow_past)
        ctx_parts = []
        for d, m in zip(docs, metas):
            url = m.get('source_url','') or ""
            include_url = url_is_relevant(url, d, q_tokens, must=set(), ban=set())
            link_line = f"POVEZAVA: {url}" if include_url else "POVEZAVA: "
            ctx_parts.append(f"--- VIR: {m.get('source','?')}\n{link_line}\nVSEBINA: {d}\n")
        if not ctx_parts: return None
        now = datetime.now()
        prompt = f"""Ti si 'Virtualni župan občine Rače-Fram'.
DIREKTIVA: Današnji datum je {now.strftime('%d.%m.%Y')}. Ne navajaj starih informacij (razen če so izrecno vprašane).
DIREKTIVA: Bodi pregleden; ključne informacije **poudari**; alineje za sezname.
DIREKTIVA: Če v kontekstu ni URL, ne izmišljuj povezav.

--- KONTEKST ---
{''.join(ctx_parts)}
---
VPRAŠANJE: "{vprasanje}"
ODGOVOR:"""
        return self._call_llm(prompt, max_tokens=450) or None

    # ---- glavni vmesnik
    def odgovori(self, uporabnikovo_vprasanje: str, session_id: str) -> str:
        self.nalozi_bazo()
        if not self.collection:
            return "Oprostite, moja baza znanja trenutno ni na voljo."

        ses = self.zgodovina_seje.setdefault(session_id, {'zgodovina': [], 'stanje': {}})
        zgodovina = ses['zgodovina']
        stanje = ses['stanje']

        # preoblikovanje
        pametno = self.preoblikuj_vprasanje_s_kontekstom(zgodovina, uporabnikovo_vprasanje)
        route = self._route(pametno)
        intent = route.get("intent", "general")
        entities = route.get("entities", {})
        logger.info(f"Router -> intent={intent}, entities={entities}, conf={route.get('confidence')}")

        # nameni
        if intent == "mayor":
            odgovor = "Župan občine Rače-Fram je **Samo Rajšp**."
        elif intent == "waste":
            odgovor = self.obravnavaj_odvoz_odpadkov(pametno, session_id, next_only=bool(entities.get("next_only")))
        elif intent == "traffic_status":
            odgovor = self.preveri_zapore_cest()
        elif intent == "traffic_form":
            odgovor = self.obravnavaj_vloga_zapora(normalize_text(pametno))
        elif intent == "awards":
            odgovor = self.obravnavaj_nagrade(normalize_text(pametno), entities.get("year"), entities.get("award_type"))
        elif intent == "fire_brigades":
            odgovor = self.obravnavaj_gasilci()
        elif intent == "fire_command":
            odgovor = self.obravnavaj_gasilci_poveljstvo()
        elif intent == "mun_tax":
            # kratek, bistven povzetek + linki iz RAG
            rag = self._rag_answer(pametno, intent_hint="mun_tax", allow_past=True) or ""
            osnovno = (
                "**Komunalni prispevek – bistveno:**\n"
                "- **Kaj je:** prispevek za del stroškov komunalne opreme.\n"
                "- **Zavezanec:** investitor/lastnik objekta.\n"
                "- **Kdaj:** praviloma **pred izdajo gradbenega dovoljenja**.\n"
                "- **Kje vložim:** občina; odločba na podlagi izračuna.\n"
                "- **Izračun/priloge:** možen **informativni izračun**; priloge po navodilih občine.\n"
                f"- **Kontakt:** {cfg.FALLBACK_CONTACT}\n"
            )
            odgovor = osnovno + ("\n" + rag if rag else "")
        elif intent == "building_permit":
            rag = self._rag_answer(pametno, intent_hint="building_permit", allow_past=True) or ""
            if cfg.GRADBENO_OFFICIAL_URL not in (rag or ""):
                rag += (("\n\n" if rag else "") + f"**Uradna navodila (eUprava):** {cfg.GRADBENO_OFFICIAL_URL}")
            odgovor = rag or "Žal nimam dovolj podatkov za natančen odgovor."
        elif intent == "summer_camp":
            # samo aktualno leto, razen če uporabnik izrecno navede drugo
            explicit_year = entities.get("year")
            allow_past = bool(explicit_year)
            odgovor = self._rag_answer(pametno, intent_hint="summer_camp", allow_past=allow_past) or "Trenutno nimam podatka o poletnem taboru."
        else:
            odgovor = self._rag_answer(pametno, intent_hint="general", allow_past=False) or "Žal o tej temi nimam informacij."

        # beleženje
        zgodovina.append((uporabnikovo_vprasanje, odgovor))
        if len(zgodovina) > 4: zgodovina.pop(0)
        self.belezi_pogovor(session_id, uporabnikovo_vprasanje, odgovor)
        return odgovor

# ------------------------------------------------------------------------------
# 6) CLI
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
