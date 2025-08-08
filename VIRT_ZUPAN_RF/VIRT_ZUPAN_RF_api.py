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
# 0. ENV
# ------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '..', '.env'))

# ------------------------------------------------------------------------------
# 1. KONFIGURACIJA
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
    
    WASTE_VARIANTS: Dict[str, List[str]] = field(default_factory=lambda: {
        "Biološki odpadki": ["bioloski odpadki", "bioloskih odpakov", "bioloskih odpadkov", "bio", "bioloski", "biološki odpadki"],
        "Mešani komunalni odpadki": ["mesani komunalni odpadki", "mesani", "mešani", "komunalni odpadki", "komunalnih odpadkov"],
        "Odpadna embalaža": ["odpadna embalaza", "odpadna embalaža", "embalaza", "embalaža", "embalaže", "rumena kanta", "rumene kante"],
        "Papir in karton": ["papir in karton", "papir", "karton"],
        "Steklena embalaža": ["steklena embalaza", "steklena embalaža", "steklo", "stekla"]
    })

    GENERIC_STREET_WORDS: Tuple[str, ...] = ("cesta", "cesti", "ulica", "ulici", "pot", "trg", "naselje")
    GENERIC_PREPS: Tuple[str, ...] = ("na", "v", "za", "ob", "pod", "pri", "nad", "do", "od", "k", "proti")

    FALLBACK_CONTACT: str = "Občina Rače-Fram: 02 609 60 10"

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
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("VirtualniZupan")

# ------------------------------------------------------------------------------
# 3. POMOŽNE FUNKCIJE (normalizacija & ujemanje)
# ------------------------------------------------------------------------------
def normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.lower()
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8')
    s = re.sub(r'[^\w\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def _ratio(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()

def sl_variant_eq(a: str, b: str) -> bool:
    a_n = normalize_text(a)
    b_n = normalize_text(b)
    if a_n == b_n:
        return True
    # odreži pogoste slovenske končnice
    a_stripped = re.sub(r'(ski|ska|sko|skega|skem|skih|ska|sko|ski)$', '', a_n)
    b_stripped = re.sub(r'(ski|ska|sko|skega|skem|skih|ska|sko|ski)$', '', b_n)
    if a_stripped and a_stripped == b_stripped:
        return True
    # še mehkejše: -a/-i/-o/-e
    if len(a_n) > 2 and len(b_n) > 2 and a_n[:-1] == b_n[:-1] and {a_n[-1], b_n[-1]} <= {"a","i","e","o","u"}:
        return True
    return False

def street_phrase_matches(query_phrase: str, street_tok: str, allow_substrings: bool = True, threshold: float = 0.88) -> bool:
    """
    Ujemanje ulice:
    - odstrani generične besede (cesta, ulica, pot...) in predloge (na, v, pod...).
    - substring ujemanje je onemogočeno za kratke besede (<4).
    """
    generic = set(normalize_text(" ".join(cfg.GENERIC_STREET_WORDS)).split()) | set(normalize_text(" ".join(cfg.GENERIC_PREPS)).split())
    qp = normalize_text(query_phrase)
    st = normalize_text(street_tok)

    if sl_variant_eq(qp, st):
        return True

    q_words = [w for w in qp.split() if w not in generic and len(w) >= 3]
    street_words = [w for w in st.split() if w not in generic and len(w) >= 3]

    if not q_words:
        # nič pametnega v frazi -> prepusti drugim frazam
        return False

    for qw in q_words:
        matched = False
        for sw in street_words:
            if sl_variant_eq(qw, sw):
                matched = True
                break
            # substring samo za dovolj dolge besede
            if allow_substrings and len(qw) >= 4 and (qw in sw or sw in qw):
                matched = True
                break
            if _ratio(qw, sw) >= threshold:
                matched = True
                break
        if not matched:
            return False
    return True

def get_canonical_waste(text: str) -> Optional[str]:
    norm = normalize_text(text)
    if ("rumen" in norm and "kant" in norm) or "embal" in norm:
        return "Odpadna embalaža"
    if "komunaln" in norm and "odpadk" in norm:
        return "Mešani komunalni odpadki"
    if "bio" in norm or "biolos" in norm:
        return "Biološki odpadki"
    if "stekl" in norm:
        return "Steklena embalaža"
    if "papir" in norm or "karton" in norm:
        return "Papir in karton"
    for canon, variants in cfg.WASTE_VARIANTS.items():
        for v in variants:
            if normalize_text(v) in norm:
                return canon
    for canon, variants in cfg.WASTE_VARIANTS.items():
        for v in variants:
            if _ratio(norm, normalize_text(v)) >= 0.90:
                return canon
    return None

def extract_locations_from_naselja(naselja_field: str) -> List[str]:
    parts = set()
    if not naselja_field:
        return []
    naselja_clean = re.sub(r'\(h\. *št\..*?\)', '', naselja_field)
    segments = re.split(r'([A-ZČŠŽ][a-zčšž]+\s*:)', naselja_clean)
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        if seg.endswith(':'):
            # naselje, a ga ne dodajamo k ulicam (da ne kvarimo ujemanja)
            continue
        else:
            for sub in seg.split(','):
                n = normalize_text(sub)
                if n:
                    parts.add(n)
    return list(parts)

def parse_dates_from_text(text: str) -> List[str]:
    dates = re.findall(r'(\d{1,2}\.\d{1,2}\.?)', text)
    seen = set()
    out = []
    for d in dates:
        d_norm = d if d.endswith('.') else (d + '.')
        if d_norm not in seen:
            seen.add(d_norm)
            out.append(d_norm)
    return out

def detect_intent_qna(q_norm: str) -> str:
    if re.search(r'\bkdo je\b', q_norm) and ('zupan' in q_norm or 'župan' in q_norm):
        return 'who_is_mayor'
    if 'kontakt' in q_norm or 'telefon' in q_norm or 'e-mail' in q_norm or 'stevilka' in q_norm or 'številka' in q_norm:
        return 'contact'
    if 'prevoz' in q_norm or 'vozni red' in q_norm or 'minibus' in q_norm or 'avtobus' in q_norm or 'kopivnik' in q_norm:
        return 'transport'
    if 'komunaln' in q_norm and 'prispev' in q_norm:
        return 'komunalni_prispevek'
    if 'poletn' in q_norm and ('tabor' in q_norm or 'kamp' in q_norm):
        return 'camp'
    return 'general'

# ------------------------------------------------------------------------------
# 4. GLAVNI RAZRED
# ------------------------------------------------------------------------------
class VirtualniZupan:
    def __init__(self) -> None:
        prefix = "PRODUCTION" if cfg.ENV_TYPE == 'production' else "DEVELOPMENT"
        logger.info(f"[{prefix}] VirtualniŽupan v44.0 inicializiran.")
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

    # ---------- infra ----------
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

    # ---------- spomin / preoblikovanje ----------
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

    # ---------- NAP / promet ----------
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

    # ---------- odpadki ----------
    def _build_location_phrases(self, vprasanje_norm: str) -> List[str]:
        waste_stop = set()
        for variants in cfg.WASTE_VARIANTS.values():
            for v in variants:
                waste_stop.add(normalize_text(v))
        extra_stop = {"kdaj","je","naslednji","odvoz","odpadkov","smeti","urnik","urniki","termini","termine",
                      "kako","kateri","katera","kaj","koga","kje","kam"} \
                      | set(normalize_text(" ".join(cfg.GENERIC_STREET_WORDS)).split()) \
                      | set(normalize_text(" ".join(cfg.GENERIC_PREPS)).split())
        stop = waste_stop.union(extra_stop)

        toks = [t for t in re.split(r'[,\s]+', vprasanje_norm) if t and t not in stop and len(t) >= 3]
        phrases = []
        # n-grami 3->2->1, a enobesedne dodamo šele na koncu, da ne prevladajo (npr. 'pod')
        for size in (3,2):
            for i in range(len(toks)-size+1):
                p = " ".join(toks[i:i+size]).strip()
                if p and p not in phrases:
                    phrases.append(p)
        # enobesedne (dolžine >=4) dodaj na koncu
        for t in toks:
            if len(t) >= 4 and t not in phrases:
                phrases.append(t)
        return phrases

    def _match_streets(self, phrases: List[str], all_streets: List[str]) -> Tuple[List[Dict[str,Any]], List[Dict[str,Any]]]:
        exact, fuzzy = [], []
        for phrase in phrases:
            for street in all_streets:
                if street_phrase_matches(phrase, street, allow_substrings=True, threshold=0.90):
                    exact.append({"phrase": phrase, "street": street, "score": 1.0})
                else:
                    # fuzzy samo, če sta fraza in ulica oba večbesedna ali beseda dolžine >=5
                    if (len(phrase.split()) > 1 or len(phrase) >= 5):
                        sc = _ratio(normalize_text(phrase), normalize_text(street))
                        if sc >= 0.92:
                            fuzzy.append({"phrase": phrase, "street": street, "score": sc})
        fuzzy.sort(key=lambda x: x['score'], reverse=True)
        return exact, fuzzy

    def _pick_best_street(self, exact: List[Dict[str,Any]], fuzzy: List[Dict[str,Any]]) -> Optional[str]:
        if exact:
            exact.sort(key=lambda x: (len(x["phrase"]), x["score"]), reverse=True)
            return exact[0]["street"]
        if fuzzy:
            return fuzzy[0]["street"]
        return None

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
                return f"Naslednji odvoz za **{tip_odpadka}** na **{street_display}** je **{cand}**."
            else:
                return f"Za **{tip_odpadka}** na **{street_display}** v tem letu ni več predvidenih terminov."
        else:
            return f"Za **{street_display}** je odvoz **{tip_odpadka}**: {', '.join(dates)}"

    def obravnavaj_odvoz_odpadkov(self, uporabnikovo_vprasanje: str, session_id: str) -> str:
        logger.info("Kličem specialista za odpadke...")
        stanje = self.zgodovina_seje[session_id].setdefault('stanje', {})
        if not self.collection:
            return "Baza urnikov ni na voljo."
        
        vprasanje_norm = normalize_text(uporabnikovo_vprasanje)
        contains_naslednji = "naslednji" in vprasanje_norm

        iskani_tip = get_canonical_waste(vprasanje_norm)
        if not iskani_tip and contains_naslednji and stanje.get('zadnji_tip'):
            iskani_tip = stanje['zadnji_tip']

        phrases = self._build_location_phrases(vprasanje_norm)
        explicit_from_memory = False
        if not phrases and stanje.get('zadnja_lokacija_display'):
            phrases = [stanje['zadnja_lokacija_display']]
            explicit_from_memory = True

        explicit_street_mode = any(len(p.split())>=2 for p in phrases) or explicit_from_memory

        vsi_urniki = self.collection.get(where={"kategorija": "Odvoz odpadkov"})
        if not vsi_urniki or not vsi_urniki.get('ids'):
            return "V bazi znanja ni podatkov o urnikih."

        kandidati = []
        for i in range(len(vsi_urniki['ids'])):
            meta = vsi_urniki['metadatas'][i]
            doc_text = vsi_urniki['documents'][i]
            tip_meta = meta.get('tip_odpadka', '')
            streets = extract_locations_from_naselja(meta.get('naselja', ''))

            if iskani_tip and get_canonical_waste(tip_meta) != iskani_tip:
                continue

            exact, fuzzy = self._match_streets(phrases, streets)
            if exact or fuzzy:
                best_street = self._pick_best_street(exact, fuzzy)
                if best_street:
                    sc = 1.0 if exact else (fuzzy[0]['score'] if fuzzy else 0.0)
                    kandidati.append({
                        "doc": doc_text,
                        "matched_street": best_street,
                        "score": sc,
                        "tip": get_canonical_waste(tip_meta) or tip_meta
                    })

        if not kandidati:
            if any(p for p in phrases):  # ulica je podana -> vprašaj za tip ali vrni, da ni
                if not iskani_tip:
                    # shrani lokacijo v spomin
                    stanje['zadnja_lokacija_display'] = phrases[0].title()
                    stanje['zadnja_lokacija_norm'] = normalize_text(phrases[0])
                    stanje['namen'] = 'odpadki'
                    stanje['caka_na'] = 'tip'
                    return "Kateri tip odpadkov te zanima? (bio, mešani, embalaža, papir, steklo)"
                return f"Za navedeno ulico žal nisem našel urnika za izbrani tip."
            # ni lokacije -> vprašaj po lokaciji
            stanje['namen'] = 'odpadki'
            stanje['caka_na'] = 'lokacija'
            return "Za katero ulico te zanima urnik? (npr. 'Bistriška cesta, Fram')"

        if explicit_street_mode:
            kandidati.sort(key=lambda x: x['score'], reverse=True)
            best = kandidati[0]
            street_disp = (best['matched_street'] or stanje.get('zadnja_lokacija_display') or phrases[0]).title()
            tip_canon = best['tip']
            ans = self._format_dates_for_tip(street_disp, tip_canon, best['doc'], contains_naslednji)
            if not ans:
                return "Žal zate ne najdem datumov v urniku."
            # spomin
            stanje['zadnja_lokacija_display'] = street_disp
            stanje['zadnja_lokacija_norm'] = normalize_text(street_disp)
            stanje['zadnji_tip'] = tip_canon
            stanje['namen'] = 'odpadki'
            stanje.pop('caka_na', None)
            return ans

        # ne-ekspl.: lahko vrnemo več (kratko)
        odgovori = []
        for k in kandidati[:3]:
            tip_canon = k['tip']
            loc_disp = k['matched_street'].title()
            msg = self._format_dates_for_tip(loc_disp, tip_canon, k['doc'], contains_naslednji)
            if msg:
                odgovori.append(msg)
        if not odgovori:
            return "Žal zate ne najdem datumov v urniku."

        prvi = kandidati[0]
        stanje['zadnja_lokacija_display'] = prvi['matched_street'].title()
        stanje['zadnja_lokacija_norm'] = normalize_text(prvi['matched_street'])
        stanje['zadnji_tip'] = prvi['tip']
        stanje['namen'] = 'odpadki'
        stanje.pop('caka_na', None)

        uniq = []
        for o in odgovori:
            if o not in uniq:
                uniq.append(o)
        return "\n\n".join(uniq)

    # ---------- RAG ----------
    def _zgradi_rag_prompt(self, vprasanje: str, zgodovina: List[Tuple[str, str]]) -> Optional[str]:
        logger.info(f"Gradim RAG prompt za vprašanje: '{vprasanje}'")
        if not self.collection:
            return None
        
        results = self.collection.query(
            query_texts=[normalize_text(vprasanje)],
            n_results=5,
            include=["documents", "metadatas"]
        )
        context = ""
        if results and results.get('documents') and results['documents'][0]:
            for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                context += f"--- VIR: {meta.get('source', '?')}\nPOVEZAVA: {meta.get('source_url', '')}\nVSEBINA: {doc}\n\n"
        if not context:
            return None

        now = datetime.now()
        zgodovina_str = "\n".join([f"U: {q}\nA: {a}" for q, a in zgodovina])
        q_norm = normalize_text(vprasanje)
        intent = detect_intent_qna(q_norm)

        extra_directives = []
        if intent == 'who_is_mayor':
            extra_directives.append("Če vprašanje sprašuje 'kdo je župan', odgovori v ENEM kratkem stavku samo z imenom in funkcijo, brez dodatkov.")
        if intent == 'contact':
            extra_directives.append(f"Če uporabnik prosi za 'kontakt' in v kontekstu ni specifičnega, uporabi generični kontakt: '{cfg.FALLBACK_CONTACT}'.")
        if intent == 'transport':
            extra_directives.append("Če je vprašanje o prevozu/voznem redu, navedi samo relacijo, ključne ure in kontakt – brez balasta.")
        if intent == 'komunalni_prispevek':
            extra_directives.append("Za komunalni prispevek odgovori v 5–7 alinejah: (1) Kaj je, (2) Kdo je zavezanec, (3) Kdaj se plača, (4) Kako oddam vlogo (kam), (5) Potrebne priloge/izračun, (6) Kontakt.")
        if intent == 'camp':
            extra_directives.append("Za poletni kamp povej datum(e), lokacijo, ceno (če je v kontekstu), in kontakt/URL – kratko in jedrnato.")

        directives = "\n".join(extra_directives)

        return f"""Ti si 'Virtualni župan občine Rače-Fram'.
DIREKTIVA #1 (DATUMI): Današnji datum je {now.strftime('%d.%m.%Y')}. Če je podatek iz leta, ki je manjše od {now.year}, ga IGNORIRAJ.
DIREKTIVA #2 (OBLIKA): Odgovor naj bo kratek in pregleden. Ključne informacije **poudari**. Kjer naštevaš, **uporabi alineje (-)**.
DIREKTIVA #3 (POVEZAVE): Če v kontekstu pod ključem 'POVEZAVA' najdeš URL, ga MORAŠ vključiti kot '[Ime vira](URL)'.
DIREKTIVA #4 (SPECIFIČNOST): Če specifičnega podatka ni, povej 'Žal nimam natančnega podatka.' Namesto balasta.
{directives}

--- KONTEKST ---
{context}---
ZGODOVINA POGOVORA:
{zgodovina_str}
---
VPRAŠANJE: "{vprasanje}"
ODGOVOR:"""

    # ---------- glavni vmesnik ----------
    def odgovori(self, uporabnikovo_vprasanje: str, session_id: str) -> str:
        self.nalozi_bazo()
        if not self.collection:
            return "Oprostite, moja baza znanja trenutno ni na voljo."

        ses = self.zgodovina_seje.setdefault(session_id, {'zgodovina': [], 'stanje': {}})
        zgodovina = ses['zgodovina']
        stanje = ses['stanje']

        q_norm = normalize_text(uporabnikovo_vprasanje)

        # hard-shortcut: župan
        if re.search(r'\bkdo je\b', q_norm) and ('zupan' in q_norm or 'župan' in q_norm):
            odgovor = "Župan občine Rače-Fram je **Samo Rajšp**."
            zgodovina.append((uporabnikovo_vprasanje, odgovor))
            if len(zgodovina) > 4: zgodovina.pop(0)
            self.belezi_pogovor(session_id, uporabnikovo_vprasanje, odgovor)
            # ko zapustimo “odpadki”, odstrani lepljivost
            if stanje.get('namen') == 'odpadki' and not stanje.get('caka_na'):
                stanje.pop('namen', None)
            return odgovor

        # preoblikovanje
        pametno_vprasanje = self.preoblikuj_vprasanje_s_kontekstom(zgodovina, uporabnikovo_vprasanje)
        vprasanje_lower = normalize_text(pametno_vprasanje)

        # domena detekcija
        is_waste_query = any(re.search(r'\b' + re.escape(k) + r'\b', vprasanje_lower) for k in cfg.KLJUCNE_ODPADKI) or (get_canonical_waste(vprasanje_lower) is not None) or ('naslednji' in vprasanje_lower)
        is_traffic_query = any(re.search(r'\b' + re.escape(k) + r'\b', vprasanje_lower) for k in cfg.KLJUCNE_PROMET)
        waste_followup_needed = (stanje.get('namen') == 'odpadki' and stanje.get('caka_na') in ('lokacija','tip'))

        if is_waste_query or waste_followup_needed:
            odgovor = self.obravnavaj_odvoz_odpadkov(pametno_vprasanje, session_id)
        elif is_traffic_query:
            # ko zapustimo “odpadki”, odstrani lepljivost
            if stanje.get('namen') == 'odpadki' and not waste_followup_needed:
                stanje.pop('namen', None)
                stanje.pop('caka_na', None)
            odgovor = self.preveri_zapore_cest()
        else:
            # čist izhod iz “odpadki”, če ni več namena
            if stanje.get('namen') == 'odpadki' and not waste_followup_needed:
                stanje.pop('namen', None)
                stanje.pop('caka_na', None)
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
            if not q:
                continue
            print("\n--- ODGOVOR ŽUPANA ---\n")
            print(zupan.odgovori(q, session_id))
            print("\n---------------------\n")
        except KeyboardInterrupt:
            break
    print("\nNasvidenje!")
