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
from difflib import SequenceMatcher

# ------------------------------------------------------------------------------
# VIRT_ZUPAN_RF_api.py — v41.0 (fokus na točno ulico + točen tip odpadkov)
# - ohranja tvojo strukturo (Config dataclass, logging, sessions, cache)
# - popravi iskanje odpadkov: robusten tip (npr. "stekla") + natančno ujemanje ulice
# - minimalni odgovori za "kdo je župan", "komunalni prispevek", "prevoz kopivnik", "kontakt"
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# 1) KONFIGURACIJA
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
    PROMET_FILTER: Tuple[str, ...] = (
        "rače", "fram", "slivnica", "brunšvik", "podova", "morje", "hoče",
        "r2-430", "r3-711", "g1-2", "priključek slivnica", "razcep slivnica",
        "letališče maribor", "odcep za rače"
    )

    # razširjeno! vključuje slovnične oblike in kolokvialne fraze (rumena kanta)
    WASTE_VARIANTS: Dict[str, List[str]] = field(default_factory=lambda: {
        "Biološki odpadki": [
            "biološki", "bioloski", "bio", "bio odpadki", "biološki odpadki", "bioloskih odpadkov", "bioloških odpadkov",
            "zeleni odpadki"
        ],
        "Mešani komunalni odpadki": [
            "mešani", "mesani", "mešani odpadki", "mesani odpadki",
            "komunalni odpadki", "komunalnih odpadkov", "mešane komunalne", "mesane komunalne"
        ],
        "Odpadna embalaža": [
            "embalaža", "embalaza", "odpadna embalaža", "odpadna embalaza",
            "rumena kanta", "rumene kante", "rumen zabojnik", "rumeni zabojnik"
        ],
        "Papir in karton": [
            "papir", "papirja", "karton", "kartona", "papir in karton", "papir in kartona"
        ],
        "Steklena embalaža": [
            "steklo", "stekla", "steklene", "steklena", "stekleni", "stekleno",
            "steklena embalaža", "steklena embalaza", "stekleno embalažo", "stekleno embalazo"
        ]
    })

    def __post_init__(self):
        object.__setattr__(self, 'CHROMA_DB_PATH', os.path.join(self.DATA_DIR, "chroma_db"))
        object.__setattr__(self, 'LOG_FILE_PATH', os.path.join(self.DATA_DIR, "zupan_pogovori.jsonl"))
        os.makedirs(self.DATA_DIR, exist_ok=True)

cfg = Config()
load_dotenv(os.path.join(cfg.BASE_DIR, '..', '.env'))

# Validacija okolja
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
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("VirtualniZupan")

# ------------------------------------------------------------------------------
# 3) POMOŽNE FUNKCIJE (normalizacija, slovenski sklon, ulica match)
# ------------------------------------------------------------------------------
def normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    s = s.lower()
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8')
    s = re.sub(r'[^\w\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def slovenian_variant_equivalent(a: str, b: str) -> bool:
    # groba heuristika: isti koren + samoglasniški končnici
    a_n = normalize_text(a)
    b_n = normalize_text(b)
    if a_n == b_n:
        return True
    if len(a_n) > 1 and len(b_n) > 1:
        if a_n[:-1] == b_n[:-1] and {a_n[-1], b_n[-1]} <= {"a", "e", "i", "o", "u"}:
            return True
    return False

def fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()

def street_phrase_matches(query_phrase: str, street_tok: str, threshold: float = 0.85) -> bool:
    generic = {"cesta", "cesti", "ulica", "ulici", "pot", "trg", "ob"}
    qp = normalize_text(query_phrase)
    st = normalize_text(street_tok)

    if slovenian_variant_equivalent(qp, st):
        return True

    q_words = [w for w in qp.split() if w not in generic]
    street_words = [w for w in st.split() if w not in generic]
    if not q_words:
        return fuzzy_ratio(qp, st) >= threshold

    for qw in q_words:
        if any(slovenian_variant_equivalent(qw, sw) or fuzzy_ratio(qw, sw) >= threshold or qw in sw or sw in qw
               for sw in street_words):
            continue
        return False
    return True

def get_canonical_waste(text: str) -> Optional[str]:
    norm = normalize_text(text)

    # heuristike/kolokvialno
    if ("rumen" in norm) and ("kant" in norm or "zaboj" in norm):
        return "Odpadna embalaža"
    if "komunaln" in norm and "odpadk" in norm:
        return "Mešani komunalni odpadki"
    if "bio" in norm or "biolosk" in norm or "biološk" in norm:
        if "odpadk" in norm or "kanta" in norm:
            return "Biološki odpadki"
    if "stekl" in norm:
        return "Steklena embalaža"
    if "papir" in norm or "karton" in norm:
        return "Papir in karton"
    if "embal" in norm:
        return "Odpadna embalaža"

    for canon, variants in cfg.WASTE_VARIANTS.items():
        for v in variants:
            if normalize_text(v) in norm:
                return canon
    # fuzzy fallback
    for canon, variants in cfg.WASTE_VARIANTS.items():
        for v in variants:
            if fuzzy_ratio(norm, v) >= 0.85:
                return canon
    return None

def extract_locations_from_naselja(field: str) -> List[str]:
    parts = []
    if not field:
        return parts
    # Odstrani "(h. št. ...)"
    clean = re.sub(r'\(h\.?\s*št\..*?\)', '', field, flags=re.IGNORECASE)
    # Razbij po "Kraj:" skupinah
    segments = re.split(r'([A-ZČŠŽ][a-zčšž]+\s*:)', clean)
    prefix = ""
    out = set()
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        if seg.endswith(':'):
            prefix = normalize_text(seg[:-1])
            if prefix:
                out.add(prefix)
        else:
            for sub in seg.split(','):
                n = normalize_text(sub)
                if n:
                    out.add(n)
                    if prefix:
                        out.add(f"{n} {prefix}")
    return list(out)

# ------------------------------------------------------------------------------
# 4) GLAVNI RAZRED
# ------------------------------------------------------------------------------
class VirtualniZupan:
    def __init__(self) -> None:
        prefix = "PRODUCTION" if cfg.ENV_TYPE == 'production' else "DEVELOPMENT"
        logger.info(f"[{prefix}] VirtualniŽupan v41.0 inicializiran.")
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

    # --------------------- infra ---------------------
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
            self.collection = client.get_collection(
                name=cfg.COLLECTION_NAME, embedding_function=ef
            )
            logger.info(f"Baza uspešno naložena. Dokumentov: {self.collection.count()}")
        except Exception:
            logger.exception("KRITIČNA NAPAKA: Baze znanja ni mogoče naložiti.")
            self.collection = None

    def belezi_pogovor(self, session_id: str, vprasanje: str, odgovor: str) -> None:
        try:
            zapis = {
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "vprasanje": vprasanje,
                "odgovor": odgovor
            }
            with open(cfg.LOG_FILE_PATH, 'a', encoding='utf-8') as f:
                f.write(json.dumps(zapis, ensure_ascii=False) + '\n')
        except Exception:
            logger.exception(f"Napaka pri beleženju pogovora (seja {session_id}).")

    # --------------------- spomin ---------------------
    def preoblikuj_vprasanje_s_kontekstom(self, zgodovina_pogovora: List[Tuple[str, str]], zadnje_vprasanje: str) -> str:
        if not zgodovina_pogovora:
            return zadnje_vprasanje
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
        return preoblikovano if "napak" not in preoblikovano.lower() else zadnje_vprasanje

    # --------------------- NAP / promet ---------------------
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

        relevantne = []
        for d in vsi_dogodki:
            txt = " ".join(str(d.get('properties', {}).get(polje, '')).lower() for polje in ['cesta', 'opis', 'imeDogodka'])
            if any(k in txt for k in cfg.PROMET_FILTER):
                relevantne.append(d.get('properties', {}))
        if not relevantne:
            return "Po podatkih portala promet.si na območju občine Rače-Fram trenutno ni zabeleženih del na cesti, zapor ali zastojev."
        unique_zapore = [dict(t) for t in {tuple(d.items()) for d in relevantne}]
        porocilo = "Našel sem naslednje **trenutne** informacije o dogodkih na cesti (vir: promet.si):\n\n"
        for z in unique_zapore:
            porocilo += f"- **Cesta:** {z.get('cesta', 'Ni podatka')}\n  **Opis:** {z.get('opis', 'Ni podatka')}\n\n"
        return porocilo.strip()

    # --------------------- odvoz odpadkov (natančno po ulici + tipu) ---------------------
    def _poisci_naslednji_datum(self, datumi_str: str, danes: datetime) -> Optional[str]:
        leto = danes.year
        datumi = []
        for del_str in re.findall(r'(\d{1,2})\.(\d{1,2})\.?', datumi_str):
            try:
                d, m = int(del_str[0]), int(del_str[1])
                datumi.append(datetime(leto, m, d))
            except Exception:
                continue
        for datum in sorted(set(datumi)):
            if datum.date() >= danes.date():
                return datum.strftime('%d.%m.%Y')
        return None

    def obravnavaj_odvoz_odpadkov(self, uporabnikovo_vprasanje: str, session_id: str) -> str:
        logger.info("Kličem specialista za odpadke (natančno ujemanje ulice + tip)...")
        stanje = self.zgodovina_seje[session_id].get('stanje', {})
        if not self.collection:
            return "Baza urnikov ni na voljo."

        vprasanje_norm = normalize_text(uporabnikovo_vprasanje)
        iskani_tip = get_canonical_waste(vprasanje_norm)
        contains_naslednji = "naslednji" in vprasanje_norm

        # zgradi lokacijske fraze (3-gram, 2-gram, 1-gram), odstrani besede tipa odpadkov in generic
        waste_stop = {normalize_text(k) for k in cfg.WASTE_VARIANTS.keys()}
        for variants in cfg.WASTE_VARIANTS.values():
            for v in variants:
                waste_stop.add(normalize_text(v))
        extra_stop = {"kdaj", "je", "naslednji", "odvoz", "odpadkov", "smeti", "na", "v", "za", "kako", "poteka",
                      "ali", "kateri", "katera", "kaj"}
        generic_single = {"cesta", "cesti", "ulica", "ulici", "pot", "trg", "ob"}
        tokens = [t for t in re.split(r'[,\s]+', vprasanje_norm) if t and t not in waste_stop and t not in extra_stop]
        # n-grami
        phrases = []
        for n in (3, 2, 1):
            for i in range(len(tokens) - n + 1):
                phr = " ".join(tokens[i:i+n]).strip()
                if n == 1 and phr in generic_single:
                    continue
                if phr and phr not in phrases:
                    phrases.append(phr)

        if not phrases and not stanje.get('caka_na'):
            stanje.update({'caka_na': 'lokacija', 'namen': 'odpadki'})
            return "Katera ulica ali naselje te zanima? (npr. 'Bistriška cesta, Fram')"

        vsi_urniki = self.collection.get(where={"kategorija": "Odvoz odpadkov"})
        if not vsi_urniki or not vsi_urniki.get('ids'):
            return "V bazi znanja ni podatkov o urnikih."

        exact_matches = []
        fuzzy_matches = []
        area_matches = []

        def score_street(phrase: str, street_tok: str) -> float:
            if slovenian_variant_equivalent(phrase, street_tok):
                return 1.0
            full = fuzzy_ratio(phrase, street_tok)
            best_word = 0.0
            for w in normalize_text(street_tok).split():
                best_word = max(best_word, fuzzy_ratio(phrase, w))
            return max(full, best_word * 0.95)

        for i in range(len(vsi_urniki['ids'])):
            meta = vsi_urniki['metadatas'][i]
            doc = vsi_urniki['documents'][i]
            lokacije = extract_locations_from_naselja(meta.get('naselja', ''))
            obm = normalize_text(meta.get('obmocje', ''))

            # filter po tipu (če uporabnik poda tip)
            meta_tip_canon = get_canonical_waste(meta.get('tip_odpadka', ''))
            if iskani_tip and meta_tip_canon != iskani_tip:
                continue

            matched_for_doc = False
            # najprej natančna ulica
            for phrase in [p for p in phrases if len(p.split()) > 1] + [p for p in phrases if len(p.split()) == 1]:
                for street_tok in lokacije:
                    if street_phrase_matches(phrase, street_tok):
                        exact_matches.append({
                            'doc': doc, 'meta': meta, 'tip_canon': meta_tip_canon,
                            'matched_street': street_tok, 'matched_phrase': phrase, 'score': 1.0
                        })
                        matched_for_doc = True
                        break
                if matched_for_doc:
                    break

            # fuzzy ulica (če še ni)
            if not matched_for_doc:
                for phrase in phrases:
                    for street_tok in lokacije:
                        sc = score_street(phrase, street_tok)
                        thresh = 0.8 if len(phrase.split()) > 1 else 0.87
                        if sc >= thresh:
                            fuzzy_matches.append({
                                'doc': doc, 'meta': meta, 'tip_canon': meta_tip_canon,
                                'matched_street': street_tok, 'matched_phrase': phrase, 'score': sc
                            })
                            matched_for_doc = True
                # fallback: območje
                if not matched_for_doc and obm:
                    for phrase in phrases:
                        if fuzzy_ratio(phrase, obm) >= 0.8 or phrase in obm or obm in phrase:
                            area_matches.append({
                                'doc': doc, 'meta': meta, 'tip_canon': meta_tip_canon,
                                'matched_area': obm, 'matched_phrase': phrase, 'score': fuzzy_ratio(phrase, obm)
                            })
                            break

        # izbira kandidatov (prioriteta: exact > fuzzy > area)
        if exact_matches:
            kandidati = exact_matches
        elif fuzzy_matches:
            kandidati = sorted(fuzzy_matches, key=lambda x: x['score'], reverse=True)
        else:
            kandidati = area_matches

        if not kandidati:
            # če je uporabnik zelo specifičen, ne vračamo vsega
            if phrases:
                return f"Za **{phrases[0].title()}** in izbrani tip odpadkov žal nisem našel urnika."
            return "Žal nisem našel ustreznega urnika za navedeno lokacijo."

        # če je uporabnik eksplicitno navedel lokacijo (ulica/pot/trg), NE vračamo drugih območij
        explicit_location = any(any(w in p for w in ["cesta", "ulica", "pot", "trg"]) or len(p.split()) > 1 for p in phrases)
        if explicit_location:
            # obdrži samo kandidate, ki so prišli iz uličnega ujemanja
            only_street = [c for c in kandidati if 'matched_street' in c]
            if only_street:
                kandidati = only_street
                # zadrži le najboljše po točki
                best_score = max(c['score'] for c in kandidati)
                kandidati = [c for c in kandidati if c['score'] >= best_score - 1e-6]

        # oblikuj odgovor: samo ZA izbrani tip in izbrano ulico/območje
        now = datetime.now()
        if contains_naslednji:
            best = None  # (datetime, tip, loc_descr)
            for info in kandidati:
                txt = info['doc']
                dtm = re.search(r':\s*([\d\.,\s]+)', txt)
                if not dtm:
                    continue
                nd = self._poisci_naslednji_datum(dtm.group(1), now)
                if not nd:
                    continue
                nd_dt = datetime.strptime(nd, "%d.%m.%Y")
                if (best is None) or (nd_dt < best[0]):
                    if 'matched_street' in info:
                        loc = info['matched_street'].title()
                    else:
                        loc = info['meta'].get('obmocje', '').title()
                    best = (nd_dt, info['meta'].get('tip_odpadka', ''), loc)
            if best:
                return f"Naslednji odvoz za **{best[1]}** na **{best[2]}** je **{best[0].strftime('%d.%m.%Y')}**."
            return "Za izbrani tip in lokacijo v tem letu ni več prihodnjih terminov."

        # brez 'naslednji' – vrni 1 jasno vrstico (ne vseh tipov!)
        # kandidati so že filtrirani po tipu, če ga je uporabnik navedel
        # če ga NI, vrni samo najrelevantnejšega (ne poplave)
        if not iskani_tip and kandidati:
            # prednost uličnemu + najvišji score
            kandidati = sorted(kandidati, key=lambda x: (0 if 'matched_street' in x else 1, -x['score']))
            kandidati = [kandidati[0]]

        odgovori = []
        for info in kandidati:
            txt = info['doc']
            tip_odp = info['meta'].get('tip_odpadka', '')
            obmocje = info['meta'].get('obmocje', '').title()
            dtm = re.search(r':\s*([\d\.,\s]+)', txt)
            datumi = dtm.group(1).strip() if dtm else txt
            if 'matched_street' in info:
                ulica = info['matched_street'].title()
                odgovori.append(f"Za **{ulica}** ({obmocje}) je odvoz **{tip_odp}**: {datumi}")
            else:
                odgovori.append(f"Za območje **{obmocje}** je odvoz **{tip_odp}**: {datumi}")

        stanje.clear()
        # vrni samo unikatne vrstice, a največ 2 (da ne poplavi)
        uniq = []
        for o in odgovori:
            if o not in uniq:
                uniq.append(o)
        return "\n\n".join(uniq[:2])

    # --------------------- RAG prompt ---------------------
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
        return f"""Ti si 'Virtualni župan občine Rače-Fram'.
DIREKTIVA #1 (VAROVALKA ZA DATUME): Današnji datum je {now.strftime('%d.%m.%Y')}. Če je podatek iz leta, ki je manjše od {now.year}, ga IGNORIRAJ.
DIREKTIVA #2 (OBLIKOVANJE): Odgovor naj bo kratek in pregleden. Ključne informacije **poudari s krepko pisavo**. Kjer naštevaš, **uporabi alineje (-)**.
DIREKTIVA #3 (POVEZAVE): Če je 'POVEZAVA' ne-prazna, jo vključi v klikljivi obliki: [Ime vira](URL).
DIREKTIVA #4 (SPECIFIČNOST): Če ne najdeš specifičnega podatka (npr. 'kontakt'), reci: "Žal nimam specifičnega kontakta za to temo."

--- KONTEKST ---
{context}---
ZGODOVINA POGOVORA:
{zgodovina_str}
---
VPRAŠANJE UPORABNIKA: "{vprasanje}"
ODGOVOR:"""

    # --------------------- usmerjanje ---------------------
    def odgovori(self, uporabnikovo_vprasanje: str, session_id: str) -> str:
        # Minimalni overridi za kratke odgovore
        ql = uporabnikovo_vprasanje.lower()

        # 1) župan — kratek odgovor
        if re.search(r'\bkdo je župan\b', ql):
            return "Župan občine Rače-Fram je **Samo Rajšp**."

        # 2) komunalni prispevek — bistveno
        if re.search(r'\bkomunaln[ia] prispev', ql):
            return (
                "**Komunalni prispevek**: plača ga investitor pred izdajo gradbenega dovoljenja.\n"
                "- **Zavezanec:** investitor/lastnik objekta\n"
                "- **Kdaj:** pred izdajo gradbenega dovoljenja\n"
                "- **Informativni izračun/vloga:** na občinski strani\n"
                "Več: https://www.race-fram.si/objava/400293"
            )

        # 3) prevoz Kopivnik — bistveno
        if re.search(r'\bprevoz\b.*\bkopivnik\b', ql):
            return (
                "**Šolski prevoz Kopivnik – OŠ Fram**\n"
                "- Popoldanski minibus: ~14:40 Fram → Kopivnik → Bukovec → Fram (~14:55)\n"
                "Več: https://www.osfram.si/prevozi"
            )

        # 4) kontakt — če ni specifično, vrni splošnega
        if re.search(r'\b(kontakt|telefon|e-?pošta|email)\b', ql):
            return (
                "Splošni kontakt **Občina Rače-Fram**:\n"
                "- Telefon: **02 609 60 10**\n"
                "- E-pošta: **obcina@race-fram.si**\n"
                "Če želiš kontakt za točno določen oddelek/temo, povej katero."
            )

        # normalni tok
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

        # spomin – shranjuj le zadnje 4
        zgodovina.append((uporabnikovo_vprasanje, odgovor))
        if len(zgodovina) > 4:
            zgodovina.pop(0)

        self.belezi_pogovor(session_id, uporabnikovo_vprasanje, odgovor)
        return odgovor

# ------------------------------------------------------------------------------
# 5) CLI za hiter test
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
