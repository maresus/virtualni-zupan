# VIRT_ZUPAN_RF_api.py — stabilni “backup” + odpadki + NAP + hiti
# Verzija 36.0 — FIX: Chroma get() brez "ids"; odpadki robustno + "naslednji"
# ---------------------------------------------------------------------------

import os
import sys
import re
import json
import unicodedata
import argparse
from datetime import datetime, timedelta
from difflib import SequenceMatcher

import chromadb
from chromadb.utils import embedding_functions
import requests
from dotenv import load_dotenv
from openai import OpenAI

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

CHROMA_DB_PATH = os.path.join(DATA_DIR, "chroma_db")
LOG_FILE_PATH = os.path.join(DATA_DIR, "zupan_pogovori.jsonl")

COLLECTION_NAME = "obcina_race_fram_prod"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
GENERATOR_MODEL_NAME = "gpt-4o-mini"

NAP_TOKEN_URL = "https://b2b.nap.si/uc/user/token"
NAP_DATA_URL = "https://b2b.nap.si/data/b2b.roadworks.geojson.sl_SI"
NAP_USERNAME = os.getenv("NAP_USERNAME")
NAP_PASSWORD = os.getenv("NAP_PASSWORD")

# -----------------------------------------------------------------------------
# Trigerji
# -----------------------------------------------------------------------------
PROMET_FILTER_KLJUCNIKI = [
    "rače", "race", "fram", "slivnica", "brunšvik", "brunsvik", "podova",
    "morje", "hoče", "hoce",
    "r2-430", "r3-711", "g1-2",
    "priključek slivnica", "razcep slivnica", "letališče maribor", "odcep za rače"
]
KLJUCNE_BESEDE_ODPADKI = [
    "smeti", "odpadki", "odvoz", "odpavkov", "komunala",
    "urnik", "rumena", "kanta", "embalaža", "embalaza",
    "steklo", "papir", "biolo", "mešani", "mesani"
]
KLJUCNE_BESEDE_PROMET = [
    "cesta", "ceste", "cesti", "promet", "dela", "delo",
    "zapora", "zapore", "zaprta", "zastoj", "gneča", "kolona", "zaprt", "oviran"
]

# -----------------------------------------------------------------------------
# Normalizacija & fuzzy
# -----------------------------------------------------------------------------
def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('utf-8')
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def fuzzy_match(a: str, b: str, threshold: float = 0.8) -> bool:
    a_n = normalize_text(a)
    b_n = normalize_text(b)
    if a_n in b_n or b_n in a_n:
        return True
    return SequenceMatcher(None, a_n, b_n).ratio() >= threshold

def slovenian_variant_equivalent(a: str, b: str) -> bool:
    a_n = normalize_text(a)
    b_n = normalize_text(b)
    if a_n == b_n:
        return True
    if len(a_n) > 1 and len(b_n) > 1:
        if a_n[:-1] == b_n[:-1] and {a_n[-1], b_n[-1]} <= {"a", "i", "e", "o", "u"}:
            return True
    return False

def street_phrase_matches(query_phrase: str, street_tok: str, threshold: float = 0.85) -> bool:
    """Ujemanje fraze (npr. 'bistriska cesta') s tok-om ('bistriska'), z ignoriranjem generičnih besed in fleksijo."""
    generic = {"cesta", "cesti", "ulica", "ulici", "pot", "trg", "ob", "naselje", "naselju"}
    qp = normalize_text(query_phrase)
    st = normalize_text(street_tok)
    if slovenian_variant_equivalent(qp, st):
        return True
    q_words = [w for w in qp.split() if w not in generic]
    street_words = [w for w in st.split() if w not in generic]
    if not q_words:
        return fuzzy_match(qp, st, threshold)
    for qw in q_words:
        if slovenian_variant_equivalent(qw, st):
            return True
        if SequenceMatcher(None, qw, st).ratio() >= threshold or (qw in st or st in qw):
            return True
        for sw in street_words:
            if slovenian_variant_equivalent(qw, sw):
                return True
            if SequenceMatcher(None, qw, sw).ratio() >= threshold or (qw in sw or sw in qw):
                return True
    return False

def gen_street_keys(street_name: str):
    """Ključi iz uličnega imena ('Bistriška cesta' -> ['bistriska','bistriske','bistriski'])."""
    base = normalize_text(street_name).replace("cesta", "").replace("ulica", "").strip()
    if not base:
        return []
    roots = [t for t in base.split() if len(t) >= 3] or [base]
    variants = set()
    for r in roots:
        variants.add(r)
        if not r.endswith("a"): variants.add(r + "a")
        if not r.endswith("i"): variants.add(r + "i")
        if not r.endswith("e"): variants.add(r + "e")
        if r.endswith("a"):
            variants.add(r[:-1] + "i")
            variants.add(r[:-1] + "e")
    return sorted(list(variants))

# -----------------------------------------------------------------------------
# Odpadki – kategorije
# -----------------------------------------------------------------------------
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

def get_canonical_waste(text: str):
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
            if SequenceMatcher(None, norm, normalize_text(v)).ratio() >= 0.85:
                return canonical
    return None

def extract_locations_from_naselja(naselja_field: str):
    """
    Robustno razbije 'naselja':
    - če so ločila (, ; \n) → delimo po njih
    - če NI ločil → razbij po presledkih, filtriraj šum (h, št, številke, generične besede)
    Vrne normalizirane tokene/fraze.
    """
    if not naselja_field:
        return []
    text = re.sub(r'\(h\.?\s*št\.?.*?\)', '', naselja_field, flags=re.IGNORECASE)

    if re.search(r'[;,\n]', text):
        parts = []
        for chunk in re.split(r'[;,\n]+', text):
            chunk = chunk.strip()
            if not chunk:
                continue
            parts.append(normalize_text(chunk))
        out, seen = [], set()
        for p in parts:
            if p and p not in seen:
                seen.add(p)
                out.append(p)
        return out

    toks = [t.strip() for t in text.split() if t.strip()]
    generic = {
        "cesta","cesti","ulica","ulici","pot","trg","naselje","obmocje","območje",
        "pri","na","v","pod","nad","k","do","od","proti"
    }
    noise = {
        "h","st","št","stev","stevilka","hiscna","hiscne","hiscni",
        "hisna","hisne","hisni","hisa","hise"
    }
    out, seen = [], set()
    for tok in toks:
        n = normalize_text(tok)
        if not n or n in generic or n in noise:
            continue
        if n.isdigit():
            continue
        if len(n) < 3:
            continue
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out

# -----------------------------------------------------------------------------
# Pomožni: varen Chroma get()
# -----------------------------------------------------------------------------
def chroma_get_safe(collection, **kwargs):
    """
    Ovije collection.get() in poskrbi, da 'include' NE vsebuje 'ids' (Render Chroma to zavrne).
    Vrne dict z 'documents' in 'metadatas' (če obstajajo).
    """
    include = kwargs.get("include")
    if include:
        include = [i for i in include if i in ("documents", "embeddings", "metadatas", "distances", "uris", "data")]
        kwargs["include"] = include
    try:
        return collection.get(**kwargs)
    except Exception:
        # failsafe brez include (privzeto vrne vse, kar sme)
        kwargs.pop("include", None)
        return collection.get(**kwargs)

# -----------------------------------------------------------------------------
# Virtualni župan
# -----------------------------------------------------------------------------
class VirtualniZupan:
    def __init__(self):
        print("Inicializacija razreda VirtualniZupan (Verzija 36.0 — RF_api backup + odpadki FIX)...")
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.collection = None
        self.zgodovina_seje = {}
        self._nap_access_token = None
        self._nap_token_expiry = None
        self._all_docs_cache = None

    # ---------------- Baza / Chroma ----------------
    def nalozi_bazo(self):
        if self.collection is None:
            try:
                print(f"Poskušam naložiti bazo znanja iz: {CHROMA_DB_PATH}")
                openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model_name=EMBEDDING_MODEL_NAME
                )
                chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
                self.collection = chroma_client.get_collection(
                    name=COLLECTION_NAME,
                    embedding_function=openai_ef
                )
                print(f"Povezano. Število dokumentov: {self.collection.count()}")
            except Exception as e:
                print(f"KRITIČNA NAPAKA: Baze znanja ni mogoče naložiti. Razlog: {e}")
                self.collection = None

    def _get_all_docs_cached(self):
        if self._all_docs_cache is not None:
            return self._all_docs_cache
        try:
            res = chroma_get_safe(self.collection, include=["documents", "metadatas"], limit=5000)
            docs = []
            if res and res.get("documents"):
                for i in range(len(res["documents"])):
                    docs.append({
                        "text": res["documents"][i],
                        "meta": res["metadatas"][i] if res.get("metadatas") else {}
                    })
            self._all_docs_cache = docs
            return docs
        except Exception:
            return []

    # ---------------- Beleženje ----------------
    def belezi_pogovor(self, session_id, vprasanje, odgovor):
        try:
            zapis = {
                "timestamp": datetime.now().isoformat(),
                "session_id": session_id,
                "vprasanje": vprasanje,
                "odgovor": odgovor
            }
            with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
                f.write(json.dumps(zapis, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Napaka pri beleženju pogovora: {e}")

    # ---------------- Spomin / preoblikovanje ----------------
    def preoblikuj_vprasanje_s_kontekstom(self, zgodovina_pogovora, zadnje_vprasanje):
        if not zgodovina_pogovora:
            return zadnje_vprasanje
        try:
            zgodovina_str = "\n".join(
                [f"Uporabnik: {q}\nAsistent: {a}" for q, a in zgodovina_pogovora]
            )
            prompt = f"""Tvoja naloga je, da glede na zgodovino pogovora preoblikuješ novo vprašanje v samostojno vprašanje. Bodi kratek in jedrnat.

Primer 1:
Uporabnik: kdaj je odvoz smeti na gortanovi ulici?
Asistent: Odvoz je vsak petek.
Novo vprašanje: "kaj pa papir?"
Samostojno vprašanje: "kdaj je odvoz za papir na gortanovi ulici?"

Primer 2:
Uporabnik: kaj je za malico 1.9?
Asistent: Za malico je francoski rogljič.
Novo vprašanje: "kaj pa naslednji dan?"
Samostojno vprašanje: "kaj je za malico 2.9?"
---
Zgodovina:
{zgodovina_str}
Novo vprašanje: "{zadnje_vprasanje}"
Samostojno vprašanje:"""
            resp = self.openai_client.chat.completions.create(
                model=GENERATOR_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100
            )
            out = (resp.choices[0].message.content or "").strip().replace('"', '')
            return out or zadnje_vprasanje
        except Exception:
            return zadnje_vprasanje

    # ---------------- NAP promet ----------------
    def _ensure_nap_token(self):
        if (self._nap_access_token and self._nap_token_expiry and
            datetime.utcnow() < self._nap_token_expiry - timedelta(seconds=60)):
            return self._nap_access_token
        if not NAP_USERNAME or not NAP_PASSWORD:
            raise RuntimeError("NAP poverilnice niso nastavljene.")
        payload = {'grant_type': 'password', 'username': NAP_USERNAME, 'password': NAP_PASSWORD}
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        response = requests.post(NAP_TOKEN_URL, data=payload, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        self._nap_access_token = data['access_token']
        self._nap_token_expiry = datetime.utcnow() + timedelta(seconds=data.get('expires_in', 3600))
        return self._nap_access_token

    def preveri_zapore_cest(self):
        if not NAP_USERNAME or not NAP_PASSWORD:
            return "Dostop do prometnih informacij ni mogoč."
        try:
            token = self._ensure_nap_token()
            headers = {'Authorization': f'Bearer {token}'}
            data_response = requests.get(NAP_DATA_URL, headers=headers, timeout=15)
            data_response.raise_for_status()
            vsi_dogodki = data_response.json().get('features', [])

            MUNICIPAL_FILTER = set(PROMET_FILTER_KLJUCNIKI)
            relevantne = []
            for dogodek in vsi_dogodki:
                props = dogodek.get('properties', {}) or {}
                cesta = str(props.get('cesta', '')).strip()
                opis = str(props.get('opis', '')).strip()
                ime = str(props.get('imeDogodka', '')).strip()
                cel = normalize_text(" ".join([cesta, opis, ime]))
                if not any(k in cel for k in MUNICIPAL_FILTER):
                    continue
                relevantne.append({
                    "cesta": cesta or "Ni podatka",
                    "opis": opis or "Ni podatka",
                    "imeDogodka": ime
                })

            if not relevantne:
                return ("Po podatkih portala promet.si na območju občine Rače-Fram "
                        "trenutno ni zabeleženih del na cesti, zapor ali zastojev.")

            merged = []
            for z in relevantne:
                added = False
                for m in merged:
                    ista_cesta = normalize_text(z['cesta']) == normalize_text(m['cesta'])
                    opis_sim = SequenceMatcher(None, normalize_text(z['opis']), normalize_text(m['opis'])).ratio()
                    if ista_cesta and opis_sim >= 0.9:
                        added = True
                        break
                if not added:
                    merged.append(z)

            ts = datetime.now().strftime("%d.%m.%Y %H:%M")
            porocilo = f"**Stanje na cestah (vir: NAP/promet.si, {ts})**\n\n"
            for z in merged:
                porocilo += f"- **Cesta:** {z['cesta']}\n  **Opis:** {z['opis']}\n"
            return porocilo.strip()
        except Exception:
            return "Žal mi neposreden vpogled v stanje na cestah trenutno ne deluje. Poskusite kasneje."

    # ---------------- Odpadki ----------------
    def obravnavaj_odvoz_odpadkov(self, uporabnikovo_vprasanje, session_id):
        print("-> Kličem specialista za odpadke...")
        # varna inicializacija seje
        self.zgodovina_seje.setdefault(session_id, {"zgodovina": [], "stanje": {}})
        stanje = self.zgodovina_seje[session_id]["stanje"]

        vprasanje_za_iskanje = (stanje.get('izvirno_vprasanje', '') + " " + (uporabnikovo_vprasanje or "")).strip()
        vprasanje_norm = normalize_text(vprasanje_za_iskanje)

        if not self.collection:
            return "Baza urnikov ni na voljo."

        vsi_urniki = chroma_get_safe(
            self.collection,
            where={"kategorija": "Odvoz odpadkov"},
            include=["documents", "metadatas"],
            limit=5000
        )
        if not vsi_urniki or not vsi_urniki.get('documents'):
            return "V bazi znanja ni podatkov o urnikih."

        iskani_tip = get_canonical_waste(vprasanje_norm)
        contains_naslednji = "naslednji" in vprasanje_norm

        waste_type_stopwords = {normalize_text(k) for k in WASTE_TYPE_VARIANTS.keys()}
        for variants in WASTE_TYPE_VARIANTS.values():
            for v in variants:
                waste_type_stopwords.add(normalize_text(v))
        extra_stop = {
            "kdaj","je","naslednji","odvoz","odpadkov","odpadke","smeti","na","v",
            "za","kako","kateri","katera","kaj","kje","rumene","rumena","kanta",
            "kante","ulici","cesti"
        }
        odstrani = waste_type_stopwords.union(extra_stop)

        raw_tokens = [t for t in re.split(r'[,\s]+', vprasanje_norm) if t and t not in odstrani]

        # lokacijske fraze (3→2→1)
        location_phrases = []
        for size in (3, 2, 1):
            for i in range(len(raw_tokens) - size + 1):
                phrase = " ".join(raw_tokens[i:i + size])
                if phrase:
                    location_phrases.append(phrase)
        # dedupe ob ohranitvi reda
        seen_set = set()
        filtered_phrases = []
        for p in location_phrases:
            if p in seen_set:
                continue
            seen_set.add(p)
            filtered_phrases.append(p)
        location_phrases = filtered_phrases

        street_indicators = {"cesta", "ulica", "pot", "trg", "naslov", "pod", "terasami"}
        is_explicit_location = (any(len(p.split()) > 1 for p in location_phrases) or
                                any(ind in vprasanje_norm for ind in street_indicators))

        def extract_datumi(doc_text: str):
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

        exact_street_matches, fuzzy_street_matches, area_matches = [], [], []

        docs = vsi_urniki['documents']
        metas = vsi_urniki.get('metadatas') or [{}] * len(docs)

        for i in range(len(docs)):
            meta = metas[i] or {}
            doc_text = docs[i] or ""

            meta_tip_raw = meta.get('tip_odpadka', '') or ''
            meta_tip_canon = get_canonical_waste(meta_tip_raw) or normalize_text(meta_tip_raw)
            if iskani_tip and meta_tip_canon != iskani_tip:
                continue

            lokacije = extract_locations_from_naselja(meta.get('naselja', '') or '')
            obm = normalize_text(meta.get('obmocje', '') or '')

            matched_for_this_doc = False

            # exact
            for phrase in location_phrases:
                for street_tok in lokacije:
                    if normalize_text(phrase) == normalize_text(street_tok) or slovenian_variant_equivalent(phrase, street_tok):
                        exact_street_matches.append({
                            'doc': doc_text, 'meta': meta, 'tip_canon': meta_tip_canon,
                            'matched_street': street_tok, 'matched_phrase': phrase,
                            'datumi': extract_datumi(doc_text)
                        })
                        matched_for_this_doc = True
                        break
                if matched_for_this_doc:
                    break

            # fuzzy
            if not matched_for_this_doc:
                for phrase in location_phrases:
                    for street_tok in lokacije:
                        thr = 0.80 if len(phrase.split()) > 1 else 0.85
                        if street_phrase_matches(phrase, street_tok, threshold=thr):
                            fuzzy_street_matches.append({
                                'doc': doc_text, 'meta': meta, 'tip_canon': meta_tip_canon,
                                'matched_street': street_tok, 'matched_phrase': phrase,
                                'datumi': extract_datumi(doc_text)
                            })
                            matched_for_this_doc = True
                            break
                    if matched_for_this_doc:
                        break

            # area
            if not matched_for_this_doc and obm:
                for phrase in location_phrases:
                    if fuzzy_match(phrase, obm, threshold=0.8):
                        area_matches.append({
                            'doc': doc_text, 'meta': meta, 'tip_canon': meta_tip_canon,
                            'matched_area': obm, 'matched_phrase': phrase,
                            'datumi': extract_datumi(doc_text)
                        })
                        break

        kandidati = exact_street_matches or fuzzy_street_matches or area_matches

        if is_explicit_location and not kandidati:
            if not iskani_tip:
                return "Kateri tip odpadka te zanima? (bio, mešani komunalni, embalaža, papir, steklo)"
            prv = (location_phrases[0] if location_phrases else "želeno lokacijo").title()
            return f"Za **{prv}** in tip **{iskani_tip}** žal nimam ustreznega urnika."

        if not kandidati:
            if not iskani_tip:
                return "Kateri tip odpadka te zanima? (bio, mešani komunalni, embalaža, papir, steklo)"
            return "Za navedeno kombinacijo tipa in lokacije žal nimam urnika."

        # “naslednji”
        if contains_naslednji:
            today = datetime.now().date()
            best_dt, best_tip, best_loc = None, None, None
            for c in kandidati:
                tip = c['meta'].get('tip_odpadka', c.get('tip_canon', ''))
                loc = c.get('matched_street') or c.get('matched_area') or c['meta'].get('obmocje', '')
                for m in re.findall(r'(\d{1,2})\.(\d{1,2})\.', c['doc']):
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
            return "Za iskani tip in lokacijo v dokumentu ni prihodnjih terminov."

        # sicer: prikaži prvo dobro ujemanje
        c = kandidati[0]
        tip = c['meta'].get('tip_odpadka', c.get('tip_canon', ''))
        loc = (c.get('matched_street') or c.get('matched_area') or c['meta'].get('obmocje', '')).title()
        datumi = c.get('datumi') or []
        if not datumi:
            datumi = []
            for m in re.findall(r'(\d{1,2})\.(\d{1,2})\.', c['doc']):
                try:
                    d, mo = int(m[0]), int(m[1])
                    s = f"{d}.{mo}."
                    if s not in datumi:
                        datumi.append(s)
                except Exception:
                    continue
        datumi_str = ", ".join(datumi) if datumi else "ni zabeleženih terminov"
        return f"Odvoz – {loc}: **{tip}**. Termini: {datumi_str}"

    # ---------------- Rule-based hiti ----------------
    def _hit_rule_based(self, vprasanje_norm: str):
        docs = self._get_all_docs_cached()
        if not docs:
            return None

        # Direktor(ica) OU
        if ("direktor" in vprasanje_norm and "obcins" in vprasanje_norm) or \
           ("direktorica" in vprasanje_norm and "obcins" in vprasanje_norm):
            ime, tel, mail = None, None, None
            for d in docs:
                t = d["text"]; n = normalize_text(t)
                if "direktorica obcinske uprave" in n or "direktor obcinske uprave" in n or "karmen kotnik" in n:
                    if "karmen kotnik" in n:
                        ime = "mag. Karmen Kotnik"
                    m = re.search(r'[\w\.-]+@race-fram\.si', t, re.IGNORECASE)
                    if m: mail = m.group(0)
                    m2 = re.search(r'\b02\s?609\s?60\s?\d{2}\b', t)
                    if m2: tel = m2.group(0)
            if ime or tel or mail:
                parts = [f"**Direktorica občinske uprave** je **{ime or 'mag. Karmen Kotnik'}**."]
                if tel:  parts.append(f"**Telefon:** {tel}")
                if mail: parts.append(f"**E-pošta:** {mail}")
                return " ".join(parts)

        # Turizem
        if "turizem" in vprasanje_norm or "turistic" in vprasanje_norm:
            oseba, mail, tel = None, None, None
            for d in docs:
                t = d["text"]; n = normalize_text(t)
                if "tanja kosi" in n and ("turizem" in n or "turistic" in n):
                    oseba = "Tanja Kosi (višja svetovalka za okolje, kmetijstvo, turizem in CZ)"
                    m = re.search(r'[\w\.-]+@race-fram\.si', t, re.IGNORECASE)
                    if m: mail = m.group(0)
                    m2 = re.search(r'\b02\s?609\s?60\s?\d{2}\b', t)
                    if m2: tel = m2.group(0)
                    break
            if oseba or mail or tel:
                out = [f"Za področje **turizma** je zadolžena **{oseba or 'Tanja Kosi'}**."]
                if mail: out.append(f"**E-pošta:** {mail}")
                if tel:  out.append(f"**Telefon:** {tel}")
                return " ".join(out)

        # E-pošta občine
        if ("e-post" in vprasanje_norm or "email" in vprasanje_norm or "e mail" in vprasanje_norm) and \
           ("obcine" in vprasanje_norm or "obcina" in vprasanje_norm):
            maili = set()
            for d in docs:
                for m in re.findall(r'[\w\.-]+@race-fram\.si', d["text"], re.IGNORECASE):
                    if m.lower() in ("obcina@race-fram.si", "info@race-fram.si"):
                        maili.add(m)
            if maili:
                maili = sorted(list(maili))
                return "**E-poštni naslov občine Rače-Fram**: " + " ali ".join(maili)

        # EV polnilnice
        if ("polnilnic" in vprasanje_norm or "polnilnice" in vprasanje_norm or "ev" in vprasanje_norm) and \
           ("obcin" in vprasanje_norm or "race" in vprasanje_norm or "fram" in vprasanje_norm):
            lok, cena = None, None
            for d in docs:
                n = normalize_text(d["text"])
                if "polnilnic" in n or "polnilnice" in n:
                    if ("grad race" in n or "dtv partizan fram" in n) and not lok:
                        lok = "Grad Rače in DTV Partizan Fram"
                    m = re.search(r'0[,\.]?25\s?eur/kwh', d["text"], re.IGNORECASE)
                    if m: cena = "0,25 EUR/kWh (+ štartnina 0,50 EUR; rezervacija 1,00 EUR)"
            if lok or cena:
                parts = ["**Električne polnilnice** v občini:"]
                if lok:  parts.append(f"- Lokacije: {lok}")
                if cena: parts.append(f"- Cene: {cena}")
                return "\n".join(parts)

        return None

    # ---------------- Glavni odgovor ----------------
    def odgovori(self, uporabnikovo_vprasanje: str, session_id: str):
        self.nalozi_bazo()
        if not self.collection:
            return "Oprostite, moja baza znanja trenutno ni na voljo."

        self.zgodovina_seje.setdefault(session_id, {'zgodovina': [], 'stanje': {}})
        stanje = self.zgodovina_seje[session_id]['stanje']
        zgodovina = self.zgodovina_seje[session_id]['zgodovina']

        pametno_vprasanje = self.preoblikuj_vprasanje_s_kontekstom(zgodovina, uporabnikovo_vprasanje)
        vprasanje_lower = normalize_text(pametno_vprasanje)

        # 1) Odpadki
        if any(re.search(r'\b' + re.escape(k) + r'\b', vprasanje_lower) for k in KLJUCNE_BESEDE_ODPADKI):
            odgovor = self.obravnavaj_odvoz_odpadkov(pametno_vprasanje, session_id)

        # 2) Promet
        elif any(re.search(r'\b' + re.escape(k) + r'\b', vprasanje_lower) for k in KLJUCNE_BESEDE_PROMET):
            odgovor = self.preveri_zapore_cest()

        # 3) Rule-based hiti
        else:
            rb = self._hit_rule_based(vprasanje_lower)
            if rb:
                odgovor = rb
            else:
                # 4) RAG + LLM
                rezultati_iskanja = self.collection.query(
                    query_texts=[vprasanje_lower],
                    n_results=5,
                    include=["documents", "metadatas"]
                )
                kontekst_baza = ""
                if rezultati_iskanja.get('documents'):
                    for doc, meta in zip(rezultati_iskanja['documents'][0], rezultati_iskanja['metadatas'][0]):
                        kontekst_baza += (
                            f"--- VIR: {meta.get('source', 'Neznan')}\n"
                            f"POVEZAVA: {meta.get('source_url', 'Brez')}\n"
                            f"VSEBINA: {doc}\n\n"
                        )

                if not kontekst_baza:
                    odgovor = "Žal o tem nimam nobenih informacij."
                else:
                    now = datetime.now()
                    prompt_za_llm = (
                        f"Ti si 'Virtualni župan občine Rače-Fram'.\n"
                        f"DIREKTIVA #1 (VAROVALKA ZA DATUME): Današnji datum je {now.strftime('%d.%m.%Y')}. "
                        f"Če je podatek iz leta, ki je manjše od {now.year}, ga IGNORIRAJ.\n"
                        "DIREKTIVA #2 (OBLIKOVANJE): Odgovor mora biti pregleden. Ključne informacije **poudari**. "
                        "Kjer naštevaš, **uporabi alineje (-)**.\n"
                        "DIREKTIVA #3 (POVEZAVE): Če najdeš URL, ga MORAŠ vključiti v klikljivi obliki: [Ime vira](URL).\n"
                        "DIREKTIVA #4 (SPECIFIČNOST): Če ne najdeš specifičnega podatka (npr. 'kontakt'), NE ponavljaj splošnih informacij. "
                        "Raje reci: \"Žal nimam specifičnega kontakta za to temo.\"\n\n"
                        f"--- KONTEKST ---\n{kontekst_baza}---\n"
                        f"VPRAŠANJE: \"{uporabnikovo_vprasanje}\"\n"
                        "ODGOVOR:"
                    )
                    try:
                        resp = self.openai_client.chat.completions.create(
                            model=GENERATOR_MODEL_NAME,
                            messages=[{"role": "user", "content": prompt_za_llm}],
                            temperature=0.0
                        )
                        odgovor = resp.choices[0].message.content
                    except Exception:
                        odgovor = "Žal odgovora trenutno ne morem sestaviti."

        # beleženje + kratka zgodovina
        zgodovina.append((uporabnikovo_vprasanje, odgovor))
        if len(zgodovina) > 4:
            zgodovina.pop(0)
        self.belezi_pogovor(session_id, uporabnikovo_vprasanje, odgovor)
        return odgovor

# -----------------------------------------------------------------------------
# CLI orodja
# -----------------------------------------------------------------------------
def _cmd_debug_street(zupan: VirtualniZupan, street: str, tip: str = "steklo"):
    q = f"KDAJ je odvoz {tip} na {street}"
    print(zupan.obravnavaj_odvoz_odpadkov(q, "debug_cli"))

def _cmd_repl(zupan: VirtualniZupan):
    print("Župan je pripravljen. Vprašajte ga ('konec' za izhod):")
    session_id = "local_cli"
    while True:
        try:
            q = input("> ").strip()
            if q.lower() in ("konec", "exit", "quit"):
                break
            if not q:
                continue
            print("\n--- ODGOVOR ŽUPANA ---\n")
            print(zupan.odgovori(q, session_id))
            print("\n----------------------\n")
        except KeyboardInterrupt:
            break

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Virtualni župan — RF_api (backup) + odpadki + NAP + hiti")
    parser.add_argument("--debug-street", type=str, help='Hitri test odpadkov, npr. --debug-street "Bistriška cesta"')
    parser.add_argument("--debug-tip", type=str, default="steklo", help='Tip odpadka za --debug-street (default: steklo)')
    args = parser.parse_args()

    vz = VirtualniZupan()
    vz.nalozi_bazo()

    if args.debug_street:
        _cmd_debug_street(vz, args.debug_street, args.debug_tip)
        sys.exit(0)

    _cmd_repl(vz)
