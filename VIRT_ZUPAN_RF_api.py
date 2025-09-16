import os
import json
import chromadb
import requests
import re
import unicodedata
from datetime import datetime, timedelta
from openai import OpenAI
from dotenv import load_dotenv
from chromadb.utils import embedding_functions
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple, Any
import time
import locale

# --- KONFIGURACIJA ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '..', '.env'))

try:
    locale.setlocale(locale.LC_TIME, 'sl_SI.UTF-8')
except Exception:
    try:
        locale.setlocale(locale.LC_TIME, 'sl_SI')
    except Exception:
        pass

if os.getenv('ENV_TYPE') == 'production':
    DATA_DIR = "/data"
    print("Zaznano produkcijsko okolje (Render). Poti so nastavljene na /data.")
else:
    DATA_DIR = os.path.join(BASE_DIR, "data")
    print("Zaznano lokalno okolje. Poti so nastavljene relativno.")
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

CHROMA_DB_PATH = os.path.join(DATA_DIR, "chroma_db")
LOG_FILE_PATH = os.path.join(DATA_DIR, "zupan_pogovori.jsonl")
IZVORNI_PODATKI_PATH = os.path.join(BASE_DIR, "izvorni_podatki")

COLLECTION_NAME = "obcina_race_fram_prod"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
GENERATOR_MODEL_NAME = "gpt-4o-mini"

NAP_TOKEN_URL = "https://b2b.nap.si/uc/user/token"
NAP_DATA_URL = "https://b2b.nap.si/data/b2b.roadworks.geojson.sl_SI"
NAP_USERNAME = os.getenv("NAP_USERNAME")
NAP_PASSWORD = os.getenv("NAP_PASSWORD")

MEAL_FILES = {
    "fram": os.path.join(IZVORNI_PODATKI_PATH, "prehrana fram.jsonl"),
    "race": os.path.join(IZVORNI_PODATKI_PATH, "prehrana race.jsonl"),
}

SCHOOL_DISPLAY_NAMES = {
    "fram": "OŠ Fram",
    "race": "OŠ Rače",
}

EMAIL_PATTERN = re.compile(r"[\w\.-]+@[\w\.-]+\.[\w]{2,}")
PHONE_PATTERN = re.compile(r"(?:\+386\s*|\b0)(?:[\d\s\/-]{5,}\d)")

# --- POMOŽNE FUNKCIJE ---
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
    ratio = SequenceMatcher(None, a_n, b_n).ratio()
    return ratio >= threshold

def extract_email(text: str) -> Optional[str]:
    if not text:
        return None
    match = EMAIL_PATTERN.search(text)
    if match:
        return match.group(0).strip().strip('.,;')
    return None

def extract_phone(text: str) -> Optional[str]:
    if not text:
        return None
    match = PHONE_PATTERN.search(text)
    if match:
        phone = match.group(0).strip().strip('.,;')
        phone = re.sub(r"\s+", " ", phone)
        return phone
    return None

def get_slovenian_day_name(date: datetime) -> str:
    days_sl = {
        0: 'ponedeljek',
        1: 'torek',
        2: 'sreda',
        3: 'četrtek',
        4: 'petek',
        5: 'sobota',
        6: 'nedelja'
    }
    return days_sl.get(date.weekday(), 'neznan dan')

def get_tomorrow_date() -> Tuple[datetime, str]:
    tomorrow = datetime.now() + timedelta(days=1)
    day_name = get_slovenian_day_name(tomorrow)
    return tomorrow, day_name

def extract_date_from_text(text: str) -> Optional[datetime]:
    patterns = [
        r'(\d{1,2})\.\s*(\d{1,2})\.\s*(\d{4})',
        r'(\d{1,2})\.\s*(\d{1,2})\.',
        r'(\d{1,2})\.\s*(september|oktober|november|december|januar|februar|marec|april|maj|junij|julij|avgust)',
    ]
    months = {
        'januar': 1, 'februar': 2, 'marec': 3, 'april': 4,
        'maj': 5, 'junij': 6, 'julij': 7, 'avgust': 8,
        'september': 9, 'oktober': 10, 'november': 11, 'december': 12
    }
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                if len(match.groups()) == 3 and match.group(3).isdigit():
                    return datetime(int(match.group(3)), int(match.group(2)), int(match.group(1)))
                if len(match.groups()) == 2 and match.group(2).isdigit():
                    current_year = datetime.now().year
                    return datetime(current_year, int(match.group(2)), int(match.group(1)))
                if len(match.groups()) == 2 and match.group(2).lower() in months:
                    current_year = datetime.now().year
                    return datetime(current_year, months[match.group(2).lower()], int(match.group(1)))
            except Exception:
                continue
    return None

WASTE_TYPE_VARIANTS = {
    "Biološki odpadki": [
        "bioloski odpadki", "bioloskih odpakov", "bioloski", "bioloskih", "bio", "biološki odpadki",
        "bioloskih odpadkov", "bioloski odpadkov", "bioloških odpadkov", "bioloških odpadki"
    ],
    "Mešani komunalni odpadki": [
        "mesani komunalni odpadki", "mešani komunalni odpadki", "mesani", "mešani",
        "mešane odpadke", "mesane odpadke", "mešani odpadki", "mešane komunalne",
        "mesane komunalne", "mešane komunalne odpadke", "mesane komunalne odpadke",
        "komunalni odpadki", "komunalnih odpadkov", "komunalne odpadke"
    ],
    "Odpadna embalaža": [
        "odpadna embalaza", "odpadna embalaža", "embalaza", "embalaža", "embalaže",
        "rumena kanta", "rumene kante", "plastika"
    ],
    "Papir in karton": [
        "papir in karton", "papir", "karton", "papirja", "kartona", "papir in kartona"
    ],
    "Steklena embalaža": [
        "steklena embalaza", "steklena embalaža", "steklo", "stekla", "stekle",
        "stekleno", "stekleni", "steklen"
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
    if "embal" in norm or "plastik" in norm:
        return "Odpadna embalaža"
    for canonical, variants in WASTE_TYPE_VARIANTS.items():
        if normalize_text(canonical) in norm:
            return canonical
        for v in variants:
            if normalize_text(v) in norm:
                return canonical
    return None
class VirtualniZupan:
    def __init__(self):
        print("🚀 Inicializacija VirtualniZupan v37 - Date Filter")
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.collection = None
        self.zgodovina_seje: Dict[str, Dict[str, Any]] = {}
        self._nap_access_token: Optional[str] = None
        self._nap_token_expiry: Optional[datetime] = None
        self.location_index: Dict[str, List[Dict[str, Any]]] = {}
        self.waste_schedule_cache: Dict[str, Any] = {}
        self.index_built = False
        self.meal_data = self._load_meal_data()

    # -------------------- NALAGANJE BAZE --------------------
    def nalozi_bazo(self):
        if self.collection is None:
            try:
                print(f"📂 Nalagam bazo znanja iz: {CHROMA_DB_PATH}")
                openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=os.getenv("OPENAI_API_KEY"),
                    model_name=EMBEDDING_MODEL_NAME
                )
                chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
                self.collection = chroma_client.get_collection(
                    name=COLLECTION_NAME,
                    embedding_function=openai_ef
                )
                print(f"✅ Povezano. Število dokumentov: {self.collection.count()}")
                if not self.index_built:
                    self._build_indexes()
            except Exception as e:
                print(f"❌ KRITIČNA NAPAKA: Baze znanja ni mogoče naložiti. Razlog: {e}")
                self.collection = None

    # -------------------- INDEKSI --------------------
    def _build_indexes(self):
        print("🔨 Gradim indekse za hitro iskanje...")
        start_time = time.time()
        try:
            try:
                all_docs = self.collection.get(where={"kategorija": "Odvoz odpadkov"}, limit=1000)
            except Exception:
                all_docs = self.collection.get(limit=1000)
            if not all_docs or not all_docs.get('ids'):
                print("⚠️ Ni dokumentov za indeksiranje")
                self.index_built = True
                return
            count = 0
            for i, doc_id in enumerate(all_docs['ids']):
                metadata = all_docs['metadatas'][i] if i < len(all_docs['metadatas']) else {}
                if metadata.get('kategorija') != 'Odvoz odpadkov':
                    doc_text = all_docs['documents'][i] if i < len(all_docs['documents']) else ""
                    if not any(w in doc_text.lower() for w in ['odvoz', 'odpadki', 'smeti']):
                        continue
                doc_text = all_docs['documents'][i] if i < len(all_docs['documents']) else ""
                locations = self._extract_all_locations(doc_text, metadata)
                tip_odpadka = metadata.get('tip_odpadka', '')
                for loc in locations:
                    loc_norm = normalize_text(loc)
                    if loc_norm not in self.location_index:
                        self.location_index[loc_norm] = []
                    self.location_index[loc_norm].append({
                        'doc_id': doc_id,
                        'doc': doc_text,
                        'meta': metadata,
                        'tip': get_canonical_waste(tip_odpadka) or tip_odpadka,
                        'original_location': loc
                    })
                    count += 1
            self.index_built = True
            elapsed = time.time() - start_time
            print(f"✅ Indeksi zgrajeni v {elapsed:.2f}s. Indeksiranih vnosov: {count}, Unikatnih lokacij: {len(self.location_index)}")
        except Exception as e:
            print(f"⚠️ Napaka pri gradnji indeksov: {e}")
            self.index_built = True

    def _extract_all_locations(self, doc_text: str, metadata: Dict[str, Any]) -> List[str]:
        locations = set()
        if 'naselja' in metadata:
            parts = re.split(r'[,:]+', metadata['naselja'])
            for part in parts:
                part = part.strip()
                if part and len(part) > 2:
                    locations.add(part)
                    if 'ulica' in part.lower():
                        clean = part.lower().replace('ulica', '').strip()
                        if clean:
                            locations.add(clean)
                    if 'cesta' in part.lower():
                        clean = part.lower().replace('cesta', '').strip()
                        if clean:
                            locations.add(clean)
        if len(doc_text) < 10000:
            street_patterns = [
                r'([A-ZČŠŽa-zčšž]+\s+(?:ulica|cesta|pot|trg))',
                r'((?:ulica|cesta|pot|trg)\s+[A-ZČŠŽa-zčšž]+)',
                r'(Pod\s+[A-ZČŠŽa-zčšž]+)',
            ]
            for pattern in street_patterns:
                matches = re.findall(pattern, doc_text[:2000], re.IGNORECASE)
                for match in matches[:10]:
                    if isinstance(match, tuple):
                        match = match[0]
                    if len(match) > 3:
                        locations.add(match.strip())
        if 'terasa' in doc_text.lower() or 'terasami' in doc_text.lower():
            locations.update(['Pod terasami', 'pod terasami', 'terasami'])
        return list(locations)

    def _load_meal_data(self) -> Dict[Tuple[str, datetime.date], List[Dict[str, Any]]]:
        meal_data: Dict[Tuple[str, datetime.date], List[Dict[str, Any]]] = {}
        for school_key, path in MEAL_FILES.items():
            if not os.path.exists(path):
                continue
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        metadata = record.get('metadata', {})
                        datum = metadata.get('datum')
                        if not datum:
                            continue
                        try:
                            date_obj = datetime.fromisoformat(datum).date()
                        except ValueError:
                            continue
                        key = (school_key, date_obj)
                        meal_data.setdefault(key, []).append({
                            'text': record.get('text', ''),
                            'metadata': metadata
                        })
            except Exception as e:
                print(f"⚠️ Napaka pri nalaganju jedilnika ({path}): {e}")
        if meal_data:
            print(f"🍽️ Naloženih jedilnikov: {len(meal_data)}")
        else:
            print("⚠️ Jedilniki niso bili naloženi iz lokalnih datotek.")
        return meal_data

    def _resolve_school_from_question(self, question_norm: str, stanje: Dict[str, Any]) -> Optional[str]:
        if 'fram' in question_norm:
            return 'fram'
        if 'race' in question_norm or 'raca' in question_norm:
            return 'race'
        if 'os fram' in question_norm:
            return 'fram'
        if 'os race' in question_norm:
            return 'race'
        return stanje.get('zadnja_sola')

    def _format_meal_entry(self, entry_text: str) -> str:
        if not entry_text:
            return ""
        body = re.sub(r'^na jedilniku[^:]*:\s*', '', entry_text, flags=re.IGNORECASE)
        parts: List[str] = []
        for chunk in re.split(r';\s*', body):
            chunk = chunk.strip()
            if not chunk:
                continue
            if 'Opomba:' in chunk:
                before, note = chunk.split('Opomba:', 1)
                if before.strip():
                    parts.append(before.strip())
                parts.append(f"Opomba: {note.strip()}")
            else:
                parts.append(chunk)
        formatted_lines: List[str] = []
        for segment in parts:
            if '-' in segment:
                label, rest = segment.split('-', 1)
                formatted_lines.append(f"- **{label.strip()}** – {rest.strip()}")
            else:
                formatted_lines.append(f"- {segment}")
        return "\n".join(formatted_lines)

    # -------------------- JEDILNIKI --------------------
    def obravnavaj_jedilnik(self, vprasanje: str, session_id: str) -> str:
        print("🍽️ Obravnavam vprašanje o jedilniku...")
        if session_id not in self.zgodovina_seje:
            self.zgodovina_seje[session_id] = {'zgodovina': [], 'stanje': {}}
        stanje = self.zgodovina_seje[session_id]['stanje']

        now = datetime.now()
        target_date: Optional[datetime] = None
        target_day: Optional[str] = None
        vprasanje_lower = vprasanje.lower()
        vprasanje_norm = normalize_text(vprasanje)

        date_match = re.search(r'(\d{1,2})\.\s*(\d{1,2})(?:\.\s*(\d{4}))?', vprasanje)
        if date_match:
            try:
                day = int(date_match.group(1))
                month = int(date_match.group(2))
                year = int(date_match.group(3)) if date_match.group(3) else now.year
                target_date = datetime(year, month, day)
                target_day = get_slovenian_day_name(target_date)
            except Exception:
                pass

        if not target_date:
            days_map = {'ponedeljek': 0, 'torek': 1, 'sred': 2, 'četrtek': 3, 'petek': 4, 'sobota': 5, 'nedelja': 6}
            for day_name, day_num in days_map.items():
                if day_name in vprasanje_lower:
                    days_ahead = (day_num - now.weekday()) % 7
                    target_date = now + timedelta(days=days_ahead)
                    target_day = 'sreda' if day_name == 'sred' else day_name
                    break
            if not target_date and 'jutri' in vprasanje_lower:
                target_date = now + timedelta(days=1)
                target_day = get_slovenian_day_name(target_date)
            if not target_date and 'danes' in vprasanje_lower:
                target_date = now
                target_day = get_slovenian_day_name(target_date)

        if not target_date:
            prev = stanje.get('zadnji_jedilnik_datum')
            if isinstance(prev, datetime):
                target_date = prev
                target_day = get_slovenian_day_name(prev)

        if not target_date:
            return "Prosim, navedite dan ali datum za jedilnik."

        school = self._resolve_school_from_question(vprasanje_norm, stanje)
        if not school:
            return "Prosim, navedite, za katero šolo (OŠ Fram ali OŠ Rače) želite jedilnik."

        stanje['zadnji_jedilnik_datum'] = target_date
        stanje['zadnja_sola'] = school

        target_day = target_day or get_slovenian_day_name(target_date)
        display_school = SCHOOL_DISPLAY_NAMES.get(school, school.title())
        print(f"  🎯 Iščem jedilnik za: {display_school}, {target_day}, {target_date.strftime('%d.%m.%Y')}")

        meal_entries = self.meal_data.get((school, target_date.date()))
        if meal_entries:
            formatted_sections = [self._format_meal_entry(entry.get('text', '')) for entry in meal_entries]
            formatted_sections = [section for section in formatted_sections if section]
            if formatted_sections:
                response = (
                    f"**Jedilnik za {display_school} – {target_day.capitalize()}, {target_date.strftime('%d.%m.%Y')}:**\n\n"
                    + "\n\n".join(formatted_sections)
                )
                metadata = meal_entries[0].get('metadata', {})
                vir = metadata.get('vir')
                source_url = metadata.get('source_url')
                if vir:
                    response += f"\n\nVir: {vir}"
                if source_url:
                    response += f"\nPovezava: {source_url}"
                return response

        if not self.collection:
            return f"Žal nimam podatkov o jedilniku za {target_day}, {target_date.strftime('%d.%m.%Y')}"

        search_parts = [display_school]
        if 'kosilo' in vprasanje_lower:
            search_parts.append('kosilo')
        elif 'malica' in vprasanje_lower:
            search_parts.append('malica')
        elif 'zajtrk' in vprasanje_lower:
            search_parts.append('zajtrk')
        search_parts.append(target_day)
        search_parts.append(target_date.strftime('%d.%m'))
        search_query = " ".join(search_parts)

        results = self.collection.query(query_texts=[search_query], n_results=20)
        if not results['documents'] or not results['documents'][0]:
            return f"Žal nimam podatkov o jedilniku za {target_day}, {target_date.strftime('%d.%m.%Y')}"

        best_match = None
        best_date_diff = float('inf')
        for doc in results['documents'][0]:
            if not any(word in doc.lower() for word in ['kosilo', 'malica', 'zajtrk', 'jedilnik']):
                continue
            doc_date = extract_date_from_text(doc)
            if doc_date:
                diff = abs((doc_date - target_date).days)
                if diff == 0:
                    best_match = doc
                    break
                if diff < best_date_diff:
                    best_date_diff = diff
                    best_match = doc

        if best_match:
            doc_date = extract_date_from_text(best_match)
            if doc_date and doc_date.date() == target_date.date():
                lines = best_match.split('\n')
                relevant_lines: List[str] = []
                capturing = False
                date_token = f"{target_date.day}.{target_date.month}"
                date_token_zero = target_date.strftime('%d.%m')
                for line in lines:
                    line_lower = line.lower()
                    if date_token in line_lower or date_token_zero in line_lower:
                        capturing = True
                    elif capturing and any(d in line_lower for d in ['ponedeljek', 'torek', 'sreda', 'četrtek', 'petek']):
                        break
                    if capturing:
                        relevant_lines.append(line)
                if relevant_lines:
                    response = f"**Jedilnik za {display_school} – {target_day}, {target_date.strftime('%d.%m.%Y')}:**\n\n"
                    response += "\n".join(relevant_lines[:10])
                    return response
                return best_match[:500]
            if doc_date:
                return (
                    f"Žal nimam jedilnika za {target_day}, {target_date.strftime('%d.%m.%Y')}. "
                    f"Najbližji podatki so za {get_slovenian_day_name(doc_date)}, {doc_date.strftime('%d.%m.%Y')}."
                )
        return f"Žal nimam podatkov o jedilniku za {target_day}, {target_date.strftime('%d.%m.%Y')}"

    # -------------------- ODPADKI --------------------
    def obravnavaj_odvoz_odpadkov_systematic(self, uporabnikovo_vprasanje: str, session_id: str) -> str:
        print("🎯 Sistemski pristop za odpadke...")
        if not self.collection:
            return "V bazi znanja ni podatkov o urnikih odpadkov."
        vprasanje_norm = normalize_text(uporabnikovo_vprasanje)
        vprasanje_norm = re.sub(r'\bnaslednj\w*\b', '', vprasanje_norm)
        iskani_tip = get_canonical_waste(vprasanje_norm)
        contains_naslednji = bool(re.search(r'\bnaslednj\w*\b', normalize_text(uporabnikovo_vprasanje)))
        print(f"  📊 Tip: {iskani_tip}, Naslednji: {contains_naslednji}")

        search_query = uporabnikovo_vprasanje + (f" {iskani_tip}" if iskani_tip else "")
        try:
            wide_results = self.collection.query(
                query_texts=[search_query],
                n_results=30,
                include=["documents", "metadatas", "distances"]
            )
        except Exception:
            wide_results = self.collection.query(query_texts=["odvoz odpadki " + uporabnikovo_vprasanje], n_results=30)

        location_candidates: List[Dict[str, Any]] = []
        potential_locations = self._extract_query_locations(vprasanje_norm)
        for loc in potential_locations:
            loc_norm = normalize_text(loc)
            if loc_norm in self.location_index:
                location_candidates.extend(self.location_index[loc_norm])
            if len(self.location_index) < 1000:
                for idx_loc in self.location_index.keys():
                    if fuzzy_match(loc_norm, idx_loc, 0.85):
                        location_candidates.extend(self.location_index[idx_loc])

        all_candidates: List[Dict[str, Any]] = []
        seen_docs = set()
        for cand in location_candidates:
            if iskani_tip and cand['tip'] and cand['tip'] != iskani_tip:
                continue
            doc_id = cand.get('doc_id', str(hash(cand['doc'])))
            if doc_id not in seen_docs:
                all_candidates.append({'doc': cand['doc'], 'meta': cand['meta'], 'score': 0.95, 'tip': cand['tip'], 'source': 'index'})
                seen_docs.add(doc_id)

        if wide_results['documents'] and wide_results['documents'][0]:
            distances = wide_results.get('distances', [[0.5] * len(wide_results['documents'][0])])[0]
            for doc, meta, dist in zip(
                wide_results['documents'][0][:15],
                wide_results['metadatas'][0][:15],
                distances[:15]
            ):
                if meta.get('kategorija') != 'Odvoz odpadkov':
                    if not any(w in doc.lower() for w in ['odvoz', 'odpadki', 'smeti']):
                        continue
                doc_tip = meta.get('tip_odpadka', '') or get_canonical_waste(doc.lower())
                if iskani_tip and doc_tip and iskani_tip != doc_tip:
                    continue
                doc_id = str(hash(doc))
                if doc_id not in seen_docs:
                    all_candidates.append({'doc': doc, 'meta': meta, 'score': 1 - dist, 'tip': doc_tip, 'source': 'semantic'})
                    seen_docs.add(doc_id)

        all_candidates.sort(key=lambda x: x['score'], reverse=True)
        print(f"  📋 Najdenih kandidatov: {len(all_candidates)}")
        if not all_candidates:
            if not iskani_tip:
                return "Kateri tip odpadka vas zanima? (npr. steklo, papir, bio, embalaža, komunalni)"
            return f"Žal nisem našel urnika za {iskani_tip}. Poskusite z bolj specifično lokacijo ali pokličite 02 609 60 10"
        return self._format_waste_answer(all_candidates, contains_naslednji, uporabnikovo_vprasanje)

    def _extract_query_locations(self, query_norm: str) -> List[str]:
        query_norm = re.sub(r'\bnaslednj\w*\b', '', query_norm)
        locations: List[str] = []
        known_patterns = {
            'pod terasa': ['pod terasami', 'Pod terasami'],
            'terasa': ['pod terasami', 'Pod terasami'],
            'bistrisk': ['bistriška cesta', 'Bistriška cesta', 'bistriska'],
            'bistriš': ['bistriška cesta', 'Bistriška cesta'],
            'mlinsk': ['mlinska ulica', 'Mlinska ulica', 'mlinska'],
            'turnerj': ['turnerjeva', 'Turnerjeva ulica', 'turnerjeva ulica'],
        }
        for pattern, locs in known_patterns.items():
            if pattern in query_norm:
                locations.extend(locs)
        stopwords = {'kdaj', 'je', 'naslednji', 'odvoz', 'odpadkov', 'smeti',
                     'na', 'v', 'za', 'kako', 'kateri', 'kaj', 'kje', 'pa'}
        for canonical in WASTE_TYPE_VARIANTS.keys():
            stopwords.add(normalize_text(canonical))
        for variants in WASTE_TYPE_VARIANTS.values():
            for v in variants:
                stopwords.add(normalize_text(v))
        tokens = [t for t in query_norm.split() if t and t not in stopwords]
        if not locations:
            for size in range(min(3, len(tokens)), 0, -1):
                for i in range(len(tokens) - size + 1):
                    phrase = " ".join(tokens[i:i + size])
                    if phrase and len(phrase) > 2:
                        locations.append(phrase)
        seen = set()
        unique_locations = []
        for loc in locations:
            if loc not in seen:
                seen.add(loc)
                unique_locations.append(loc)
        return unique_locations

    def _format_waste_answer(self, candidates: List[Dict[str, Any]], contains_naslednji: bool, original_query: str) -> str:
        now = datetime.now()
        if contains_naslednji:
            najblizji = None
            najblizji_info = None
            for kandidat in candidates[:5]:
                doc = kandidat['doc']
                tip = kandidat['tip']
                for match in re.findall(r'(\d{1,2})\.(\d{1,2})\.', doc):
                    try:
                        dan, mesec = int(match[0]), int(match[1])
                        datum = datetime(now.year, mesec, dan)
                        if datum.date() < now.date():
                            datum = datetime(now.year + 1, mesec, dan)
                        if not najblizji or datum < najblizji:
                            najblizji = datum
                            lokacija = self._find_location_in_doc(doc, original_query)
                            najblizji_info = (datum, tip, lokacija)
                    except Exception:
                        continue
            if najblizji_info:
                datum, tip, lokacija = najblizji_info
                if lokacija:
                    return f"Naslednji odvoz **{tip}** ({lokacija}) je **{datum.strftime('%d.%m.%Y')}**"
                return f"Naslednji odvoz **{tip}** je **{datum.strftime('%d.%m.%Y')}**"
            return "Žal ne najdem prihodnjih terminov odvoza."
        best_candidate = None
        best_location_match = False
        query_locations = self._extract_query_locations(normalize_text(original_query))
        index_candidates = [c for c in candidates if c.get('source') == 'index']
        if index_candidates:
            best_candidate = index_candidates[0]
            best_location_match = True
        else:
            for kandidat in candidates[:5]:
                doc = kandidat['doc'].lower()
                for loc in query_locations:
                    if loc in doc:
                        best_candidate = kandidat
                        best_location_match = True
                        break
                if best_location_match:
                    break
            if not best_candidate:
                best_candidate = candidates[0]
        if best_candidate:
            doc = best_candidate['doc']
            tip = best_candidate['tip'] or "Neznano"
            datumi = re.findall(r'\d{1,2}\.\d{1,2}\.', doc)
            if datumi:
                lokacija = self._find_location_in_doc(doc, original_query)
                if lokacija:
                    return f"**{lokacija}** - odvoz **{tip}**:\nTermini: {', '.join(datumi[:10])}"
                if best_location_match and query_locations:
                    return f"**{query_locations[0].title()}** - odvoz **{tip}**:\nTermini: {', '.join(datumi[:10])}"
                return f"Odvoz **{tip}**:\nTermini: {', '.join(datumi[:10])}"
            return doc[:400] + "..."
        return "Žal nisem našel konkretnih podatkov o odvozu."

    def _find_location_in_doc(self, doc: str, query: str) -> Optional[str]:
        query_words = set(normalize_text(query).split())
        patterns = [
            r'([A-ZČŠŽa-zčšž]+\s+(?:ulica|cesta|pot|trg))',
            r'Pod\s+[A-ZČŠŽa-zčšž]+',
            r'[A-ZČŠŽa-zčšž]+\s+\d+',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, doc, re.IGNORECASE)
            for match in matches:
                match_norm = normalize_text(match)
                if any(word in match_norm for word in query_words if len(word) > 3):
                    return match.title()
        return None

    # -------------------- ZAPORE CEST --------------------
    def _ensure_nap_token(self):
        if self._nap_access_token and self._nap_token_expiry and datetime.now() < self._nap_token_expiry - timedelta(seconds=60):
            return self._nap_access_token
        print("-> Pridobivam/osvežujem NAP API žeton...")
        payload = {'grant_type': 'password', 'username': NAP_USERNAME, 'password': NAP_PASSWORD}
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        response = requests.post(NAP_TOKEN_URL, data=payload, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        self._nap_access_token = data['access_token']
        self._nap_token_expiry = datetime.now() + timedelta(seconds=data['expires_in'])
        return self._nap_access_token

    def preveri_zapore_cest(self):
        if not NAP_USERNAME or not NAP_PASSWORD:
            return "Dostop do prometnih informacij ni mogoč."
        print("-> Kličem specialista za promet (NAP API)...")
        try:
            token = self._ensure_nap_token()
            headers = {'Authorization': f'Bearer {token}'}
            data_response = requests.get(NAP_DATA_URL, headers=headers, timeout=15)
            data_response.raise_for_status()
            vsi_dogodki = data_response.json().get('features', [])

            MUNICIPAL_FILTER = {"rače", "fram", "slivnica", "brunšvik", "podova", "morje", "hoče"}
            normalized_filter = {normalize_text(m) for m in MUNICIPAL_FILTER}
            relevantne_zapore_raw = []
            for dogodek in vsi_dogodki:
                lastnosti = dogodek.get('properties', {})
                cesta = str(lastnosti.get('cesta', '')).strip()
                opis = str(lastnosti.get('opis', '')).strip()
                ime = str(lastnosti.get('imeDogodka', '')).strip()
                celotno_besedilo_norm = normalize_text(" ".join([cesta, opis, ime]))
                if not any(k in celotno_besedilo_norm for k in normalized_filter):
                    continue
                relevantne_zapore_raw.append({
                    'cesta': cesta or "Ni podatka",
                    'opis': opis or "Ni podatka",
                    'imeDogodka': ime,
                    'full_props': lastnosti
                })
            if not relevantne_zapore_raw:
                return "Po podatkih portala promet.si na območju občine Rače-Fram trenutno ni zabeleženih del na cesti, zapor ali zastojev."

            merged: List[Dict[str, Any]] = []
            for z in relevantne_zapore_raw:
                added = False
                for m in merged:
                    ista_cesta = normalize_text(z['cesta']) == normalize_text(m['cesta'])
                    opis_sim = SequenceMatcher(None, normalize_text(z['opis']), normalize_text(m['opis'])).ratio()
                    if ista_cesta and opis_sim >= 0.9:
                        added = True
                        break
                if not added:
                    merged.append(z)

            def priority_key(z):
                text = normalize_text(" ".join([z['cesta'], z['opis'], z['imeDogodka']]))
                return 0 if ("race" in text or "fram" in text) else 1

            merged.sort(key=priority_key)

            porocilo = "Našel sem naslednje **trenutne** informacije o dogodkih na cesti (vir: promet.si):\n\n"
            for z in merged:
                porocilo += f"- **Cesta:** {z['cesta']}\n  **Opis:** {z['opis']}\n\n"
            porocilo = porocilo.strip() + "\n\nZa več informacij obiščite: https://www.race-fram.si/objave/274"
            return porocilo
        except Exception:
            return "Žal mi neposreden vpogled v stanje na cestah trenutno ne deluje. Poskusite kasneje."

    # -------------------- BELEŽENJE --------------------
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

    # -------------------- PREOBLIKOVANJE VPRAŠANJ --------------------
    def preoblikuj_vprasanje_s_kontekstom(self, zgodovina_pogovora, zadnje_vprasanje):
        if not zgodovina_pogovora:
            return zadnje_vprasanje
        print("-> Kličem specialista za spomin...")
        zgodovina_str = "\n".join([f"Uporabnik: {q}\nAsistent: {a}" for q, a in zgodovina_pogovora])
        prompt = f"""Tvoja naloga je, da glede na zgodovino pogovora preoblikuješ novo vprašanje v samostojno vprašanje. Bodi kratek in jedrnat.

Zgodovina:
{zgodovina_str}

Novo vprašanje: \"{zadnje_vprasanje}\"

Samostojno vprašanje:"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100
            )
            preoblikovano = response.choices[0].message.content.strip().replace('"', '')
            print(f"Originalno: '{zadnje_vprasanje}' -> Preoblikovano: '{preoblikovano}'")
            return preoblikovano
        except Exception:
            return zadnje_vprasanje

    # -------------------- KONTAKTNE INFORMACIJE --------------------
    def _derive_role_from_text(self, text: str, person: Optional[str]) -> Optional[str]:
        if not text:
            return None
        sentence = text.split('.', 1)[0]
        if person and person in sentence:
            sentence = sentence.split(person, 1)[-1]
        match = re.search(r'je\s+([^\.,]+)', sentence)
        if match:
            role = match.group(1).strip()
            role = re.sub(r'^na\s+ob[cč]ini\s+ra[cč]e-fram\s+', '', role, flags=re.IGNORECASE)
            return role
        return None

    def _extract_contact_entries(self, docs: List[str], metas: List[Dict[str, Any]], original_question: str) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        query_words = {w for w in normalize_text(original_question).split() if len(w) > 3}
        for doc, meta in zip(docs, metas):
            if not doc:
                continue
            doc_norm = normalize_text(doc)
            tema_norm = normalize_text(meta.get('tema', ''))
            if query_words and not any(word in doc_norm or word in tema_norm for word in query_words):
                continue
            name = meta.get('oseba') or meta.get('ime') or meta.get('naziv')
            if not name:
                name_match = re.search(r'([A-ZŠŽČ][a-zšžčćđ]+(?:\s+[A-ZŠŽČ][a-zšžčćđ]+)+)', doc)
                if name_match:
                    name = name_match.group(1)
            role = meta.get('funkcija') or meta.get('delovno_mesto') or meta.get('opis_funkcije') or meta.get('tema')
            if not role:
                role = self._derive_role_from_text(doc, name)
            phone = meta.get('telefon') or meta.get('phone') or extract_phone(doc)
            email = meta.get('email') or meta.get('e_posta') or extract_email(doc)
            if not any([name, role, phone, email]):
                continue
            entries.append({
                'name': name,
                'role': role,
                'phone': phone,
                'email': email,
                'source': meta.get('source'),
                'url': meta.get('source_url')
            })
        unique_entries: List[Dict[str, Any]] = []
        seen = set()
        for entry in entries:
            key = (entry.get('name'), entry.get('role'))
            if key in seen:
                continue
            seen.add(key)
            unique_entries.append(entry)
        return unique_entries

    def _format_contact_entries(self, entries: List[Dict[str, Any]], original_question: str) -> str:
        if not entries:
            return "Žal ne najdem konkretnega kontakta."
        lead = "Našel sem naslednje kontaktne informacije."
        question_lower = original_question.lower()
        if 'šport' in question_lower:
            lead = "Za področje športa so na voljo naslednji kontakti:"
        elif 'direktor' in question_lower and len(entries) == 1:
            lead = "Direktor občinske uprave:"
        lines = [lead, ""]
        for entry in entries[:3]:
            name = entry.get('name') or 'Kontakt'
            role = entry.get('role')
            phone = entry.get('phone')
            email = entry.get('email')
            url = entry.get('url')
            bullet = f"- **{name}**"
            if role:
                bullet += f" – {role}"
            detail_lines = []
            if phone:
                detail_lines.append(f"Telefon: {phone}")
            if email:
                detail_lines.append(f"E-pošta: {email}")
            if url:
                detail_lines.append(f"Vir: [{url}]({url})")
            if detail_lines:
                bullet += "\n  " + "\n  ".join(detail_lines)
            lines.append(bullet)
        return "\n".join(lines).strip()

    # -------------------- GLAVNI ODGOVOR --------------------
    def odgovori(self, uporabnikovo_vprasanje: str, session_id: str):
        self.nalozi_bazo()
        if session_id not in self.zgodovina_seje:
            self.zgodovina_seje[session_id] = {'zgodovina': [], 'stanje': {}}
        stanje = self.zgodovina_seje[session_id]['stanje']
        zgodovina = self.zgodovina_seje[session_id]['zgodovina']

        norm_original = normalize_text(uporabnikovo_vprasanje)
        if re.fullmatch(r'kdo\s+pa?\s+je\s+to\??', norm_original):
            last_person = stanje.get('zadnja_oseba')
            if last_person:
                uporabnikovo_vprasanje = f"Kdo je {last_person}?"

        pametno_vprasanje = self.preoblikuj_vprasanje_s_kontekstom(zgodovina, uporabnikovo_vprasanje)
        vprasanje_lower = pametno_vprasanje.lower()

        if any(word in vprasanje_lower for word in ['kosilo', 'malica', 'zajtrk', 'jedilnik']):
            odgovor = self.obravnavaj_jedilnik(pametno_vprasanje, session_id)
        elif any(re.search(r'\b' + re.escape(k) + r'\b', vprasanje_lower) for k in ['smeti', 'odpadki', 'odvoz', 'odpadk', 'komunala']) or stanje.get('namen') == 'odpadki':
            odgovor = self.obravnavaj_odvoz_odpadkov_systematic(pametno_vprasanje, session_id)
        elif any(re.search(r'\b' + re.escape(k) + r'\b', vprasanje_lower) for k in ['cesta', 'promet', 'zapora', 'zastoj']):
            odgovor = self.preveri_zapore_cest()
        else:
            rezultati_iskanja = self.collection.query(
                query_texts=[vprasanje_lower],
                n_results=5,
                include=["documents", "metadatas"]
            ) if self.collection else {'documents': [[]], 'metadatas': [[]]}

            role_match = re.search(r'kdo\s+je\s+na\s+ob[cč]ini\s+zadol[zž]en\s+za\s+([^?]+)', vprasanje_lower)
            if role_match and rezultati_iskanja.get('documents') and rezultati_iskanja['documents'][0]:
                domain_norm = normalize_text(role_match.group(1))
                filtered_docs = []
                filtered_metas = []
                for doc, meta in zip(rezultati_iskanja['documents'][0], rezultati_iskanja['metadatas'][0]):
                    if domain_norm in normalize_text(meta.get('tema', '')) or domain_norm in normalize_text(doc):
                        filtered_docs.append(doc)
                        filtered_metas.append(meta)
                if filtered_docs:
                    rezultati_iskanja['documents'][0] = filtered_docs
                    rezultati_iskanja['metadatas'][0] = filtered_metas

            kontaktni_odgovor = None
            if rezultati_iskanja.get('documents') and rezultati_iskanja['documents'][0]:
                contact_entries = self._extract_contact_entries(
                    rezultati_iskanja['documents'][0],
                    rezultati_iskanja['metadatas'][0],
                    pametno_vprasanje
                ) if re.search(r'\b(kontakt|telefon|številka|stevilka|zadolžen|odgovoren|direktor)\b', vprasanje_lower) else []
                if contact_entries:
                    kontaktni_odgovor = self._format_contact_entries(contact_entries, pametno_vprasanje)
                    for entry in contact_entries:
                        if entry.get('name'):
                            stanje['zadnja_oseba'] = entry['name']
                            break

            if kontaktni_odgovor:
                odgovor = kontaktni_odgovor
            else:
                kontekst_baza = ""
                if rezultati_iskanja.get('documents'):
                    for doc, meta in zip(rezultati_iskanja['documents'][0], rezultati_iskanja['metadatas'][0]):
                        if meta.get('oseba'):
                            stanje['zadnja_oseba'] = meta['oseba']
                        kontekst_baza += (
                            f"--- VIR: {meta.get('source', 'Neznan')}\n"
                            f"POVEZAVA: {meta.get('source_url', 'Brez')}\n"
                            f"VSEBINA: {doc}\n\n"
                        )

                if not kontekst_baza:
                    odgovor = "Žal o tem nimam nobenih informacij."
                else:
                    now = datetime.now()
                    current_day = get_slovenian_day_name(now)
                    tomorrow, tomorrow_day = get_tomorrow_date()
                    contact_query = bool(re.search(r'\b(kontakt|telefon|številka|stevilka|zadolžen|odgovoren|direktor)\b', vprasanje_lower))

                    if 'investicij' in vprasanje_lower:
                        odgovor = """Za investicije je zadolžena:
**Mateja Frešer**
Telefon: 02 609 60 10
E-pošta: obcina@race-fram.si"""
                    else:
                        prompt_za_llm = (
                            f"Ti si 'Virtualni župan občine Rače-Fram'.\n"
                            f"DIREKTIVA #1: Danes je {current_day}, {now.strftime('%d.%m.%Y')}. Jutri je {tomorrow_day}, {tomorrow.strftime('%d.%m.%Y')}.\n"
                            f"DIREKTIVA #2: Če je podatek iz leta, ki je manjše od {now.year}, ga IGNORIRAJ.\n"
                            "DIREKTIVA #3: Odgovor mora biti pregleden. Ključne informacije **poudari**.\n"
                            "DIREKTIVA #4: Če najdeš URL, ga vključi v klikljivi obliki.\n"
                            "DIREKTIVA #5: Če ne najdeš specifičnega podatka, NE ponavljaj splošnih informacij.\n"
                        )
                        if contact_query:
                            prompt_za_llm += "DIREKTIVA #6: Za kontakte podaj IME, FUNKCIJO, TELEFON in EMAIL. Bodi jedrnat.\n"
                        if role_match:
                            prompt_za_llm += "DIREKTIVA #7: Pri odgovoru vedno navedi IME in PRIIMEK odgovorne osebe.\n"
                        if 'dan' in vprasanje_lower and 'danes' in vprasanje_lower:
                            prompt_za_llm += f"\nODGOVORI: Danes je **{current_day}**, {now.strftime('%d.%m.%Y')}.\n"
                        prompt_za_llm += (
                            f"\n--- KONTEKST ---\n{kontekst_baza}---\n"
                            f"VPRAŠANJE: \"{uporabnikovo_vprasanje}\"\n"
                            "ODGOVOR:"
                        )

                        try:
                            response = self.openai_client.chat.completions.create(
                                model=GENERATOR_MODEL_NAME,
                                messages=[{"role": "user", "content": prompt_za_llm}],
                                temperature=0.0,
                                max_tokens=500
                            )
                            odgovor = response.choices[0].message.content
                        except Exception as e:
                            print(f"LLM napaka: {e}")
                            odgovor = "Prišlo je do napake pri obdelavi vprašanja. Poskusite kasneje."

                if contact_query and not kontaktni_odgovor:
                    odgovor = re.sub(r'- Storitev:.*\n', '', odgovor)
                    odgovor = re.sub(r'- Lokacija:.*\n', '', odgovor)
                    odgovor = re.sub(r'mag\.\s*sci\.\s*', '', odgovor, flags=re.IGNORECASE)
                    odgovor = re.sub(r'dr\.\s*med\.\s*', 'Dr. ', odgovor, flags=re.IGNORECASE)
                    if len(odgovor) > 800:
                        lines = odgovor.split('\n')
                        new_lines, char_count = [], 0
                        for line in lines:
                            if char_count + len(line) < 750:
                                new_lines.append(line)
                                char_count += len(line)
                            else:
                                new_lines.append("\nZa dodatne informacije pokličite 02 609 60 10.")
                                break
                        odgovor = '\n'.join(new_lines)

        zgodovina.append((uporabnikovo_vprasanje, odgovor))
        if len(zgodovina) > 4:
            zgodovina.pop(0)
        self.belezi_pogovor(session_id, uporabnikovo_vprasanje, odgovor)
        return odgovor

# --- MAIN FUNKCIJA ---
def main():
    print("\n" + "="*60)
    print("🏛️  VIRTUALNI ŽUPAN RAČE-FRAM v37")
    print("    Pametno filtriranje jedilnikov po datumu")
    print("="*60)

    zupan = VirtualniZupan()
    DEBUG = False

    now = datetime.now()
    day_name = get_slovenian_day_name(now)
    print(f"\n📅 Danes je {day_name}, {now.strftime('%d.%m.%Y')}")
    print("\n💬 Pripravljen za vprašanja!")
    print("📝 Ukazi: 'izhod' za končanje | 'test' za testiranje | 'debug on/off'\n")
    session_id = f"cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    while True:
        try:
            vprasanje = input("\n🤔 Vaše vprašanje: ").strip()
            if not vprasanje:
                continue
            if vprasanje.lower() in ['izhod', 'quit', 'exit', 'konec', 'q']:
                print("\n👋 Nasvidenje!")
                break
            if vprasanje.lower() == 'debug on':
                DEBUG = True
                print("🔍 Debug način VKLOPLJEN")
                continue
            if vprasanje.lower() == 'debug off':
                DEBUG = False
                print("🔍 Debug način IZKLOPLJEN")
                continue
            if vprasanje.lower() == 'test':
                print("\n🧪 TESTIRANJE SISTEMA...")
                test_questions = [
                    "kaj je za kosilo v sredo v oš fram",
                    "kaj je za kosilo v sredo 10.9 v oš fram",
                    "kdaj je odvoz stekla pod terasami",
                    "kdaj je odvoz papirja na bistriški",
                    "ali imamo v občini zobozdravnika",
                    "kateri dan je danes",
                    "kdo je direktor občinske uprave"
                ]
                for i, test_q in enumerate(test_questions, 1):
                    print(f"\n{i}. TEST: {test_q}")
                    print("=" * 60)
                    odgovor = zupan.odgovori(test_q, f"test_{i}")
                    print(odgovor[:500] + ("..." if len(odgovor) > 500 else ""))
                continue
            if vprasanje.lower() == 'test jedilnik':
                print("\n🍽️ TEST: Jedilniki z datumi")
                test_queries = [
                    "kaj je za kosilo danes v oš fram",
                    "kaj je za kosilo jutri v oš fram",
                    "kaj je za kosilo v sredo v oš fram",
                    "kaj je za kosilo v sredo 10.9 v oš fram"
                ]
                for q in test_queries:
                    print(f"\n❓ {q}")
                    print("-" * 40)
                    odgovor = zupan.odgovori(q, "test_meal")
                    print(odgovor)
                continue

            print("\n" + "="*70)
            print("🤖 ODGOVOR:")
            print("="*70)

            if DEBUG:
                print(f"🔍 DEBUG: vprasanje_lower = '{vprasanje.lower()}'")

            odgovor = zupan.odgovori(vprasanje, session_id)
            print(odgovor)
            print("="*70)

        except KeyboardInterrupt:
            print("\n\n⚠️ Prekinitev... Nasvidenje!")
            break
        except Exception as e:
            print(f"\n❌ Napaka: {e}")
            if DEBUG:
                import traceback
                traceback.print_exc()
            continue

if __name__ == "__main__":
    main()
