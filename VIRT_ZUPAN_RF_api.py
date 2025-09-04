import os
import sys
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

# ---- KONFIGURACIJA ----
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '..', '.env'))

# Pametno določanje poti glede na okolje
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

# Filtri in ključne besede
PROMET_FILTER_KLJUCNIKI = [
    "rače", "fram", "slivnica", "brunšvik", "podova", "morje", "hoče",
    "r2-430", "r3-711", "g1-2",
    "priključek slivnica", "razcep slivnica", "letališče maribor", "odcep za rače"
]
KLJUCNE_BESEDE_ODPADKI = ["smeti", "odpadki", "odvoz", "odpavkov", "komunala"]
KLJUCNE_BESEDE_PROMET = ["cesta", "ceste", "cesti", "promet", "dela", "delo", "zapora", "zapore", "zaprta", "zastoj", "gneča", "kolona"]

# ---- POMOŽNE FUNKCIJE ----
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

def slovenian_variant_equivalent(a: str, b: str) -> bool:
    """IZBOLJŠANA heuristika za slovenske fleksijske variante + kratke oblike"""
    a_n = normalize_text(a)
    b_n = normalize_text(b)
    
    if a_n == b_n:
        return True
    
    # NOVO: Mapping kratkih oblik na polna imena
    street_short_forms = {
        # Kratka oblika -> možne polne oblike
        "bistriski": ["bistriska cesta", "bistriška cesta", "bistriska", "bistriški"],
        "bistriška": ["bistriska cesta", "bistriška cesta", "bistriska", "bistriški"], 
        "bistriska": ["bistriska cesta", "bistriška cesta", "bistriški"],
        "mlinski": ["mlinska ulica", "mlinska", "mlinsko", "mlinske"],
        "mlinska": ["mlinska ulica", "mlinski", "mlinsko", "mlinske"],
        "turnerjeva": ["turnerjeva ulica", "turnerjevi", "turnerjev"],
        "turnerjevi": ["turnerjeva ulica", "turnerjeva", "turnerjev"],
        "framski": ["framska cesta", "framska", "framsko", "framske"],
        "framska": ["framska cesta", "framski", "framsko", "framske"],
        "grajski": ["grajski trg", "grajska", "grajsko", "grajske"],
        "terasami": ["pod terasami", "terase", "terasa", "terasah"],
        "terase": ["pod terasami", "terasami", "terasa", "terasah"]
    }
    
    # Preverimo kratke oblike - a je kratka oblika od b?
    if a_n in street_short_forms:
        for full_form in street_short_forms[a_n]:
            if full_form in b_n or normalize_text(full_form) == b_n:
                return True
    
    # Preverimo kratke oblike - b je kratka oblika od a?
    if b_n in street_short_forms:
        for full_form in street_short_forms[b_n]:
            if full_form in a_n or normalize_text(full_form) == a_n:
                return True
    
    # NOVO: Povratno ujemanje - če "bistriski" v polnem imenu "bistriska cesta"
    a_words = set(a_n.split())
    b_words = set(b_n.split())
    
    # Če a je del b (npr. "mlinski" je v "mlinska ulica")
    if len(a_words) < len(b_words):
        for a_word in a_words:
            if any(slovenian_word_match(a_word, b_word) for b_word in b_words):
                # Dodatno preveri, če je to smiselno ujemanje
                if any(generic in b_n for generic in ["cesta", "ulica", "pot", "trg"]):
                    return True
    
    # Povratno - če b je del a
    elif len(b_words) < len(a_words):
        for b_word in b_words:
            if any(slovenian_word_match(b_word, a_word) for a_word in a_words):
                if any(generic in a_n for generic in ["cesta", "ulica", "pot", "trg"]):
                    return True
    
    # Preddefinirane variante (ohranjen original)
    known_variants = {
        frozenset(["bistriska", "bistriski", "bistriška", "bistriške", "bistriske"]),
        frozenset(["mlinska", "mlinski", "mlinsko", "mlinske"]),
        frozenset(["framska", "framski", "framsko", "framske"]),
        frozenset(["grajski", "grajska", "grajsko", "grajske"]),
        frozenset(["terasami", "terase", "terasa", "terasah"]),
        frozenset(["turnerjeva", "turnerjevi", "turnerjev"])
    }
    
    for variant_set in known_variants:
        if a_n in variant_set and b_n in variant_set:
            return True
    
    # Splošna fleksijska heuristika (ohranjena)
    if len(a_n) > 3 and len(b_n) > 3:
        min_len = min(len(a_n), len(b_n))
        stem_len = int(min_len * 0.75)
        if a_n[:stem_len] == b_n[:stem_len]:
            endings = {"a", "i", "e", "o", "u", "ih", "imi", "ega", "emu"}
            a_end = a_n[stem_len:]
            b_end = b_n[stem_len:]
            if a_end in endings and b_end in endings:
                return True
    
    return False

def slovenian_word_match(word1: str, word2: str) -> bool:
    """Pomožna funkcija za ujemanje posameznih slovenskih besed"""
    if word1 == word2:
        return True
    
    # Slovenski končniki
    if len(word1) > 3 and len(word2) > 3:
        stem1 = word1[:-2] if len(word1) > 4 else word1[:-1]
        stem2 = word2[:-2] if len(word2) > 4 else word2[:-1]
        if stem1 == stem2:
            return True
    
    return SequenceMatcher(None, word1, word2).ratio() >= 0.85

def street_phrase_matches(query_phrase: str, street_tok: str, threshold: float = 0.85) -> bool:
    """Napredna funkcija za ujemanje ulic (iz skripte 2)"""
    generic = {"cesta", "cesti", "ulica", "ulici", "pot", "trg", "ob"}
    qp = normalize_text(query_phrase)
    st = normalize_text(street_tok)

    if slovenian_variant_equivalent(qp, st):
        return True

    q_words = [w for w in qp.split() if w not in generic]
    street_words = [w for w in st.split() if w not in generic]
    
    if not q_words:
        return fuzzy_match(qp, st, threshold)
    
    for qw in q_words:
        matched = False
        for sw in street_words:
            if slovenian_variant_equivalent(qw, sw):
                matched = True
                break
            if SequenceMatcher(None, qw, sw).ratio() >= threshold or qw in sw or sw in qw:
                matched = True
                break
        if not matched:
            return False
    return True

# Waste type handling (iz skripte 2)
WASTE_TYPE_VARIANTS = {
    "Biološki odpadki": [
        "bioloski odpadki", "bioloskih odpakov", "bioloski", "bioloskih", "bio", "biološki odpadki",
        "bioloskih odpadkov", "bioloski odpadkov", "bioloških odpadkov", "bioloških odpadki"
    ],
    "Mešani komunalni odpadki": [
        "mesani komunalni odpadki", "mešani komunalni odpadki", "mesani", "mešani",
        "mešane odpadke", "mesane odpadke", "mešani odpadki", "mešane komunalne", "mesane komunalne",
        "mešane komunalne odpadke", "mesane komunalne odpadke",
        "komunalni odpadki", "komunalnih odpadkov", "komunalne odpadke"
    ],
    "Odpadna embalaža": [
        "odpadna embalaza", "odpadna embalaža", "embalaza", "embalaža", "embalaže"
    ],
    "Papir in karton": [
        "papir in karton", "papir", "karton", "papirja", "kartona", "papir in kartona"
    ],
    "Steklena embalaža": [
        "steklena embalaza", "steklena embalaža", "steklo", "stekla", "stekle", "stekleno", "stekleni", "steklen"
    ],
}

def get_canonical_waste(text: str):
    """Izboljšana funkcija za prepoznavanje tipov odpadkov (iz skripte 2)"""
    norm = normalize_text(text)
    
    # Heuristike
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

    # Direktno ujemanje
    for canonical, variants in WASTE_TYPE_VARIANTS.items():
        if normalize_text(canonical) in norm:
            return canonical
        for v in variants:
            if normalize_text(v) in norm:
                return canonical
    
    # Fuzzy fallback
    for canonical, variants in WASTE_TYPE_VARIANTS.items():
        for v in variants:
            if SequenceMatcher(None, norm, normalize_text(v)).ratio() >= 0.85:
                return canonical
    return None

def extract_locations_from_naselja(naselja_field: str):
    """Ekstraktiranje lokacij iz naselja field (iz skripte 2)"""
    parts = []
    if ':' in naselja_field:
        prefix, rest = naselja_field.split(':', 1)
        parts.append(prefix.strip())
        for chunk in rest.split(','):
            chunk = chunk.strip()
            if ':' in chunk:
                subprefix, subrest = chunk.split(':', 1)
                parts.append(subprefix.strip())
                parts.append(subrest.strip())
            else:
                parts.append(chunk)
    else:
        for chunk in naselja_field.split(','):
            parts.append(chunk.strip())
    return list({normalize_text(p) for p in parts if p})

def obravnavaj_jedilnik(vprasanje: str, collection):
    """Funkcija za jedilnike"""
    vprasanje_lower = vprasanje.lower()
    
    school = "OŠ Rače"
    if "fram" in vprasanje_lower:
        school = "OŠ Fram"
    
    today = datetime.now()
    target_date = None
    
    # Parsing datuma iz vprašanja
    date_match = re.search(r'(\d{1,2})\.(\d{1,2})', vprasanje_lower)
    if date_match:
        dan = int(date_match.group(1))
        mesec = int(date_match.group(2))
        target_date = datetime(today.year, mesec, dan)
    elif "jutri" in vprasanje_lower:
        target_date = today + timedelta(days=1)
    elif "pojutrišnjem" in vprasanje_lower:
        target_date = today + timedelta(days=2)
    elif "danes" in vprasanje_lower:
        target_date = today
    elif "ponedeljek" in vprasanje_lower:
        days_ahead = 0 - today.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        target_date = today + timedelta(days=days_ahead)
    elif "torek" in vprasanje_lower:
        days_ahead = 1 - today.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        target_date = today + timedelta(days=days_ahead)
    else:
        target_date = today

    print(f"🗓️ DEBUG: Iščem {school} za datum {target_date.strftime('%d.%m.%Y')}")

    try:
        search_queries = [
            f"malica {target_date.strftime('%d.%m')} {school}",
            f"jedilnik {target_date.strftime('%d.%m')} {school}",
            f"malica {target_date.strftime('%-d.%-m')} {school}",
        ]
        
        for query in search_queries:
            results = collection.query(
                query_texts=[query],
                n_results=20,
                include=["documents", "metadatas"]
            )
            
            if results['documents'] and results['documents'][0]:
                date_patterns = [
                    target_date.strftime('%d.%m.%Y'),
                    target_date.strftime('%d.%m.'),
                    target_date.strftime('%-d.%-m.%Y'),
                    target_date.strftime('%-d.%-m.'),
                ]
                
                for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                    doc_contains_school = school.lower() in doc.lower()
                    doc_contains_date = any(pattern in doc for pattern in date_patterns)
                    
                    if doc_contains_school and doc_contains_date:
                        print(f"✅ NAJDEN točen match: {school} + datum")
                        return f"**{school} za {target_date.strftime('%d.%m.%Y')}:**\n\n{doc}"
                        
    except Exception as e:
        print(f"Napaka pri iskanju jedilnika: {e}")
    
    return f"Žal nimam podatkov o malici za **{school}** na datum **{target_date.strftime('%d.%m.%Y')}**."

class VirtualniZupan:
    def __init__(self):
        print("Inicializacija razreda VirtualniZupan (Verzija 37.0 - združena končna verzija)...")
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.collection = None
        self.zgodovina_seje = {}
        self._nap_access_token = None
        self._nap_token_expiry = None
        self.jsonl_cache = {}

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
                print(f"✅ ChromaDB povezan: {self.collection.count()} dokumentov")
            except Exception as e:
                print(f"❌ KRITIČNA NAPAKA: Baze znanja ni mogoče naložiti. Razlog: {e}")
                self.collection = None

    def load_jsonl_data(self, filename):
        """Naloži JSONL podatke s cache sistemom"""
        cache_key = filename
        
        if cache_key in self.jsonl_cache:
            return self.jsonl_cache[cache_key]
        
        filepath = os.path.join(IZVORNI_PODATKI_PATH, filename)
        data = []
        
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f):
                        line = line.strip()
                        if line:
                            try:
                                data.append(json.loads(line))
                            except json.JSONDecodeError as e:
                                print(f"JSON napaka v {filename} vrstica {line_num + 1}: {e}")
                                continue
                        
                self.jsonl_cache[cache_key] = data
                print(f"📂 Naloženo {len(data)} zapisov iz {filename}")
                
            except Exception as e:
                print(f"Napaka pri branju {filename}: {e}")
        else:
            print(f"Datoteka {filename} ne obstaja na poti {filepath}")
        
        return data

    def get_health_data_direct(self, query_lower=""):
        """ZDRAVSTVENI PODATKI - ohranjena funkcionalnost iz obeh skript"""
        health_data = self.load_jsonl_data("zdravstvo.jsonl")
        
        if not health_data:
            return "Žal nimam dostopa do zdravstvenih podatkov."
        
        if any(word in query_lower for word in ["osebni", "splošna", "družinski", "ambulanta splošne"]):
            doctors = [item for item in health_data if 
                      "splošna medicina" in item.get("text", "").lower() and
                      not any(zobni in item.get("text", "").lower() for zobni in ["zobni", "zobna", "dentalna"])]
            title = "Osebni zdravniki (splošna medicina) v občini Rače-Fram:\n\n"
            
        elif any(word in query_lower for word in ["zobni", "zobozdravnik", "dentalna", "zobna"]):
            doctors = [item for item in health_data if 
                      any(word in item.get("text", "").lower() for word in ["zobni", "zobna", "dentalna", "madens", "zobozdravstvo"])]
            title = "Zobozdravniki v občini Rače-Fram:\n\n"
            
        else:
            doctors = [item for item in health_data if 
                      any(word in item.get("text", "").lower() for word in ["dr.", "doktor", "specialistka"]) and
                      not any(word in item.get("text", "").lower() for word in ["patronažna", "fizioterapija", "lekarna"])]
            title = "Zdravstvene storitve v občini Rače-Fram:\n\n"
        
        if not doctors:
            return "Žal nisem našel ustreznih zdravstvenih podatkov."
        
        # Združevanje podatkov (rešeno podvajanje)
        doctor_profiles = {}
        
        for item in doctors:
            text = item.get("text", "")
            metadata = item.get("metadata", {})
            
            doctor_name = None
            
            patterns = [
                r'Dr\.\s+([^,\.]+(?:\s+[^,\.]+)*)',
                r'mag\.\s*sci\.\s+([^,\.]+(?:\s+[^,\.]+)*)',
                r'doktor[ai]ca?\s+([^,\.]+(?:\s+[^,\.]+)*)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    doctor_name = match.group(1).strip()
                    doctor_name = re.sub(r'\s*\([^)]*\).*', '', doctor_name)
                    doctor_name = re.sub(r'\s*,.*', '', doctor_name)
                    break
            
            if not doctor_name:
                doctor_name = metadata.get("zdravnik") or metadata.get("oseba")
            
            if doctor_name:
                existing_profile = None
                for existing_name in doctor_profiles.keys():
                    if (normalize_text(doctor_name) in normalize_text(existing_name) or 
                        normalize_text(existing_name) in normalize_text(doctor_name) or
                        slovenian_variant_equivalent(doctor_name, existing_name) or
                        SequenceMatcher(None, normalize_text(doctor_name), normalize_text(existing_name)).ratio() > 0.85):
                        existing_profile = existing_name
                        break
                
                if existing_profile:
                    if metadata.get('tip') == 'urnik' and not doctor_profiles[existing_profile].get('urnik'):
                        doctor_profiles[existing_profile]['urnik'] = text
                    continue
                
                doctor_profiles[doctor_name] = {
                    'osnovni_podatki': text if metadata.get('tip') != 'urnik' else None,
                    'urnik': text if metadata.get('tip') == 'urnik' else None,
                    'telefon': None,
                    'email': None,
                    'naslov': None,
                    'storitev': None,
                    'lokacija': metadata.get('lokacija', ''),
                    'tip': metadata.get('tip', '')
                }
                
                profile = doctor_profiles[doctor_name]
                
                if metadata.get('tip') != 'urnik':
                    telefon_match = re.search(r'Telefon:\s*([0-9\/\-\s]+)', text)
                    if telefon_match:
                        profile['telefon'] = telefon_match.group(1).strip()
                    
                    email_match = re.search(r'E-pošta:\s*([a-zA-Z0-9\._-]+@[a-zA-Z0-9\.-]+\.[a-zA-Z]{2,})', text)
                    if email_match:
                        profile['email'] = email_match.group(1).strip()
                    elif re.search(r'E-pošta:\s*([^\s,\.]+)', text):
                        partial = re.search(r'E-pošta:\s*([^\s,\.]+)', text).group(1).strip()
                        if 'sebastijan' in partial and '@' not in partial:
                            profile['email'] = 'sebastijan.sketa@zd-mb.si'
                        elif 'boris' in partial and '@' not in partial:
                            profile['email'] = 'boris.sapac@zd-mb.si'
                        elif partial == 'info@madens':
                            profile['email'] = 'info@madens.eu'
                        else:
                            profile['email'] = partial
                    
                    naslov_match = re.search(r'Naslov:\s*([^\.]+?)(?:\s*\.\s*Telefon|\s*\.\s*E-pošta|\.\s*$|$)', text)
                    if naslov_match:
                        naslov_raw = naslov_match.group(1).strip().rstrip(',.')
                        if naslov_raw == 'Nova ul':
                            profile['naslov'] = 'Nova ul. 5, Rače'
                        elif naslov_raw == 'Nova ulica 5':
                            profile['naslov'] = 'Nova ulica 5, Rače'
                        elif naslov_raw == 'Cafova ul':
                            profile['naslov'] = 'Cafova ul. 1, Fram'
                        else:
                            profile['naslov'] = naslov_raw
                    
                    if "splošna medicina" in text.lower():
                        profile['storitev'] = "Splošna medicina"
                    elif "zasebna" in text.lower():
                        profile['storitev'] = "Zasebna ambulanta"  
                    elif any(word in text.lower() for word in ["zobni", "zobna", "dentalna"]):
                        profile['storitev'] = "Zobozdravstvo"
        
        response = title
        
        for doctor_name, profile in doctor_profiles.items():
            response += f"**Dr. {doctor_name}**\n"
            
            if profile['telefon']:
                response += f"- Telefon: {profile['telefon']}\n"
            if profile['email']:
                response += f"- E-pošta: {profile['email']}\n"
            if profile['naslov']:
                response += f"- Naslov: {profile['naslov']}\n"
            if profile['storitev']:
                response += f"- Storitev: {profile['storitev']}\n"
            if profile['lokacija']:
                response += f"- Lokacija: {profile['lokacija']}\n"
            
            if profile['urnik']:
                urnik_match = re.search(r'Ordinacijski čas[:\s]*(.+)', profile['urnik'], re.IGNORECASE | re.DOTALL)
                if urnik_match:
                    urnik_text = urnik_match.group(1).strip()
                    response += f"- Ordinacijski čas: {urnik_text}\n"
            
            response += "\n"
        
        response += "Za aktualne informacije o razpoložljivosti pokličite direktno na navedene številke."
        return response

    def get_contacts_data_direct(self, query_lower=""):
        """KONTAKTI - izboljšana funkcionalnost iz skripte 1"""
        print(f"🔍 Iščem kontakte za: '{query_lower}'")
        
        # PRIORITETA 1: Direktni odgovori za ključna vprašanja
        if any(word in query_lower for word in ["direktor", "direktorica", "direktor občinske uprave", "vodja uprave"]):
            print("🎯 Zaznano: direktor občinske uprave")
            return """**Direktorica občinske uprave:**

**mag. Karmen Kotnik**  
📧 E-pošta: karmen.kotnik@race-fram.si  
📞 Telefon: 02 609 60 10

_direktorica občinske uprave občine Rače-Fram_

Za direkten kontakt pokličite glavno številko občine."""

        if any(word in query_lower for word in ["župan", "zupan", "mayor"]):
            print("🎯 Zaznano: župan")
            return """**Župan občine Rače-Fram:**

**Samo Rajšp**  
📞 Telefon: 02 609 60 10  
📧 E-pošta: obcina@race-fram.si

_župan občine Rače-Fram od leta 2018_"""
        
        # PRIORITETA 2: Mapiranje področij dela
        field_keywords = {
            "kmetijstvo": ["kmetijstvo", "kmetijski", "kmet", "subvencije", "razpis", "poljedelstvo", "agronomija", "kmetijski razpis"],
            "sport": ["telovadnica", "dvorana", "šport", "sport", "rekreacija", "atletika", "športni objekti", "termine telovadnice"],
            "turizem": ["turizem", "turistični", "promocija", "prireditve", "gostinstvo"],
            "gradnja": ["gradnja", "gradbeni", "dovoljenja", "investicije", "objekti", "infrastruktura"],
            "finance": ["finance", "računovodstvo", "proračun", "davki", "plače"],
            "pravno": ["pravne", "pravni", "pogodbe", "javna naročila", "kadrovsko"],
            "sociala": ["šolstvo", "zdravstvo", "socialno", "varstvo", "dijaki", "študenti"]
        }
        
        detected_field = None
        for field, keywords in field_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected_field = field
                print(f"🎯 Zaznano specifično področje: {detected_field}")
                break
        
        # Preddefinirani kontakti
        field_contacts = {
            "kmetijstvo": {
                "name": "Tanja Kosi", 
                "email": "tanja.kosi@race-fram.si",
                "description": "diplomirana inženirka agronomije, pristojna za kmetijstvo, zaščito okolja in turizem"
            },
            "sport": {
                "name": "Klaudia Sovdat", 
                "email": "klaudia.sovdat@race-fram.si",
                "description": "referentka za področje športa, pripravlja letne programe športa in upravlja s športnimi objekti"
            },
            "turizem": {
                "name": "Tanja Kosi", 
                "email": "tanja.kosi@race-fram.si",
                "description": "pristojna za turizem in promocijo občine"
            },
            "gradnja": {
                "name": "Mateja Frešer", 
                "email": "mateja.freser@race-fram.si",
                "description": "diplomirana inženirka gradbeništva, vodi občinske investicije"
            },
            "finance": {
                "name": "Rosvita Robar", 
                "email": "rosvita.robar@race-fram.si",
                "description": "magistra ekonomskih ved, skrbi za proračun in finance"
            },
            "pravno": {
                "name": "Anja Čelan", 
                "email": "anja.celan@race-fram.si",
                "description": "univerzitetna diplomirana pravnica, javna naročila in pogodbe"
            },
            "sociala": {
                "name": "Monika Skledar", 
                "email": "monika.skledar@race-fram.si",
                "description": "izvaja postopke na področju šolstva, zdravstva in socialnega varstva"
            }
        }
        
        if detected_field and detected_field in field_contacts:
            contact = field_contacts[detected_field]
            return f"""**Kontaktna oseba za {detected_field}:**

**{contact['name']}**
📧 E-pošta: {contact['email']}
📞 Telefon: 02 609 60 10

_{contact['description']}_

Za direkten kontakt pokličite glavno številko občine in prosite za povezavo z {contact['name']}."""
        
        # PRIORITETA 3: Fallback na JSONL iskanje
        contacts_data = self.load_jsonl_data("imenik_zaposlenih_in_ure.jsonl")
        
        if not contacts_data:
            return """**Splošni kontaktni podatki Občine Rače-Fram:**

📞 **Telefon:** 02 609 60 10
📧 **E-pošta:** obcina@race-fram.si
📍 **Naslov:** Grajski trg 14, 2327 Rače

*Za specifične poizvedbe navedite področje dela (npr. šport, kmetijstvo, turizem).*"""
        
        # Iskanje v JSONL podatkih
        relevant_items = []
        for item in contacts_data:
            text = str(item.get('text', '')).lower()
            
            if not query_lower:
                relevant_items.append(item)
            else:
                if (any(term in text for term in query_lower.split()) or
                    any(term in str(item.get('name', '')).lower() for term in query_lower.split()) or
                    any(term in str(item.get('role', '')).lower() for term in query_lower.split())):
                    relevant_items.append(item)
        
        if not relevant_items:
            return """**Splošni kontaktni podatki Občine Rače-Fram:**

📞 **Telefon:** 02 609 60 10
📧 **E-pošta:** obcina@race-fram.si
📍 **Naslov:** Grajski trg 14, 2327 Rače

*Za specifične poizvedbe navedite področje dela (npr. šport, kmetijstvo, turizem).*"""
        
        # Formatiraj odgovor iz JSONL podatkov
        response = "**Kontaktni podatki:**\n\n"
        
        for item in relevant_items[:3]:  # Omeji na 3 rezultate
            name = item.get('name', '')
            text = item.get('text', '')
            
            if name:
                response += f"**{name}**\n"
            
            # Poišči telefon
            phone_match = re.search(r'0[2-9][/\-\s]*\d{3}[/\-\s]*\d{2}[/\-\s]*\d{2}', text)
            if phone_match:
                response += f"📞 Telefon: {phone_match.group()}\n"
            
            # Poišči email
            email_match = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)
            if email_match:
                response += f"📧 E-pošta: {email_match.group()}\n"
            
            # Dodaj opis dela
            if len(text) > 50:
                response += f"_{text[:100]}..._\n"
            
            response += "\n"
        
        return response.strip()

    def get_office_hours_direct(self, query_lower=""):
        """Direktno pridobi uradne ure iz krajevni_urad_aktualno.jsonl"""
        office_data = self.load_jsonl_data("krajevni_urad_aktualno.jsonl")
        
        if not office_data:
            return "Žal nimam dostopa do podatkov o uradnih urah."
        
        response = "**Uradne ure Občine Rače-Fram:**\n\n"
        
        dan_mapping = {
            "ponedeljek": "ponedeljek", "pon": "ponedeljek",
            "torek": "torek", "tor": "torek", 
            "sreda": "sreda", "sre": "sreda",
            "četrtek": "četrtek", "čet": "četrtek",
            "petek": "petek", "pet": "petek"
        }
        
        asked_day = None
        for day_variant, canonical_day in dan_mapping.items():
            if day_variant in query_lower:
                asked_day = canonical_day
                break
        
        found_specific = False
        for item in office_data:
            text = item.get("text", "")
            
            if asked_day and asked_day in text.lower():
                response += f"**{asked_day.title()}:** {text}\n"
                found_specific = True
            elif not asked_day and any(word in text.lower() for word in ["ura", "odprt", "odpr"]):
                response += f"• {text}\n"
        
        if asked_day and not found_specific:
            response += f"Žal nimam specifičnih podatkov za {asked_day}.\n"
        
        # Fallback na osnovna data
        if "•" not in response and not found_specific:
            response += "📞 **Telefon:** 02 609 60 10\n"
            response += "📍 **Naslov:** Grajski trg 14, 2327 Rače\n\n"
            response += "• **Ponedeljek:** 8:00–12:00 in 13:00–15:00\n"
            response += "• **Sreda:** 8:00–12:00 in 13:00–17:00\n"
            response += "• **Petek:** 8:00–13:00\n"
        
        return response

    def preoblikuj_vprasanje_s_kontekstom(self, zgodovina_pogovora, zadnje_vprasanje):
        """IZBOLJŠANA kontekstualna logika - brez bluzenja"""
        if not zgodovina_pogovora:
            return zadnje_vprasanje

        # Preverimo, če je vprašanje resnično kontekstualno
        q_norm = normalize_text(zadnje_vprasanje)
        short_contextual_words = ["kaj", "pa", "kdaj", "kje", "kako", "koga", "ali", "koliko"]
        
        # Če vprašanje ni kratko in kontekstualno, ne preoblikuj
        if len(zadnje_vprasanje.split()) > 6:
            return zadnje_vprasanje
        
        # Samo če vsebuje kontekstualne besede
        if not any(word in q_norm for word in short_contextual_words):
            return zadnje_vprasanje

        print("→ Kličem specialista za spomin...")
        
        # Vzemi samo zadnji Q/A par za kontekst (ne celotne zgodovine)
        if len(zgodovina_pogovora) > 0:
            last_q, last_a = zgodovina_pogovora[-1]
            zgodovina_str = f"Uporabnik: {last_q}\nAsistent: {last_a[:200]}..." # Omeji dolžino
        else:
            return zadnje_vprasanje

        prompt = f"""Preoblikuj novo vprašanje v samostojno vprašanje glede na zadnji pogovor.

Zadnji pogovor:
{zgodovina_str}

Novo vprašanje: "{zadnje_vprasanje}"

Samostojno vprašanje (kratko in jasno):"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=80  # Krajši odgovor
            )

            preoblikovano = response.choices[0].message.content.strip().replace('"', '').replace("'", "")
            
            # Preveri, če preoblikovanje sploh spremeni nekaj
            if normalize_text(preoblikovano) == normalize_text(zadnje_vprasanje):
                return zadnje_vprasanje
                
            print(f"Kontekst: '{zadnje_vprasanje}' → '{preoblikovano}'")
            return preoblikovano
        except Exception as e:
            print(f"Napaka pri kontekstualnem preoblikovanju: {e}")
            return zadnje_vprasanje

    def _ensure_nap_token(self):
        """Zagotovi veljaven NAP API token"""
        if (self._nap_access_token and self._nap_token_expiry and 
            datetime.now() < self._nap_token_expiry - timedelta(seconds=60)):
            return self._nap_access_token

        print("→ Pridobivam NAP API token...")
        try:
            response = requests.post(
                NAP_TOKEN_URL,
                data={
                    'grant_type': 'password',
                    'username': NAP_USERNAME,
                    'password': NAP_PASSWORD
                },
                headers={'Content-Type': 'application/x-www-form-urlencoded'},
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            self._nap_access_token = data['access_token']
            self._nap_token_expiry = datetime.now() + timedelta(seconds=data.get('expires_in', 3600))
            return self._nap_access_token
        except Exception as e:
            print(f"Napaka pri pridobivanju NAP tokena: {e}")
            raise

    def preveri_zapore_cest(self):
        """Pridobi aktualne prometne informacije prek NAP API"""
        if not NAP_USERNAME or not NAP_PASSWORD:
            return "Dostop do prometnih informacij ni mogoč."

        print("→ Kličem specialista za promet (NAP API)...")

        try:
            token = self._ensure_nap_token()
            headers = {'Authorization': f'Bearer {token}'}
            data_response = requests.get(NAP_DATA_URL, headers=headers, timeout=15)
            data_response.raise_for_status()
            vsi_dogodki = data_response.json().get('features', [])

            # Lokalni filter za občino
            MUNICIPAL_FILTER = {"rače", "fram", "slivnica", "brunšvik", "podova", "morje", "hoče"}

            relevantni_dogodki = []
            for dogodek in vsi_dogodki:
                props = dogodek.get('properties', {})
                location_text = " ".join([
                    str(props.get('cesta', '')),
                    str(props.get('opis', '')),
                    str(props.get('imeDogodka', ''))
                ]).lower()

                if any(keyword in location_text for keyword in MUNICIPAL_FILTER):
                    relevantni_dogodki.append(props)

            if not relevantni_dogodki:
                return "Po podatkih NAP trenutno ni zabeleženih del na cesti, zapor ali zastojev na območju občine Rače-Fram."

            # Deduplikacija
            merged = []
            for z in relevantni_dogodki:
                added = False
                for m in merged:
                    ista_cesta = normalize_text(z.get('cesta', '')) == normalize_text(m.get('cesta', ''))
                    opis_sim = SequenceMatcher(None, 
                                             normalize_text(z.get('opis', '')), 
                                             normalize_text(m.get('opis', ''))).ratio()
                    if ista_cesta and opis_sim >= 0.9:
                        added = True
                        break
                if not added:
                    merged.append(z)

            # Prioriteta (Rače, Fram prvo)
            def priority_key(z):
                text = " ".join([z.get('cesta', ''), z.get('opis', ''), z.get('imeDogodka', '')]).lower()
                if "rače" in text or "fram" in text:
                    return 0
                return 1

            merged.sort(key=priority_key)

            # Formatiraj poročilo
            timestamp = datetime.now().strftime("%d.%m.%Y %H:%M")
            porocilo = f"🚗 **Promet v občini Rače-Fram** _(posodobljeno: {timestamp})_\n\n"

            for ereignis in merged[:5]:  # Max 5 dogodkov
                cesta = ereignis.get('cesta', 'Neznana cesta')
                opis = ereignis.get('opis', 'Brez opisa')
                porocilo += f"**{cesta}**\n{opis}\n\n"

            porocilo += "_Vir: NAP / promet.si_"
            return porocilo

        except Exception as e:
            print(f"Napaka pri NAP API: {e}")
            return "⚠️ Prometne informacije trenutno niso dostopne. Poskusite kasneje."

    def obravnavaj_odvoz_odpadkov(self, uporabnikovo_vprasanje, session_id):
        """ZDRUŽENA FUNKCIJA ZA ODVOZ ODPADKOV - kombinira obe skripti"""
        print("→ Kličem specialista za odpadke...")
        
        if not self.collection:
            return "V bazi znanja ni podatkov o urnikih odpadkov."
        
        stanje = self.zgodovina_seje[session_id].get('stanje', {})
        vprasanje_za_iskanje = (stanje.get('izvirno_vprasanje', '') + " " + uporabnikovo_vprasanje).strip()
        vprasanje_norm = normalize_text(vprasanje_za_iskanje)

        try:
            vsi_urniki = self.collection.get(where={"kategorija": "Odvoz odpadkov"})
        except:
            # Fallback brez filtra če where ne deluje
            print("⚠️ Filter ne deluje, iščem brez where")
            results = self.collection.query(
                query_texts=[vprasanje_norm + " odvoz odpadki smeti"],
                n_results=10,
                include=["documents", "metadatas"]
            )
            
            if results['documents'] and results['documents'][0]:
                # Poišči dokumente o odpadkih
                waste_docs = []
                for doc in results['documents'][0]:
                    doc_lower = doc.lower()
                    if any(word in doc_lower for word in ['odvoz', 'odpadki', 'smeti', 'steklo', 'papir', 'bio', 'embalaža']):
                        waste_docs.append(doc)
                
                if waste_docs:
                    return waste_docs[0]  # Vzemi prvi relevanten dokument
            
            return "Žal nisem našel podatkov o odvozu za to lokacijo. Pokličite občino: 02 609 60 10"

        if not vsi_urniki or not vsi_urniki.get('ids'):
            return "V bazi znanja ni podatkov o urnikih."

        # Kompleksna logika iz skripte 2
        iskani_tip = get_canonical_waste(vprasanje_norm)
        contains_naslednji = "naslednji" in vprasanje_norm

        # Ekstraktiraj lokacije iz vprašanja
        waste_type_stopwords = {normalize_text(k) for k in WASTE_TYPE_VARIANTS.keys()}
        for variants in WASTE_TYPE_VARIANTS.values():
            for v in variants:
                waste_type_stopwords.add(normalize_text(v))

        extra_stop = {"kdaj", "je", "naslednji", "odvoz", "odpadkov", "smeti", "na", "v", "za", "kako", "kateri", "katera", "kaj", "kje", "rumene", "rumena", "kanta", "kante"}

        odstrani = waste_type_stopwords.union(extra_stop)
        raw_tokens = [t for t in re.split(r'[,\s]+', vprasanje_norm) if t and t not in odstrani]

        # Zgradi lokacijske fraze
        location_phrases = []
        for size in (3, 2, 1):
            for i in range(len(raw_tokens) - size + 1):
                phrase = " ".join(raw_tokens[i:i + size])
                location_phrases.append(phrase)

        # Odstrani podvojene
        seen = set()
        filtered_phrases = []
        for p in location_phrases:
            if p in seen:
                continue
            seen.add(p)
            filtered_phrases.append(p)
        location_phrases = filtered_phrases

        generic_single = {"cesta", "cesti", "ulica", "ulici", "pot", "trg", "ob"}
        multi_word_phrases = [p for p in location_phrases if len(p.split()) > 1]
        single_word_phrases = [p for p in location_phrases if len(p.split()) == 1 and p not in generic_single]

        # Poisci ujemanja
        exact_street_matches = []
        fuzzy_street_matches = []
        area_matches = []

        def score_street(phrase: str, street_tok: str) -> float:
            if slovenian_variant_equivalent(phrase, street_tok):
                return 1.0
            norm_phrase = normalize_text(phrase)
            norm_street = normalize_text(street_tok)
            street_words = [w for w in norm_street.split() if w]
            best_word_ratio = 0.0
            for w in street_words:
                r = SequenceMatcher(None, norm_phrase, w).ratio()
                if r > best_word_ratio:
                    best_word_ratio = r
            full_ratio = SequenceMatcher(None, norm_phrase, norm_street).ratio()
            return max(full_ratio, best_word_ratio * 0.95)

        # Faze iskanja
        phrase_groups = []
        if multi_word_phrases:
            phrase_groups.append(("multi", multi_word_phrases))
        phrase_groups.append(("single", single_word_phrases))

        for i in range(len(vsi_urniki['ids'])):
            meta = vsi_urniki['metadatas'][i]
            doc_text = vsi_urniki['documents'][i]

            meta_tip_raw = meta.get('tip_odpadka', '')
            meta_tip_canon = get_canonical_waste(meta_tip_raw) or normalize_text(meta_tip_raw)

            if iskani_tip and meta_tip_canon != iskani_tip:
                continue

            lokacije = extract_locations_from_naselja(meta.get('naselja', ''))
            obm = normalize_text(meta.get('obmocje', ''))

            matched_for_this_doc = False

            # Exact matches
            for phase, phrases in phrase_groups:
                if not phrases:
                    continue
                for phrase in phrases:
                    for street_tok in lokacije:
                        if normalize_text(phrase) == normalize_text(street_tok) or slovenian_variant_equivalent(phrase, street_tok):
                            candidate = {
                                'doc': doc_text,
                                'meta': meta,
                                'tip_canon': meta_tip_canon,
                                'matched_street': street_tok,
                                'matched_phrase': phrase,
                                'score': 1.0
                            }
                            exact_street_matches.append(candidate)
                            matched_for_this_doc = True
                            break
                    if matched_for_this_doc:
                        break
                if matched_for_this_doc:
                    break

            # Fuzzy matches
            if not matched_for_this_doc:
                for phase, phrases in phrase_groups:
                    if not phrases:
                        continue
                    for phrase in phrases:
                        for street_tok in lokacije:
                            sc = score_street(phrase, street_tok)
                            threshold = 0.75 if phase == "multi" else 0.85
                            if sc >= threshold:
                                candidate = {
                                    'doc': doc_text,
                                    'meta': meta,
                                    'tip_canon': meta_tip_canon,
                                    'matched_street': street_tok,
                                    'matched_phrase': phrase,
                                    'score': sc
                                }
                                fuzzy_street_matches.append(candidate)
                                matched_for_this_doc = True
                        if matched_for_this_doc and phase == "multi":
                            break

            # Area matches jako fallback
            has_street_for_doc = any(
                c['meta'] is meta and ('matched_street' in c) for c in exact_street_matches + fuzzy_street_matches
            )
            if not has_street_for_doc:
                for phrase in multi_word_phrases + single_word_phrases:
                    if obm:
                        ratio_full = SequenceMatcher(None, normalize_text(phrase), obm).ratio()
                        if ratio_full >= 0.75 or fuzzy_match(phrase, obm, threshold=0.8):
                            candidate = {
                                'doc': doc_text,
                                'meta': meta,
                                'tip_canon': meta_tip_canon,
                                'matched_area': obm,
                                'matched_phrase': phrase,
                                'score': ratio_full
                            }
                            area_matches.append(candidate)
                            break

        # Izberi najboljše kandidate
        if exact_street_matches:
            kandidati = exact_street_matches
        elif fuzzy_street_matches:
            fuzzy_street_matches.sort(key=lambda x: x.get('score', 0), reverse=True)
            kandidati = fuzzy_street_matches
        else:
            kandidati = area_matches

        if not kandidati:
            return "Za navedeno kombinacijo tipa in lokacije žal nisem našel ustreznega urnika. Pokličite 02 609 60 10."

        # Generiraj odgovor
        now = datetime.now()
        if contains_naslednji:
            best = None
            for info in kandidati:
                doc_text = info['doc']
                tip_odpadka = info['meta'].get('tip_odpadka', '')
                
                if 'matched_street' in info:
                    lokacija_descr = info['matched_street'].title()
                elif 'matched_area' in info:
                    lokacija_descr = info['matched_area'].title()
                else:
                    lokacija_descr = info['meta'].get('obmocje', '')

                datumi = []
                for match in re.findall(r"(\d{1,2})\.(\d{1,2})\.", doc_text):
                    try:
                        dan, mesec = int(match[0]), int(match[1])
                        dt = datetime(now.year, mesec, dan)
                        datumi.append(dt)
                    except Exception:
                        continue

                datumi = sorted(set(datumi))
                naslednji = None
                for d in datumi:
                    if d.date() >= now.date():
                        naslednji = d
                        break

                if naslednji:
                    if best is None or naslednji < best[0]:
                        best = (naslednji, tip_odpadka, lokacija_descr)

            if best:
                dt_obj, tip_odpadka, lok_descr = best
                return f"Naslednji odvoz za **{tip_odpadka}** na **{lok_descr}** je **{dt_obj.strftime('%-d.%m.%Y')}**."
            else:
                return f"Za iskani tip in lokacijo ni več prihodnjih terminov v letu {now.year}."

        else:
            odgovori = []
            for info in kandidati:
                doc_text = info['doc']
                tip_odpadka = info['meta'].get('tip_odpadka', '')

                if 'matched_street' in info:
                    street_display = info['matched_street'].title()
                    datumi_match = re.search(
                        r'je odvoz za .*? predviden ob naslednjih terminih:\s*(.*)',
                        doc_text, flags=re.IGNORECASE
                    )
                    if datumi_match:
                        datumi_str = datumi_match.group(1).strip()
                        odgovori.append(
                            f"Za ulico **{street_display}** je odvoz **{tip_odpadka}** predviden ob naslednjih terminih: {datumi_str}"
                        )
                    else:
                        odgovori.append(doc_text)
                elif 'matched_area' in info:
                    odgovori.append(doc_text)
                else:
                    odgovori.append(doc_text)

            stanje.clear()
            unique = []
            for o in odgovori:
                if o not in unique:
                    unique.append(o)
            return "\n\n".join(unique) if unique else "Žal mi ni uspelo najti ustreznega urnika."

    def belezi_pogovor(self, session_id, vprasanje, odgovor):
        """Beleži pogovor v log datoteko"""
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
            print(f"Napaka pri beleženju: {e}")

    def odgovori(self, uporabnikovo_vprasanje: str, session_id: str):
        """GLAVNA METODA - ZDRUŽENA LOGIKA"""
        
        # Inicializiraj bazo
        self.nalozi_bazo()
        
        if not self.collection:
            return "Oprostite, moja baza znanja trenutno ni na voljo."

        # Inicializiraj session
        if session_id not in self.zgodovina_seje:
            self.zgodovina_seje[session_id] = {'zgodovina': [], 'stanje': {}}

        session_data = self.zgodovina_seje[session_id]
        zgodovina = session_data['zgodovina']
        stanje = session_data['stanje']

        print(f"🔍 DEBUG: vprasanje_lower = '{uporabnikovo_vprasanje.lower()}'")

        vprasanje_lower = uporabnikovo_vprasanje.lower()
        
        # KLJUČNA IZBOLJŠAVA: Boljši kontekstni routing
        if any(phrase in vprasanje_lower for phrase in ["kaj pa", "kdaj pa", "in kaj", "kako pa"]):
            if zgodovina:
                zadnje_user_q, _ = zgodovina[-1]
                # Če je zadnje vprašanje o odpadkih in novo vsebuje lokacijo
                if (any(word in zadnje_user_q.lower() for word in ["odvoz", "smeti", "odpadk", "steklo", "papir", "bio", "embal"]) and
                    any(word in vprasanje_lower for word in ["cesta", "ulica", "na ", "v ", "cesti", "ulici", "turnerjevi", "bistriska", "mlinska"])):
                    print("🔄 KONTEKST: 'kaj pa' + lokacija -> odvoz odpadkov")
                    kontekstno_vprasanje = f"kdaj je odvoz stekla {uporabnikovo_vprasanje.lower().replace('kdaj pa', '').replace('kaj pa', '').strip()}"
                    odgovor = self.obravnavaj_odvoz_odpadkov(kontekstno_vprasanje, session_id)
                    
                    zgodovina.append((uporabnikovo_vprasanje, odgovor))
                    if len(zgodovina) > 8:  # Povečan limit
                        zgodovina.pop(0)
                    
                    self.belezi_pogovor(session_id, uporabnikovo_vprasanje, odgovor)
                    return odgovor

        # LAYER 1: Zdravstveni podatki (direktno iz JSONL)
        if any(word in vprasanje_lower for word in ["zdravnik", "zdravnica", "osebni zdravnik", "zobozdravnik", "zobni", "ambulanta", "zdravstvo", "ordinacija", "medicina"]):
            print("⚕️ ZAZNANO: Zdravstveno vprašanje - direktno iz JSONL!")
            odgovor = self.get_health_data_direct(vprasanje_lower)
            
            zgodovina.append((uporabnikovo_vprasanje, odgovor))
            if len(zgodovina) > 6:  # Zdravstvo - daljša zgodovina
                zgodovina.pop(0)
            
            self.belezi_pogovor(session_id, uporabnikovo_vprasanje, odgovor)
            return odgovor

        # LAYER 2: Kontakti (izboljšano z direktnimi odgovori)
        if any(word in vprasanje_lower for word in ["kontakt", "kontaktiram", "koga", "kdo je odgovoren", "telefon", "mail", "email", "naslov", "zaposleni", "direktor", "župan"]):
            print("📞 ZAZNANO: Kontaktno vprašanje - direktno iz JSONL!")
            odgovor = self.get_contacts_data_direct(vprasanje_lower)
            
            zgodovina.append((uporabnikovo_vprasanje, odgovor))
            if len(zgodovina) > 4:  # Kontakti - srednja zgodovina
                zgodovina.pop(0)
            
            self.belezi_pogovor(session_id, uporabnikovo_vprasanje, odgovor)
            return odgovor

        # LAYER 3: OŠ Rače govorilne ure
        if (any(word in vprasanje_lower for word in ["govorilne ure", "govorilnih ur", "govorilne", "naročanje"]) and 
            any(word in vprasanje_lower for word in ["rače", "race"])):
            print("🏫 ZAZNANO: Govorilne ure OŠ Rače!")
            odgovor = """**Govorilne ure v OŠ Rače:**

Za govorilne ure v OŠ Rače je **obvezno predhodno spletno naročanje**.

🔗 **Povezava za naročanje:** [Govorilne ure OŠ Rače](https://www.osrace.si/?p=1235)

Prosimo, da se naročite vnaprej preko zgornje povezave."""
            
            zgodovina.append((uporabnikovo_vprasanje, odgovor))
            if len(zgodovina) > 3:  # Kratka zgodovina za to
                zgodovina.pop(0)
            
            self.belezi_pogovor(session_id, uporabnikovo_vprasanje, odgovor)
            return odgovor

        # LAYER 4: Uradne ure
        if any(word in vprasanje_lower for word in ["ura", "odprt", "kdaj odprt", "uradne ure", "krajevni urad"]):
            print("🏢 ZAZNANO: Uradne ure vprašanje - direktno iz JSONL!")
            odgovor = self.get_office_hours_direct(vprasanje_lower)
            
            zgodovina.append((uporabnikovo_vprasanje, odgovor))
            if len(zgodovina) > 4:
                zgodovina.pop(0)
            
            self.belezi_pogovor(session_id, uporabnikovo_vprasanje, odgovor)
            return odgovor

        # KONTEKSTUALNO PREOBLIKOVANJE (samo za smiselne primere)
        pametno_vprasanje = self.preoblikuj_vprasanje_s_kontekstom(zgodovina, uporabnikovo_vprasanje)
        vprasanje_lower = pametno_vprasanje.lower()

        # LAYER 5: Odvoz odpadkov (ZDRUŽENA FUNKCIONALNOST)
        if (any(re.search(r'\b' + re.escape(k) + r'\b', vprasanje_lower) for k in KLJUCNE_BESEDE_ODPADKI) or 
            stanje.get('namen') == 'odpadki'):
            odgovor = self.obravnavaj_odvoz_odpadkov(pametno_vprasanje, session_id)

        # LAYER 6: Promet
        elif any(re.search(r'\b' + re.escape(k) + r'\b', vprasanje_lower) for k in KLJUCNE_BESEDE_PROMET):
            odgovor = self.preveri_zapore_cest()

        # LAYER 7: Jedilniki
        elif any(word in vprasanje_lower for word in ["malica", "jedilnik", "kosilo", "zajtrk"]):
            print("🍽️ ZAZNANO: Jedilnik vprašanje")
            odgovor = obravnavaj_jedilnik(vprasanje_lower, self.collection)

        # LAYER 8: Splošne poizvedbe (RAG sistem)
        else:
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
                        f"VSEBINA: {doc}\n\n"
                    )

            if not kontekst_baza:
                odgovor = """Žal o tem nimam informacij. 

Za pomoč se obrnite na:
📞 02 609 60 10
📧 obcina@race-fram.si"""
            else:
                now = datetime.now()
                prompt_za_llm = (
                    f"Ti si virtualni župan občine Rače-Fram. Današnji datum je {now.strftime('%d.%m.%Y')}.\n"
                    "Odgovori kratko in jasno na podlagi konteksta. Ključne informacije **poudari**.\n\n"
                    f"KONTEKST:\n{kontekst_baza}\n"
                    f"VPRAŠANJE: {uporabnikovo_vprasanje}\n"
                    "ODGOVOR:"
                )

                try:
                    response = self.openai_client.chat.completions.create(
                        model=GENERATOR_MODEL_NAME,
                        messages=[{"role": "user", "content": prompt_za_llm}],
                        temperature=0.1,
                        max_tokens=400
                    )
                    odgovor = response.choices[0].message.content
                except Exception as e:
                    print(f"LLM napaka: {e}")
                    odgovor = "Prišlo je do napake pri obdelavi vprašanja. Poskusite kasneje."

        # DODAJ V ZGODOVINO z pametnim omejevanjem
        zgodovina.append((uporabnikovo_vprasanje, odgovor))
        
        # KLJUČNO: Ne briši kontekst preagresivno
        max_history = 6  # Povečan iz 4
        if len(zgodovina) > max_history:
            zgodovina.pop(0)

        # Shrani in logiraj
        self.belezi_pogovor(session_id, uporabnikovo_vprasanje, odgovor)
        
        return odgovor

def test_system():
    """Testna funkcija za preverjanje delovanja združene skripte"""
    print("\n🧪 TESTIRANJE ZDRUŽENE SKRIPTE:")
    print("=" * 50)
    
    zupan = VirtualniZupan()
    zupan.nalozi_bazo()
    
    test_questions = [
        ("Kdo je direktor občinske uprave?", "kontakti - direktor"),
        ("Koga kontaktiram za kmetijski razpis?", "kontakti - kmetijstvo"),
        ("Koga kontaktiram za termine telovadnice?", "kontakti - šport"),
        ("Ali imamo v občini zobozdravnika?", "zdravstvo"),
        ("Kdaj je odvoz stekla na Bistriški cesti?", "odpadki - kompleksno"),
        ("Kaj pa na Mlinski ulici?", "kontekst->odpadki")
    ]
    
    session_id = "test_session"
    
    for i, (question, expected_type) in enumerate(test_questions, 1):
        print(f"\n{i}. {question} (tip: {expected_type})")
        print("-" * 40)
        
        try:
            answer = zupan.odgovori(question, session_id)
            print(f"Odgovor: {answer[:150]}{'...' if len(answer) > 150 else ''}")
        except Exception as e:
            print(f"❌ Napaka: {e}")
    
    print(f"\n📊 Statistika: {len(zupan.zgodovina_seje)} aktivnih sej")
    print(f"Cache vnosi: {len(zupan.jsonl_cache)}")

def main():
    """Glavni CLI vmesnik"""
    print("\n" + "="*70)
    print("🏛️  VIRTUALNI ŽUPAN OBČINE RAČE-FRAM")
    print("    Verzija 37.0 - ZDRUŽENA KONČNA SKRIPTA")
    print("    ✅ Kontakti iz skripte 1 + Odpadki iz skripte 2")
    print("="*70)
    
    try:
        zupan = VirtualniZupan()
        zupan.nalozi_bazo()
        
        # Testiraj sistem
        test_system()
        
        print("\n💬 CLI vmesnik pripravljen! (vnesite 'izhod' za konec)")
        print("📊 Za statistike vnesite 'stats'")
        
        session_id = f"cli_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print("\n" + "="*50)
        print("Pozdravljeni! Sem vaš virtualni župan občine Rače-Fram.")
        print("Lahko me vprašate karkoli o naši občini.")
        print("="*50)
        
        while True:
            try:
                vprasanje = input(f"\n🤔 Vaše vprašanje: ").strip()
                
                if not vprasanje:
                    continue
                    
                if vprasanje.lower() in ['izhod', 'exit', 'quit', 'konec']:
                    print("\n👋 Hvala za uporabo! Nasvidenje!")
                    break
                
                if vprasanje.lower() == 'stats':
                    print(f"\n📊 STATISTIKE:")
                    print(f"   • Session ID: {session_id}")
                    print(f"   • Cache entries: {len(zupan.jsonl_cache)}")
                    if zupan.collection:
                        print(f"   • Documents in ChromaDB: {zupan.collection.count()}")
                    continue
                
                print("\n" + "="*70)
                print("🤖 ODGOVOR:")
                print("="*70)
                
                odgovor = zupan.odgovori(vprasanje, session_id)
                print(odgovor)
                
                print("="*70)
                
            except KeyboardInterrupt:
                print("\n\n👋 Prekinitev... Nasvidenje!")
                break
            except Exception as e:
                print(f"\n❌ Napaka: {e}")
                continue
        
        print(f"\n📊 KONČNA STATISTIKA:")
        print(f"   • Aktivnih sej: {len(zupan.zgodovina_seje)}")
        print(f"   • Cache vnosov: {len(zupan.jsonl_cache)}")
        
    except Exception as e:
        print(f"\n❌ Kritična napaka: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())