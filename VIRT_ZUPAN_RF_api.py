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

# --- KONFIGURACIJA ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '..', '.env'))

# --- Pametno določanje poti glede na okolje ---
if os.getenv('ENV_TYPE') == 'production':
    # Produkcijsko okolje na Renderju
    DATA_DIR = "/data"
    print("Zaznano produkcijsko okolje (Render). Poti so nastavljene na /data.")
else:
    # Lokalno razvojno okolje
    DATA_DIR = os.path.join(BASE_DIR, "data")
    print("Zaznano lokalno okolje. Poti so nastavljene relativno.")
    # Zagotovimo, da lokalna mapa obstaja
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

CHROMA_DB_PATH = os.path.join(DATA_DIR, "chroma_db")
LOG_FILE_PATH = os.path.join(DATA_DIR, "zupan_pogovori.jsonl")
IZVORNI_PODATKI_PATH = os.path.join(BASE_DIR, "izvorni_podatki")

# --- Konec pametnega določanja poti ---

COLLECTION_NAME = "obcina_race_fram_prod"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
GENERATOR_MODEL_NAME = "gpt-4o-mini"
NAP_TOKEN_URL = "https://b2b.nap.si/uc/user/token"
NAP_DATA_URL = "https://b2b.nap.si/data/b2b.roadworks.geojson.sl_SI"
NAP_USERNAME = os.getenv("NAP_USERNAME")
NAP_PASSWORD = os.getenv("NAP_PASSWORD")

# --- FILTRI IN KLJUČNE BESEDE ---
PROMET_FILTER_KLJUCNIKI = [
    "rače", "fram", "slivnica", "brunšvik", "podova", "morje", "hoče",
    "r2-430", "r3-711", "g1-2",
    "priključek slivnica", "razcep slivnica", "letališče maribor", "odcep za rače"
]
KLJUCNE_BESEDE_ODPADKI = ["smeti", "odpadki", "odvoz", "odpavkov", "komunala"]
KLJUCNE_BESEDE_PROMET = ["cesta", "ceste", "cesti", "promet", "dela", "delo", "zapora", "zapore", "zaprta", "zastoj", "gneča", "kolona"]

# --- POMOŽNE FUNKCIJE ZA ODPADKE ---
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
    a_n = normalize_text(a)
    b_n = normalize_text(b)
    
    if a_n == b_n:
        return True
    
    # DODAJ TA DEL - odstrani generične besede
    generic_words = {"cesta", "ulica", "pot", "trg"}
    
    # Očisti generične besede za primerjanje
    a_clean = " ".join(word for word in a_n.split() if word not in generic_words).strip()
    b_clean = " ".join(word for word in b_n.split() if word not in generic_words).strip()
    
    known_variants = {
        # Bistriška cesta variante
        frozenset(["bistriska", "bistriski", "bistriška", "bistriške", "bistriske"]),
        # Mlinska cesta variante  
        frozenset(["mlinska", "mlinski", "mlinsko", "mlinske"]),
        # Framska cesta variante
        frozenset(["framska", "framski", "framsko", "framske"]),
        # Grajski trg variante
        frozenset(["grajski", "grajska", "grajsko", "grajske"]),
        # Pod terasami variante
        frozenset(["terasami", "terase", "terasa", "terasah"]),
        # Turnerjeva variante
        frozenset(["turnerjeva", "turnerjevi", "turnerjev"])
    }
    
    # Preveri če sta obe besedi v istem setu variant (UPORABI OČIŠČENE)
    for variant_set in known_variants:
        if a_clean in variant_set and b_clean in variant_set:
            return True
    
    # Splošna fleksijska heuristika - končnice
    if len(a_n) > 3 and len(b_n) > 3:
        # Če se ujemata v prvih 75% znakov
        min_len = min(len(a_n), len(b_n))
        stem_len = int(min_len * 0.75)
        if a_n[:stem_len] == b_n[:stem_len]:
            # In končnici sta slovenske
            endings = {"a", "i", "e", "o", "u", "ih", "imi", "ega", "emu"}
            a_end = a_n[stem_len:]
            b_end = b_n[stem_len:]
            if a_end in endings and b_end in endings:
                return True
    
    return False

def has_explicit_location(text: str) -> bool:
    """Preveri ali besedilo vsebuje eksplicitno lokacijo"""
    text_norm = normalize_text(text)
    
    # Ključne fraze ki kažejo na specifično lokacijo
    explicit_location_phrases = [
        "pod terasami", "pri terasami", "na terasami", "terasami",
        "na bistriški", "bistriška cesta", "bistriški cesti",
        "na turnerjevi", "turnerjeva ulica", "turnerjevi ulici",
        "na framski", "framska cesta", "framski cesti",
        "na mlinski", "mlinska cesta", "mlinski cesti",
        "grajski trg", "pri grajskem", "na grajskem",
        "mariborska cesta", "na mariborski",
        "ptujska cesta", "na ptujski"
    ]
    
    # Preveri če katera od fraz obstaja v besedilu
    return any(phrase in text_norm for phrase in explicit_location_phrases)

def extract_location_from_text(text: str) -> str:
    """Ekstraktira lokacijo iz besedila"""
    text_norm = normalize_text(text)
    
    # Slovar znanih lokacij z njihovimi variantami
    location_map = {
        "pod terasami": ["pod terasami", "pri terasami", "na terasami", "terasami", "terasa", "terase"],
        "bistriška cesta": ["bistriška", "bistriški", "bistriska", "na bistriški", "bistriška cesta"],
        "turnerjeva ulica": ["turnerjeva", "turnerjevi", "turnerjev", "na turnerjevi", "turnerjeva ulica"],
        "framska cesta": ["framska", "framski", "na framski", "framska cesta"],
        "mlinska cesta": ["mlinska", "mlinski", "na mlinski", "mlinska cesta"],
        "grajski trg": ["grajski", "grajska", "pri grajskem", "na grajskem", "grajski trg"],
        "mariborska cesta": ["mariborska", "na mariborski", "mariborska cesta"],
        "ptujska cesta": ["ptujska", "na ptujski", "ptujska cesta"]
    }
    
    # Poišči ujemanje
    for canonical_name, variants in location_map.items():
        if any(variant in text_norm for variant in variants):
            return canonical_name
    
    return ""

def street_phrase_matches(query_phrase: str, street_tok: str, threshold: float = 0.85) -> bool:
    """
    Natančnejše ujemanje: razbijemo query_phrase in street_tok na besede,
    ignoriramo generične (npr. 'cesta') pri preverjanju,
    in zahtevamo, da se vsi pomembni deli ujemajo z vsaj enim delom street_tok.
    Vključena slovenska fleksijska heuristika.
    """
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

# canonicalne različice tipov odpadkov z možnimi variacijami (razširjene)
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
    norm = normalize_text(text)

    # heuristike / kolokvialno
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

    # direktno ujemanje variant
    for canonical, variants in WASTE_TYPE_VARIANTS.items():
        if normalize_text(canonical) in norm:
            return canonical
        for v in variants:
            if normalize_text(v) in norm:
                return canonical
    # fuzzy fallback
    for canonical, variants in WASTE_TYPE_VARIANTS.items():
        for v in variants:
            if SequenceMatcher(None, norm, normalize_text(v)).ratio() >= 0.85:
                return canonical
    return None

def extract_locations_from_naselja(naselja_field: str):
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
    # normalize and dedupe
    return list({normalize_text(p) for p in parts if p})

def obravnavaj_jedilnik(vprasanje: str, collection):
    """Izboljšana obdelava za jedilnike/malice - vrne SAMO iskani datum"""
    vprasanje_lower = vprasanje.lower()
    
    # Določi šolo
    school = "OŠ Rače"  # Privzeto
    if "fram" in vprasanje_lower:
        school = "OŠ Fram"
    
    # Določi datum
    today = datetime.now()
    target_date = None
    
    # Preveri ali je v vprašanju numerični datum (2.9, 1.9, itd.)
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
    elif "sreda" in vprasanje_lower:
        days_ahead = 2 - today.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        target_date = today + timedelta(days=days_ahead)
    elif "četrtek" in vprasanje_lower:
        days_ahead = 3 - today.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        target_date = today + timedelta(days=days_ahead)
    elif "petek" in vprasanje_lower:
        days_ahead = 4 - today.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        target_date = today + timedelta(days=days_ahead)
    else:
        target_date = today  # Danes

    print(f"🗓️ DEBUG: Iščem {school} za datum {target_date.strftime('%d.%m.%Y')}")

    # STRATEGIJA 1: Poskuši iskanje po metadatah
    try:
        # Poskuši z različnimi formati datuma
        date_formats = [
            target_date.strftime('%Y-%m-%d'),
            target_date.strftime('%d.%m.%Y'),
            target_date.strftime('%d.%m.'),
            target_date.strftime('%-d.%-m.%Y'),
            target_date.strftime('%-d.%-m.')
        ]
        
        for date_format in date_formats:
            try:
                results = collection.get(
                    where={"datum": date_format},
                    include=["documents", "metadatas"]
                )
                
                if results['documents']:
                    for doc in results['documents']:
                        if school in doc:
                            return f"**{school} za {target_date.strftime('%d.%m.%Y')}:**\n\n{doc}"
            except Exception:
                continue
                
    except Exception as e:
        print(f"Metadata search error: {e}")
    
    # STRATEGIJA 2: Semantic search s STROGIM datumskim filterjem
    try:
        search_queries = [
            f"malica {target_date.strftime('%d.%m')} {school}",
            f"jedilnik {target_date.strftime('%d.%m')} {school}",
            f"malica {target_date.strftime('%-d.%-m')} {school}",
            f"{school} {target_date.strftime('%d')} {target_date.strftime('%-m')} malica",
        ]
        
        for query in search_queries:
            results = collection.query(
                query_texts=[query],
                n_results=20,  # Povečaj za boljše iskanje
                include=["documents", "metadatas"]
            )
            
            if results['documents'] and results['documents'][0]:
                # STROŽJI FILTER: išči TOČNO iskani datum
                date_patterns = [
                    target_date.strftime('%d.%m.%Y'),
                    target_date.strftime('%d.%m.'),
                    target_date.strftime('%-d.%-m.%Y'),
                    target_date.strftime('%-d.%-m.'),
                    f"{target_date.day}.{target_date.month}.",
                    f"{target_date.day}. {target_date.month}."
                ]
                
                for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
                    # Preveri ali dokument vsebuje TOČNO iskani datum in šolo
                    doc_contains_school = school.lower() in doc.lower()
                    doc_contains_date = any(pattern in doc for pattern in date_patterns)
                    
                    if doc_contains_school and doc_contains_date:
                        print(f"✅ NAJDEN točen match: {school} + datum")
                        return f"**{school} za {target_date.strftime('%d.%m.%Y')}:**\n\n{doc}"
                        
    except Exception as e:
        print(f"Semantic search error: {e}")
    
    # STRATEGIJA 3: Če ni našel točnega datuma, NE vrni ničesar
    print(f"❌ Ni najden točen podatek za {school} na {target_date.strftime('%d.%m.%Y')}")
    
    return f"Žal nimam podatkov o malici za **{school}** na datum **{target_date.strftime('%d.%m.%Y')}**.\n\nPoskusite:\n- Preveriti ali je datum pravilen\n- Kontaktirati šolo direktno\n- Vprašati za drug datum"

class VirtualniZupan:
    def __init__(self):
        print("Inicializacija razreda VirtualniZupan (Verzija 35.0 - direktni JSONL pristop)...")
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.collection = None
        self.zgodovina_seje = {}
        self._nap_access_token = None
        self._nap_token_expiry = None
        
        # NOVO: Cache za JSONL podatke
        self.jsonl_cache = {}

    def nalozi_bazo(self):
        if self.collection is None:
            try:
                print(f"Poskušam naložiti bazo znanja iz: {CHROMA_DB_PATH}")
                openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"), model_name=EMBEDDING_MODEL_NAME)
                chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
                self.collection = chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=openai_ef)
                print(f"Povezano. Število dokumentov: {self.collection.count()}")
            except Exception as e:
                print(f"KRITIČNA NAPAKA: Baze znanja ni mogoče naložiti. Razlog: {e}")
                self.collection = None

    def belezi_pogovor(self, session_id, vprasanje, odgovor):
        try:
            zapis = {"timestamp": datetime.now().isoformat(), "session_id": session_id, "vprasanje": vprasanje, "odgovor": odgovor}
            with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
                f.write(json.dumps(zapis, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"Napaka pri beleženju pogovora: {e}")

    # NOVO: Direktno branje JSONL datotek
    def load_jsonl_data(self, filename):
        """Direktno preberi JSONL datoteko z cache sistemom"""
        cache_key = filename
        
        # Preveri cache
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
                                print(f"JSON error in {filename} line {line_num + 1}: {e}")
                                continue
                        
                # Cache podatke
                self.jsonl_cache[cache_key] = data
                print(f"📂 Naloženo {len(data)} zapisov iz {filename}")
                
            except Exception as e:
                print(f"Napaka pri branju {filename}: {e}")
        else:
            print(f"Datoteka {filename} ne obstaja na poti {filepath}")
        
        return data

    def get_health_data_direct(self, query_lower=""):
        """Direktno pridobi zdravstvene podatke iz zdravstvo.jsonl"""
        health_data = self.load_jsonl_data("zdravstvo.jsonl")
        
        if not health_data:
            return "Žal nimam dostopa do zdravstvenih podatkov."
        
        # Klasificiraj tip poizvedbe
        if any(word in query_lower for word in ["osebni", "splošna", "družinski"]):
            # Samo osebni zdravniki
            doctors = [item for item in health_data if 
                      "splošna medicina" in item.get("text", "").lower() and
                      "zobni" not in item.get("text", "").lower()]
            title = "**Osebni zdravniki (splošna medicina) v občini Rače-Fram:**\n\n"
            
        elif any(word in query_lower for word in ["zobni", "zobozdravnik", "dentalna"]):
            # Samo zobozdravniki  
            doctors = [item for item in health_data if 
              any(word in item.get("text", "").lower() for word in ["zobni", "zobna", "dentalna", "madens"])]
            title = "**Zobozdravniki v občini Rače-Fram:**\n\n"
            
        else:
            # Vsi zdravniki
            doctors = health_data
            title = "**Zdravstvene storitve v občini Rače-Fram:**\n\n"
        
        if not doctors:
            return "Žal nisem našel ustreznih zdravstvenih podatkov."
        
        # Formatiraj odgovor
        response = title
        
        # Grupiraj po zdravnikih
        processed_doctors = set()
        
        for item in doctors:
            text = item.get("text", "")
            
            # Izvleci ime zdravnika
            doctor_match = re.search(r'Dr\.\s+([^,]+)', text)
            if doctor_match:
                doctor_name = doctor_match.group(1).strip()
                
                # Preveri če je že obdelan
                if doctor_name in processed_doctors:
                    continue
                processed_doctors.add(doctor_name)
                
                # Poišči vse podatke za tega zdravnika
                doctor_info = []
                doctor_schedule = []
                
                for related_item in health_data:
                    related_text = related_item.get("text", "")
                    if doctor_name in related_text:
                        if "Ordinacijski čas" in related_text:
                            doctor_schedule.append(related_text)
                        else:
                            doctor_info.append(related_text)
                
                # Sestavi profil zdravnika
                if doctor_info:
                    main_info = doctor_info[0]  # Prvi vnos z osnovnimi podatki
                    
                    response += f"**Dr. {doctor_name}**\n"
                    
                    # Izvleci kontaktne podatke
                    telefon_match = re.search(r'Telefon:\s*([^\.]+)', main_info)
                    if telefon_match:
                        response += f"- Telefon: {telefon_match.group(1).strip()}\n"
                    
                    email_match = re.search(r'E-pošta:\s*([^\.]+)', main_info)
                    if email_match:
                        response += f"- E-pošta: {email_match.group(1).strip()}\n"
                    
                    naslov_match = re.search(r'Naslov:\s*([^\.]+)', main_info)
                    if naslov_match:
                        response += f"- Naslov: {naslov_match.group(1).strip()}\n"
                    
                    # Dodaj tip storitve
                    if "splošna medicina" in main_info.lower():
                        response += f"- Storitev: Splošna medicina\n"
                    elif "zasebna" in main_info.lower():
                        response += f"- Storitev: Zasebna ambulanta\n"
                    elif "zobni" in main_info.lower() or "dentalna" in main_info.lower():
                        response += f"- Storitev: Zobozdravstvo\n"
                    
                    # Dodaj ordinacijski čas če obstaja
                    if doctor_schedule:
                        schedule_text = doctor_schedule[0]
                        schedule_match = re.search(r'Ordinacijski čas[^:]*:\s*(.+)', schedule_text)
                        if schedule_match:
                            response += f"- Ordinacijski čas: {schedule_match.group(1).strip()}\n"
                    
                    response += "\n"
        
        # Če ni bilo najdenih zdravnikov z imeni, prikaži osnovne podatke
        if not processed_doctors:
            for item in doctors[:5]:  # Max 5 vnosov
                text = item.get("text", "")
                if text:
                    response += f"- {text}\n\n"
        
        response += "\n*Za aktualne informacije o razpoložljivosti pokličite direktno na navedene številke.*"
        
        return response.strip()

    def get_contacts_data_direct(self, query_lower=""):
        """Direktno pridobi kontaktne podatke iz imenik_zaposlenih_in_ure.jsonl"""
        contacts_data = self.load_jsonl_data("imenik_zaposlenih_in_ure.jsonl")
        
        if not contacts_data:
            return "**Splošni kontakt Občine Rače-Fram:**\n\n- **Telefon**: 02 609 60 10\n- **E-pošta**: obcina@race-fram.si"
        
        response = "**Kontaktni podatki Občine Rače-Fram:**\n\n"
        
        for item in contacts_data:
            text = item.get("text", "")
            if text:
                # Formatiraj lepše
                if "direktorica" in text.lower():
                    response += f"**🏢 {text}**\n\n"
                else:
                    response += f"- {text}\n"
        
        response += "\n**Splošni kontakt:**\n- **Telefon**: 02 609 60 10\n- **E-pošta**: obcina@race-fram.si"
        
        return response

    def get_office_hours_direct(self):
        """Direktno pridobi uradne ure iz krajevni_urad_aktualno.jsonl"""
        office_data = self.load_jsonl_data("krajevni_urad_aktualno.jsonl")
        
        for item in office_data:
            text = item.get("text", "")
            if "URADNE URE" in text and "KRAJEVNI URAD RAČE" in text:
                # Parsiraj ure iz besedila
                if "Ponedeljek" in text and "Sreda" in text:
                    return """**Krajevni urad Rače - delovni čas:**

**Ponedeljek:**
- 8:00-12:00
- 13:00-14:30

**Sreda:**
- 8:00-12:00  
- 13:00-17:00

📞 **Kontakt**: 02 609 60 21
📍 **Naslov**: Grajski trg 14, Rače"""
        
        # Fallback če direktno branje ne uspe
        return """**Krajevni urad Rače - delovni čas:**

**Ponedeljek:** 8:00-12:00 in 13:00-14:30
**Sreda:** 8:00-12:00 in 13:00-17:00

Za več informacij pokličite 02 609 60 10."""

    def preoblikuj_vprasanje_s_kontekstom(self, zgodovina_pogovora, zadnje_vprasanje):
        if not zgodovina_pogovora:
            return zadnje_vprasanje
            
        # KLJUČNI POPRAVEK: Preveri ali novo vprašanje vsebuje eksplicitno lokacijo
        if has_explicit_location(zadnje_vprasanje):
            print("-> Nova eksplicitna lokacija zaznana, ne dodajam konteksta iz zgodovine")
            return zadnje_vprasanje
            
        print("-> Kličem specialista za spomin...")
        zgodovina_str = "\n".join([f"Uporabnik: {q}\nAsistent: {a}" for q, a in zgodovina_pogovora])
        prompt = f"""Tvoja naloga je, da glede na zgodovino pogovora preoblikuješ novo vprašanje v samostojno vprašanje. Bodi kratek in jedrnat.

POMEMBNO: Če novo vprašanje že vsebuje lokacijo (kot "pod terasami", "na bistriški", "turnerjeva"), ne dodajaj druge lokacije iz zgodovine!

Primer 1:
Zgodovina:
Uporabnik: kdaj je odvoz smeti na gortanovi ulici?
Asistent: Odvoz je vsak petek.
Novo vprašanje: "kaj pa papir?"
Samostojno vprašanje: "kdaj je odvoz za papir na gortanovi ulici?"

Primer 2:
Zgodovina:
Uporabnik: kaj je za malico 1.9?
Asistent: Za malico je francoski rogljič.
Novo vprašanje: "kaj pa naslednji dan?"
Samostojno vprašanje: "kaj je za malico 2.9?"

Primer 3 (POMEMBNO):
Zgodovina:
Uporabnik: kdaj je odvoz papirja na turnerjevi?
Asistent: Odvoz papirja je...
Novo vprašanje: "kdaj je naslednji odvoz rumene kante pod terasami"
Samostojno vprašanje: "kdaj je naslednji odvoz rumene kante pod terasami" (BREZ dodajanja "turnerjeva")

---
Zgodovina:
{zgodovina_str}
Novo vprašanje: "{zadnje_vprasanje}"
Samostojno vprašanje:"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=100
            )
            preoblikovano = response.choices[0].message.content.strip().replace('"', '')
            
            # DODATNA VAROVALKA: Če preoblikovano vprašanje vsebuje novo lokacijo, uporabi originalno
            original_location = extract_location_from_text(zadnje_vprasanje)
            transformed_location = extract_location_from_text(preoblikovano)
            
            if original_location and transformed_location and original_location != transformed_location:
                print(f"-> OPOZORILO: Zaznana mešanja lokacij ({original_location} vs {transformed_location}), uporabim originalno vprašanje")
                return zadnje_vprasanje
            
            print(f"Originalno: '{zadnje_vprasanje}' -> Preoblikovano: '{preoblikovano}'")
            return preoblikovano
        except Exception:
            return zadnje_vprasanje

    def _ensure_nap_token(self):
        if self._nap_access_token and self._nap_token_expiry and datetime.utcnow() < self._nap_token_expiry - timedelta(seconds=60):
            return self._nap_access_token
        print("-> Pridobivam/osvežujem NAP API žeton...")
        payload = {'grant_type': 'password', 'username': NAP_USERNAME, 'password': NAP_PASSWORD}
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        response = requests.post(NAP_TOKEN_URL, data=payload, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        self._nap_access_token = data['access_token']
        self._nap_token_expiry = datetime.utcnow() + timedelta(seconds=data['expires_in'])
        return self._nap_access_token

    def preveri_zapore_cest(self):
        if not NAP_USERNAME or not NAP_PASSWORD:
            return "Dostop do prometnih informacij ni mogoč."
        print("-> Kličem specialista za promet (NAP API) z ultra-natančnim filtrom...")
        try:
            token = self._ensure_nap_token()
            headers = {'Authorization': f'Bearer {token}'}
            data_response = requests.get(NAP_DATA_URL, headers=headers, timeout=15)
            data_response.raise_for_status()
            vsi_dogodki = data_response.json().get('features', [])

            # ožji lokalni filter: le dogodki, ki omenjajo ključne kraje v občini
            MUNICIPAL_FILTER = {"rače", "fram", "slivnica", "brunšvik", "podova", "morje", "hoče"}

            relevantne_zapore_raw = []
            for dogodek in vsi_dogodki:
                lastnosti = dogodek.get('properties', {})
                cesta = str(lastnosti.get('cesta', '')).strip()
                opis = str(lastnosti.get('opis', '')).strip()
                ime = str(lastnosti.get('imeDogodka', '')).strip()

                celotno_besedilo = " ".join([cesta, opis, ime]).lower()
                if not any(k in celotno_besedilo for k in MUNICIPAL_FILTER):
                    continue

                relevantne_zapore_raw.append({
                    'cesta': cesta or "Ni podatka",
                    'opis': opis or "Ni podatka",
                    'imeDogodka': ime,
                    'full_props': lastnosti
                })

            if not relevantne_zapore_raw:
                return "Po podatkih portala promet.si na območju občine Rače-Fram trenutno ni zabeleženih del na cesti, zapor ali zastojev."

            # deduplikacija: isti cesta + zelo podoben opis
            merged = []
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

            # prioriteta: tisti, ki eksplicitno omenjajo "rače" ali "fram" najprej
            def priority_key(z):
                text = " ".join([z['cesta'], z['opis'], z['imeDogodka']]).lower()
                if "rače" in text or "fram" in text:
                    return 0
                return 1

            merged.sort(key=priority_key)

            porocilo = "Našel sem naslednje **trenutne** informacije o dogodkih na cesti (vir: promet.si):\n\n"
            for z in merged:
                porocilo += f"- **Cesta:** {z['cesta']}\n  **Opis:** {z['opis']}\n\n"
            porocilo = porocilo.strip() + "\n\nZa več informacij obiščite: https://www.race-fram.si/objave/274"
            return porocilo
        except Exception as e:
            return f"Žal mi neposreden vpogled v stanje na cestah trenutno ne deluje. Poskusite kasneje."

    def _poisci_naslednji_datum(self, datumi_str: str, danes: datetime):
        leto = danes.year
        datumi = []
        for del_str in datumi_str.replace('.', ' ').split(','):
            try:
                dan, mesec = map(int, del_str.strip().split())
                datumi.append(datetime(leto, mesec, dan))
            except (ValueError, IndexError):
                continue
        for datum in sorted(datumi):
            if datum.date() >= danes.date():
                return datum.strftime('%d.%m.%Y')
        return None

    def priority_search(self, query, n_results=5):
        """Iskanje ki upošteva prioriteto dokumentov"""
        # Najprej poiščimo high priority dokumente
        try:
            high_priority = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where={"priority": "high"}
            )
            if high_priority['documents'] and high_priority['documents'][0]:
                return high_priority
        except:
            pass
        
        # Če ni high priority, običajno iskanje
        return self.collection.query(
            query_texts=[query],
            n_results=n_results
        )

    def obravnavaj_odvoz_odpadkov(self, uporabnikovo_vprasanje, session_id):
        print("-> Kličem specialista za odpadke...")
        if session_id not in self.zgodovina_seje:
            self.zgodovina_seje[session_id] = {'zgodovina': [], 'stanje': {}}
        stanje = self.zgodovina_seje[session_id].get('stanje', {})
        
        # KLJUČNI POPRAVEK: Če vprašanje vsebuje eksplicitno lokacijo, POČISTI STANJE
        if has_explicit_location(uporabnikovo_vprasanje):
            print("-> Zaznana eksplicitna lokacija, počistim stanje seje")
            stanje.clear()  # Odstrani vse prejšnje kontekstne podatke
        
        vprasanje_za_iskanje = (stanje.get('izvirno_vprasanje', '') + " " + uporabnikovo_vprasanje).strip()
        vprasanje_norm = normalize_text(vprasanje_za_iskanje)

        vsi_urniki = self.collection.get(where={"kategorija": "Odvoz odpadkov"})
        if not vsi_urniki or not vsi_urniki.get('ids'):
            return "V bazi znanja ni podatkov o urnikih."

        # prepoznaj tip odpadka (kanonično), z dodatno heuristiko za rumene kante
        iskani_tip = get_canonical_waste(vprasanje_norm)

        # ali uporabnik hoče "naslednji"
        contains_naslednji = "naslednji" in vprasanje_norm

        # gradimo stopwords iz tipov, da ne pridejo v lokacijske fraze
        waste_type_stopwords = {normalize_text(k) for k in WASTE_TYPE_VARIANTS.keys()}
        for variants in WASTE_TYPE_VARIANTS.values():
            for v in variants:
                waste_type_stopwords.add(normalize_text(v))
        extra_stop = {"kdaj", "je", "naslednji", "odvoz", "odpadkov", "smeti", "na", "v", "za", "kako", "kateri", "katera", "kaj", "kje", "rumene", "rumena", "kanta", "kante"}
        odstrani = waste_type_stopwords.union(extra_stop)

        raw_tokens = [t for t in re.split(r'[,\s]+', vprasanje_norm) if t and t not in odstrani]

        # zgradi n-gram lokacij (prioriteta večbesednih)
        location_phrases = []
        for size in (3, 2, 1):
            for i in range(len(raw_tokens) - size + 1):
                phrase = " ".join(raw_tokens[i:i + size])
                location_phrases.append(phrase)
        # dedupe while preserving order
        seen = set()
        filtered_phrases = []
        for p in location_phrases:
            if p in seen:
                continue
            seen.add(p)
            filtered_phrases.append(p)
        location_phrases = filtered_phrases

        # odstranimo generične enobesedne fraze (npr. "cesta", "ulica")
        generic_single = {"cesta", "cesti", "ulica", "ulici", "pot", "trg", "ob"}
        multi_word_phrases = [p for p in location_phrases if len(p.split()) > 1]
        single_word_phrases = [p for p in location_phrases if len(p.split()) == 1 and p not in generic_single]

        # določimo, ali je v vprašanju eksplicitna lokacija
        street_indicators = {"cesta", "ulica", "pot", "trg", "naslov", "pod", "terasami"}
        is_explicit_location = any(len(p.split()) > 1 or any(ind in p for ind in street_indicators) for p in location_phrases)

        exact_street_matches = []
        fuzzy_street_matches = []
        area_matches = []

        # helper za scoring med frazo in uličnim tokenom z upoštevanjem slovenske variante
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

        # faze: najprej multi-word, potem single-word
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

            for phase, phrases in phrase_groups:
                if not phrases:
                    continue

                # exact match
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

                # fuzzy match
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

            # fallback na območje (samo če ni ulični kandidat za ta dokument)
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

        # izbira kandidatov po prioriteti
        if exact_street_matches:
            kandidati = exact_street_matches
        elif fuzzy_street_matches:
            fuzzy_street_matches.sort(key=lambda x: x.get('score', 0), reverse=True)
            kandidati = fuzzy_street_matches
        else:
            kandidati = area_matches

        # **NOVA LOGIKA**: če je eksplicitna lokacija in ni "naslednji", omeji na ujemanja, ki imajo matched_phrase iz primarnih (najdaljših) fraz
        if is_explicit_location and not contains_naslednji and kandidati:
            primary_phrases = multi_word_phrases if multi_word_phrases else single_word_phrases
            if primary_phrases:
                explicit = [c for c in kandidati if 'matched_phrase' in c and any(
                    normalize_text(c['matched_phrase']) == normalize_text(p) for p in primary_phrases
                )]
                if explicit:
                    max_score = max(c.get('score', 1.0) for c in explicit)
                    kandidati = [c for c in explicit if c.get('score', 1.0) >= max_score - 1e-6]

        # če je eksplicitna lokacija in ni kandidatov, ne fallbackaj
        if is_explicit_location and not kandidati:
            if not iskani_tip and 'caka_na' not in stanje:
                stanje.update({'caka_na': 'tip', 'namen': 'odpadki', 'izvirno_vprasanje': uporabnikovo_vprasanje})
                return "Kateri tip odpadka te zanima? (npr. biološki, mešani komunalni, embalaža, papir in karton, steklo)"
            if iskani_tip:
                prvi = (multi_word_phrases + single_word_phrases)[0] if (multi_word_phrases + single_word_phrases) else "želeno lokacijo"
                return f"Za **{prvi.title()}** in tip **{iskani_tip}** žal nisem našel ustreznega urnika."
            return "Za navedeno kombinacijo tipa in lokacije žal nisem našel ustreznega urnika."

        if not kandidati:
            if not iskani_tip and 'caka_na' not in stanje:
                stanje.update({'caka_na': 'tip', 'namen': 'odpadki', 'izvirno_vprasanje': uporabnikovo_vprasanje})
                return "Kateri tip odpadka te zanima? (npr. biološki, mešani komunalni, embalaža, papir in karton, steklo)"
            if iskani_tip and 'caka_na' not in stanje:
                stanje.update({'caka_na': 'lokacija', 'namen': 'odpadki', 'izvirno_vprasanje': uporabnikovo_vprasanje})
                return "Katero ulico ali območje misliš? Npr. 'Bistriška cesta, Fram'."
            return "Za navedeno kombinacijo tipa in lokacije žal nisem našel ustreznega urnika."

        # sestavi odgovor
        now = datetime.now()
        if contains_naslednji:
            best = None  # (datetime, tip, street_or_area)
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

    def odgovori(self, uporabnikovo_vprasanje: str, session_id: str):
        self.nalozi_bazo()
        if not self.collection:
            return "Oprostite, moja baza znanja trenutno ni na voljo."

        if session_id not in self.zgodovina_seje:
            self.zgodovina_seje[session_id] = {'zgodovina': [], 'stanje': {}}

        stanje = self.zgodovina_seje[session_id]['stanje']
        zgodovina = self.zgodovina_seje[session_id]['zgodovina']

        # PREVERI KATEGORIJE PRED PREOBLIKOVANJEM
        vprasanje_lower = uporabnikovo_vprasanje.lower()
        print(f"🔍 DEBUG: vprasanje_lower = '{vprasanje_lower}'")
        
        # NOVO: Direktno preverjanje JSONL kategorij
        if any(word in vprasanje_lower for word in ["zdravnik", "zdravnica", "zobozdravnik", "zobni", "ambulanta", "ordinacija", "madens", "sapač", "lobnik", "repolusk", "strojnik", "naskovska"]):
            print("⚕️ ZAZNANO: Zdravstveno vprašanje - direktno iz JSONL!")
            odgovor = self.get_health_data_direct(vprasanje_lower)
            
        elif any(word in vprasanje_lower for word in ["kontakt", "telefon", "mail", "email", "zaposleni", "direktorica", "svetovalka"]) and not any(word in vprasanje_lower for word in ["zdravnik", "zobozdravnik"]):
            print("📞 ZAZNANO: Kontaktno vprašanje - direktno iz JSONL!")
            odgovor = self.get_contacts_data_direct(vprasanje_lower)
            
        elif any(word in vprasanje_lower for word in ["uradne ure", "odprt", "odprto", "krajevni urad", "delovni čas"]):
            print("🏢 ZAZNANO: Uradne ure vprašanje - direktno iz JSONL!")
            if "ponedeljek" in vprasanje_lower or ("kaj pa" in vprasanje_lower and any("ponedel" in str(item) for item in zgodovina)):
                odgovor = """**Krajevni urad Rače** je ob ponedeljkih odprt:

- **8:00-12:00**
- **13:00-14:30**

📞 **Kontakt**: 02 609 60 21"""
            elif "sreda" in vprasanje_lower:
                odgovor = """**Krajevni urad Rače** je ob sredah odprt:

- **8:00-12:00**  
- **13:00-17:00**

📞 **Kontakt**: 02 609 60 21"""
            else:
                odgovor = self.get_office_hours_direct()
                
        # Preveri če gre za jedilnik/malico
        elif any(word in vprasanje_lower for word in ["malica", "malico", "kosilo", "kosila", "jedilnik", "jedilnika", "hrana", "hrane", "meni", "menija"]):
            print("🍽️ ZAZNANO: Jedilnik vprašanje!")
            odgovor = obravnavaj_jedilnik(uporabnikovo_vprasanje, self.collection)
            
        elif any(re.search(r'\b' + re.escape(k) + r'\b', vprasanje_lower) for k in KLJUCNE_BESEDE_ODPADKI) or stanje.get('namen') == 'odpadki':
            pametno_vprasanje = self.preoblikuj_vprasanje_s_kontekstom(zgodovina, uporabnikovo_vprasanje)
            odgovor = self.obravnavaj_odvoz_odpadkov(pametno_vprasanje, session_id)
            
        elif any(re.search(r'\b' + re.escape(k) + r'\b', vprasanje_lower) for k in KLJUCNE_BESEDE_PROMET):
            odgovor = self.preveri_zapore_cest()
            
        else:
            pametno_vprasanje = self.preoblikuj_vprasanje_s_kontekstom(zgodovina, uporabnikovo_vprasanje)
            vprasanje_lower = pametno_vprasanje.lower()

            # UPORABI IZBOLJŠANO ISKANJE NAMESTO OBIČAJNEGA
            rezultati_iskanja = self.priority_search(vprasanje_lower)
            
            kontekst_baza = ""
            if rezultati_iskanja.get('documents'):
                for doc, meta in zip(rezultati_iskanja['documents'][0], rezultati_iskanja['metadatas'][0]):
                    kontekst_baza += (
                        f"--- VIR: {meta.get('source', 'Neznan')}\n"
                        f"POVEZAVA: {meta.get('source_url', 'Brez')}\n"
                        f"VSEBINA: {doc}\n\n"
                    )

            if not kontekst_baza:
                return "Žal o tem nimam nobenih informacij."

            now = datetime.now()
            prompt_za_llm = (
                "Ti si 'Virtualni župan občine Rače-Fram'.\n"
                f"KRITIČNA DIREKTIVA - DATUM FILTER: Danes je {now.strftime('%d.%m.%Y')}. "
                f"POPOLNOMA IGNORIRAJ vse podatke, dokumente, informacije za leto 2024 ali starejše. "
                f"NIKOLI ne omenjaj letnic 2024, 2023, 2022 ali katerekoli starejše v svojem odgovoru. "
                f"Če dokument vsebuje zastarele datume, RECI: 'Za aktualne podatke za leto 2025 se obrnite na občino.'\n"
                "DIREKTIVA #2 (OBLIKOVANJE): Odgovor mora biti pregleden. Ključne informacije **poudari**. Kjer naštevaš, **uporabi alineje (-)**.\n"
                "DIREKTIVA #3 (POVEZAVE): Če najdeš URL, ga MORAŠ vključiti v klikljivi obliki: [Ime vira](URL).\n"
                "DIREKTIVA #4 (SPECIFIČNOST): Če ne najdeš specifičnega podatka (npr. 'kontakt'), NE ponavljaj splošnih informacij. Raje reci: \"Žal nimam specifičnega kontakta za to temo.\"\n\n"
                f"--- KONTEKST ---\n{kontekst_baza}---\n"
                f"VPRAŠANJE: \"{uporabnikovo_vprasanje}\"\n"
                "ODGOVOR:"
            )

            response = self.openai_client.chat.completions.create(
                model=GENERATOR_MODEL_NAME,
                messages=[{"role": "user", "content": prompt_za_llm}],
                temperature=0.0
            )
            odgovor = response.choices[0].message.content

        # **NOVA LOGIKA ZA KONTAKT**: če uporabnik sprašuje po kontaktu, ne dajemo osebnih imen ampak splošni občinski
        contact_query = bool(re.search(r'\b(kontakt|telefon|številka|stevilka)\b', uporabnikovo_vprasanje.lower()))
        if contact_query and not any(word in vprasanje_lower for word in ["zdravnik", "zobozdravnik"]):
            # znebimo se osebnih imen kot "mag. Karmen Kotnik" in morebitnih emailov
            odgovor = re.sub(r'(?i)mag\.?\s*karmen\s+kotnik', 'občina Rače-Fram', odgovor)
            odgovor = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '', odgovor)
            # odstranimo specifične alternative telefonskih številk (npr. 02 609 60 19)
            odgovor = re.sub(r'\b02[\s\-]*609[\s\-]*60[\s\-]*1[0-9]\b', '', odgovor)
            # pospravimo podvojene prazne vrstice/odvečne presledke
            odgovor = re.sub(r'\n{2,}', '\n\n', odgovor).strip()
            # zagotovimo splošni kontakt
            generic_line = "Za več informacij pokličite občino Rače-Fram na 02 609 60 10."
            if generic_line not in odgovor:
                odgovor = odgovor + "\n\n" + generic_line

        zgodovina.append((uporabnikovo_vprasanje, odgovor))
        if len(zgodovina) > 4:
            zgodovina.pop(0)

        self.belezi_pogovor(session_id, uporabnikovo_vprasanje, odgovor)
        return odgovor


if __name__ == "__main__":
    import sys
    
    print("Inicializacija Virtualnega župana...")
    zupan = VirtualniZupan()
    
    # Test če dela
    print("Testiram povezavo...")
    zupan.nalozi_bazo()
    
    if not zupan.collection:
        print("❌ NAPAKA: Ne morem naložiti baze!")
        sys.exit(1)
    
    print(f"✅ Sistem pripravljen! ({zupan.collection.count()} dokumentov)")
    
    # Test JSONL sistema
    print("\n🧪 TESTIRANJE DIREKTNEGA JSONL DOSTOPA:")
    print("=" * 50)
    
    # Test zdravniki
    test_questions = [
        "mamo v občini zobozdravnika?",
        "kontakte od zdravnikov", 
        "kdaj je odprti krajevni urad rače",
        "kaj pa ob ponedeljkih",
        "kontakti občine"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. {question}")
        print("-" * 40)
        try:
            answer = zupan.odgovori(question, f"test_{i}")
            # Pokaži samo prve 200 znakov
            preview = answer[:200] + ("..." if len(answer) > 200 else "")
            print(preview)
        except Exception as e:
            print(f"NAPAKA: {e}")
    
    print("\n" + "=" * 50)
    print("CLI VMESNIK (vpišite 'quit' za izhod):\n")
    
    session_id = "cli_test"
    
    while True:
        try:
            vprasanje = input("> ").strip()
            
            if not vprasanje:
                continue
                
            if vprasanje.lower() in ['quit', 'exit', 'q']:
                print("Nasvidenje!")
                break
            
            odgovor = zupan.odgovori(vprasanje, session_id)
            print("\n" + "="*50)
            print(odgovor)
            print("="*50 + "\n")
            
        except KeyboardInterrupt:
            print("\nNasvidenje!")
            break
        except Exception as e:
            print(f"Napaka: {e}")