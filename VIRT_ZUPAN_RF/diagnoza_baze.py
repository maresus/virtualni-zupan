import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Nastavi poti
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '..', '.env'))

DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DB_PATH = os.path.join(DATA_DIR, "chroma_db")
COLLECTION_NAME = "obcina_race_fram_prod"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"

print("=== DIAGNOSTIKA CHROMADB BAZE ===")
print(f"Path: {CHROMA_DB_PATH}")

# Poveži se z bazo
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"), 
    model_name=EMBEDDING_MODEL_NAME
)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=openai_ef)

print(f"\nSkupaj dokumentov: {collection.count()}")

# PREVERI JEDILNIKE
print("\n=== JEDILNIKI ===")
jedilnik_results = collection.query(
    query_texts=["jedilnik malica OŠ"],
    n_results=10,
    include=["documents", "metadatas"]
)

print(f"Najdenih jedilnikov: {len(jedilnik_results['documents'][0])}")

# Analiziraj jedilnike
os_race_count = 0
os_fram_count = 0
datum_2_9_found = False

for i, (doc, meta) in enumerate(zip(jedilnik_results['documents'][0], jedilnik_results['metadatas'][0])):
    print(f"\n--- DOKUMENT {i+1} ---")
    print(f"Vir: {meta.get('vir', 'N/A')}")
    print(f"Datum: {meta.get('datum', 'N/A')}")
    print(f"Kategorija: {meta.get('kategorija', 'N/A')}")
    
    # Preveri katere šole so omenjene
    if "OŠ Rače" in doc:
        os_race_count += 1
        print("ŠOLA: OŠ Rače ✓")
    elif "OŠ Fram" in doc:
        os_fram_count += 1
        print("ŠOLA: OŠ Fram ✓")
    else:
        print("ŠOLA: Ni jasno")
    
    # Preveri za 2.9
    if "2. septembra 2025" in doc:
        datum_2_9_found = True
        print("DATUM 2.9: NAJDEN ✓")
    
    # Prikaži kratko vsebino
    print(f"Vsebina: {doc[:100]}...")

print(f"\n=== POVZETEK ===")
print(f"OŠ Rače dokumentov: {os_race_count}")
print(f"OŠ Fram dokumentov: {os_fram_count}")
print(f"2. september najden: {datum_2_9_found}")

# SPECIFIČEN TEST ZA 2.9
print(f"\n=== TEST ZA 2. SEPTEMBER ===")
september_2_results = collection.query(
    query_texts=["2. septembra 2025"],
    n_results=3,
    include=["documents", "metadatas"]
)

for i, doc in enumerate(september_2_results['documents'][0]):
    print(f"\nRezultat {i+1}:")
    if "OŠ Rače" in doc:
        print("ŠOLA: OŠ Rače")
    elif "OŠ Fram" in doc:
        print("ŠOLA: OŠ Fram")
    print(f"Vsebina: {doc[:150]}...")

# PREVERI METADATA FILTERE
print(f"\n=== TEST METADATA FILTROV ===")
try:
    meta_results = collection.get(
        where={"datum": "2025-09-02"},
        include=["documents", "metadatas"]
    )
    print(f"Dokumenti z datumom 2025-09-02: {len(meta_results['documents'])}")
    for doc in meta_results['documents']:
        print(f"- {doc[:100]}...")
except Exception as e:
    print(f"Metadata filter napaka: {e}")

# PREVERI VIR PODATKOV
print(f"\n=== VIR PODATKOV ===")
unique_sources = set()
vir_results = collection.get(include=["metadatas"])
for meta in vir_results['metadatas']:
    if meta and 'vir' in meta:
        unique_sources.add(meta['vir'])

print("Najdeni viri:")
for source in sorted(unique_sources):
    print(f"- {source}")