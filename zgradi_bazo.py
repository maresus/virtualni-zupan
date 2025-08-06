import os
import json
import chromadb
from dotenv import load_dotenv
from chromadb.utils import embedding_functions

# --- KONFIGURACIJA ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))

# --- Pametno določanje poti glede na okolje ---
if os.getenv('ENV_TYPE') == 'production':
    # Produkcijsko okolje na Renderju
    DATA_DIR = "/data"
    print("Zaznano produkcijsko okolje (Render). Poti so nastavljene na /data.")
else:
    # Lokalno razvojno okolje
    DATA_DIR = os.path.join(BASE_DIR, "VIRT_ZUPAN_RF", "data")
    print("Zaznano lokalno okolje. Poti so nastavljene relativno.")
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

CHROMA_DB_PATH = os.path.join(DATA_DIR, "chroma_db")
# --- POPRAVEK: Dodana manjkajoča definicija za SOURCE_DIRECTORY ---
SOURCE_DIRECTORY = os.path.join(BASE_DIR, "izvorni_podatki")
# --- Konec pametnega določanja poti ---

COLLECTION_NAME = "obcina_race_fram_prod" 
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
MAX_CHUNK_LENGTH_CHARS = 2000 

def robustno_sekanje(text: str, max_dolzina: int):
    chunks = []
    paragraphs = text.split('\n\n')
    for p in paragraphs:
        p_strip = p.strip()
        if not p_strip: continue
        if len(p_strip) <= max_dolzina:
            chunks.append(p_strip)
        else:
            sentences = p_strip.replace('\n', ' ').split('. ')
            current_chunk = ""
            for sentence in sentences:
                if not sentence: continue
                sentence_with_dot = sentence + "."
                if len(current_chunk) + len(sentence_with_dot) > max_dolzina:
                    if current_chunk.strip(): chunks.append(current_chunk.strip())
                    current_chunk = sentence_with_dot
                else:
                    current_chunk += " " + sentence_with_dot
            if current_chunk.strip(): chunks.append(current_chunk.strip())
    return chunks

def zgradi_bazo():
    print(f"Baza znanja bo ustvarjena v mapi: {CHROMA_DB_PATH}")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Napaka: OpenAI API ključ ni najden."); return

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=openai_api_key, model_name=EMBEDDING_MODEL_NAME)
    
    if not os.path.exists(CHROMA_DB_PATH):
        os.makedirs(CHROMA_DB_PATH)

    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    print(f"Brisanje stare kolekcije '{COLLECTION_NAME}'...");
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        print("Stara kolekcija ni obstajala.")

    collection = chroma_client.create_collection(name=COLLECTION_NAME, embedding_function=openai_ef)

    vsi_dokumenti, vsi_metapodatki, vsi_ids = [], [], []
    doc_id_counter = 0

    print(f"Branje dokumentov iz: '{SOURCE_DIRECTORY}'...")
    if not os.path.isdir(SOURCE_DIRECTORY):
        print(f"NAPAKA: Mapa z izvornimi podatki '{SOURCE_DIRECTORY}' ne obstaja!")
        return

    for filename in os.listdir(SOURCE_DIRECTORY):
        file_path = os.path.join(SOURCE_DIRECTORY, filename)
        if filename.endswith(".txt"):
            with open(file_path, 'r', encoding='utf-8') as f:
                vsebina = f.read()
                kosi = robustno_sekanje(vsebina, MAX_CHUNK_LENGTH_CHARS)
                for kos in kosi:
                    vsi_dokumenti.append(kos); vsi_metapodatki.append({'source': filename}); vsi_ids.append(f'id_{doc_id_counter}'); doc_id_counter += 1
        elif filename.endswith(".jsonl"):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        text = data.get("text")
                        if text: vsi_dokumenti.append(text); vsi_metapodatki.append(data.get("metadata", {'source': filename})); vsi_ids.append(f'id_{doc_id_counter}'); doc_id_counter += 1
                    except json.JSONDecodeError: print(f"    Opozorilo: Napačna vrstica v {filename}")
        elif filename.endswith(".json"):
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    if "content" in data and "metadata" in data:
                        title = data.get("metadata", {}).get("title", "")
                        content = data.get("content", "")
                        vsi_dokumenti.append(f"Naslov: {title}\nVsebina: {content}"); vsi_metapodatki.append(data.get("metadata", {'source': filename})); vsi_ids.append(f'id_{doc_id_counter}'); doc_id_counter += 1
                    else:
                        for kljuc, vrednost in data.items():
                            vsi_dokumenti.append(f"Tema: {kljuc}. Podrobnosti: {json.dumps(vrednost, ensure_ascii=False)}"); vsi_metapodatki.append({'source': filename, 'kategorija': kljuc}); vsi_ids.append(f'id_{doc_id_counter}'); doc_id_counter += 1
                except json.JSONDecodeError: print(f"    Opozorilo: Datoteka {filename} ni veljaven JSON.")
    
    if not vsi_dokumenti: print("V virih ni podatkov."); return
    
    print(f"Najdenih {len(vsi_dokumenti)} veljavnih segmentov. Dodajam v bazo v paketih...")
    velikost_paketa = 100
    for i in range(0, len(vsi_dokumenti), velikost_paketa):
        konec = min(i + velikost_paketa, len(vsi_dokumenti))
        print(f"  -> Dodajam paket: dokumenti od {i} do {konec}")
        collection.add(
            documents=vsi_dokumenti[i:konec],
            metadatas=vsi_metapodatki[i:konec],
            ids=vsi_ids[i:konec]
        )
    print(f"\nGradnja končana! V bazi je {collection.count()} dokumentov.")

if __name__ == "__main__":
    zgradi_bazo()