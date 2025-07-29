import os
import json
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
from chromadb.utils import embedding_functions

# POPRAVLJENO: Pravilna skladnja za uvoz orodja
from Google Search import Google Search

# --- KONFIGURACIJA ---
load_dotenv()
CHROMA_DB_PATH = "/Users/markosatler/Documents/ZUPAN JULIJ 2025/chroma_db_prod" 
COLLECTION_NAME = "obcina_race_fram_prod" 
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
GENERATOR_MODEL_NAME = "gpt-4o-mini"

class VirtualniZupan:
    def __init__(self):
        print("Pripravljam virtualnega župana (verzija z dostopom do spleta)...")
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        try:
            openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                            api_key=os.getenv("OPENAI_API_KEY"),
                            model_name=EMBEDDING_MODEL_NAME
                        )
            chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            self.collection = chroma_client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=openai_ef
            )
            print(f"Povezan z bazo znanja: '{COLLECTION_NAME}'.")
        except Exception as e:
            print(f"Kritična napaka: Baze znanja ni mogoče naložiti. {e}")
            self.collection = None
        
        self.zgodovina_pogovora = []
        print("\nVirtualni župan je pripravljen. Pozdravljeni! Kako vam lahko pomagam?")
        print('Za konec pogovora vpišite "adijo" ali "konec".')

    def poisci_na_spletu(self, vprasanje: str):
        """Izvede iskanje po spletu za aktualne informacije."""
        print("-> Iščem sveže podatke na spletu...")
        try:
            # Izvedemo iskanje z več različnimi poizvedbami za boljše rezultate
            search_queries = [
                f"zapore cest občina Rače-Fram",
                f"prometne informacije Rače-Fram",
                f"{vprasanje} race-fram.si"
            ]
            rezultati_iskanja = Google Search(queries=search_queries)
            
            spletni_kontekst = ""
            # Združimo vse najdene snippete v en sam kontekst
            for rezultat in rezultati_iskanja:
                for r in rezultat.results:
                    spletni_kontekst += f"Vir: {r.source_title}\nPovzetek: {r.snippet}\nURL: {r.url}\n---\n"
            
            return spletni_kontekst if spletni_kontekst else "Na spletu nisem našel relevantnih informacij."
        except Exception as e:
            print(f"Napaka pri spletnem iskanju: {e}")
            return "Prišlo je do napake med iskanjem po spletu."


    def odgovori(self, uporabnikovo_vprasanje: str):
        if not self.collection:
            return "Oprostite, nisem povezan z bazo znanja."
        
        print("1. Iščem informacije v interni bazi...")
        rezultati_iskanja_baza = self.collection.query(
            query_texts=[uporabnikovo_vprasanje],
            n_results=5, # Povečamo število internih rezultatov za boljši kontekst
            include=["documents"]
        )
        
        kontekst_baza = "\n\n---\n\n".join(rezultati_iskanja_baza['documents'][0]) if rezultati_iskanja_baza and rezultati_iskanja_baza['documents'] else "V interni bazi ni bilo najdenih podatkov."

        # Preverimo, ali vprašanje vsebuje ključne besede za spletno iskanje
        spletni_kontekst = ""
        kljucne_besede = ["zapora", "ceste", "promet", "aktualno", "danes", "stanje"]
        if any(beseda in uporabnikovo_vprasanje.lower() for beseda in kljucne_besede):
            spletni_kontekst = self.poisci_na_spletu(uporabnikovo_vprasanje)

        print("2. Pripravljam celovit odgovor...")

        formatirana_zgodovina = ""
        for vpr, odg in self.zgodovina_pogovora:
            formatirana_zgodovina += f"Uporabnik: {vpr}\nŽupan: {odg}\n"

        prompt_za_llm = f"""
        Ti si 'Virtualni župan občine Rače-Fram'. Bodi prijazen in ustrežljiv.
        Tvoj cilj je, da združiš informacije iz dveh virov: (1) interne baze znanja in (2) svežih podatkov s spleta.

        --- ZGODOVINA Dosedanjega Pogovora ---
        {formatirana_zgodovina}
        --- KONEC ZGODOVINE ---

        --- INFORMACIJE IZ INTERNE BAZE ZNANJA ---
        {kontekst_baza}
        ---

        --- SVEŽE INFORMACIJE S SPLETA ---
        {spletni_kontekst}
        ---
        
        Na podlagi VSEH zgornjih informacij odgovori na ZADNJE vprašanje uporabnika. Najprej povzemi, kaj si našel na spletu, nato pa dodaj morebitne dodatne informacije iz interne baze.
        Če najdeš spletno povezavo (URL), jo vedno vključi v svoj odgovor.
        Če v nobenem viru ni odgovora, prijazno povej, da informacije nimaš.

        ZADNJE VPRAŠANJE UPORABNIKA: "{uporabnikovo_vprasanje}"

        TVOJ ODGOVOR:
        """
        
        response = self.openai_client.chat.completions.create(
            model=GENERATOR_MODEL_NAME,
            messages=[{"role": "user", "content": prompt_za_llm}],
            temperature=0.1,
        )
        koncni_odgovor = response.choices[0].message.content
        self.zgodovina_pogovora.append((uporabnikovo_vprasanje, koncni_odgovor))
        
        return koncni_odgovor

# --- GLAVNA ZANKA ZA INTERAKTIVNI POGOVOR ---
if __name__ == "__main__":
    zupan = VirtualniZupan()
    
    if zupan.collection:
        while True:
            vprasanje_uporabnika = input("\nVi: ")
            if vprasanje_uporabnika.lower() in ["konec", "adijo", "exit"]:
                print("Hvala za pogovor. Nasvidenje!")
                break
            
            odgovor_zupana = zupan.odgovori(vprasanje_uporabnika)
            print(f"Župan: {odgovor_zupana}")