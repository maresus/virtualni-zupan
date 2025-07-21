import os
# ... (ostali importi) ...
from dotenv import load_dotenv
from chromadb.utils import embedding_functions

# --- KONFIGURACIJA ZA RENDER ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, '.env'))
# Pot do TRAJNEGA DISKA na Renderju (mora biti enaka kot v zgradi_bazo.py)
CHROMA_DB_PATH = "/data/chroma_db_prod"
COLLECTION_NAME = "obcina_race_fram_prod"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
GENERATOR_MODEL_NAME = "gpt-4o-mini"

# NAP API KONFIGURACIJA
NAP_TOKEN_URL = "https://b2b.nap.si/uc/user/token"
NAP_DATA_URL = "https://b2b.nap.si/data/b2b.roadworks_si.json"
LOKACIJE_ZA_FILTER = ["Rače", "Fram", "Slivnica"]
NAP_USERNAME = os.getenv("NAP_USERNAME")
NAP_PASSWORD = os.getenv("NAP_PASSWORD")

class VirtualniZupan:
    def __init__(self):
        print("Pripravljam virtualnega župana (verzija 5.1)...")
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.collection = None
        self.nap_access_token = None
        try:
            print(f"Poskušam naložiti bazo znanja iz poti: {CHROMA_DB_PATH}")
            openai_ef = embedding_functions.OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"), model_name=EMBEDDING_MODEL_NAME)
            chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            self.collection = chroma_client.get_collection(name=COLLECTION_NAME, embedding_function=openai_ef)
            print(f"USPEH: Povezan z bazo znanja. V bazi je {self.collection.count()} dokumentov.")
        except Exception as e:
            print(f"KRITIČNA NAPAKA: Baze znanja ni mogoče naložiti. Razlog: {e}")
            traceback.print_exc()

        self.zgodovina_pogovora = []
        if self.collection:
            print("\nVirtualni župan je pripravljen. Pozdravljeni! Kako vam lahko pomagam?")
            print('Za konec pogovora vpišite "adijo" ali "konec".')

    def pridobi_nap_zeton(self):
        if not NAP_USERNAME or not NAP_PASSWORD:
            return "Dostop do prometnih informacij ni mogoč, ker niso vpisani prijavni podatki za NAP API v .env datoteki."
        print("-> Pridobivam nov žeton za dostop do NAP...")
        payload = {'grant_type': 'password', 'username': NAP_USERNAME, 'password': NAP_PASSWORD}
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        try:
            response = requests.post(NAP_TOKEN_URL, data=payload, headers=headers, timeout=15)
            response.raise_for_status()
            self.nap_access_token = response.json().get('access_token')
            if not self.nap_access_token:
                return "Dostop do prometnih informacij ni uspel (ni bilo mogoče pridobiti žetona)."
            print("-> Žeton uspešno pridobljen.")
            return True
        except requests.exceptions.RequestException as e:
            return f"Napaka pri pridobivanju žetona: {e}"

    def preveri_zapore_cest(self):
        if not self.nap_access_token:
            rezultat_pridobivanja_zetona = self.pridobi_nap_zeton()
            if rezultat_pridobivanja_zetona is not True:
                return rezultat_pridobivanja_zetona

        print("-> Pridobivam podatke o zaporah cest...")
        headers = {'Authorization': f'Bearer {self.nap_access_token}'}
        try:
            data_response = requests.get(NAP_DATA_URL, headers=headers, timeout=15)
            data_response.raise_for_status()
            vsi_dogodki = data_response.json()
            relevantne_zapore = [
                dogodek['properties']
                for dogodek in vsi_dogodki.get('features', [])
                if any(lok.lower() in dogodek.get('properties', {}).get('description', '').lower() for lok in LOKACIJE_ZA_FILTER)
            ]
            if not relevantne_zapore:
                return "Na območju občine Rače-Fram trenutno ni zabeleženih del na cesti s strani Nacionalne točke dostopa."
            
            porocilo = "Našel sem naslednje aktualne informacije o delih na cesti:\n\n"
            for z in relevantne_zapore:
                porocilo += f"- Lokacija: {z.get('locationDescription', 'Ni podatka')}\n"
                porocilo += f"  Opis: {z.get('description', 'Ni podatka')}\n\n"
            return porocilo
        except requests.exceptions.RequestException as e:
            return f"Žal mi neposreden vpogled v stanje na cestah trenutno ne deluje. Za najnovejše informacije obiščite https://www.promet.si. Tehnični razlog: {e}"

    def odgovori(self, uporabnikovo_vprasanje: str):
        if not self.collection or self.collection.count() == 0:
            return "Oprostite, zdi se, da moja baza znanja ni na voljo ali pa je prazna. Prosim, preverite, ali je bila pravilno zgrajena."

        spletni_kontekst = ""
        kljucne_besede_promet = ["zapora", "ceste", "promet", "stanje na cestah", "dela na cesti"]
        if any(beseda in uporabnikovo_vprasanje.lower() for beseda in kljucne_besede_promet):
            spletni_kontekst = self.preveri_zapore_cest()
            if spletni_kontekst:
                return spletni_kontekst

        print(f"1. Iščem informacije v bazi z vprašanjem: '{uporabnikovo_vprasanje}'")
        rezultati_iskanja = self.collection.query(
            query_texts=[uporabnikovo_vprasanje],
            n_results=7,
            include=["documents"]
        )
        kontekst_baza = "\n\n---\n\n".join(rezultati_iskanja['documents'][0]) if rezultati_iskanja and rezultati_iskanja['documents'] else ""
        if not kontekst_baza:
            return "Žal o tem nimam nobenih informacij."

        print("2. Pripravljam celovit in časovno ozaveščen odgovor...")
        
        # --- POPRAVLJEN IN POENOSTAVLJEN POZIV ---
        prompt_za_llm = f"""
        Ti si 'Virtualni župan občine Rače-Fram'.
        Odgovori na uporabnikovo vprašanje samo na podlagi spodnjih informacij.
        Bodi prijazen in jedrnat. Če informacije ni, povej, da podatka nimaš.

        --- INFORMACIJE IZ BAZE ZNANJA ---
        {kontekst_baza}
        ---

        VPRAŠANJE UPORABNIKA: "{uporabnikovo_vprasanje}"
        
        TVOJ ODGOVOR:
        """
        # --- KONEC POPRAVKA ---
        
        response = self.openai_client.chat.completions.create(model=GENERATOR_MODEL_NAME, messages=[{"role": "user", "content": prompt_za_llm}], temperature=0.1)
        koncni_odgovor = response.choices[0].message.content
        self.zgodovina_pogovora.append((uporabnikovo_vprasanje, koncni_odgovor))
        return koncni_odgovor

if __name__ == "__main__":
    zupan = VirtualniZupan()
    if zupan.collection:
        while True:
            vprasanje = input("\nVi: ")
            if vprasanje.lower() in ["konec", "adijo", "exit"]:
                print("Hvala za pogovor. Nasvidenje!")
                break
            if not vprasanje.strip():
                print("Župan: Prosim, vnesite vprašanje.")
                continue
            
            odgovor = zupan.odgovori(vprasanje)
            print(f"Župan: {odgovor}")