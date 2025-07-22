import os
import requests
from dotenv import load_dotenv

# --- KONFIGURACIJA ---
load_dotenv()

TOKEN_URL = "https://b2b.nap.si/uc/user/token"
USERNAME = os.getenv("NAP_USERNAME")
PASSWORD = os.getenv("NAP_PASSWORD")

# Seznam možnih URL-jev, ki jih bomo preizkusili na podlagi vaših dovoljenj
MOZNI_URLJI = {
    "Splošni (iz dokumentacije)": "https://b2b.nap.si/data/b2b.roadworks",
    "Slovenski JSON": "https://b2b.nap.si/data/b2b.roadworks_si.json",
    "Slovenski GeoJSON (stari)": "https://b2b.nap.si/data/b2b.roadworks_si.geojson",
    "Slovenski GeoJSON (nov)": "https://b2b.nap.si/data/b2b.roadworks.geojson.sl_SI" # Vaš nov predlog
}
LOKACIJE_ZA_FILTER = ["Rače", "Fram", "Slivnica"]


def testiraj_nap_api_v2():
    """
    Izboljšana testna skripta, ki preizkusi več možnih URL-jev za dostop do podatkov.
    """
    print("--- ZAČETEK TESTIRANJA NAP API POVEZAVE (V2) ---")

    if not USERNAME or not PASSWORD:
        print("\nNAPAKA: Manjkata NAP_USERNAME ali NAP_PASSWORD v .env datoteki.")
        return

    # 1. KORAK: Pridobivanje žetona
    print("\n1. Korak: Pridobivanje žetona...")
    try:
        payload = {'grant_type': 'password', 'username': USERNAME, 'password': PASSWORD}
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        response = requests.post(TOKEN_URL, data=payload, headers=headers, timeout=15)
        response.raise_for_status()  # Preveri za napake kot 4xx ali 5xx

        access_token = response.json().get('access_token')
        if not access_token:
            print("   -> NAPAKA: Žetona ni bilo mogoče pridobiti.")
            return

        print("   -> USPEH! Žeton uspešno pridobljen.")

    except Exception as e:
        print(f"   -> NAPAKA pri pridobivanju žetona: {e}")
        return

    # 2. KORAK: Testiranje različnih URL-jev
    print("\n2. Korak: Testiranje dostopa do podatkov...")
    data_headers = {'Authorization': f'Bearer {access_token}'}

    uspesen_url = None
    for ime, url in MOZNI_URLJI.items():
        print(f"\nPoskušam dostopiti do: '{ime}' na naslovu {url}")
        try:
            response = requests.get(url, headers=data_headers, timeout=15)
            response.raise_for_status()

            podatki = response.json()
            st_dogodkov = len(podatki.get('features', []))

            print(f"   -> USPEH! Strežnik je vrnil {st_dogodkov} dogodkov.")
            print(f"   -> PRAVILEN URL JE: {url}")
            uspesen_url = url

            # Dodatni korak: Filtriranje
            print(f"   -> Filtriram za ključne besede: {LOKACIJE_ZA_FILTER}...")
            relevantne_zapore = [
                dogodek['properties']
                for dogodek in podatki.get('features', [])
                if any(lok.lower() in dogodek.get('properties', {}).get('opis', '').lower() for lok in LOKACIJE_ZA_FILTER)
            ]
            if relevantne_zapore:
                print(f"   -> USPEH FILTRIRANJA! Najdenih {len(relevantne_zapore)} relevantnih del na cesti.")
            else:
                print(f"   -> Filter ni našel relevantnih del v teh podatkih.")

            break # Ustavimo se, ko najdemo delujoč URL

        except requests.exceptions.HTTPError as e:
            print(f"   -> NAPAKA: {e}")
        except Exception as e:
            print(f"   -> PRIŠLO JE DO NEZNANE NAPAKE: {e}")

    if not uspesen_url:
        print("\nSKLEP: Noben od preizkušenih URL-jev ni deloval. Težava je verjetno še vedno v dovoljenjih na strani NAP portala.")

    print("\n--- KONEC TESTIRANJA ---")

if __name__ == "__main__":
    testiraj_nap_api_v2()