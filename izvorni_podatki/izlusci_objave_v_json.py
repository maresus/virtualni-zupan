import requests
from bs4 import BeautifulSoup
import json
import time

# Seznam URL-jev (tvojih 31)
URLJI = [
    "https://www.race-fram.si/objava/400259",
    "https://www.race-fram.si/objava/1097301",
    "https://www.race-fram.si/objava/400260",
    "https://www.race-fram.si/objava/400259",
    "https://www.race-fram.si/objava/400314",
    "https://www.race-fram.si/objava/400309",
    "https://www.race-fram.si/objava/400307",
    "https://www.race-fram.si/objava/400299",
    "https://www.race-fram.si/objava/400303",
    "https://www.race-fram.si/objava/400301",
    "https://www.race-fram.si/objava/400274",
    "https://www.race-fram.si/objava/400266",
    "https://www.race-fram.si/objava/400295",
    "https://www.race-fram.si/objava/400264",
    "https://www.race-fram.si/objava/400277",
    "https://www.race-fram.si/objava/400294",
    "https://www.race-fram.si/objava/400276",
    "https://www.race-fram.si/objava/400265",
    "https://www.race-fram.si/objava/400293",
    "https://www.race-fram.si/objava/400296",
    "https://www.race-fram.si/objava/400297",
    "https://www.race-fram.si/objava/400270",
    "https://www.race-fram.si/objava/400273",
    "https://www.race-fram.si/objava/400278",
    "https://www.race-fram.si/objava/400272",
    "https://www.race-fram.si/objava/400271",
    "https://www.race-fram.si/objava/400275",
    "https://www.race-fram.si/objava/400312",
    "https://www.race-fram.si/objava/400269",
    "https://www.race-fram.si/objava/400268",
    "https://www.race-fram.si/objava/400262"
]

headers = {
    "User-Agent": "Mozilla/5.0"
}

rezultat = {}

for url in URLJI:
    try:
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")

        # Naslov objave
        naslov = soup.
