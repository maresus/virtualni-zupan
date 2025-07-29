import os
import time
import csv
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from urllib.parse import urljoin
from webdriver_manager.chrome import ChromeDriverManager
import re

# Mape za shranjevanje
os.makedirs("objave", exist_ok=True)
os.makedirs("priloge", exist_ok=True)

# Vsi URL-ji za obdelavo
URLJI = [
      "https://www.osfram.si/prevozi/",
]




chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

def shrani_prilogo(url_priloge, naziv_datoteke):
    try:
        odgovor = requests.get(url_priloge, timeout=20)
        odgovor.raise_for_status()
        with open(naziv_datoteke, "wb") as f:
            f.write(odgovor.content)
        print(f"📥 Shranjeno: {naziv_datoteke}")
        return True
    except Exception as e:
        print(f"❌ Napaka pri prenosu: {url_priloge} -> {e}")
        return False

def pridobi_vsebino_iz_div(driver):
    try:
        elementi = driver.find_elements(By.CSS_SELECTOR, "div.opis.obogatena_vsebina.colored_links")
        if elementi:
            vsebine = [el.text.strip() for el in elementi if el.text.strip()]
            if vsebine:
                return "\n\n".join(vsebine)
    except Exception:
        pass
    return None

def pridobi_obicno_vsebino(driver):
    try:
        telo = driver.find_element(By.TAG_NAME, "body")
        return telo.text.strip()
    except Exception:
        return ""

def obdelaj_url(url):
    print(f"▶️ Obiskujem (Selenium): {url}")
    driver.get(url)
    time.sleep(3)

    id_str = re.findall(r"(\d+)", url)
    id_ = id_str[0] if id_str else "neznan"

    vsebina = pridobi_vsebino_iz_div(driver)
    if not vsebina:
        vsebina = pridobi_obicno_vsebino(driver)

    vsebina += f"\n\nURL: {url}"

    txt_pot = f"objave/{id_}.txt"
    with open(txt_pot, "w", encoding="utf-8") as f:
        f.write(vsebina)
    print(f"✅ Shrani {txt_pot}")

    priloge_stevilo = 0
    elementi = driver.find_elements(By.CSS_SELECTOR, "div.files a")
    for a in elementi:
        href = a.get_attribute("href")
        if not href:
            continue
        href = href.strip()
        if href.startswith("/"):
            href = urljoin(url, href)

        if any(href.lower().endswith(ext) for ext in [".pdf", ".doc", ".docx"]) or "DownloadFile?id=" in href:
            basename = os.path.basename(href)
            filename = f"{id_}_{basename}"
            filename = filename.replace("?", "_").replace("&", "_")
            pot_priloge = os.path.join("priloge", filename)

            shrani_prilogo(href, pot_priloge)
            priloge_stevilo += 1

    print(f"📄 PDF in DOCX prilog: {priloge_stevilo}")

for url in URLJI:
    try:
        obdelaj_url(url)
    except Exception as e:
        print(f"❌ Napaka pri obdelavi URL: {url} -> {e}")

driver.quit()
print("✅ Zajem končan.")
