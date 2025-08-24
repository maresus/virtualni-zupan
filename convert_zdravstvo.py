#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pretvornik zdravstvo.txt v JSONL format za ChromaDB
"""

import json
import re

# Originalni tekst
zdravstvo_text = """
ZP Rače - referenčna ambulanta, Nova ul. 5, Rače
Nataša NASKOVSKA ILIEVSKA, dr. med., spec.
tel.: 02/63-00-877
e-pošta: marjeta.loncaric@zd-mb.si

Delovni čas za paciente:
PONEDELJEK - 7.00 – 14.00
TOREK - 12.30 – 19.30
SREDA - 7.00 – 14.00
ČETRTEK - 12.30 – 19.30
PETEK - 7.00 – 14.00

Naročanje:
PONEDELJEK - 7.00 – 11.00
TOREK - 13.00 – 17.00
SREDA - 7.00 – 11.00
ČETRTEK - 13.00 – 17.00
PETEK - 7.00 – 11.00

---

ZP Rače - referenčna ambulanta, Nova ul. 5, Rače
Marija REPOLUSK, dr. med., spec.
tel.: 02/63-00-870
e-pošta: sebastijan.sketa@zd-mb.si
PONEDELJEK - 12.30 – 19.30
TOREK - 7.00 – 14.00
SREDA - 12.30 – 19.30
ČETRTEK - 7.00 – 14.00
PETEK - 7.00 – 14.00

Hišni obiski - po predhodnem naročilu: dopoldne od 13.00 – 14.00, popoldne od 18.30 do 19.30

---

Zasebna ambulanta splošne medicine Fram,
Cafova ul. 1, Fram
Irena STROJNIK, dr. med., spec.
- splošna ambulanta
- referenčna ambulanta
tel.: 02/601-20-20
e-pošta: narocila.strojnik@gmail.com

---

Patronažna zdravstvena nega in babištvo Zlatka
Napast, d. m. s., ZP Fram, Cafova ul. 1, Fram
(Cesta talcev 16, Rače)
Zlatka NAPAST, dipl. med. sestra
gsm: 031-562-849
e-pošta: zlatka.napast@amis.net

Delovni čas: 7.30 – 14.30 (pon-pet)
Po potrebi – po dogovoru, prav tako v popoldanskem času in med prazniki

---

Patronažna zdravstvena nega in babištvo Anita
Gajšt v. m. s., ZP Fram, Cafova ul. 1, Fram
Anita GAJŠT, viš. med. sestra
gsm: 041-221-059
e-pošta: anita.gajst@siol.net

Delovni čas: 7.00 – 14.00 (pon-pet)
Dežurstvo – obiski po dogovoru, samo v nujnih primerih

---

Zasebna zobna ordinacija mag. sci. Lobnik-Gomilšek Bojana, dr. dent. med., Nova ul. 5, Rače
mag. sci. Bojana LOBNIK-GOMILŠEK, dr. dent. med.
tel.: 02/609-50-30
e-pošta: zobna.ambulanta@siol.net

PONEDELJEK - 13.00 – 19.30
TOREK - 13.00 – 19.30
SREDA - 7.00 – 13.30
ČETRTEK - 7.00 – 13.30
PETEK - 7.00 – 13.30

Naročanje v ordinacijskem času

---

MADENS d. o. o., PE Fram, Cafova ul. 1, Fram
Andrej MALEK, dr. dent. med.
tel.: 02/601-71-00
e-pošta: info@madens.eu

PONEDELJEK - 12.30 – 19.00
TOREK - 12.30 – 19.30
SREDA - 7.00 – 13.30
ČETRTEK - 7.00 – 13.30
PETEK - 7.00 – 13.30

Naročanje prve štiri ure ordinacijskega časa

---

ZD Maribor – zob. ordinacija Rače,
Nova ulica 5, Rače
Boris SAPAČ, dr. dent. med.
tel.: 02/630-08-72
e-pošta: boris.sapac@zd-mb.si

PONEDELJEK - 12.30 – 19.30
TOREK - 7.30 – 14.30
SREDA - 7.30 – 14.30
ČETRTEK - 12.30 – 19.30
PETEK - 7.30 – 14.30

Naročanje v ordinacijskem času

---

VIVAGIB, zasebna fizioterapija d.o.o.
Gozdna ul. 24, Rače
Alenka DAJČMAN, dipl. fiziot., terapevt ortopedske medicine in manualne terapije
tel.: 02/609-73-81, gsm: 040-799-733
e-pošta: vivagib@gmail.com

Delovni čas: 6.30 – 13.00 (pon-pet)
Naročanje 8.00 – 12.00 (za paciente z delovnim nalogom in kartico zdravstvenega zavarovanja)
Samoplačniške storitve opravljajo izven ordinacijskega časa – po dogovoru.

---

Lekarna Rače, Ljubljanska c. 14 A, Rače
Miran GOLUB, mag. farm.
tel.: 02/609-71-20
e-pošta: lekarna.race@siol.net

PONEDELJEK-PETEK: 8.00 – 12.30, 13.00 – 18.00
SOBOTA: 8.00 – 12.00
NEDELJA in PRAZNIKI: zaprto
"""

def create_jsonl_entries():
    """Ustvari JSONL vnose iz zdravstvenih podatkov"""
    
    entries = []
    
    # 1. SPLOŠNA MEDICINA - Dr. Nataša Naskovska Ilievska
    entries.append({
        "text": "ZP Rače - referenčna ambulanta splošne medicine. Dr. Nataša Naskovska Ilievska, specialistka. Naslov: Nova ul. 5, Rače. Telefon: 02/63-00-877. E-pošta: marjeta.loncaric@zd-mb.si",
        "metadata": {
            "kategorija": "Zdravstvo",
            "tip": "splošna medicina", 
            "zdravnik": "Nataša Naskovska Ilievska",
            "lokacija": "Rače",
            "vir": "Ordinacijski čas zdravstvenih storitev"
        }
    })
    
    entries.append({
        "text": "Ordinacijski čas dr. Nataše Naskovska Ilievska (ZP Rače): Ponedeljek 7.00-14.00, Torek 12.30-19.30, Sreda 7.00-14.00, Četrtek 12.30-19.30, Petek 7.00-14.00. Naročanje: Po 7.00-11.00, To 13.00-17.00, Sr 7.00-11.00, Če 13.00-17.00, Pe 7.00-11.00.",
        "metadata": {
            "kategorija": "Zdravstvo",
            "tip": "urnik",
            "zdravnik": "Nataša Naskovska Ilievska", 
            "lokacija": "Rače",
            "vir": "Ordinacijski čas zdravstvenih storitev"
        }
    })
    
    # 2. SPLOŠNA MEDICINA - Dr. Marija Repolusk
    entries.append({
        "text": "ZP Rače - referenčna ambulanta splošne medicine. Dr. Marija Repolusk, specialistka. Naslov: Nova ul. 5, Rače. Telefon: 02/63-00-870. E-pošta: sebastijan.sketa@zd-mb.si",
        "metadata": {
            "kategorija": "Zdravstvo",
            "tip": "splošna medicina",
            "zdravnik": "Marija Repolusk",
            "lokacija": "Rače", 
            "vir": "Ordinacijski čas zdravstvenih storitev"
        }
    })
    
    entries.append({
        "text": "Ordinacijski čas dr. Marije Repolusk (ZP Rače): Ponedeljek 12.30-19.30, Torek 7.00-14.00, Sreda 12.30-19.30, Četrtek 7.00-14.00, Petek 7.00-14.00. Hišni obiski po predhodnem naročilu: dopoldne 13.00-14.00, popoldne 18.30-19.30.",
        "metadata": {
            "kategorija": "Zdravstvo", 
            "tip": "urnik",
            "zdravnik": "Marija Repolusk",
            "lokacija": "Rače",
            "vir": "Ordinacijski čas zdravstvenih storitev"
        }
    })
    
    # 3. ZASEBNA MEDICINA - Dr. Irena Strojnik
    entries.append({
        "text": "Zasebna ambulanta splošne medicine Fram. Dr. Irena Strojnik, specialistka - splošna in referenčna ambulanta. Naslov: Cafova ul. 1, Fram. Telefon: 02/601-20-20. E-pošta: narocila.strojnik@gmail.com",
        "metadata": {
            "kategorija": "Zdravstvo",
            "tip": "zasebna medicina",
            "zdravnik": "Irena Strojnik", 
            "lokacija": "Fram",
            "vir": "Ordinacijski čas zdravstvenih storitev"
        }
    })
    
    # 4. PATRONAŽNA NEGA - Zlatka Napast
    entries.append({
        "text": "Patronažna zdravstvena nega in babištvo Zlatka Napast. Zlatka Napast, diplomirana medicinska sestra. Naslov: Cafova ul. 1, Fram (Cesta talcev 16, Rače). GSM: 031-562-849. E-pošta: zlatka.napast@amis.net",
        "metadata": {
            "kategorija": "Zdravstvo",
            "tip": "patronažna nega",
            "oseba": "Zlatka Napast",
            "lokacija": "Fram",
            "vir": "Ordinacijski čas zdravstvenih storitev"
        }
    })
    
    entries.append({
        "text": "Delovni čas patronažne sestre Zlatke Napast: 7.30-14.30 (ponedeljek do petek). Po potrebi - po dogovoru, prav tako v popoldanskem času in med prazniki.",
        "metadata": {
            "kategorija": "Zdravstvo",
            "tip": "urnik",
            "oseba": "Zlatka Napast",
            "lokacija": "Fram", 
            "vir": "Ordinacijski čas zdravstvenih storitev"
        }
    })
    
    # 5. PATRONAŽNA NEGA - Anita Gajšt  
    entries.append({
        "text": "Patronažna zdravstvena nega in babištvo Anita Gajšt. Anita Gajšt, višja medicinska sestra. Naslov: Cafova ul. 1, Fram. GSM: 041-221-059. E-pošta: anita.gajst@siol.net",
        "metadata": {
            "kategorija": "Zdravstvo",
            "tip": "patronažna nega",
            "oseba": "Anita Gajšt",
            "lokacija": "Fram",
            "vir": "Ordinacijski čas zdravstvenih storitev"  
        }
    })
    
    entries.append({
        "text": "Delovni čas patronažne sestre Anite Gajšt: 7.00-14.00 (ponedeljek do petek). Dežurstvo - obiski po dogovoru, samo v nujnih primerih.",
        "metadata": {
            "kategorija": "Zdravstvo",
            "tip": "urnik",
            "oseba": "Anita Gajšt",
            "lokacija": "Fram",
            "vir": "Ordinacijski čas zdravstvenih storitev"
        }
    })
    
    # 6. ZOBOZDRAVSTVO - Dr. Bojana Lobnik-Gomilšek
    entries.append({
        "text": "Zasebna zobna ordinacija mag. sci. Bojana Lobnik-Gomilšek, doktorica dentalne medicine. Naslov: Nova ul. 5, Rače. Telefon: 02/609-50-30. E-pošta: zobna.ambulanta@siol.net",
        "metadata": {
            "kategorija": "Zdravstvo",
            "tip": "zobozdravstvo",
            "zdravnik": "Bojana Lobnik-Gomilšek",
            "lokacija": "Rače",
            "vir": "Ordinacijski čas zdravstvenih storitev"
        }
    })
    
    entries.append({
        "text": "Ordinacijski čas dr. Bojane Lobnik-Gomilšek (zobna ordinacija, Rače): Ponedeljek 13.00-19.30, Torek 13.00-19.30, Sreda 7.00-13.30, Četrtek 7.00-13.30, Petek 7.00-13.30. Naročanje v ordinacijskem času.",
        "metadata": {
            "kategorija": "Zdravstvo",
            "tip": "urnik", 
            "zdravnik": "Bojana Lobnik-Gomilšek",
            "lokacija": "Rače",
            "vir": "Ordinacijski čas zdravstvenih storitev"
        }
    })
    
    # 7. ZOBOZDRAVSTVO - Dr. Andrej Malek
    entries.append({
        "text": "MADENS d. o. o., PE Fram - zobna ordinacija. Dr. Andrej Malek, doktor dentalne medicine. Naslov: Cafova ul. 1, Fram. Telefon: 02/601-71-00. E-pošta: info@madens.eu",
        "metadata": {
            "kategorija": "Zdravstvo",
            "tip": "zobozdravstvo",
            "zdravnik": "Andrej Malek",
            "lokacija": "Fram",
            "vir": "Ordinacijski čas zdravstvenih storitev"
        }
    })
    
    entries.append({
        "text": "Ordinacijski čas dr. Andreja Maleka (MADENS, Fram): Ponedeljek 12.30-19.00, Torek 12.30-19.30, Sreda 7.00-13.30, Četrtek 7.00-13.30, Petek 7.00-13.30. Naročanje prve štiri ure ordinacijskega časa.",
        "metadata": {
            "kategorija": "Zdravstvo",
            "tip": "urnik",
            "zdravnik": "Andrej Malek", 
            "lokacija": "Fram",
            "vir": "Ordinacijski čas zdravstvenih storitev"
        }
    })
    
    # 8. ZOBOZDRAVSTVO - Dr. Boris Sapač
    entries.append({
        "text": "ZD Maribor - zobna ordinacija Rače. Dr. Boris Sapač, doktor dentalne medicine. Naslov: Nova ulica 5, Rače. Telefon: 02/630-08-72. E-pošta: boris.sapac@zd-mb.si",
        "metadata": {
            "kategorija": "Zdravstvo",
            "tip": "zobozdravstvo",
            "zdravnik": "Boris Sapač",
            "lokacija": "Rače",
            "vir": "Ordinacijski čas zdravstvenih storitev"
        }
    })
    
    entries.append({
        "text": "Ordinacijski čas dr. Borisa Sapača (ZD Maribor, zobna ordinacija Rače): Ponedeljek 12.30-19.30, Torek 7.30-14.30, Sreda 7.30-14.30, Četrtek 12.30-19.30, Petek 7.30-14.30. Naročanje v ordinacijskem času.",
        "metadata": {
            "kategorija": "Zdravstvo",
            "tip": "urnik",
            "zdravnik": "Boris Sapač",
            "lokacija": "Rače", 
            "vir": "Ordinacijski čas zdravstvenih storitev"
        }
    })
    
    # 9. FIZIOTERAPIJA - Alenka Dajčman
    entries.append({
        "text": "VIVAGIB - zasebna fizioterapija d.o.o. Alenka Dajčman, diplomirana fizioterapevtka, terapevtka ortopedske medicine in manualne terapije. Naslov: Gozdna ul. 24, Rače. Telefon: 02/609-73-81, GSM: 040-799-733. E-pošta: vivagib@gmail.com",
        "metadata": {
            "kategorija": "Zdravstvo",
            "tip": "fizioterapija",
            "oseba": "Alenka Dajčman",
            "lokacija": "Rače",
            "vir": "Ordinacijski čas zdravstvenih storitev"
        }
    })
    
    entries.append({
        "text": "Delovni čas fizioterapevtke Alenke Dajčman (VIVAGIB): 6.30-13.00 (ponedeljek do petek). Naročanje 8.00-12.00 za paciente z delovnim nalogom in kartico zdravstvenega zavarovanja. Samoplačniške storitve opravljajo izven ordinacijskega časa po dogovoru.",
        "metadata": {
            "kategorija": "Zdravstvo",
            "tip": "urnik", 
            "oseba": "Alenka Dajčman",
            "lokacija": "Rače",
            "vir": "Ordinacijski čas zdravstvenih storitev"
        }
    })
    
    # 10. LEKARNA - Miran Golub
    entries.append({
        "text": "Lekarna Rače. Miran Golub, magister farmacije. Naslov: Ljubljanska c. 14 A, Rače. Telefon: 02/609-71-20. E-pošta: lekarna.race@siol.net",
        "metadata": {
            "kategorija": "Zdravstvo",
            "tip": "lekarna",
            "oseba": "Miran Golub",
            "lokacija": "Rače",
            "vir": "Ordinacijski čas zdravstvenih storitev"
        }
    })
    
    entries.append({
        "text": "Delovni čas Lekarne Rače: Ponedeljek-Petek 8.00-12.30 in 13.00-18.00. Sobota 8.00-12.00. Nedelja in prazniki zaprto.",
        "metadata": {
            "kategorija": "Zdravstvo", 
            "tip": "urnik",
            "oseba": "Miran Golub",
            "lokacija": "Rače",
            "vir": "Ordinacijski čas zdravstvenih storitev"
        }
    })
    
    # 11. SPLOŠNI POVZETEK
    entries.append({
        "text": "V občini Rače-Fram je na voljo celovita zdravstvena oskrba: splošna medicina (3 zdravniki), zobozdravstvo (3 zobozdravniki), patronažna nega (2 medicinski sestri), fizioterapija in lekarna. Večina storitev je v Račah, nekatere tudi v Framu.",
        "metadata": {
            "kategorija": "Zdravstvo",
            "tip": "povzetek",
            "lokacija": "Rače-Fram",
            "vir": "Ordinacijski čas zdravstvenih storitev"
        }
    })
    
    return entries

def generate_jsonl():
    """Generiraj JSONL output"""
    entries = create_jsonl_entries()
    
    jsonl_output = []
    for entry in entries:
        jsonl_output.append(json.dumps(entry, ensure_ascii=False))
    
    return "\n".join(jsonl_output)

# Generate the JSONL
if __name__ == "__main__":
    jsonl_content = generate_jsonl()
    print(jsonl_content)
    print(f"\n\n--- GENERIRANO {len(create_jsonl_entries())} JSONL VNOSOV ---")