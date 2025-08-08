# zupan_smoke_tests_v50.py
import re
import sys
import time
import argparse
from typing import List, Dict, Any

def bail(msg: str, code: int = 2):
    print(msg, file=sys.stderr)
    sys.exit(code)

# Dinamični import VirtualniZupan z jasnimi napakami
try:
    from VIRT_ZUPAN_RF_api import VirtualniZupan  # tvoja glavna skripta
except ModuleNotFoundError as e1:
    # Če manjka requests/chromadb itd., povej uporabniku naj namesti requirements
    if "No module named 'requests'" in str(e1):
        bail("Manjka paket 'requests' (in verjetno še drugi). Zaženi:\n"
             "  python -m pip install -r requirements.txt")
    try:
        from VIRT_ZUPAN_core_v50 import VirtualniZupan
    except ModuleNotFoundError:
        try:
            from VIRT_ZUPAN_core import VirtualniZupan
        except ModuleNotFoundError as e3:
            bail(
                "Ne najdem razreda VirtualniZupan.\n"
                "- Preveri, da je datoteka 'VIRT_ZUPAN_RF_api.py' v isti mapi kot ta test.\n"
                "- Namesti pakete: python -m pip install -r requirements.txt\n"
                f"Originalni import error: {e1}"
            )

OK = "\x1b[32mOK\x1b[0m"
FAIL = "\x1b[31mFAIL\x1b[0m"

def contains_any(text: str, needles: List[str]) -> bool:
    t = text.lower() if text else ""
    return any(n.lower() in t for n in needles)

def contains_all(text: str, needles: List[str]) -> bool:
    t = text.lower() if text else ""
    return all(n.lower() in t for n in needles)

def contains_date(text: str) -> bool:
    # npr. 12.3. ali 12.03.
    return re.search(r"\b\d{1,2}\.\d{1,2}\.\b", text or "") is not None

def run_case(zupan: VirtualniZupan, case: Dict[str, Any], show_output: bool = False) -> bool:
    name = case["name"]
    session_id = f"test_{name}_{int(time.time()*1000)}"
    passed_all = True

    print(f"• {name} …", end=" ", flush=True)

    for i, q in enumerate(case["queries"]):
        try:
            out = zupan.odgovori(q, session_id=session_id)
        except Exception as e:
            print(f"\n  {FAIL} — exception pri '{q}': {e}")
            return False

        if show_output:
            print(f"\n  Q{i+1}: {q}\n  ---\n  {out}\n  ---")

        # MUST (all)
        for token in case.get("must_all", []):
            if not contains_all(out, [token]):
                print(f"\n  {FAIL} — manjka obvezni niz: '{token}'")
                if not show_output:
                    print(f"  Odgovor: {out}")
                passed_all = False

        # MUST (any)
        must_any = case.get("must_any", [])
        if must_any:
            if not contains_any(out, must_any):
                print(f"\n  {FAIL} — noben od pričakovanih nizov ni prisoten: {must_any}")
                if not show_output:
                    print(f"  Odgovor: {out}")
                passed_all = False

        # MUST date
        if case.get("must_date", False):
            if not contains_date(out):
                print(f"\n  {FAIL} — pričakovani datum (npr. 12.3.) ni zaznan.")
                if not show_output:
                    print(f"  Odgovor: {out}")
                passed_all = False

        # MUST NOT
        for bad in case.get("must_not", []):
            if contains_any(out, [bad]):
                print(f"\n  {FAIL} — ne sme se pojaviti: '{bad}'")
                if not show_output:
                    print(f"  Odgovor: {out}")
                passed_all = False

        # regex MUST (all)
        for rx in case.get("must_regex_all", []):
            if not re.search(rx, out or "", flags=re.IGNORECASE):
                print(f"\n  {FAIL} — regex '{rx}' se ni ujemal.")
                if not show_output:
                    print(f"  Odgovor: {out}")
                passed_all = False

    if passed_all:
        print(OK)
    return passed_all

def main(show_output: bool = False, only: str | None = None) -> int:
    zupan = VirtualniZupan()

    tests: List[Dict[str, Any]] = [
        # 1) Odpadki — steklo na Mlinska ulica
        {
            "name": "odpadki_steklo_mlinski",
            "queries": ["kdaj je odvoz steklene embalaže na mlinski ulici"],
            "must_all": ["mlinska", "stekl"],  # robustno: 'stekl' lovi 'Steklena'
            "must_not": ["odpadna embalaža"],   # ne zamenjaj za rumene kante
        },
        # 2) Odpadki — mešani Pod terasami
        {
            "name": "odpadki_mesani_pod_terasami",
            "queries": ["kdaj je odvoz mešanih odpadkov pod terasami"],
            "must_all": ["pod terasami", "mešan"],
        },
        # 3) Odpadki — BISTRIŠKA diakritika
        {
            "name": "odpadki_bistriska_diacritics",
            "queries": [
                "kdaj je odvoz stekla na bistriski cesti",
                "kdaj je odvoz stekla na bistriški cesti",
            ],
            "must_all": ["bistri", "stekl"],
        },
        # 4) Odpadki — naslednji datum (papir, Pod terasami)
        {
            "name": "odpadki_next_papir_pod_terasami",
            "queries": ["kdaj je naslednji odvoz papirja pod terasami"],
            "must_all": ["naslednji odvoz", "pod terasami", "papir"],
            "must_date": True,
        },
        # 5) Vloga za zaporo — nikoli promet
        {
            "name": "vloga_zapora_form",
            "queries": [
                "kje najdem vlogo za zaporo ceste",
                "kje najdme vlogo za zaprtje ceste?",
            ],
            "must_any": ["vloga za zaporo ceste", "obrazec", "povezav"],
            "must_regex_all": [r"400297|zapora\-ceste"],
            "must_not": ["promet.si", "a1", "zastoj", "nap.si", "roadworks", "geojson"],
        },
        # 6) Stanje prometa — izpis zapor, ne obrazec
        {
            "name": "promet_status",
            "queries": ["ali v občini potekajo kakšna dela na cesti?"],
            "must_any": ["našel sem", "cesta", "opis", "promet.si"],
            "must_not": ["vloga za zaporo ceste"],
        },
        # 7) Župan — enostaven stavek
        {
            "name": "who_is_mayor",
            "queries": ["kdo je župan občine"],
            "must_any": ["samo rajšp", "samo rajsp", "rajšp"],
            "must_not": ["direktorica", "uprav", "malica", "odvoz"],
        },
        # 8) Nagrade — Županova petica 2012 (ne sme vrniti “župan je…”)
        {
            "name": "awards_petica_2012",
            "queries": ["kdo so dobitniki županove petice za 2012"],
            "must_any": ["2012", "petic", "zapis o **županovi petici**", "zapis o županovi petici"],
            "must_not": ["župan občine rače-fram je"],
        },
        # 9) PGD — vsa društva + kontakti
        {
            "name": "pgd_all",
            "queries": ["katera gasilska društva imamo v občini?"],
            "must_all": ["pgd rače", "pgd fram", "pgd podova", "pgd spodnja in zgornja gorica"],
        },
        # 10) PGD poveljstvo — naj bo “poveljnik” v izpisu
        {
            "name": "pgd_poveljstvo",
            "queries": ["daj mi poveljniške podatke pgd"],
            "must_any": ["poveljnik"],
        },
        # 11) Gradbeno — uradni eUprava link v odgovoru
        {
            "name": "building_permit_euprava",
            "queries": ["kako dobim gradbeno dovoljenje"],
            "must_any": ["e-uprava.gov.si", "gradbeno-dovoljenje"],
        },
        # 12) Fram info — brez gasilskih/linkov na PGD
        {
            "name": "fram_info_clean",
            "queries": ["povej mi nekaj informacij o framu"],
            "must_any": ["fram", "pohor", "framsk"],
            "must_not": ["pgd", "gasil"],
        },
        # 13) Poletni kamp — brez 2024 (razen če uporabnik vpraša izrecno)
        {
            "name": "summer_camp_no_old_years",
            "queries": ["ali imamo v občini poletni kamp?"],
            "must_not": ["2024"],
        },
    ]

    if only:
        tests = [t for t in tests if only.lower() in t["name"].lower()]
        if not tests:
            print(f"Ni testa z imenom, ki vsebuje: '{only}'")
            return 1

    total = len(tests)
    ok = 0
    for case in tests:
        if run_case(zupan, case, show_output=show_output):
            ok += 1

    print(f"\nRezultat: {ok}/{total} OK")
    return 0 if ok == total else 1

def cli() -> int:
    parser = argparse.ArgumentParser(description="Smoke testi za VirtualniŽupan")
    parser.add_argument("--show", action="store_true", help="Pokaži celotne odgovore za vsako poizvedbo.")
    parser.add_argument("--only", type=str, default=None, help="Zaženi le teste, katerih ime vsebuje ta niz.")
    args = parser.parse_args()
    return main(show_output=args.show, only=args.only)

if __name__ == "__main__":
    sys.exit(cli())
