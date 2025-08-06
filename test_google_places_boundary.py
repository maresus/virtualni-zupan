#!/usr/bin/env python3
import os
import sys
import json
import time
import argparse
import requests
from shapely.geometry import shape, Point, LineString
from shapely import ops
from pyproj import Transformer
from collections import defaultdict

# ------------ KONFIGURACIJA ------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # mora biti v .env ali okolju
BOUNDARY_FILE = "race_fram_boundary.geojson"  # datoteka s GeoJSON mejo (FeatureCollection) ali Overpass output

# ključne storitve in kategorije
SERVICE_KEYWORDS = [
    "avtomehanik", "avto mehanik", "keramičar", "mizar",
    "vodovodar", "električar", "zdravnik", "frizer", "kozmetičarka",
    "zobozdravnik", "računalniško popravilo", "računalničar", "gradbenik",
    "inštalater", "računovodja", "pravnik", "storitve", "lokalne storitve"
]

PLACE_QUERIES = {
    "restavracije": {"type": "restaurant"},
    "bare": {"type": "bar"},
    "kavarne": {"type": "cafe"},
    "bencinski_servisi": {"type": "gas_station"},
    "trgovine": {"combine": ["store", "storitve"]},  # združi trgovine + vse storitve
    "storitve": {"keywords": SERVICE_KEYWORDS},
    "keramičarji": {"keywords": ["keramičar", "keramičarstvo", "keramičarji"]},
    "avtomehaniki": {"keywords": ["avtomehanik", "servis avtomobilov", "avto mehanik"]},
    "mizarji": {"keywords": ["mizar", "lesenar", "mizarstvo"]},
}

# ------------ POMOŽNE FUNKCIJE ------------

def load_boundary(path):
    """
    Naloži mejo iz GeoJSON FeatureCollection ali Overpass JSON in vrni shapely polygon.
    Prav tako ustvari buffered polygon (500 m) z uporabo projekcije.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    # Pridobi osnovno geometrijo
    if "features" in raw:  # standardni GeoJSON
        base_geom = shape(raw["features"][0]["geometry"])
    elif "elements" in raw:  # Overpass format: sestavi iz way-ov
        # poišči relation
        rel = next((e for e in raw["elements"] if e.get("type") == "relation"), None)
        if not rel:
            raise ValueError("V Overpass JSON ni relation.")
        # zberi vse outer ways z geometrijo
        line_strings = []
        for member in rel.get("members", []):
            if member.get("type") == "way" and member.get("role") in ("outer", "") and "geometry" in member:
                coords = [(pt["lon"], pt["lat"]) for pt in member["geometry"]]
                line_strings.append(LineString(coords))
        if not line_strings:
            raise ValueError("Ni outer way geometrij v Overpass podatkih.")
        merged = ops.linemerge(line_strings)
        polygons = list(ops.polygonize(merged))
        if not polygons:
            # fallback: združi vse kot multipolygon-like
            base_geom = ops.unary_union(line_strings).buffer(0)
        else:
            # vzemi največji
            base_geom = max(polygons, key=lambda p: p.area)
    else:
        raise ValueError("Nepodprt format meje. Pričakovan GeoJSON ali Overpass JSON.")

    # transformacija v metrični sistem (Web Mercator) za buffer
    proj_to_merc = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform
    proj_to_ll = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform

    geom_merc = ops.transform(proj_to_merc, base_geom)
    buffered_merc = geom_merc.buffer(500)  # 500 metrov
    buffered_ll = ops.transform(proj_to_ll, buffered_merc)

    return base_geom, buffered_ll, geom_merc, buffered_merc

def compute_search_center_and_radius(buffered_merc):
    """
    Izračunaj centroid in maksimalno razdaljo v metrih za začetni radius iskanja.
    Vrne: (lat, lon), radius_m
    """
    # centroid v metričnih
    centroid_merc = buffered_merc.centroid
    # poišči najdaljšo razdaljo do roba
    max_dist = 0.0
    for coord in buffered_merc.exterior.coords:
        pt = Point(coord)
        d = centroid_merc.distance(pt)
        if d > max_dist:
            max_dist = d
    radius = max_dist + 500  # malo pad
    # omeji na 50000 (Google Places max)
    radius = min(radius, 50000)
    # transformiraj centroid nazaj v lat/lon
    proj_to_ll = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform
    lon, lat = proj_to_ll(centroid_merc.x, centroid_merc.y)
    return (lat, lon), int(radius)

def fetch_nearby_places(place_type, keyword, location, radius, api_key):
    """
    Pokliče Google Places Nearby Search z danim tipom ali keywordom.
    Vrne seznam place dictov.
    """
    lat, lng = location
    base_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "radius": radius,
        "key": api_key,
    }
    if place_type:
        params["type"] = place_type
    if keyword:
        params["keyword"] = keyword

    results = []
    while True:
        resp = requests.get(base_url, params=params, timeout=10)
        if resp.status_code != 200:
            print(f"Napaka API: {resp.status_code} {resp.text}", file=sys.stderr)
            break
        data = resp.json()
        results.extend(data.get("results", []))
        if "next_page_token" in data:
            # pri čakanju, da se token aktivira
            time.sleep(2)
            params = {
                "pagetoken": data["next_page_token"],
                "key": api_key
            }
            continue
        break
    return results

def fetch_place_details(place_id, api_key):
    """
    Dobi dodatne podatke (telefon, splet) preko Place Details.
    """
    url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "fields": "formatted_phone_number,website",
        "key": api_key
    }
    resp = requests.get(url, params=params, timeout=10)
    if resp.status_code != 200:
        return {}
    data = resp.json().get("result", {})
    return {
        "phone": data.get("formatted_phone_number"),
        "website": data.get("website")
    }

def dedupe_places(places):
    seen = {}
    deduped = []
    for p in places:
        pid = p.get("place_id")
        if not pid:
            continue
        if pid in seen:
            continue
        seen[pid] = True
        deduped.append(p)
    return deduped

# ------------ GLAVNI DEL ------------
def main():
    parser = argparse.ArgumentParser(description="Iskanje krajev z Google Places omejeno na občino Rače-Fram + 500 m.")
    parser.add_argument("category", help="Kategorija (restavracije, bare, kavarne, bencinski_servisi, trgovine, storitve, keramičarji, avtomehaniki, mizarji)")
    parser.add_argument("--details", action="store_true", help="Pridobi še telefon in splet za vsak rezultat")
    args = parser.parse_args()

    if not GOOGLE_API_KEY:
        print("Ni določen GOOGLE_API_KEY v okolju.", file=sys.stderr)
        sys.exit(1)

    key = args.category.lower()
    if key not in PLACE_QUERIES:
        print(f"Ne poznam kategorije '{args.category}'. Razpoložljive: {', '.join(PLACE_QUERIES.keys())}", file=sys.stderr)
        sys.exit(1)

    # naloži mejo in buffered polygon
    try:
        base_geom, buffered_poly_ll, base_merc, buffered_merc = load_boundary(BOUNDARY_FILE)
    except Exception as e:
        print(f"Napaka pri nalaganju meje: {e}", file=sys.stderr)
        sys.exit(1)

    # izračunaj center in radius za iskanje
    (center_lat, center_lon), radius = compute_search_center_and_radius(buffered_merc)
    print(f"Iščem okoli koordinat: {center_lat:.6f}, {center_lon:.6f} z radiusom {radius} m (filtriram na občino +500m)")

    raw_places = []
    spec = PLACE_QUERIES[key]
    if "combine" in spec:
        for sub in spec["combine"]:
            if sub == "store":
                raw_places.extend(fetch_nearby_places("store", None, (center_lat, center_lon), radius, GOOGLE_API_KEY))
            elif sub == "storitve":
                # vse storitve
                for kw in SERVICE_KEYWORDS:
                    raw_places.extend(fetch_nearby_places(None, kw, (center_lat, center_lon), radius, GOOGLE_API_KEY))
    else:
        place_type = spec.get("type")
        keywords = spec.get("keywords")
        if keywords:
            for kw in keywords:
                raw_places.extend(fetch_nearby_places(None, kw, (center_lat, center_lon), radius, GOOGLE_API_KEY))
        else:
            raw_places = fetch_nearby_places(place_type, None, (center_lat, center_lon), radius, GOOGLE_API_KEY)

    # deduplikacija
    raw_places = dedupe_places(raw_places)

    # filtriraj po bufferju (lat/lon)
    filtered = []
    for p in raw_places:
        loc = p.get("geometry", {}).get("location", {})
        if "lat" not in loc or "lng" not in loc:
            continue
        point = Point(loc["lng"], loc["lat"])
        if buffered_poly_ll.contains(point):
            filtered.append(p)

    if not filtered:
        print("Ni najdenih rezultatov znotraj meje + 500 m.")
        return

    # po potrebi poberi detajle
    output = []
    for p in filtered:
        entry = {
            "name": p.get("name"),
            "address": p.get("vicinity") or p.get("formatted_address"),
            "types": p.get("types", []),
            "rating": p.get("rating"),
            "user_ratings_total": p.get("user_ratings_total"),
            "place_id": p.get("place_id"),
        }
        if args.details:
            details = fetch_place_details(p["place_id"], GOOGLE_API_KEY)
            entry["phone"] = details.get("phone")
            entry["website"] = details.get("website")
        output.append(entry)

    # sortiraj po ratingu (če obstaja), nato po imenu
    def sort_key(e):
        return (-(e["rating"] or 0), e["name"] or "")

    output.sort(key=sort_key)

    # izpiši
    for e in output:
        print("--------------------------------------------------")
        print(f"Ime: {e.get('name')}")
        print(f"Naslov: {e.get('address')}")
        if e.get("rating") is not None:
            print(f"Ocenjeno: {e.get('rating')} ({e.get('user_ratings_total', 0)} ocen)")
        print(f"Vrste: {', '.join(e.get('types', []))}")
        if args.details:
            if e.get("phone"):
                print(f"Telefon: {e.get('phone')}")
            if e.get("website"):
                print(f"Splet: {e.get('website')}")
        print(f"Place ID: {e.get('place_id')}")
    print("--------------------------------------------------")
    print(f"Skupno najdenih (po filtriranju): {len(output)}")

if __name__ == "__main__":
    main()
