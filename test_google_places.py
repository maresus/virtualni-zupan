#!/usr/bin/env python3
import os
import sys
import json
import argparse
import time
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher

import requests
from shapely.geometry import shape, Point
from shapely import ops
from pyproj import Transformer

# -------------------- KONFIGURACIJA --------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # mora biti v okolju
if not GOOGLE_API_KEY:
    print("Napaka: nastavi okoljsko spremenljivko GOOGLE_API_KEY z veljavnim ključem.")
    sys.exit(1)

TOP_N = 10  # koliko najboljših vrnemo
SEARCH_RADIUS = 1500  # privzeti radius za non-legacy mode (m)
MAX_SEED_CENTERS = 9  # koliko centrov generira grid

# -------------------- NALOŽI MEJO OBČINE --------------------
with open("race_fram_boundary.geojson", encoding="utf-8") as f:
    geo = json.load(f)

boundary_geom = shape(geo["features"][0]["geometry"])  # WGS84 (lon, lat)

to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32633", always_xy=True)
from_utm = Transformer.from_crs("EPSG:32633", "EPSG:4326", always_xy=True)

def project_to_utm(geom):
    return ops.transform(lambda x, y: to_utm.transform(x, y), geom)

boundary_utm = project_to_utm(boundary_geom)
boundary_buffer_utm = boundary_utm.buffer(500)
boundary_buffer_geo = ops.transform(lambda x, y: from_utm.transform(x, y), boundary_buffer_utm)

def is_within_race_fram_plus_500m(lat: float, lon: float) -> bool:
    pt = Point(lon, lat)
    return boundary_buffer_geo.contains(pt) or boundary_geom.contains(pt)

# -------------------- POMOŽNE FUNKCIJE --------------------
def normalize_name(name: str) -> str:
    return "".join(c for c in (name or "").lower() if c.isalnum() or c.isspace()).strip()

def dedupe_similar(places, distance_threshold_m=30, name_threshold=0.85):
    clusters = []
    used = set()

    def score_key(p):
        return ((p.get("rating") or 0), (p.get("user_ratings_total") or 0))

    sorted_places = sorted(places, key=lambda p: score_key(p), reverse=True)
    for p in sorted_places:
        pid = p.get("place_id")
        if pid in used:
            continue
        group = [p]
        used.add(pid)
        name_p = normalize_name(p.get("name", ""))
        loc = p.get("geometry", {}).get("location", {})
        lat_p, lon_p = loc.get("lat"), loc.get("lng")
        if lat_p is None or lon_p is None:
            clusters.append(p)
            continue
        x1, y1 = to_utm.transform(lon_p, lat_p)
        for q in sorted_places:
            qid = q.get("place_id")
            if qid in used or qid == pid:
                continue
            name_q = normalize_name(q.get("name", ""))
            loc_q = q.get("geometry", {}).get("location", {})
            lat_q, lon_q = loc_q.get("lat"), loc_q.get("lng")
            if lat_q is None or lon_q is None:
                continue
            x2, y2 = to_utm.transform(lon_q, lat_q)
            dist = math.hypot(x1 - x2, y1 - y2)
            name_sim = SequenceMatcher(None, name_p, name_q).ratio()
            if dist <= distance_threshold_m and name_sim >= name_threshold:
                group.append(q)
                used.add(qid)
        clusters.append(group[0])
    return clusters

def sort_and_take_top(places, top_n=TOP_N):
    def key(p):
        rating = p.get("rating") or 0
        count = p.get("user_ratings_total") or 0
        return (-rating, -count)
    unique = dedupe_similar(places)
    sorted_list = sorted(unique, key=key)
    return sorted_list[:top_n]

# -------------------- GOOGLE PLACES API --------------------
def google_nearby_search(lat, lon, place_type=None, keyword=None, radius=None, page_token=None):
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "key": GOOGLE_API_KEY,
        "location": f"{lat},{lon}",
        "radius": radius or SEARCH_RADIUS,
    }
    if place_type:
        params["type"] = place_type
    if keyword:
        params["keyword"] = keyword
    if page_token:
        params["pagetoken"] = page_token

    resp = requests.get(url, params=params, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(f"Google Places error: {resp.status_code} {resp.text}")
    return resp.json()

def run_type_nearby(lat, lon, typ, radius, debug=False):
    out = []
    resp = google_nearby_search(lat, lon, place_type=typ, radius=radius)
    if debug:
        print(f"[DEBUG] nearby type={typ} center=({lat},{lon}) raw count:", len(resp.get("results", [])))
    out.extend(resp.get("results", []))
    token = resp.get("next_page_token")
    if token:
        time.sleep(2)
        resp2 = google_nearby_search(lat, lon, place_type=typ, radius=radius, page_token=token)
        out.extend(resp2.get("results", []))
    return out

def run_text_nearby(lat, lon, text, radius, debug=False):
    out = []
    resp = google_nearby_search(lat, lon, keyword=text, radius=radius)
    if debug:
        print(f"[DEBUG] nearby keyword='{text}' center=({lat},{lon}) raw count:", len(resp.get("results", [])))
    out.extend(resp.get("results", []))
    token = resp.get("next_page_token")
    if token:
        time.sleep(2)
        resp2 = google_nearby_search(lat, lon, keyword=text, radius=radius, page_token=token)
        out.extend(resp2.get("results", []))
    return out

# -------------------- SEED CENTRI --------------------
def generate_seed_centers(max_centers=MAX_SEED_CENTERS):
    minx, miny, maxx, maxy = boundary_buffer_geo.bounds  # lon/lat
    lat_step = 0.01
    lon_step = 0.01
    candidates = []
    lat = miny
    while lat <= maxy:
        lon = minx
        while lon <= maxx:
            if is_within_race_fram_plus_500m(lat, lon):
                candidates.append((lat, lon))
            lon += lon_step
        lat += lat_step
    centroid = boundary_geom.centroid
    rep = boundary_geom.representative_point()
    seeds = [(centroid.y, centroid.x), (rep.y, rep.x)]
    for c in candidates:
        if c not in seeds:
            seeds.append(c)
        if len(seeds) >= max_centers:
            break
    return seeds[:max_centers]

# -------------------- DEFINICIJE POIZVEDB --------------------
PREDEFINED = {
    "bar": {
        "types": ["restaurant", "bar", "cafe"],
        "texts": ["kavarna", "lokal", "bar", "pijača", "drink bar", "gostilna"]
    },
    "restavracija": {
        "types": ["restaurant"],
        "texts": ["gostilna", "jedilnica", "pizzerija", "restavracija", "lokal s hrano"]
    },
    "cafe": {
        "types": ["cafe"],
        "texts": ["kavarna", "coffee shop", "espresso", "barista"]
    },
    "turisticne_kmetije": {
        "texts": ["turistična kmetija", "agroturizem", "turizem na kmetiji"],
        "target_min": 3
    },
    "trgovine": {
        "types": [
            "grocery_or_supermarket", "convenience_store", "department_store",
            "hardware_store", "home_goods_store", "clothing_store", "electronics_store",
            "furniture_store", "pet_store", "bakery", "store"
        ]
    },
}

# -------------------- ISKANJE --------------------
def perform_search_for_spec(spec, seed_centers, debug=False, no_boundary=False, target_min=None):
    collected = {}
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = []
        for lat, lon in seed_centers:
            if "types" in spec:
                for t in spec["types"]:
                    futures.append(executor.submit(run_type_nearby, lat, lon, t, SEARCH_RADIUS, debug))
            if "type" in spec:
                futures.append(executor.submit(run_type_nearby, lat, lon, spec["type"], SEARCH_RADIUS, debug))
            if "texts" in spec:
                for txt in spec["texts"]:
                    futures.append(executor.submit(run_text_nearby, lat, lon, txt, SEARCH_RADIUS, debug))

        for fut in as_completed(futures):
            try:
                results = fut.result()
            except Exception as e:
                print(f"opozorilo: napaka pri poizvedbi: {e}")
                continue
            for p in results:
                pid = p.get("place_id")
                if not pid:
                    continue
                loc = p.get("geometry", {}).get("location", {})
                lat_p, lon_p = loc.get("lat"), loc.get("lng")
                if lat_p is None or lon_p is None:
                    continue
                if not no_boundary and not is_within_race_fram_plus_500m(lat_p, lon_p):
                    continue
                existing = collected.get(pid)
                if existing:
                    def score(x): return ((x.get("rating") or 0), (x.get("user_ratings_total") or 0))
                    if score(p) > score(existing):
                        collected[pid] = p
                else:
                    collected[pid] = p
            if target_min:
                uniq = dedupe_similar(list(collected.values()))
                if len(uniq) >= target_min:
                    break
    return list(collected.values())

# -------------------- LEGACY NAČIN (en center + velik radij) --------------------
def legacy_search(key, spec, center_lat, center_lon, legacy_radius, debug=False, no_boundary=False, target_min=None):
    collected = {}
    type_list = []
    if "types" in spec:
        type_list.extend(spec["types"])
    if "type" in spec:
        type_list.append(spec["type"])
    text_list = spec.get("texts", [])

    print(f"Iščem po tipih: {type_list} z začetno lokacijo ({center_lat}, {center_lon}) in radijem {legacy_radius} m\n")
    # po tipu
    for t in type_list:
        print(f"--- poizvedba za type='{t}' ---")
        results = run_type_nearby(center_lat, center_lon, t, legacy_radius, debug)
        print(f"  Najdenih (pred filtrom): {len(results)}")
        for p in results:
            pid = p.get("place_id")
            if not pid:
                continue
            loc = p.get("geometry", {}).get("location", {})
            lat_p, lon_p = loc.get("lat"), loc.get("lng")
            if lat_p is None or lon_p is None:
                continue
            if not no_boundary and not is_within_race_fram_plus_500m(lat_p, lon_p):
                continue
            existing = collected.get(pid)
            if existing:
                def score(x): return ((x.get("rating") or 0), (x.get("user_ratings_total") or 0))
                if score(p) > score(existing):
                    collected[pid] = p
            else:
                collected[pid] = p

    # po tekstu (fallback/razširjeno)
    for txt in text_list:
        print(f"--- poizvedba za keyword='{txt}' ---")
        results = run_text_nearby(center_lat, center_lon, txt, legacy_radius, debug)
        print(f"  Najdenih (pred filtrom): {len(results)}")
        for p in results:
            pid = p.get("place_id")
            if not pid:
                continue
            loc = p.get("geometry", {}).get("location", {})
            lat_p, lon_p = loc.get("lat"), loc.get("lng")
            if lat_p is None or lon_p is None:
                continue
            if not no_boundary and not is_within_race_fram_plus_500m(lat_p, lon_p):
                continue
            existing = collected.get(pid)
            if existing:
                def score(x): return ((x.get("rating") or 0), (x.get("user_ratings_total") or 0))
                if score(p) > score(existing):
                    collected[pid] = p
            else:
                collected[pid] = p

    places = list(collected.values())
    filtered = [p for p in places if no_boundary or is_within_race_fram_plus_500m(p["geometry"]["location"]["lat"], p["geometry"]["location"]["lng"])]
    print(f"\n***** Filtrirano: {len(filtered)} objektov znotraj občine Rače-Fram +500 m *****\n")
    top = sort_and_take_top(filtered, TOP_N)
    if not top:
        print("Ni najdenih ustreznih mest.")
        return
    for p in top:
        name = p.get("name", "brez imena")
        rating = p.get("rating", "n/a")
        total = p.get("user_ratings_total", 0)
        types = ", ".join(p.get("types", []))
        loc = p.get("geometry", {}).get("location", {})
        lat_p, lon_p = loc.get("lat"), loc.get("lng")
        vicinity = p.get("vicinity") or p.get("formatted_address", "ni naslova")
        print(f"- {name} ({types})")
        print(f"  Naslov: {vicinity}")
        print(f"  Lokacija: {lat_p}, {lon_p}")
        print(f"  Ocena: {rating} (št. ocen: {total})")
        print(f"  place_id: {p.get('place_id')}\n")

# -------------------- GLAVNI KLIC --------------------
def query_and_display(key, legacy=False, center=None, legacy_radius=8000, debug=False, no_boundary=False):
    key = key.lower()
    if key not in PREDEFINED:
        print(f"Ne razumem poizvedbe '{key}'. Na voljo: {', '.join(PREDEFINED.keys())}")
        return

    spec = PREDEFINED[key]
    if legacy:
        if center:
            try:
                lat_str, lon_str = center.split(",")
                center_lat = float(lat_str.strip())
                center_lon = float(lon_str.strip())
            except:
                print("Nepravilen format za --center. Uporabi: --center LAT,LON")
                return
        else:
            centroid = boundary_geom.centroid
            center_lat, center_lon = centroid.y, centroid.x
        legacy_search(key, spec, center_lat, center_lon, legacy_radius, debug=debug, no_boundary=no_boundary, target_min=spec.get("target_min"))
        return

    # nov način: grid seed centers
    seed_centers = generate_seed_centers()
    all_places = perform_search_for_spec(spec, seed_centers, debug=debug, no_boundary=no_boundary, target_min=spec.get("target_min"))
    top = sort_and_take_top(all_places, TOP_N)
    if not top:
        print("Ni najdenih ustreznih mest.")
        return
    print(f"\nNajdenih {len(top)} najboljših mest (znotraj{' brez' if no_boundary else ''} omejitve):\n")
    for p in top:
        name = p.get("name", "brez imena")
        rating = p.get("rating", "n/a")
        total = p.get("user_ratings_total", 0)
        types = ", ".join(p.get("types", []))
        vicinity = p.get("vicinity") or p.get("formatted_address", "ni naslova")
        print(f"- {name} (⭐ {rating}, {total} ocen)")
        print(f"    *Naslov:* {vicinity}")
        print(f"    *Tipi:* {types}")
        print(f"    *Place ID:* {p.get('place_id')}")
        print("")

def main():
    parser = argparse.ArgumentParser(description="Iskanje POI znotraj občine Rače-Fram +500m.")
    parser.add_argument("query", help="bar, restavracija, cafe, turisticne_kmetije, trgovine")
    parser.add_argument("--debug", action="store_true", help="Izpiše surove Google odgovore za diagnostiko")
    parser.add_argument("--no-boundary-filter", action="store_true", dest="no_boundary", help="Ne filtriraj znotraj meje")
    parser.add_argument("--legacy", action="store_true", help="Uporabi en center + velik radij (kot prej)")
    parser.add_argument("--center", type=str, help="LAT,LON za legacy način overridanje (npr. 46.448777,15.645564)")
    parser.add_argument("--legacy-radius", type=int, default=8000, help="Radij v metrih za legacy način")
    args = parser.parse_args()
    query_and_display(args.query, legacy=args.legacy, center=args.center, legacy_radius=args.legacy_radius, debug=args.debug, no_boundary=args.no_boundary)

if __name__ == "__main__":
    main()
