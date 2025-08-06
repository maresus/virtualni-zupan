import json

# Prebere Overpass JSON (tisto, kar si dobil iz Overpass API)
with open("race_fram_boundary.osm.json", encoding="utf-8") as f:
    over = json.load(f)

# Najdi relacijo občine Rače-Fram
relation = next(
    (e for e in over.get("elements", []) if e.get("type") == "relation" and e.get("id") == 1676323),
    None
)
if not relation:
    raise RuntimeError("Relacija Rače-Fram ni najdena v Overpass JSON-u.")

# Zberi vse outer 'way' geometrije iz članov relacije
outer_ways = [m for m in relation.get("members", []) if m.get("type") == "way" and m.get("role") == "outer"]

# Sestavi poligone: vsak way kot lista [ [lon, lat], ... ]
polygons = []
for way in outer_ways:
    geometry = way.get("geometry", [])
    coords = [(pt["lon"], pt["lat"]) for pt in geometry]
    if not coords:
        continue
    # Zapri poligon, če ni že zaprt
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    polygons.append(coords)

# GeoJSON format: MultiPolygon zahteva strukturo [[[ [lon,lat], ... ]]], zato vsak polygon v svoji lupini
geojson = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"name": "Rače-Fram"},
            "geometry": {
                "type": "MultiPolygon",
                "coordinates": [[polygon] for polygon in polygons]
            }
        }
    ]
}

# Shrani kot veljaven GeoJSON
with open("race_fram_boundary.geojson", "w", encoding="utf-8") as f:
    json.dump(geojson, f, ensure_ascii=False, indent=2)

print("Uspešno ustvarjen race_fram_boundary.geojson")
