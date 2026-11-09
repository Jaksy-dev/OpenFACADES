#!/usr/bin/env python3
"""
Tag buildings in an OSM extract with facade data from a CSV.

Reads a CSV (building_id, levels/floors, surface_material,
construction_material, facade_colour, colour_agreement, ...) and writes
an .osc (OSM change) file with modified objects, then applies it with
`osmium apply-changes`.

Semantics (important):
- CSV building_ids carry the OSM element type as prefix: 'w123' (way),
  'r123' (relation), 'n123' (node, skipped: no footprint). Ids without a
  prefix (older pipeline output) are looked up as a way first, then as a
  relation. Non-OSM ids (Overture UUIDs) are skipped.
- Only ADDS tags that are missing on the object. Existing tags are never
  modified or removed (checked against the actual OSM data).
- A <modify> block carries the object's complete new state: all node refs
  / members and all tags (otherwise apply-changes would destroy geometry).
- The version attribute is bumped (+1) from the object's real version in
  the data file, otherwise apply-changes silently ignores the change.
- Only tags OSM2World actually reads are written (verified against
  OSM2World source):
    building:levels   <- levels / floors           (LevelAndHeightData)
    building:material <- surface_material          (BuildingPart.createWallMaterial)
                         (fallback: construction_material)
    building:colour   <- facade_colour             (same, + CSS/BUILDING colors)
- Value normalization:
    levels: integer >= 1
    material: lowercase, spaces -> underscores, whitelist of materials
              OSM2World can resolve (buildMaterial / surfaceMaterialMap /
              PropertyStyle default materials)
    colour: lowercase, spaces removed (CSS names are single words:
            "light green" -> "lightgreen"), whitelist of colors OSM2World's
            BUILDING_COLORS/CSS_COLORS can parse; known color words are
            salvaged out of LLM junk sentences; anything else is dropped.

- Note for rendering (default OSM2World style): the BRICK texture is not
  colorable, so building:colour has no visible effect on brick walls.
  "plaster" has no dedicated material and renders as the default
  (plaster-textured, colorable) building wall.

Pipeline step 7 (after run_vlm.py). The input cutout is never modified; a
tagged copy is written next to it:
  python run_osm_tags.py output/bbox_...
    reads  <bbox_dir>/03_vlm/buildings.csv
           <bbox_dir>/01_data/extract.osm
    writes <bbox_dir>/04_osm/extract_tags.osc
           <bbox_dir>/04_osm/extract_tagged.osm

Standalone usage:
  python run_osm_tags.py --csv buildings.csv --osm extract.osm.pbf \
      [--out extract_tagged.osm.pbf] [--min-agreement 0.0] \
      [--dry-run] [--osc-only] [--include-untagged]
"""

import os

import argparse
import csv
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# OSM2World-supported values (verified against OSM2World source code)
# ---------------------------------------------------------------------------

# building:material values OSM2World can turn into a material:
# BuildingPart.buildMaterial branches (brick, glass, wood, ...) plus
# DefaultMaterials.surfaceMaterialMap plus the default materials exposed
# through PropertyStyle (STONE, SANDSTONE, CONCRETE, ... - resolved
# case-insensitively).
KNOWN_MATERIALS = {
    "adobe", "clay", "mud", "rammed_earth",
    "brick", "glass", "mirror", "copper", "thatch", "reed",
    "wood", "bamboo",
    "concrete", "concrete:plates", "stone", "granite", "marble",
    "sandstone", "limestone", "slate", "metal", "steel", "iron",
    "tiles", "plaster", "sand", "dirt", "earth", "gravel",
    "pebblestone", "rock", "asphalt", "paving_stones", "sett",
    "cobblestone", "unhewn_cobblestone", "woodchips", "plastic",
    "scrub", "grass", "carpet", "tartan", "ice", "snow", "salt",
    "shells", "laterite",
}

# Colours OSM2World parses for building walls: the BUILDING_COLORS switch
# (white, black, grey/gray, red, green, blue, yellow, pink, orange, brown)
# falls back to CSS_COLORS, so every CSS colour name is valid.
KNOWN_COLOURS = {
    # BUILDING_COLORS
    "white", "black", "grey", "gray", "red", "green", "blue",
    "yellow", "pink", "orange", "brown",
    # common CSS colours (single-word names only)
    "beige", "gold", "goldenrod", "burlywood", "bisque", "cornsilk",
    "ivory", "khaki", "lavender", "linen", "peru", "saddlebrown",
    "sienna", "tan", "teal", "turquoise", "violet", "wheat",
    "aliceblue", "antiquewhite", "azure", "coral", "crimson",
    "darkred", "firebrick", "gainsboro", "ghostwhite", "honeydew",
    "indianred", "navajowhite", "rosybrown", "salmon", "seashell",
    "silver", "skyblue", "snow", "steelblue", "whitesmoke",
    "lightgreen", "lightblue", "lightgray", "lightgrey",
    "darkgreen", "darkblue", "darkgray", "darkgrey",
    "lightyellow", "lemonchiffon", "papayawhip", "blanchedalmond",
    "sandybrown", "chocolate", "mistyrose", "peachpuff", "maroon",
    "navy", "olive", "plum", "purple", "magenta", "fuchsia", "aqua",
    "indigo", "orchid",
}

# CSV column aliases (first present, non-empty column wins)
LEVEL_COLS = ["levels", "floors", "building:levels"]
MATERIAL_COLS = ["surface_material", "facade_material", "building_material"]
CONSTRUCTION_COLS = ["construction_material", "alternate_construction_material"]
COLOUR_COLS = ["facade_colour", "facade_color", "colour", "color",
               "building_colour", "building_color"]
AGREEMENT_COLS = ["colour_agreement", "color_agreement", "agreement"]

INVALID_VALUES = {"", "nan", "none", "null", "unknown", "n/a", "-", "na"}


def first_value(row, cols):
    for c in cols:
        v = row.get(c)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return None


def norm_id(building_id):
    """
    Normalize a building id to 'w123' / 'r123' / 'n123', or a bare '123'
    when the type is unknown (legacy data). Accepts n/w/r, node/way/relation
    prefixes (optionally with '/'), and '.0' float suffixes. Returns None for
    non-OSM ids such as Overture UUIDs.
    """
    bid = str(building_id).strip().lower()
    m = re.fullmatch(r"(?:(node|way|relation|n|w|r)\s*[/_]?\s*)?(\d+)(?:\.0+)?", bid)
    if not m:
        return None
    return (m.group(1)[0] if m.group(1) else "") + m.group(2)


def split_id(key):
    """'w123' -> ('w', '123'); '123' -> ('', '123')."""
    return (key[0], key[1:]) if key[0].isalpha() else ("", key)


def norm_levels(value):
    if value is None:
        return None
    v = value.strip().lower()
    if v in INVALID_VALUES:
        return None
    m = re.match(r"^(\d+)", v)
    if not m:
        return None
    n = int(m.group(1))
    if n < 1 or n > 200:
        return None
    return str(n)


def norm_material(value):
    if value is None:
        return None
    v = value.strip().lower().replace(" ", "_")
    if v in INVALID_VALUES or v not in KNOWN_MATERIALS:
        return None
    return v


def norm_colour(value, dropped):
    """Normalize to a single-word CSS colour OSM2World can parse."""
    if value is None:
        return None
    v = value.strip().lower()
    if v in INVALID_VALUES:
        return None
    # CSS colour names contain no spaces: "light green" -> "lightgreen"
    compact = re.sub(r"[\s_]+", "", v)
    if compact in KNOWN_COLOURS:
        return compact
    # salvage a known colour word out of LLM junk ("...facade is beige")
    words = re.findall(r"[a-z]+", v)
    for w in words:
        if w in KNOWN_COLOURS:
            return w
    dropped.append(value)
    return None


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------

def read_csv(path, min_agreement):
    """Return ({id: {tag: value}}, stats Counter)."""
    tag_data = {}
    stats = Counter()
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "building_id" not in (reader.fieldnames or []):
            sys.exit("CSV is missing the building_id column")
        for row in reader:
            stats["csv_rows"] += 1
            bid = norm_id(row.get("building_id"))
            if bid is None:
                stats["bad_id"] += 1
                continue
            agr_raw = first_value(row, AGREEMENT_COLS)
            if agr_raw is not None:
                try:
                    if float(agr_raw) < min_agreement:
                        stats["low_agreement"] += 1
                        continue
                except ValueError:
                    pass
            dropped = []
            tags = {}
            levels = norm_levels(first_value(row, LEVEL_COLS))
            if levels:
                tags["building:levels"] = levels
            else:
                stats["no_levels"] += 1
            material = norm_material(first_value(row, MATERIAL_COLS))
            if material is None:
                material = norm_material(first_value(row, CONSTRUCTION_COLS))
                if material:
                    stats["material_from_construction"] += 1
            if material:
                tags["building:material"] = material
            else:
                stats["no_material"] += 1
            colour = norm_colour(first_value(row, COLOUR_COLS), dropped)
            if colour:
                tags["building:colour"] = colour
            else:
                stats["no_colour"] += 1
            if dropped:
                stats["junk_colour"] += len(dropped)
            if tags:
                tag_data.setdefault(bid, {}).update(tags)
            else:
                stats["empty_row"] += 1
    return tag_data, stats


# ---------------------------------------------------------------------------
# OSM extract reading (osmium cat -> XML stream)
# ---------------------------------------------------------------------------

def parse_extract(osm_path, ids):
    """
    Stream `osmium cat -f osm` and collect the ways/relations whose id is in
    `ids`. Returns ({id: {"version", "nds", "tags"}},
                     {id: {"version", "members", "tags"}}).
    """
    ways, relations = {}, {}
    proc = subprocess.Popen(
        ["osmium", "cat", "--no-progress", "-f", "osm", osm_path],
        stdout=subprocess.PIPE,
    )
    ids = set(ids)
    for _event, elem in ET.iterparse(proc.stdout, events=("end",)):
        if elem.tag == "way":
            wid = elem.get("id")
            if wid in ids:
                ways[wid] = {
                    "version": int(elem.get("version") or 0),
                    "nds": [nd.get("ref") for nd in elem.findall("nd")],
                    "tags": {t.get("k"): t.get("v") for t in elem.findall("tag")},
                }
            elem.clear()
        elif elem.tag == "relation":
            rid = elem.get("id")
            if rid in ids:
                relations[rid] = {
                    "version": int(elem.get("version") or 0),
                    "members": [(m.get("type"), m.get("ref"), m.get("role"))
                                for m in elem.findall("member")],
                    "tags": {t.get("k"): t.get("v") for t in elem.findall("tag")},
                }
            elem.clear()
    if proc.wait() != 0:
        sys.exit("osmium cat failed on the OSM input file")
    return ways, relations


# ---------------------------------------------------------------------------
# .osc generation
# ---------------------------------------------------------------------------

def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


def is_building(tags):
    return "building" in tags or "building:part" in tags


def build_osc(ways, relations, tag_data, include_untagged):
    """Build the osmChange XML. Returns (xml_string, stats Counter)."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    stats = Counter()
    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<osmChange version="0.6" generator="tag_buildings.py">',
        '  <modify>',
    ]

    def emit(obj_type, oid, obj, new_tags):
        version = obj["version"] + 1
        attrs = f'{obj_type} id="{oid}" version="{version}" timestamp="{now}" changeset="1"'
        parts.append(f'    <{attrs}>')
        if obj_type == "way":
            for ref in obj["nds"]:
                parts.append(f'      <nd ref="{esc(ref)}" />')
        else:
            for mtype, mref, mrole in obj["members"]:
                role = f' role="{esc(mrole)}"' if mrole else ""
                parts.append(f'      <member type="{mtype}" ref="{esc(mref)}"{role} />')
        for k, v in obj["tags"].items():
            parts.append(f'      <tag k="{esc(k)}" v="{esc(v)}" />')
        for k, v in new_tags.items():
            parts.append(f'      <tag k="{esc(k)}" v="{esc(v)}" />')
        parts.append(f'    </{obj_type}>')

    for key in sorted(tag_data, key=lambda k: (split_id(k)[0], int(split_id(k)[1]))):
        new = tag_data[key]
        prefix, oid = split_id(key)
        if prefix == "n":
            stats["skipped_node"] += 1  # node buildings have no footprint
            continue
        # typed ids only match their own element type; untyped (legacy) ids
        # are tried as way first, then as relation
        for obj_type, store in (("way", ways), ("relation", relations)):
            if prefix and prefix != obj_type[0]:
                continue
            obj = store.get(oid)
            if obj is None:
                continue
            if not is_building(obj["tags"]):
                if include_untagged:
                    obj = dict(obj)
                    obj["tags"] = dict(obj["tags"])
                    obj["tags"].setdefault("building", "yes")
                else:
                    stats[f"skipped_not_building_{obj_type}"] += 1
                    continue
            # only tags missing on the object are added - never overwrite
            to_add = {k: v for k, v in new.items() if k not in obj["tags"]}
            stats[f"already_tagged_{obj_type}"] += len(new) - len(to_add)
            for k, v in to_add.items():
                stats[f"added_{k.replace(':', '_')}"] += 1
            if to_add:
                emit(obj_type, oid, obj, to_add)
                stats[f"modified_{obj_type}"] += 1

    parts += ['  </modify>', '</osmChange>']
    return "\n".join(parts), stats


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Tag OSM buildings from a facade CSV (OSM2World tags only)")
    ap.add_argument("bbox_dir", nargs="?",
                    help="pipeline output folder (output/bbox_...); sets --csv/--osm/--out")
    ap.add_argument("--csv", help="input CSV with building_id")
    ap.add_argument("--osm", help="OSM file (pbf/osm/xml)")
    ap.add_argument("--out", default=None,
                    help="output file (default: <osm stem>_tagged.osm.pbf)")
    ap.add_argument("--min-agreement", type=float, default=0.0,
                    help="skip rows whose colour_agreement is below this")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the .osc to stdout, change nothing")
    ap.add_argument("--osc-only", action="store_true",
                    help="write the .osc file but do not apply it")
    ap.add_argument("--include-untagged", action="store_true",
                    help="also tag matching objects that lack building tags")
    args = ap.parse_args()

    osc_dir = None
    if args.bbox_dir:
        args.csv = args.csv or os.path.join(args.bbox_dir, "03_vlm", "buildings.csv")
        args.osm = args.osm or os.path.join(args.bbox_dir, "01_data", "extract.osm")
        osc_dir = os.path.join(args.bbox_dir, "04_osm")
        os.makedirs(osc_dir, exist_ok=True)
        args.out = args.out or os.path.join(osc_dir, "extract_tagged.osm")
    if not args.csv or not args.osm:
        ap.error("give a bbox_dir, or both --csv and --osm")

    tag_data, cstats = read_csv(args.csv, args.min_agreement)
    print(f"CSV rows: {cstats['csv_rows']}, usable building ids: {len(tag_data)}")
    for k in ("bad_id", "low_agreement", "empty_row", "junk_colour",
              "no_levels", "no_material", "no_colour",
              "material_from_construction"):
        if cstats[k]:
            print(f"  {k}: {cstats[k]}")

    ways, relations = parse_extract(args.osm, {split_id(k)[1] for k in tag_data})
    found = {"w" + i for i in ways} | {"r" + i for i in relations}
    missing = {k for k in tag_data
               if k not in found and not (split_id(k)[0] == ""
                                          and ("w" + k in found or "r" + k in found))}
    print(f"ids found: {len(tag_data) - len(missing)} / {len(tag_data)}"
          f" (in file: {len(ways)} ways, {len(relations)} relations with these numbers)")
    if missing:
        print(f"  not in OSM file: {len(missing)} ids, "
              f"e.g. {sorted(missing)[:10]}")

    osc_xml, ostats = build_osc(ways, relations, tag_data,
                                args.include_untagged)
    print("objects to modify:",
          ostats.get("modified_way", 0) + ostats.get("modified_relation", 0))
    for k in sorted(ostats):
        if k.startswith(("added_", "skipped_", "already_tagged_")) and ostats[k]:
            print(f"  {k}: {ostats[k]}")

    n_mods = ostats.get("modified_way", 0) + ostats.get("modified_relation", 0)
    if n_mods == 0:
        print("Nothing to change.")
        return

    if args.dry_run:
        print(osc_xml)
        return

    osc_path = re.sub(r"\.[^.]+$", "", args.osm) + "_tags.osc"
    if osc_dir:
        osc_path = os.path.join(osc_dir, "extract_tags.osc")
    with open(osc_path, "w", encoding="utf-8") as f:
        f.write(osc_xml)
    print(f"wrote {osc_path}")
    if args.osc_only:
        return

    out = args.out or re.sub(r"\.[^.]+$", "", args.osm) + "_tagged.osm.pbf"
    cmd = ["osmium", "apply-changes", args.osm, osc_path, "-o", out, "-O"]
    print("applying:", " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit("osmium apply-changes failed")
    print(f"done: {out}")


if __name__ == "__main__":
    main()
