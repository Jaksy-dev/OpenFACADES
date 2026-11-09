"""
Label selected building images with the OpenFACADES fine-tuned InternVL model.

Reads   <bbox_dir>/02_img/individual_building_select.csv
Writes  <bbox_dir>/03_vlm/attributes.csv   (one row per image)
        <bbox_dir>/03_vlm/buildings.csv    (one row per building: levels, materials, colour)
        <bbox_dir>/03_vlm/buildings.geojson (same, joined to footprints)

Example:
    python run_vlm.py output/bbox_19.0564_47.4908_19.0608_47.4931 --limit 5
"""
import argparse
import json
import os
import re

import pandas as pd
from tqdm import tqdm

# Prompt format used in the OpenFACADES training data (seshing/openfacades-dataset).
ATTRIBUTE_PROMPT = """<image>
Provide concise labels for each category using the following JSON format. Select appropriate values from the provided options for each category:
{
    "building_type": "(choose one option from: 'apartments', 'house', 'retail', 'office', 'hotel', 'industrial', 'religious', 'education', 'public', 'garage')",
    "alternate_building_type": "(choose another option from: 'apartments', 'house', 'retail', 'office', 'hotel', 'industrial', 'religious', 'education', 'public', 'garage')",
    "building_age": "(a 4-digit year indicating the approximate construction date of the building)",
    "floors": "(a numeric value representing the total number of floors)",
    "surface_material": "(choose one option from: 'brick', 'wood', 'concrete', 'metal', 'stone', 'glass', 'plaster')",
    "alternate_surface_material": "(choose another option from: 'brick', 'wood', 'concrete', 'metal', 'stone', 'glass', 'plaster')",
    "construction_material": "(choose one option from: 'brick', 'wood', 'concrete', 'steel', 'other')",
    "alternate_construction_material": "(choose another option from: 'brick', 'wood', 'concrete', 'steel', 'other')"
}"""

# The fine-tuned model was never trained on colour and just repeats the material,
# so colour is asked separately of the stock InternVL3 model, which answers well.

# building_age is still requested (the fine-tuned model expects its trained prompt) but dropped.
DROP_FIELDS = ["building_age"]


def parse_json(response):
    if not response:
        return {}
    match = re.search(r"\{.*\}", response, re.DOTALL)
    try:
        if not match:
            raise json.JSONDecodeError("no closed JSON object", response, 0)
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        # The model occasionally emits malformed JSON (e.g. a stray ","),
        # so fall back to pulling out "key": value pairs individually.
        pairs = re.findall(r'"(\w+)"\s*:\s*"?([^",\n}]*)"?', response)
        return {k: v.strip() for k, v in pairs}


BASE_COLOURS = ["white", "beige", "yellow", "orange", "red", "pink", "brown", "gray",
                "black", "blue", "green", "purple", "gold"]


# Asked open-ended on purpose: listing the options made the model over-answer "brown"
# (worse on 5 of 40 test photos, better on 2). normalise_colour() maps the answer
# onto BASE_COLOURS instead, so the output is still a fixed set.
COLOUR_PROMPT = "<image>\nWhat is the main colour of this building's facade? Answer with one colour word."


def normalise_colour(answer):
    """Reduce a free-text answer ('The main color of the building's facade is light grey.')
    to one base colour word ('gray'). Returns None if no colour is named (e.g. 'glass')."""
    if not isinstance(answer, str):
        return None
    text = answer.lower().replace("grey", "gray").replace("colour", "color")
    words = re.findall(r"[a-z]+", text.split(" is ")[-1])  # the part after "... is"
    found = [w for w in words if w in BASE_COLOURS]
    return found[-1] if found else None


def summarise_buildings(bbox_dir, attributes_csv):
    """One row per building: median floors, most common material/colour across its views."""
    df = pd.read_csv(attributes_csv)
    df["floors"] = pd.to_numeric(df.get("floors"), errors="coerce")
    df["facade_colour"] = df["facade_colour"].map(normalise_colour)

    def mode(s):
        s = s.dropna().astype(str).str.strip().str.lower()
        s = s[s != ""]
        return s.mode().iloc[0] if not s.empty else None

    def agreement(s):
        s = s.dropna().astype(str).str.strip().str.lower()
        return round((s == s.mode().iloc[0]).mean(), 2) if not s.empty else None

    out = df.groupby("building_id").agg(
        n_views=("image_name", "count"),
        levels=("floors", "median"),
        surface_material=("surface_material", mode),
        construction_material=("construction_material", mode),
        facade_colour=("facade_colour", mode),
        colour_agreement=("facade_colour", agreement),
    ).reset_index()
    out["levels"] = out["levels"].round().astype("Int64")

    out_dir = os.path.dirname(attributes_csv)
    out.to_csv(os.path.join(out_dir, "buildings.csv"), index=False)

    footprint = os.path.join(bbox_dir, "01_data", "footprint.geojson")
    if os.path.exists(footprint):
        import geopandas as gpd
        fp = gpd.read_file(footprint)[["building_id", "source", "geometry"]]
        fp["building_id"] = fp["building_id"].astype(str)
        out["building_id"] = out["building_id"].astype(str)
        fp.merge(out, on="building_id", how="inner").to_file(
            os.path.join(out_dir, "buildings.geojson"), driver="GeoJSON")
    print(f"Summarised {len(out)} buildings -> {out_dir}/buildings.csv / buildings.geojson")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("bbox_dir", help="Pipeline output folder for one bbox (output/bbox_...)")
    parser.add_argument("--model", default="seshing/openfacades-internvl3-2b",
                        help="Hugging Face model id or local path (the published 1b repo has no weights)")
    parser.add_argument("--colour-model", default="OpenGVLab/InternVL3-2B",
                        help="Model used for facade colour")
    parser.add_argument("--limit", type=int, help="Only label the first N images (for testing)")
    parser.add_argument("--device", help="cuda / cpu (default: GPU if available)")
    args = parser.parse_args()

    img_dir = os.path.join(args.bbox_dir, "02_img", "individual_building")
    select = pd.read_csv(os.path.join(args.bbox_dir, "02_img", "individual_building_select.csv"))
    if args.limit:
        select = select.head(args.limit)

    out_dir = os.path.join(args.bbox_dir, "03_vlm")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "attributes.csv")

    # Resume: skip images already labelled.
    done = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame(columns=["image_name"])
    todo = select[~select["image_name"].isin(done["image_name"])]
    print(f"{len(todo)} images to label ({len(done)} already done) -> {out_csv}")
    if todo.empty:
        summarise_buildings(args.bbox_dir, out_csv)
        return

    from openfacades.vlm import InternVLProcessor
    processor = InternVLProcessor(args.model, device=args.device)
    colour_processor = InternVLProcessor(args.colour_model, device=args.device)
    print(f"Models {args.model} + {args.colour_model} on {processor.device}")

    rows = []
    for _, r in tqdm(todo.iterrows(), total=len(todo), desc="Labelling"):
        path = os.path.join(img_dir, r["image_name"])
        response = processor.process_image(path, ATTRIBUTE_PROMPT)
        parsed = parse_json(response)
        if not parsed.get("surface_material"):
            # The model sometimes loops on "," until max tokens; retry discouraging repetition.
            retry = processor.process_image(path, ATTRIBUTE_PROMPT, generation_config={
                "max_new_tokens": 300, "do_sample": False, "repetition_penalty": 1.2})
            retry_parsed = parse_json(retry)
            if retry_parsed.get("surface_material"):
                response, parsed = retry, {**parsed, **retry_parsed}
        attrs = {k: v for k, v in parsed.items() if k not in DROP_FIELDS}
        colour_raw = colour_processor.process_image(path, COLOUR_PROMPT) or ""
        colour = normalise_colour(colour_raw)
        rows.append({"image_name": r["image_name"], "pid": r["pid"], "building_id": r["building_id"],
                     "facade_colour": colour, "facade_colour_raw": colour_raw, **attrs, "raw_response": response})
        # Save after every image so an interrupted run loses nothing.
        pd.concat([done, pd.DataFrame(rows)], ignore_index=True).to_csv(out_csv, index=False)

    print(f"Saved {len(rows)} new labels to {out_csv}")
    summarise_buildings(args.bbox_dir, out_csv)


if __name__ == "__main__":
    main()
