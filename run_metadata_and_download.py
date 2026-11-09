"""
Re-run only the metadata and image download steps for OpenFACADES.
"""

import os

def prepare_mly_metadata(bbox, output_path, mly_api_key):
    """
    Generate a bounding-box GeoJSON, then download Mapillary metadata.
    """
    from zensvi.download import MLYDownloader
    from openfacades.streetview import bbox_to_geojson, process_metadata

    print("=== Preparing Bounding Box GeoJSON ===")
    bbox_gdf = bbox_to_geojson(bbox)
    bbox_path = os.path.join(output_path, "bbox.geojson")
    bbox_gdf.to_file(bbox_path, driver="GeoJSON")
    kwarg = {"image_type": "pano"}

    print("=== Downloading MLY Metadata ===")
    mly_downloader = MLYDownloader(mly_api_key)
    mly_downloader.download_svi(
        output_path,  # output directory
        input_shp_file=os.path.join(output_path, 'bbox.geojson'),  # path to the input shapefile
        input_place_name="",  # name of the location to download
        resolution=2048,  # resolution of the image
        batch_size=1000,  # batch size for downloading images
        start_date=None,  # start date for downloading images (YYYY-MM-DD)
        end_date=None,  # end date for downloading images (YYYY-MM-DD)
        metadata_only=True,  # if True, only metadata is downloaded
        use_cache=True,  # if True, the cache is used
        additional_fields=["camera_type", "captured_at",
                           "computed_compass_angle", "computed_geometry",
                           "geometry", "quality_score"],  # Additional fields to fetch from the API.
        **kwarg
    )

    # Clean the raw metadata
    raw_csv = os.path.join(output_path, "pids_urls.csv")
    cleaned_csv = os.path.join(output_path, "pids_urls_clean.csv")
    print(f"=== Cleaning MLY Metadata ===\nFrom: {raw_csv}\nTo:   {cleaned_csv}")
    process_metadata(raw_csv, cleaned_csv)

    return cleaned_csv

def compute_and_filter_aov(cleaned_metadata, output_path):
    """
    Compute the AoV for each panorama and filter by constraints.
    """
    from openfacades.streetview import geo_aov_calculator, filter_img_by_aov

    aov_csv_path = os.path.join(output_path, "aov.csv")
    if not os.path.exists(aov_csv_path):
        print("=== Computing AoV ===")
        geo_aov_calculator(
            img_metadata=cleaned_metadata,
            json_dir=os.path.join(output_path, "footprint.geojson"),
            buffer=30,
            distance_between_points=2,
            out_dir=aov_csv_path,
            batch_size=20,
            max_workers=4
        )
    else:
        print("AoV CSV already exists, skipping: ", aov_csv_path)

    print("=== Filtering AoV (by max/min) ===")
    out_select = os.path.join(output_path, "pids_urls_select.csv")
    filter_img_by_aov(
        aov_csv_path,
        cleaned_metadata,
        max_aov=130,
        min_aov=10,
        max_image_per_building=5,
        method='by_aov'
    ).to_csv(out_select, index=False)

    return aov_csv_path, out_select

def download_selected_images(output_path, path_pid, mly_api_key):
    """
    Download selected panorama images.
    """
    from zensvi.download import MLYDownloader

    print("=== Downloading Selected Panoramic Images ===")
    mly_downloader = MLYDownloader(mly_api_key)
    mly_downloader.download_svi(
        output_path,
        input_csv_file=path_pid,
        resolution=2048,
        batch_size=500,
        metadata_only=False,
        use_cache=True,
        cache_dir=os.path.join(output_path, "cache"),
        buffer=30  # Add buffer
    )

def main():
    bbox = [19.05085, 47.48667, 19.06508, 47.49492]
    mly_api_key = "MLY|28359316630418089|d97008e9e9df81199d89f77536931500"
    output_path = os.path.join(os.getcwd(), "output")
    output_data_path = os.path.join(output_path, '01_data')
    output_img_path = os.path.join(output_path, '02_img')
    os.makedirs(output_data_path, exist_ok=True)
    os.makedirs(output_img_path, exist_ok=True)

    # === 2) Prepare MLY Metadata ===
    cleaned_csv = prepare_mly_metadata(bbox, output_data_path, mly_api_key)

    # === 3) Compute + Filter AoV ===
    aov_csv_path, out_select = compute_and_filter_aov(cleaned_csv, output_data_path)

    # === 4) Download Panoramic Images (selected) ===
    download_selected_images(output_img_path, out_select, mly_api_key)

if __name__ == "__main__":
    main()