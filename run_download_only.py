"""
Re-run only the image download step for OpenFACADES.
"""

import os

def download_selected_images(output_path, path_pid, mly_api_key):
    """
    Download selected panorama images (i.e., .png files) from the 'pids_urls_select.csv'.
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
        cache_dir=os.path.join(output_path, "cache")
    )

def main():
    output_img_path = os.path.join(os.getcwd(), "output/02_img")
    out_select = os.path.join(os.getcwd(), "output/01_data/pids_urls_select.csv")
    mly_api_key = "MLY|28359316630418089|d97008e9e9df81199d89f77536931500"
    
    print("=== Downloading Selected Panoramic Images ===")
    download_selected_images(output_img_path, out_select, mly_api_key)

if __name__ == "__main__":
    main()