"""
Retry only the image download step for OpenFACADES.
"""

import os

def download_selected_images(output_path, path_pid, mly_api_key):
    """
    Download selected panorama images, skipping already downloaded ones.
    """
    import sys
    sys.path.append('/home/jaksy/Project/OpenFACADES/.venv/lib/python3.14/site-packages')
    from zensvi.download import MLYDownloader
    import pandas as pd
    import os

    # Load the list of PIDs to download
    df = pd.read_csv(path_pid)
    pids_to_download = set(df['id'])  # Column is 'id', not 'pid'

    # Get already downloaded PIDs
    downloaded_pids = {
        os.path.splitext(f)[0]
        for f in os.listdir(output_path)
        if f.endswith('.png')
    }

    # Skip if all images are already downloaded
    missing_pids = pids_to_download - downloaded_pids
    if not missing_pids:
        print("All images already downloaded. Skipping.")
        return

    print(f"Downloading {len(missing_pids)} missing images...")
    mly_downloader = MLYDownloader(mly_api_key)
    mly_downloader.download_svi(
        output_path,
        input_csv_file=path_pid,
        resolution=2048,
        batch_size=500,
        metadata_only=False,
        use_cache=True,
        cache_dir=os.path.join(output_path, "cache"),
        buffer=30
    )

def main():
    output_img_path = os.path.join(os.getcwd(), "output/02_img")
    out_select = os.path.join(os.getcwd(), "output/01_data/pids_urls_select.csv")
    mly_api_key = "MLY|28359316630418089|d97008e9e9df81199d89f77536931500"
    os.makedirs(output_img_path, exist_ok=True)
    os.makedirs(os.path.join(output_img_path, "cache"), exist_ok=True)

    download_selected_images(output_img_path, out_select, mly_api_key)

if __name__ == "__main__":
    main()