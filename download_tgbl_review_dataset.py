import argparse
import shutil
import zipfile
from pathlib import Path
import requests
from tqdm import tqdm


# the url hardcoded in tgb/utils/info.py points at the old Compute Canada bucket, which no longer
# exists (the whole bucket answers NoSuchBucket), so the current TGB location is used instead
URL = "https://object-arbutus.alliancecan.ca/swift/v1/14c95234f6cd4a21a47deafe20cce2a7/tgb/tgbl-review-v2.zip"
# LinkPropPredDataset resolves its root as PROJ_DIR + dataset_path, and PROJ_DIR is the tgb package
# directory, so with the default --dataset_path ./data/ the raw files must live here
DATA_DIR = Path("./tgb/data/tgbl_review")
# utils/DataLoader.py caches the test negative samples under the dataset path itself, in a different
# directory, which has to exist before the first run can write to it
CACHE_DIR = Path("./data/tgbl_review")
EDGELIST_NAME = "tgbl-review_edgelist_v2.csv"


def download_file(url: str, target: Path):
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target.with_suffix(target.suffix + ".tmp")
    headers = {"User-Agent": "Mozilla/5.0"}
    print(f"Downloading {url} -> {target}")
    try:
        with requests.get(url, stream=True, timeout=120, headers=headers) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            chunk_size = 1024 * 1024  # 1 MB chunk
            with open(temp_target, "wb") as f, tqdm(
                desc=target.name,
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
                ncols=100
            ) as bar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        size = f.write(chunk)
                        bar.update(size)
        temp_target.replace(target)
        print("Download complete")
    except Exception:
        if temp_target.exists():
            temp_target.unlink()
        raise


def extract_archive(archive_path: Path, target_dir: Path):
    """
    Extract the dataset archive, flattening it if the files sit inside a folder, because
    LinkPropPredDataset expects them directly under its root directory.
    """
    print(f"Extracting {archive_path} -> {target_dir}")
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        members = [name for name in zip_ref.namelist() if not name.endswith("/")]
        for name in tqdm(members, desc="extracting", unit=" file", ncols=100):
            out_path = target_dir / Path(name).name
            with zip_ref.open(name) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
    print(f"Extracted {len(members)} files")


def process_dataset(keep_archive: bool = False):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = DATA_DIR / "tgbl-review-v2.zip"
    edgelist_path = DATA_DIR / EDGELIST_NAME

    if edgelist_path.exists():
        print(f"Raw file found at {edgelist_path}, skipping download")
    else:
        if not archive_path.exists():
            download_file(URL, archive_path)
        extract_archive(archive_path, DATA_DIR)
        if not keep_archive and archive_path.exists():
            archive_path.unlink()
            print(f"Removed {archive_path}")

    if not edgelist_path.exists():
        raise FileNotFoundError(f"{EDGELIST_NAME} not found in {DATA_DIR} after extraction")

    print(f"Dataset directory: {DATA_DIR.resolve()}")
    for path in sorted(DATA_DIR.iterdir()):
        print(f"  {path.name} ({path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"Negative sample cache directory: {CACHE_DIR.resolve()}")
    print("Download completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Download and extract the TGB tgbl-review dataset")
    parser.add_argument("--keep_archive", action="store_true", default=False, help="Keep the downloaded zip file")
    args = parser.parse_args()
    process_dataset(keep_archive=args.keep_archive)


if __name__ == "__main__":
    main()
