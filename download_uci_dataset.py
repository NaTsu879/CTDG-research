import argparse
import tarfile
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


URL = "http://konect.cc/files/download.tsv.opsahl-ucsocial.tar.bz2"
DATA_DIR = Path("./data/uci")
RAW_MEMBER = "opsahl-ucsocial/out.opsahl-ucsocial"


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


def extract_raw(archive_path: Path, raw_path: Path):
    """
    Extract the KONECT edge list (out.opsahl-ucsocial) from the tar.bz2 archive.
    """
    print(f"Extracting {RAW_MEMBER} from {archive_path}...")
    with tarfile.open(archive_path, "r:bz2") as tar:
        member = tar.extractfile(RAW_MEMBER)
        with open(raw_path, "wb") as f:
            f.write(member.read())


def preprocess(raw_path: str):
    """
    Read the KONECT edge list (src dst weight timestamp, '%' comment lines) and
    extract user, item, timestamp, label and edge indices, sorted by timestamp.
    """
    print(f"Parsing raw dataset from {raw_path}...")
    u_list, i_list, ts_list = [], [], []

    with open(raw_path, "r") as f:
        for line in f:
            if line.startswith("%") or not line.strip():
                continue
            e = line.split()
            u_list.append(int(e[0]))
            i_list.append(int(e[1]))
            ts_list.append(float(e[3]))

    df = pd.DataFrame({"u": u_list, "i": i_list, "ts": ts_list})
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    df.ts = df.ts - df.ts.min()
    df["label"] = 0.0
    df["idx"] = np.arange(len(df))
    return df


def reindex(df: pd.DataFrame):
    """
    Non-bipartite graph: users and items share one id space.
    Map node ids to a contiguous range and use 1-based indexing for edges/nodes.
    """
    new_df = df.copy()
    nodes = np.unique(np.concatenate([df.u.values, df.i.values]))
    node_map = {n: k for k, n in enumerate(nodes)}

    new_df.u = df.u.map(node_map) + 1
    new_df.i = df.i.map(node_map) + 1
    new_df.idx += 1
    return new_df


def process_dataset(node_feat_dim: int = 172, edge_feat_dim: int = 172):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = DATA_DIR / "download.tsv.opsahl-ucsocial.tar.bz2"
    raw_path = DATA_DIR / "out.opsahl-ucsocial"
    out_df_path = DATA_DIR / "ml_uci.csv"
    out_edge_feat_path = DATA_DIR / "ml_uci.npy"
    out_node_feat_path = DATA_DIR / "ml_uci_node.npy"

    if not raw_path.exists():
        if not archive_path.exists():
            download_file(URL, archive_path)
        extract_raw(archive_path, raw_path)

    df = preprocess(str(raw_path))
    new_df = reindex(df)

    # Edge features (UCI has none), first row is zero index padding (1-indexed)
    edge_feats = np.zeros((len(new_df) + 1, edge_feat_dim), dtype=np.float32)

    # Node features (1-indexed)
    max_node_idx = max(new_df.u.max(), new_df.i.max())
    node_feats = np.zeros((max_node_idx + 1, node_feat_dim), dtype=np.float32)

    print(f"Total nodes: {node_feats.shape[0] - 1}")
    print(f"Node feature shape: {node_feats.shape}")
    print(f"Total edges: {edge_feats.shape[0] - 1}")
    print(f"Edge feature shape: {edge_feats.shape}")

    new_df[["u", "i", "ts", "label", "idx"]].to_csv(out_df_path, index=False)
    np.save(out_edge_feat_path, edge_feats)
    np.save(out_node_feat_path, node_feats)
    print("Preprocessing completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess the UCI (KONECT opsahl-ucsocial) dataset")
    parser.add_argument("--node_feat_dim", type=int, default=172, help="Node feature dimension")
    parser.add_argument("--edge_feat_dim", type=int, default=172, help="Edge feature dimension")
    args = parser.parse_args()
    process_dataset(node_feat_dim=args.node_feat_dim, edge_feat_dim=args.edge_feat_dim)


if __name__ == "__main__":
    main()
