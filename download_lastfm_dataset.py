import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


URL = "https://snap.stanford.edu/jodie/lastfm.csv"
DATA_DIR = Path("./data/lastfm")


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


def preprocess(raw_csv_path: str):
    """
    Read raw CSV and extract user, item, timestamp, label, edge indices, and edge features.
    """
    print(f"Parsing raw dataset from {raw_csv_path}...")
    u_list, i_list, ts_list, label_list, idx_list = [], [], [], [], []
    feat_l = []

    with open(raw_csv_path, "r") as f:
        _ = next(f)  # header
        previous_time = -1.0
        for idx, line in enumerate(f):
            e = line.strip().split(",")
            u = int(e[0])
            i = int(e[1])
            ts = float(e[2])
            assert ts >= previous_time, f"Timestamps not ascending at line {idx}: {ts} < {previous_time}"
            previous_time = ts
            label = float(e[3])
            feat = [float(x) for x in e[4:]]

            u_list.append(u)
            i_list.append(i)
            ts_list.append(ts)
            label_list.append(label)
            idx_list.append(idx)
            feat_l.append(feat)

    df = pd.DataFrame({
        "u": u_list,
        "i": i_list,
        "ts": ts_list,
        "label": label_list,
        "idx": idx_list
    })
    edge_feats = np.array(feat_l, dtype=np.float32)
    return df, edge_feats


def reindex(df: pd.DataFrame, bipartite: bool = True):
    """
    Reindex bipartite nodes and 1-based indexing for edges/nodes.
    """
    new_df = df.copy()
    if bipartite:
        assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
        assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))
        assert df.u.min() == df.i.min() == 0

        upper_u = df.u.max() + 1
        new_df.i = df.i + upper_u

    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1
    return new_df


def process_dataset(node_feat_dim: int = 172):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = DATA_DIR / "lastfm.csv"
    out_df_path = DATA_DIR / "ml_lastfm.csv"
    out_edge_feat_path = DATA_DIR / "ml_lastfm.npy"
    out_node_feat_path = DATA_DIR / "ml_lastfm_node.npy"

    if not raw_path.exists():
        download_file(URL, raw_path)

    df, edge_feats = preprocess(str(raw_path))
    new_df = reindex(df, bipartite=True)

    # Edge feature for zero index padding (1-indexed)
    empty_edge = np.zeros((1, edge_feats.shape[1]), dtype=np.float32)
    edge_feats = np.vstack([empty_edge, edge_feats])

    # Node features (1-indexed)
    max_node_idx = max(new_df.u.max(), new_df.i.max())
    node_feats = np.zeros((max_node_idx + 1, node_feat_dim), dtype=np.float32)

    print(f"Total nodes: {node_feats.shape[0] - 1}")
    print(f"Node feature shape: {node_feats.shape}")
    print(f"Total edges: {edge_feats.shape[0] - 1}")
    print(f"Edge feature shape: {edge_feats.shape}")

    new_df.to_csv(out_df_path, index=False)
    np.save(out_edge_feat_path, edge_feats)
    np.save(out_node_feat_path, node_feats)
    print("Preprocessing completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess JODIE/DyGLib LastFM dataset")
    parser.add_argument("--node_feat_dim", type=int, default=172, help="Node feature dimension")
    args = parser.parse_args()
    process_dataset(node_feat_dim=args.node_feat_dim)


if __name__ == "__main__":
    main()
