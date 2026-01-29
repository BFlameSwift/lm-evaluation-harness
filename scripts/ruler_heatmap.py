from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a depth×length heatmap from lm-eval log_samples jsonl.")
    ap.add_argument("--samples", type=str, required=True, help="Path to samples_*.jsonl produced by lm_eval --log_samples.")
    ap.add_argument("--metric", type=str, default="score", help="Metric field to read from each row (default: score).")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for heatmap artifacts.")
    ap.add_argument("--title", type=str, default="", help="Optional plot title.")
    args = ap.parse_args()

    samples_path = Path(args.samples)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(samples_path)

    # Collect (length, depth) -> metric list
    grid: Dict[Tuple[int, int], List[float]] = {}
    lengths = set()
    depths = set()
    for r in rows:
        doc = r.get("doc", {})
        length = int(doc.get("max_length", -1))
        depth = int(doc.get("depth_percent", -1))
        if length < 0 or depth < 0:
            continue
        val = r.get(args.metric)
        if val is None:
            # Some harness versions store metrics under a dict.
            val = (r.get("metrics") or {}).get(args.metric)
        if val is None:
            continue
        lengths.add(length)
        depths.add(depth)
        grid.setdefault((length, depth), []).append(float(val))

    length_list = sorted(lengths)
    depth_list = sorted(depths)

    matrix: List[List[float]] = []
    for length in length_list:
        row: List[float] = []
        for depth in depth_list:
            row.append(_mean(grid.get((length, depth), [])))
        matrix.append(row)

    # Save CSV
    csv_path = out_dir / "heatmap.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("max_length," + ",".join(str(d) for d in depth_list) + "\n")
        for length, row in zip(length_list, matrix):
            f.write(str(length) + "," + ",".join(f"{x:.6f}" for x in row) + "\n")

    # Save JSON
    json_path = out_dir / "heatmap.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"max_lengths": length_list, "depths": depth_list, "matrix": matrix},
            f,
            ensure_ascii=False,
            indent=2,
        )

    # Optional plot (best-effort)
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        arr = np.array(matrix, dtype=float)
        fig, ax = plt.subplots(figsize=(max(6, len(depth_list) * 0.35), max(4, len(length_list) * 0.35)))
        im = ax.imshow(arr, aspect="auto", origin="lower")
        ax.set_xticks(list(range(len(depth_list))))
        ax.set_xticklabels([str(d) for d in depth_list], rotation=45, ha="right")
        ax.set_yticks(list(range(len(length_list))))
        ax.set_yticklabels([str(l) for l in length_list])
        ax.set_xlabel("Depth (%)")
        ax.set_ylabel("Max Length (tokens)")
        if args.title:
            ax.set_title(args.title)
        fig.colorbar(im, ax=ax, label=args.metric)
        fig.tight_layout()
        fig.savefig(out_dir / "heatmap.png", dpi=200)
        plt.close(fig)
    except Exception as e:
        print(f"[ruler_heatmap] plot skipped: {e}")

    print(f"[ruler_heatmap] wrote -> {out_dir}")


if __name__ == "__main__":
    main()

