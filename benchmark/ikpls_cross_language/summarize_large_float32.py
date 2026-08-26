#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd


root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument(
    "results",
    nargs="?",
    type=Path,
    default=root / "benchmark_results" / "ikpls_large_float32_20260826",
)
parser.add_argument(
    "--fastpls-imagenet",
    type=Path,
    default=root / "benchmark_results/frozen_release_0.99.25/imagenet/imagenet_all_results.csv",
)
args = parser.parse_args()
out = args.results
rows = []

preprocess = pd.read_csv(out / "preprocessing.tsv", sep="\t", header=None, index_col=0)[1]
preprocess_sec = float(preprocess["preprocess_sec"])

for path in sorted(out.glob("imagenet_ikpls_f32_n*.csv")):
    row = pd.read_csv(path).iloc[0]
    rows.append({
        "dataset": "ImageNet",
        "implementation": "IKPLS 6.1.2 cross-product",
        "precision": "float32",
        "ncomp": int(row.ncomp),
        "fit_sec": row.fit_sec,
        "predict_sec": row.predict_sec,
        "model_total_sec": row.total_sec,
        "preprocess_sec": preprocess_sec,
        "total_including_preprocess_sec": row.total_sec + preprocess_sec,
        "top1_accuracy": row.top1_accuracy_or_rmsd,
        "top5_accuracy": row.top5_accuracy,
        "rmsd": None,
        "peak_process_rss_mib": row.peak_rss_mib,
        "status": row.status,
        "notes": "Public NumPy Algorithm 2; one CPU thread; centred float32 arrays; blocked prediction",
    })

if args.fastpls_imagenet.exists():
    fast = pd.read_csv(args.fastpls_imagenet)
    fast = fast[(fast.classifier == "argmax") & fast.ncomp.isin([100, 200, 500, 1000])]
    for _, row in fast.iterrows():
        rows.append({
            "dataset": "ImageNet",
            "implementation": "fastPLS 0.99.25 SIMPLS CUDA-rSVD argmax",
            "precision": "float32",
            "ncomp": int(row.ncomp),
            "fit_sec": row.fit_time_sec,
            "predict_sec": row.predict_time_sec,
            "model_total_sec": row.total_time_sec,
            "preprocess_sec": None,
            "total_including_preprocess_sec": row.total_time_sec,
            "top1_accuracy": row.top1_accuracy,
            "top5_accuracy": row.top5_accuracy,
            "rmsd": None,
            "peak_process_rss_mib": row.rss_peak_predict_mb,
            "status": row.status,
            "notes": "One maximal 1000-component fit supplied all prefixes; timing is shared and not a per-prefix refit",
        })

for path in sorted(out.glob("nmr_ikpls_f32_n*.csv")):
    row = pd.read_csv(path).iloc[0]
    rows.append({
        "dataset": "NMR",
        "implementation": "IKPLS 6.1.2 cross-product",
        "precision": "float32",
        "ncomp": int(row.ncomp),
        "fit_sec": row.fit_sec,
        "predict_sec": row.predict_sec,
        "model_total_sec": row.total_sec,
        "preprocess_sec": None,
        "total_including_preprocess_sec": row.total_sec,
        "top1_accuracy": None,
        "top5_accuracy": None,
        "rmsd": row.top1_accuracy_or_rmsd,
        "peak_process_rss_mib": row.peak_rss_mib,
        "status": row.status,
        "notes": row.error if row.status != "success" else "First component collapsed under IKPLS float32 stability criterion",
    })

if not rows:
    raise SystemExit(f"No benchmark rows found in {out}")
summary = pd.DataFrame(rows).sort_values(["dataset", "implementation", "ncomp"])
summary.to_csv(out / "ikpls_fastpls_large_float32_summary.csv", index=False)

with (out / "README.md").open("w") as handle:
    handle.write("# IKPLS large-case float32 benchmark\n\n")
    handle.write("IKPLS 6.1.2 NumPy cross-product (Algorithm 2) was tested with one CPU thread. ")
    handle.write("ImageNet centring and dense centred one-hot construction took ")
    handle.write(f"{preprocess_sec:.2f} s and are reported separately from model timing.\n\n")
    handle.write("NMR at one component completed but collapsed to the training-mean prediction; ")
    handle.write("five components failed under a 10 GiB virtual-memory guard. The selected ")
    handle.write("50-component NMR model would require 68.66 GiB for IKPLS's retained B tensor alone.\n\n")
    handle.write("ImageNet fastPLS timings come from one maximal 1000-component path, so the same fit ")
    handle.write("and prediction time is attached to every prefix and is not a per-prefix refit comparison. ")
    handle.write("IKPLS and fastPLS implement different PLS estimators.\n")
