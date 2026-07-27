from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/Users/stefano/Documents/GPUPLS/fastPLS_fresh_github")
OUT = ROOT / "benchmark_results" / "manuscript_revision_cycle13_20260725"
MULTI = ROOT / "benchmark_results" / "manuscript_multidataset_summary_20260725"
WORK = Path("/Users/stefano/Documents/GPUPLS/manuscript_work_20260722/evidence")


def q25(x):
    return np.nanquantile(pd.to_numeric(x, errors="coerce"), 0.25)


def q75(x):
    return np.nanquantile(pd.to_numeric(x, errors="coerce"), 0.75)


def selected_backend_uncertainty():
    paths = [
        MULTI / "source" / "multidataset_selected_cpu_raw.csv",
        MULTI / "source" / "multidataset_selected_gpu_raw.csv",
    ]
    raw = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    raw = raw[(raw["status"] == "ok") & (raw["classifier"] != "cknn")].copy()
    keys = [
        "dataset", "task_type", "method_panel", "variant_name", "engine",
        "backend", "classifier", "requested_ncomp", "effective_ncomp",
        "precision", "execution_precision", "metric_name",
    ]
    summary = raw.groupby(keys, dropna=False).agg(
        n_runs=("replicate", "size"),
        metric_median=("metric_value", "median"),
        metric_q25=("metric_value", q25),
        metric_q75=("metric_value", q75),
        total_time_sec_median=("total_time_ms", lambda x: np.nanmedian(x) / 1000),
        total_time_sec_q25=("total_time_ms", lambda x: q25(x) / 1000),
        total_time_sec_q75=("total_time_ms", lambda x: q75(x) / 1000),
        host_rss_mb_median=("peak_host_rss_mb", "median"),
        host_rss_mb_q25=("peak_host_rss_mb", q25),
        host_rss_mb_q75=("peak_host_rss_mb", q75),
        gpu_mem_mb_median=("peak_gpu_mem_mb", "median"),
        gpu_mem_mb_q25=("peak_gpu_mem_mb", q25),
        gpu_mem_mb_q75=("peak_gpu_mem_mb", q75),
    ).reset_index()
    summary.to_csv(OUT / "selected_backend_with_uncertainty.csv", index=False)
    return summary


def external_simpls_matched():
    data = pd.read_csv(MULTI / "source" / "external_float64_summary.csv")
    keep_ids = {
        "fastPLS_simpls_cpu_irlba",
        "fastPLS_simpls_cpu_rsvd",
        "fastPLS_simpls_cuda_rsvd",
        "pls_simpls_fit",
    }
    data = data[
        data["method_id"].isin(keep_ids)
        & data["input_precision"].eq("float64")
        & data["classifier"].eq("argmax")
        & data["reps_ok"].gt(0)
        & data["requested_estimator"].eq("simpls")
        & data["executed_estimator"].eq("simpls")
    ].copy()
    data["implementation"] = data["method_id"].map({
        "fastPLS_simpls_cpu_irlba": "fastPLS CPU IRLBA",
        "fastPLS_simpls_cpu_rsvd": "fastPLS CPU rSVD",
        "fastPLS_simpls_cuda_rsvd": "fastPLS CUDA rSVD",
        "pls_simpls_fit": "pls::simpls.fit",
    })
    columns = [
        "dataset", "implementation", "ncomp_requested", "reps_ok",
        "median_time_ms", "iqr_time_ms", "median_peak_host_rss_mb",
        "iqr_peak_host_rss_mb", "metric_name", "median_metric", "iqr_metric",
        "median_accuracy", "median_balanced_accuracy", "median_macro_f1",
    ]
    data = data[columns].sort_values(["dataset", "implementation"])
    data.to_csv(OUT / "external_simpls_float64_matched.csv", index=False)

    rows = []
    for dataset, group in data.groupby("dataset"):
        reference = group[group["implementation"] == "pls::simpls.fit"]
        candidates = group[group["implementation"] != "pls::simpls.fit"]
        if reference.empty or candidates.empty:
            continue
        fastest = candidates.loc[candidates["median_time_ms"].idxmin()]
        ref = reference.iloc[0]
        rows.append({
            "dataset": dataset,
            "fastest_fastpls": fastest["implementation"],
            "ncomp": fastest["ncomp_requested"],
            "fastpls_time_ms": fastest["median_time_ms"],
            "fastpls_time_iqr_ms": fastest["iqr_time_ms"],
            "reference_time_ms": ref["median_time_ms"],
            "reference_time_iqr_ms": ref["iqr_time_ms"],
            "speedup_vs_pls": ref["median_time_ms"] / fastest["median_time_ms"],
            "fastpls_metric": fastest["median_metric"],
            "reference_metric": ref["median_metric"],
            "metric_difference": fastest["median_metric"] - ref["median_metric"],
            "metric_name": fastest["metric_name"],
        })
    compact = pd.DataFrame(rows).sort_values("dataset")
    compact.to_csv(OUT / "external_simpls_float64_compact.csv", index=False)
    return compact


def cv_quantitative():
    path = OUT / "pipeline3_cv_vs_fit" / "pipeline3_cv_vs_fit_comparison.csv"
    data = pd.read_csv(path)
    data = data[~data["method_id"].str.contains("cknn", case=False, na=False)].copy()
    data["classifier"] = np.where(
        data["method_id"].str.endswith("_lda"), "LDA", "argmax/regression"
    )
    data["method"] = data["method_id"].str.extract(
        r"fastPLS_(plssvd|simpls|opls|kernelpls)_"
    )[0]
    data["backend"] = np.select(
        [
            data["method_id"].str.contains("cuda"),
            data["method_id"].str.contains("irlba"),
        ],
        ["CUDA rSVD", "CPU IRLBA"],
        default="CPU rSVD",
    )
    data.to_csv(OUT / "cv10_vs_fit_filtered.csv", index=False)
    complete = data.dropna(subset=["fit_predict_sec", "cv10_sec", "cv_over_fit_ratio"])
    summary = complete.groupby(
        ["dataset", "method", "backend", "classifier"], dropna=False
    ).agg(
        fit_predict_sec=("fit_predict_sec", "median"),
        cv10_sec=("cv10_sec", "median"),
        cv_over_fit_ratio=("cv_over_fit_ratio", "median"),
        fit_metric=("fit_predict_metric", "median"),
        cv_metric=("cv10_metric", "median"),
    ).reset_index()
    summary.to_csv(OUT / "cv10_vs_fit_summary.csv", index=False)

    dataset_summary = complete.groupby("dataset").agg(
        comparisons=("method_id", "size"),
        cv_over_fit_median=("cv_over_fit_ratio", "median"),
        cv_over_fit_min=("cv_over_fit_ratio", "min"),
        cv_over_fit_max=("cv_over_fit_ratio", "max"),
    ).reset_index()
    dataset_summary.to_csv(OUT / "cv10_dataset_summary.csv", index=False)
    return dataset_summary


def precision_summary():
    source = WORK / "precision_memory_final_cycle7" / "precision_memory_summary_cycle7.csv"
    data = pd.read_csv(source)
    data.to_csv(OUT / "float32_float64_precision_memory_summary.csv", index=False)
    pairs = []
    keys = ["dataset", "task_type", "method", "backend", "requested_ncomp", "metric_name"]
    for key, group in data.groupby(keys):
        f32 = group[group["precision"] == "float32"]
        f64 = group[group["precision"] == "float64"]
        if f32.empty or f64.empty:
            continue
        a, b = f32.iloc[0], f64.iloc[0]
        pairs.append(dict(zip(keys, key)) | {
            "f32_runs": a["n_runs"],
            "f64_runs": b["n_runs"],
            "f32_time_sec": a["elapsed_sec_median"],
            "f64_time_sec": b["elapsed_sec_median"],
            "f32_metric": a["metric_median"],
            "f64_metric": b["metric_median"],
            "metric_difference_f32_minus_f64": a["metric_median"] - b["metric_median"],
            "f32_input_mb": a["input_storage_mb_median"],
            "f64_input_mb": b["input_storage_mb_median"],
            "f32_host_rss_mb": a["host_rss_mb_median"],
            "f64_host_rss_mb": b["host_rss_mb_median"],
            "f32_gpu_mb": a["gpu_mem_mb_median"],
            "f64_gpu_mb": b["gpu_mem_mb_median"],
        })
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(OUT / "float32_float64_pairs.csv", index=False)
    return pairs


def metal_portability():
    base = ROOT / "benchmark_results_backend_reproducibility_20260722"
    agreement = pd.read_csv(base / "metref_cpu_thread_metal_reproducibility.csv")
    rows = []
    for path in sorted((base / "metref_macos" / "rows").glob("*.csv")):
        row = pd.read_csv(path).iloc[0]
        if str(row["classifier"]) == "cknn":
            continue
        rows.append(row)
    timing = pd.DataFrame(rows)
    timing.to_csv(OUT / "metal_metref_raw.csv", index=False)
    agreement.to_csv(OUT / "metal_metref_agreement.csv", index=False)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    selected_backend_uncertainty()
    external_simpls_matched()
    cv_quantitative()
    precision_summary()
    metal_portability()
    print(OUT)


if __name__ == "__main__":
    main()
