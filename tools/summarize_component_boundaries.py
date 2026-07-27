#!/usr/bin/env python3

import csv
from collections import Counter
from pathlib import Path


ROOT = Path("/Users/stefano/Documents/GPUPLS/fastPLS_fresh_github")
SOURCE = (
    ROOT
    / "benchmark_results"
    / "manuscript_revision_cycle17_20260725"
    / "component_selection_status.csv"
)
OUT = (
    ROOT
    / "benchmark_results"
    / "manuscript_revision_cycle43_20260726"
)


def classify(value):
    text = value.strip().lower()
    if text == "not evaluated":
        return "not_evaluated"
    if "upper tested-grid boundary" in text:
        return "upper_boundary"
    if "lower tested-grid boundary" in text:
        return "lower_boundary"
    if "response-rank limit" in text:
        return "response_rank_limit"
    if "interior tested value" in text or "one-se rule" in text:
        return "interior"
    raise ValueError(f"Unrecognized component-selection status: {value}")


def main():
    families = ("plssvd", "simpls", "opls", "kernelpls")
    records = []
    counts = {family: Counter() for family in families}

    with SOURCE.open(newline="") as handle:
        for row in csv.DictReader(handle):
            for family in families:
                status = classify(row[family])
                counts[family][status] += 1
                records.append(
                    {
                        "dataset": row["dataset"],
                        "family": family,
                        "evaluated_grid": row["evaluated_grid"],
                        "selection": row[family],
                        "status": status,
                    }
                )

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "component_boundary_detail.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    statuses = (
        "interior",
        "lower_boundary",
        "upper_boundary",
        "response_rank_limit",
        "not_evaluated",
    )
    summary = []
    for family in families:
        evaluated = sum(counts[family][s] for s in statuses if s != "not_evaluated")
        summary.append(
            {
                "family": family,
                "evaluated": evaluated,
                **{status: counts[family][status] for status in statuses},
            }
        )

    overall = Counter()
    for family in families:
        overall.update(counts[family])
    evaluated = sum(overall[s] for s in statuses if s != "not_evaluated")
    summary.append(
        {
            "family": "all",
            "evaluated": evaluated,
            **{status: overall[status] for status in statuses},
        }
    )
    with (OUT / "component_boundary_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    boundary = overall["lower_boundary"] + overall["upper_boundary"]
    constrained = boundary + overall["response_rank_limit"]
    print(
        f"Evaluated={evaluated}; tested-grid boundaries={boundary} "
        f"({100 * boundary / evaluated:.1f}%); response-rank limits="
        f"{overall['response_rank_limit']} "
        f"({100 * overall['response_rank_limit'] / evaluated:.1f}%); "
        f"interior={overall['interior']} "
        f"({100 * overall['interior'] / evaluated:.1f}%); "
        f"boundary-or-rank-constrained={constrained} "
        f"({100 * constrained / evaluated:.1f}%)."
    )


if __name__ == "__main__":
    main()
