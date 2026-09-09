"""Aggregate immutable StrictColdCal pilot evidence for the IEEE manuscript.

The script reads only the persisted run selected below.  It never re-runs a
model and never treats a configured-but-unexecuted dataset as experimental
evidence.  Output contains source digests so every manuscript number can be
traced back to a concrete JSON report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


RUN_ID = "run_5875420e9572"
PROJECT_SLUG = "dta-live-pipeline-check-v2"
MAIN_STAGE_ID = "5c24af30b53f"
SUPPLEMENTARY_STAGE_ID = "f689d48cd91b"
T_975_DF2 = 4.302652729911275


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected object in {path}")
    return value


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    vector = [float(value) for value in values]
    if not vector:
        raise ValueError("Cannot summarize an empty vector")
    mean = statistics.fmean(vector)
    sd = statistics.stdev(vector) if len(vector) > 1 else 0.0
    half_width = T_975_DF2 * sd / math.sqrt(len(vector)) if len(vector) == 3 else 0.0
    return {
        "n": len(vector),
        "mean": mean,
        "sample_sd": sd,
        "t95_ci_lower": mean - half_width,
        "t95_ci_upper": mean + half_width,
        "minimum": min(vector),
        "maximum": max(vector),
    }


def _metric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float | int]:
    return _summary(row[key] for row in rows)


def _compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metrics.get(key)
        for key in (
            "coverage",
            "mean_normalized_interval_width",
            "efficiency_ratio",
            "mae",
            "rmse",
            "spearman_width_error",
            "width_error_spearman_defined",
            "interval_width_constant",
        )
    }


def _source(path: Path, root: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(root)),
        "sha256": _sha256(path),
    }


def aggregate(root: Path) -> dict[str, Any]:
    local_root = (
        root
        / "project/var/local-compute/projects"
        / PROJECT_SLUG
        / "runs"
        / RUN_ID
        / MAIN_STAGE_ID
    )
    main_paths = sorted((local_root / "jobs").glob("*/pilot_report.json"))
    if len(main_paths) != 3:
        raise RuntimeError(f"Expected 3 primary reports, found {len(main_paths)}")

    primary_rows: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    for path in main_paths:
        report = _load(path)
        sources.append(_source(path, root))
        row = {
            "seed": int(report["seed"]),
            "validation_rmse": float(report["validation_rmse"]),
            "initial_loss": float(report["train_history"]["initial_loss"]),
            "final_loss": float(report["train_history"]["final_loss"]),
            "epochs": int(report["train_history"]["epochs"]),
            "split_cell_counts": report["split_cell_counts"],
            "label_scale": float(report["label_scale"]),
            "test_by_nominal_level": {
                level: _compact_metrics(values["test_metrics"])
                for level, values in report["per_level_metrics"].items()
            },
            "q_hat_by_nominal_level": {
                level: float(values["q_hat"])
                for level, values in report["per_level_metrics"].items()
            },
        }
        primary_rows.append(row)
    primary_rows.sort(key=lambda row: row["seed"])

    level_summaries: dict[str, Any] = {}
    for level in ("0.8", "0.9", "0.95"):
        level_rows = [row["test_by_nominal_level"][level] for row in primary_rows]
        level_summaries[level] = {
            key: _metric_summary(level_rows, key)
            for key in (
                "coverage",
                "mean_normalized_interval_width",
                "efficiency_ratio",
                "mae",
                "rmse",
                "spearman_width_error",
            )
        }

    launch_path = (
        root
        / "project/var/runs"
        / RUN_ID
        / "experiment_code_run"
        / SUPPLEMENTARY_STAGE_ID
        / "launch.json"
    )
    launch = _load(launch_path)
    sources.append(_source(launch_path, root))
    seed_by_job = {job["job_id"]: int(job["seed"]) for job in launch["jobs"]}
    supplementary_paths = sorted(
        (local_root / "experiment-runs" / SUPPLEMENTARY_STAGE_ID / "jobs").glob(
            "*/pilot_report_*.json"
        )
    )
    if len(supplementary_paths) != 12:
        raise RuntimeError(
            f"Expected 12 supplementary reports, found {len(supplementary_paths)}"
        )

    supplementary_reports: list[dict[str, Any]] = []
    for path in supplementary_paths:
        report = _load(path)
        job_id = path.parent.name
        report["_seed"] = seed_by_job[job_id]
        report["_job_id"] = job_id
        supplementary_reports.append(report)
        sources.append(_source(path, root))

    ablation_rows: list[dict[str, Any]] = []
    one_sided_rows: list[dict[str, Any]] = []
    negative_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    for report in supplementary_reports:
        experiment_id = report["experiment_id"]
        seed = int(report["_seed"])
        if experiment_id == "experiment-ablation-calibration-variants":
            for method, field in (
                ("normalized", "normalized_split_conformal_per_level"),
                ("absolute-residual", "unnormalized_absolute_residual_control_per_level"),
            ):
                metrics = _compact_metrics(report[field]["0.9"]["test_metrics"])
                ablation_rows.append({"seed": seed, "method": method, **metrics})
        elif experiment_id == "experiment-one-sided-diagnostics":
            for regime in ("drug_only", "target_only"):
                metrics = _compact_metrics(
                    report[regime]["per_level_metrics"]["0.9"]["test_metrics"]
                )
                one_sided_rows.append({"seed": seed, "regime": regime, **metrics})
        elif experiment_id == "experiment-negative-control-label-permutation":
            for condition, field in (
                ("real-label", "real_label_per_level_metrics"),
                ("permuted-train-label", "shuffled_label_per_level_metrics"),
            ):
                metrics = _compact_metrics(report[field]["0.9"]["test_metrics"])
                negative_rows.append({"seed": seed, "condition": condition, **metrics})
        elif experiment_id == "experiment-case-study-repurposing-intervals":
            metrics = _compact_metrics(
                report["family_per_level_metrics"]["0.9"]["test_metrics"]
            )
            case_rows.append(
                {
                    "seed": seed,
                    "target_family": report["selected_target_family"],
                    "pair_count": int(report["family_pair_count"]),
                    **metrics,
                }
            )

    def grouped_summary(
        rows: list[dict[str, Any]], group_key: str, metrics: tuple[str, ...]
    ) -> dict[str, Any]:
        groups = sorted({str(row[group_key]) for row in rows})
        return {
            group: {
                metric: _metric_summary(
                    [row for row in rows if str(row[group_key]) == group], metric
                )
                for metric in metrics
            }
            for group in groups
        }

    dataset_manifest_path = (
        root / "data" / PROJECT_SLUG / "manifests/00db90d9c8cd.json"
    )
    dataset_manifest = _load(dataset_manifest_path)
    sources.append(_source(dataset_manifest_path, root))
    datasets = []
    for record in dataset_manifest["records"]:
        profile = record["profile"]
        label_stats = profile["label_statistics"][0]
        datasets.append(
            {
                "requirement_key": record["requirement_key"],
                "status": record["status"],
                "rows": profile["row_count"],
                "unique_pairs": profile["unique_pairs"],
                "unique_compounds": profile["unique_compounds"],
                "unique_targets": profile["unique_targets"],
                "materialized_size_bytes": record["materialized_size_bytes"],
                "label_mean": label_stats["mean"],
                "label_std": label_stats["standard_deviation"],
                "label_min": label_stats["minimum"],
                "label_max": label_stats["maximum"],
                "executed_in_this_study": record["requirement_key"] == "dataset-davis",
            }
        )

    paired_negative_deltas: dict[str, Any] = {}
    for metric in ("rmse", "coverage", "mean_normalized_interval_width", "spearman_width_error"):
        real = {
            row["seed"]: float(row[metric])
            for row in negative_rows
            if row["condition"] == "real-label"
        }
        shuffled = {
            row["seed"]: float(row[metric])
            for row in negative_rows
            if row["condition"] == "permuted-train-label"
        }
        paired_negative_deltas[metric] = _summary(
            shuffled[seed] - real[seed] for seed in sorted(real)
        )

    result = {
        "schema_version": "1.0",
        "project_slug": PROJECT_SLUG,
        "run_id": RUN_ID,
        "evidence_scope": (
            "Observed local-CPU TinyMLPRegressor pilot on a deterministic Davis cap; "
            "KIBA and Metz are materialized but were not executed."
        ),
        "datasets": datasets,
        "primary": {
            "rows": primary_rows,
            "validation_rmse": _metric_summary(primary_rows, "validation_rmse"),
            "test_by_nominal_level": level_summaries,
        },
        "supplementary": {
            "ablation_90_rows": sorted(ablation_rows, key=lambda row: (row["method"], row["seed"])),
            "ablation_90_summary": grouped_summary(
                ablation_rows,
                "method",
                ("coverage", "mean_normalized_interval_width", "rmse", "spearman_width_error"),
            ),
            "one_sided_90_rows": sorted(one_sided_rows, key=lambda row: (row["regime"], row["seed"])),
            "one_sided_90_summary": grouped_summary(
                one_sided_rows,
                "regime",
                ("coverage", "mean_normalized_interval_width", "rmse", "spearman_width_error"),
            ),
            "negative_control_90_rows": sorted(
                negative_rows, key=lambda row: (row["condition"], row["seed"])
            ),
            "negative_control_90_summary": grouped_summary(
                negative_rows,
                "condition",
                ("coverage", "mean_normalized_interval_width", "rmse", "spearman_width_error"),
            ),
            "negative_control_paired_delta_permuted_minus_real": paired_negative_deltas,
            "case_study_90_rows": sorted(case_rows, key=lambda row: row["seed"]),
            "case_study_90_summary": {
                metric: _metric_summary(case_rows, metric)
                for metric in (
                    "pair_count",
                    "coverage",
                    "mean_normalized_interval_width",
                    "rmse",
                    "spearman_width_error",
                )
            },
        },
        "source_reports": sorted(sources, key=lambda source: source["path"]),
        "statistical_note": (
            "Means and sample standard deviations use three deterministic seeds. "
            "The t intervals use t(0.975, df=2)=4.30265 and are descriptive because n=3."
        ),
    }
    return result


def write_outputs(summary: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "evidence_summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    csv_path = output_dir / "primary_per_seed.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "seed",
            "validation_rmse",
            "test_rmse_90",
            "coverage_90",
            "normalized_width_90",
            "width_error_spearman_90",
            "train_pairs",
            "calibration_pairs",
            "validation_pairs",
            "test_pairs",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary["primary"]["rows"]:
            metrics = row["test_by_nominal_level"]["0.9"]
            cells = row["split_cell_counts"]
            writer.writerow(
                {
                    "seed": row["seed"],
                    "validation_rmse": row["validation_rmse"],
                    "test_rmse_90": metrics["rmse"],
                    "coverage_90": metrics["coverage"],
                    "normalized_width_90": metrics["mean_normalized_interval_width"],
                    "width_error_spearman_90": metrics["spearman_width_error"],
                    "train_pairs": cells["train"],
                    "calibration_pairs": cells["calibration"],
                    "validation_pairs": cells["validation"],
                    "test_pairs": cells["test"],
                }
            )
    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = aggregate(args.root.resolve())
    json_path, csv_path = write_outputs(summary, args.output_dir.resolve())
    print(json.dumps({
        "json": str(json_path),
        "json_sha256": _sha256(json_path),
        "csv": str(csv_path),
        "csv_sha256": _sha256(csv_path),
    }, indent=2))


if __name__ == "__main__":
    main()
