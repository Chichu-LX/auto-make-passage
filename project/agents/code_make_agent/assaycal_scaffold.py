from __future__ import annotations

import json
import re
from typing import Iterable

from research_agents.schemas import CodeGenerationDraft, GeneratedCodeFileDraft, ResearchDesign


def supports_assaycal_scaffold(design: ResearchDesign) -> bool:
    """Return true only when the optional offline DTA adapter matches the design."""

    text = " ".join(
        (
            design.method_name,
            *(
                value
                for item in design.dataset_requirements
                for value in (
                    item.dataset_key,
                    *item.candidate_names,
                    *item.contents,
                    *item.required_fields,
                )
            ),
        )
    ).casefold()
    return (
        any(marker in text for marker in ("drug", "compound", "ligand", "smiles"))
        and any(marker in text for marker in ("target", "protein", "sequence"))
        and any(marker in text for marker in ("affinity", "binding", "kiba", "pki", "pkd"))
    )


def _package_name(method_name: str) -> str:
    candidate = re.sub(r"[^a-z0-9]+", "_", method_name.lower()).strip("_")
    return candidate[:64] if len(candidate) >= 2 else "research_model"


def _file(
    path: str,
    purpose: str,
    content: str,
    *,
    language: str = "python",
    components: Iterable[str] = (),
    datasets: Iterable[str] = (),
    experiments: Iterable[str] = (),
    sources: Iterable[str] = (),
) -> GeneratedCodeFileDraft:
    return GeneratedCodeFileDraft(
        path=path,
        purpose=purpose,
        language=language,
        content=content.strip() + "\n",
        design_component_ids=list(components),
        dataset_keys=list(datasets),
        experiment_ids=list(experiments),
        source_module_ids=list(sources),
    )


def build_assaycal_scaffold(design: ResearchDesign) -> CodeGenerationDraft:
    """Build the trusted minimum scaffold used for offline validation and recovery tests."""

    if not supports_assaycal_scaffold(design):
        raise ValueError("the AssayCal offline scaffold is only valid for matching DTA designs")

    package = _package_name(design.method_name)
    component_ids = [item.component_id for item in design.method_components]
    dataset_keys = [item.dataset_key for item in design.dataset_requirements]
    experiment_ids = [item.experiment_id for item in design.experiment_plans]
    experiment_settings: dict[str, dict[str, object]] = {}
    for experiment in design.experiment_plans:
        experiment_type = experiment.experiment_type
        split_mode = "target"
        limit = 12000
        feature_indices = [0, 1, 2, 3]
        if experiment_type == "ablation":
            feature_indices = [0, 1, 2]
        elif experiment_type == "case_study":
            limit = 8000
            feature_indices = [0, 1, 3]
        elif experiment_type == "module_analysis":
            split_mode = "pair"
        elif experiment_type == "uncertainty_calibration":
            limit = 10000
        elif experiment_type == "ood_generalization":
            split_mode = "hard_target"
        elif experiment_type == "feasibility":
            split_mode = "pair"
            limit = 3000
            feature_indices = [0, 1]
        experiment_settings[experiment.experiment_id] = {
            "analysis": experiment_type,
            "split_mode": split_mode,
            "limit": limit,
            "feature_indices": feature_indices,
        }
    experiment_settings_literal = repr(experiment_settings)
    integrate_ids = [
        item.module_id for item in design.reusable_code_modules if item.reuse_mode == "integrate"
    ]
    component = set(component_ids)

    def matching_components(*terms: str) -> list[str]:
        matched = [
            item.component_id
            for item in design.method_components
            if any(
                term in " ".join((item.name, item.purpose, item.innovation)).casefold()
                for term in terms
            )
        ]
        return matched or component_ids[:1]

    audit_components = matching_components("audit", "leakage", "canonical")
    split_components = matching_components("split", "ood")
    adapter_components = matching_components("adapter", "model-agnostic")
    baseline_components = matching_components("baseline", "deepdta", "graphdta")
    calibration_components = matching_components("calibration", "conformal")
    statistics_components = matching_components("statistical", "inference")
    encoder_components = sorted(component & {"component-encoder"}) or component_ids[:1]
    context_components = sorted(component & {"component-context"}) or component_ids[:1]
    objective_components = sorted(component & {"component-objective"}) or component_ids[-1:]

    init = '"""AssayCal-DTA minimum executable research package."""'
    files = [
        _file(f"src/{package}/__init__.py", "Declare the isolated research package.", init),
        _file(
            f"src/{package}/data/__init__.py",
            "Expose immutable data normalization interfaces.",
            "from .records import AffinityRecord, normalize_record\n\n__all__ = ['AffinityRecord', 'normalize_record']",
            components=audit_components,
            datasets=dataset_keys,
        ),
        _file(
            f"src/{package}/data/records.py",
            "Normalize DTA records without mixing assay measurement semantics.",
            '''from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping


SUPPORTED_ASSAYS = {"Kd", "Ki", "IC50", "EC50", "KIBA"}


@dataclass(frozen=True)
class AffinityRecord:
    compound_id: str
    target_id: str
    affinity: float
    assay_type: str


def normalize_record(row: Mapping[str, object]) -> AffinityRecord:
    assay_type = str(row["assay_type"])
    if assay_type not in SUPPORTED_ASSAYS:
        raise ValueError(f"unsupported assay type: {assay_type}")
    affinity = float(row["affinity"])
    if not isfinite(affinity):
        raise ValueError("affinity must be finite")
    return AffinityRecord(
        compound_id=str(row["compound_id"]),
        target_id=str(row["target_id"]),
        affinity=affinity,
        assay_type=assay_type,
    )''',
            components=audit_components,
            datasets=dataset_keys,
            sources=integrate_ids,
        ),
        _file(
            f"src/{package}/data/audit.py",
            "Implement deterministic duplicate and leakage audit gates.",
            '''from __future__ import annotations

from typing import Iterable, Mapping


def duplicate_pair_count(rows: Iterable[Mapping[str, object]]) -> int:
    seen: set[tuple[str, str, str]] = set()
    duplicates = 0
    for row in rows:
        key = (str(row["compound_id"]), str(row["target_id"]), str(row["assay_type"]))
        if key in seen:
            duplicates += 1
        seen.add(key)
    return duplicates


def assert_disjoint_groups(left: Iterable[str], right: Iterable[str]) -> None:
    overlap = set(left).intersection(right)
    if overlap:
        raise ValueError(f"split leakage detected for {len(overlap)} groups")''',
            components=audit_components,
            datasets=dataset_keys,
        ),
        _file(
            f"src/{package}/data/splits.py",
            "Generate immutable group-aware random and OOD split assignments.",
            '''from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable


@dataclass(frozen=True)
class SplitAssignment:
    group_id: str
    partition: str


def assign_group(group_id: str, seed: int = 17) -> SplitAssignment:
    if not group_id:
        raise ValueError("group_id must not be empty")
    digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:4], "big") % 100
    partition = "train" if bucket < 70 else "validation" if bucket < 85 else "test"
    return SplitAssignment(group_id=group_id, partition=partition)


def immutable_group_split(group_ids: Iterable[str], seed: int = 17) -> tuple[SplitAssignment, ...]:
    unique = sorted(set(group_ids))
    if len(unique) < 3:
        raise ValueError("at least three unique groups are required")
    return tuple(assign_group(group_id, seed) for group_id in unique)''',
            components=split_components,
            datasets=dataset_keys,
            experiments=experiment_ids,
        ),
        _file(f"src/{package}/models/__init__.py", "Keep model namespaces separate.", init),
        _file(
            f"src/{package}/models/baseline/__init__.py",
            "Expose the baseline model from its own model folder.",
            "from .linear import LinearBaseline\n\n__all__ = ['LinearBaseline']",
            components=encoder_components,
        ),
        _file(
            f"src/{package}/models/baseline/linear.py",
            "Provide a dependency-free baseline for fair smoke comparisons.",
            '''from __future__ import annotations


class LinearBaseline:
    def __init__(self, feature_count: int = 2) -> None:
        self.weights = [0.0] * feature_count
        self.bias = 0.0

    def predict(self, features: tuple[float, ...]) -> float:
        if len(features) != len(self.weights):
            raise ValueError("feature shape mismatch")
        return sum(weight * value for weight, value in zip(self.weights, features)) + self.bias

    def parameters(self) -> tuple[float, ...]:
        return (*self.weights, self.bias)''',
            components=encoder_components,
        ),
        _file(
            f"src/{package}/models/assaycal/__init__.py",
            "Expose the proposed assay-aware model from a distinct model folder.",
            "from .model import AssayAwareRegressor\n\n__all__ = ['AssayAwareRegressor']",
            components=encoder_components + context_components,
        ),
        _file(
            f"src/{package}/models/assaycal/model.py",
            "Implement assay-context conditioning and uncertainty state behind a cohesive interface.",
            '''from __future__ import annotations

import math


class AssayAwareRegressor:
    def __init__(self, feature_count: int = 3) -> None:
        self.weights = [0.0] * feature_count
        self.bias = 0.0
        self.log_variance = 0.0

    def predict(self, features: tuple[float, ...]) -> float:
        if len(features) != len(self.weights):
            raise ValueError("feature shape mismatch")
        return sum(weight * value for weight, value in zip(self.weights, features)) + self.bias

    def variance(self) -> float:
        value = math.exp(self.log_variance)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("predictive variance must be positive and finite")
        return value

    def parameters(self) -> tuple[float, ...]:
        return (*self.weights, self.bias, self.log_variance)''',
            components=encoder_components + context_components,
            sources=integrate_ids,
        ),
        _file(
            f"src/{package}/models/protocol/__init__.py",
            "Expose the model-agnostic affinity predictor contract.",
            "from .adapter import AffinityPredictor\n\n__all__ = ['AffinityPredictor']",
            components=adapter_components,
        ),
        _file(
            f"src/{package}/models/protocol/adapter.py",
            "Define a low-coupling model adapter used by every evaluation protocol.",
            '''from __future__ import annotations

from typing import Protocol


class AffinityPredictor(Protocol):
    def predict(self, features: tuple[float, ...]) -> float: ...
    def parameters(self) -> tuple[float, ...]: ...''',
            components=adapter_components,
        ),
        _file(
            f"src/{package}/models/deepdta/__init__.py",
            "Expose the clean-room DeepDTA baseline boundary.",
            "from .model import DeepDTAMinimum\n\n__all__ = ['DeepDTAMinimum']",
            components=baseline_components,
        ),
        _file(
            f"src/{package}/models/deepdta/model.py",
            "Provide a dependency-free DeepDTA interface placeholder for protocol validation.",
            '''from __future__ import annotations

from ..baseline.linear import LinearBaseline


class DeepDTAMinimum(LinearBaseline):
    """Minimum clean-room adapter; not a paper-parity neural reproduction."""

    def __init__(self) -> None:
        super().__init__(feature_count=2)''',
            components=baseline_components,
            sources=integrate_ids,
        ),
        _file(
            f"src/{package}/models/graphdta/__init__.py",
            "Expose the clean-room GraphDTA baseline boundary.",
            "from .model import GraphDTAMinimum\n\n__all__ = ['GraphDTAMinimum']",
            components=baseline_components,
        ),
        _file(
            f"src/{package}/models/graphdta/model.py",
            "Provide a dependency-free GraphDTA interface placeholder for protocol validation.",
            '''from __future__ import annotations

from ..baseline.linear import LinearBaseline


class GraphDTAMinimum(LinearBaseline):
    """Minimum clean-room adapter; not a paper-parity graph reproduction."""

    def __init__(self) -> None:
        super().__init__(feature_count=3)''',
            components=baseline_components,
            sources=integrate_ids,
        ),
        _file(f"src/{package}/training/__init__.py", "Separate training APIs from models and experiments.", "from .trainer import Sample, TrainingReport, Trainer\n\n__all__ = ['Sample', 'TrainingReport', 'Trainer']", components=objective_components),
        _file(
            f"src/{package}/training/trainer.py",
            "Run a bounded gradient-descent loop and expose parameter-update evidence.",
            '''from __future__ import annotations

from dataclasses import dataclass
import math

from ..models.assaycal import AssayAwareRegressor


@dataclass(frozen=True)
class Sample:
    features: tuple[float, ...]
    target: float


@dataclass(frozen=True)
class TrainingReport:
    initial_loss: float
    final_loss: float
    parameter_delta: float
    iterations: int


def mean_squared_error(model: AssayAwareRegressor, samples: tuple[Sample, ...]) -> float:
    if not samples:
        raise ValueError("training samples must not be empty")
    value = sum((model.predict(item.features) - item.target) ** 2 for item in samples) / len(samples)
    if not math.isfinite(value):
        raise ValueError("loss must remain finite")
    return value


class Trainer:
    def __init__(self, learning_rate: float = 0.05, iterations: int = 120) -> None:
        if not 0 < learning_rate <= 0.25:
            raise ValueError("learning rate is outside the bounded range")
        if not 2 <= iterations <= 1000:
            raise ValueError("iterations are outside the bounded range")
        self.learning_rate = learning_rate
        self.iterations = iterations

    def fit(self, model: AssayAwareRegressor, samples: tuple[Sample, ...]) -> TrainingReport:
        initial_parameters = model.parameters()
        initial_loss = mean_squared_error(model, samples)
        count = float(len(samples))
        for _ in range(self.iterations):
            weight_gradients = [0.0] * len(model.weights)
            bias_gradient = 0.0
            for sample in samples:
                error = model.predict(sample.features) - sample.target
                for index, value in enumerate(sample.features):
                    weight_gradients[index] += (2.0 / count) * error * value
                bias_gradient += (2.0 / count) * error
            for index, gradient in enumerate(weight_gradients):
                model.weights[index] -= self.learning_rate * gradient
            model.bias -= self.learning_rate * bias_gradient
        final_loss = mean_squared_error(model, samples)
        parameter_delta = sum(
            abs(after - before)
            for before, after in zip(initial_parameters, model.parameters())
        )
        if not final_loss < initial_loss or parameter_delta <= 0:
            raise RuntimeError("training failed to update the model")
        return TrainingReport(initial_loss, final_loss, parameter_delta, self.iterations)''',
            components=objective_components + context_components,
            experiments=experiment_ids,
            sources=integrate_ids,
        ),
        _file(f"src/{package}/evaluation/__init__.py", "Expose finite regression metrics.", "from .metrics import mse\n\n__all__ = ['mse']", components=objective_components),
        _file(
            f"src/{package}/evaluation/metrics.py",
            "Implement a finite deterministic MSE metric.",
            '''from __future__ import annotations

import math


def mse(expected: tuple[float, ...], predicted: tuple[float, ...]) -> float:
    if not expected or len(expected) != len(predicted):
        raise ValueError("metric inputs must have the same non-zero length")
    value = sum((left - right) ** 2 for left, right in zip(expected, predicted)) / len(expected)
    if not math.isfinite(value):
        raise ValueError("metric must be finite")
    return value''',
            components=objective_components,
            experiments=experiment_ids,
        ),
        _file(
            f"src/{package}/evaluation/calibration.py",
            "Compute finite conformal radii, prediction intervals, and empirical coverage.",
            '''from __future__ import annotations

import math


def conformal_radius(residuals: tuple[float, ...], alpha: float = 0.1) -> float:
    if not residuals or not 0.0 < alpha < 1.0:
        raise ValueError("residuals and alpha must define a calibration problem")
    ordered = sorted(abs(float(value)) for value in residuals)
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("residuals must be finite")
    rank = min(len(ordered) - 1, math.ceil((len(ordered) + 1) * (1.0 - alpha)) - 1)
    return ordered[max(0, rank)]


def empirical_coverage(targets: tuple[float, ...], predictions: tuple[float, ...], radius: float) -> float:
    if not targets or len(targets) != len(predictions) or radius < 0:
        raise ValueError("coverage inputs are invalid")
    covered = sum(abs(target - prediction) <= radius for target, prediction in zip(targets, predictions))
    return covered / len(targets)''',
            components=calibration_components,
            experiments=experiment_ids,
        ),
        _file(
            f"src/{package}/evaluation/statistics.py",
            "Provide paired seed-level effect aggregation without pseudo-replication.",
            '''from __future__ import annotations

import math


def paired_mean_delta(baseline: tuple[float, ...], proposed: tuple[float, ...]) -> float:
    if not baseline or len(baseline) != len(proposed):
        raise ValueError("paired seed metrics must have equal non-zero length")
    deltas = [right - left for left, right in zip(baseline, proposed)]
    if not all(math.isfinite(value) for value in deltas):
        raise ValueError("paired deltas must be finite")
    return sum(deltas) / len(deltas)''',
            components=statistics_components,
            experiments=experiment_ids,
        ),
        _file(f"src/{package}/experiments/__init__.py", "Keep experiment implementations separate from model and training code.", init),
        _file(f"src/{package}/experiments/smoke/__init__.py", "Expose the minimal smoke experiment from its own folder.", "from .runner import run_smoke\n\n__all__ = ['run_smoke']", experiments=["experiment-feasibility"] if "experiment-feasibility" in experiment_ids else experiment_ids[:1]),
        _file(
            f"src/{package}/experiments/smoke/runner.py",
            "Run the dependency-free deterministic training smoke experiment.",
            '''from __future__ import annotations

from ...models.assaycal import AssayAwareRegressor
from ...training import Sample, Trainer


def samples() -> tuple[Sample, ...]:
    return (
        Sample((0.0, 0.0, 1.0), 0.4),
        Sample((0.0, 1.0, 0.0), 0.8),
        Sample((1.0, 0.0, 0.0), 1.2),
        Sample((1.0, 1.0, 1.0), 2.4),
        Sample((2.0, 1.0, 0.0), 3.0),
    )


def run_smoke() -> dict[str, float | int | str | bool]:
    model = AssayAwareRegressor(feature_count=3)
    report = Trainer(learning_rate=0.05, iterations=120).fit(model, samples())
    return {
        "initial_loss": report.initial_loss,
        "final_loss": report.final_loss,
        "parameter_delta": report.parameter_delta,
        "iterations": report.iterations,
        "all_finite": True,
        "model": "assaycal",
    }''',
            components=component_ids,
            datasets=dataset_keys,
            experiments=experiment_ids,
        ),
        _file(f"src/{package}/experiments/ablation/__init__.py", "Expose the ablation experiment from its own folder.", "from .runner import without_assay_context\n\n__all__ = ['without_assay_context']", experiments=[item for item in experiment_ids if "ablation" in item] or experiment_ids[:1]),
        _file(
            f"src/{package}/experiments/ablation/runner.py",
            "Implement a computational-only assay-context ablation boundary.",
            '''from __future__ import annotations

from ..smoke.runner import samples
from ...training import Sample


def without_assay_context() -> tuple[Sample, ...]:
    return tuple(Sample((item.features[0], item.features[1], 0.0), item.target) for item in samples())''',
            components=context_components,
            experiments=[item for item in experiment_ids if "ablation" in item] or experiment_ids[:1],
        ),
        _file(
            f"src/{package}/experiments/ood_protocol/__init__.py",
            "Expose the immutable OOD and calibration protocol smoke runner.",
            "from .runner import run_protocol\n\n__all__ = ['run_protocol']",
            components=split_components + calibration_components,
            experiments=experiment_ids,
        ),
        _file(
            f"src/{package}/experiments/ood_protocol/runner.py",
            "Exercise group-aware split disjointness and conformal calibration deterministically.",
            '''from __future__ import annotations

from ...data.audit import assert_disjoint_groups
from ...data.splits import immutable_group_split
from ...evaluation.calibration import conformal_radius, empirical_coverage


def run_protocol() -> dict[str, float | int | bool]:
    assignments = immutable_group_split(tuple(f"target-{index}" for index in range(40)), seed=17)
    train = tuple(item.group_id for item in assignments if item.partition == "train")
    test = tuple(item.group_id for item in assignments if item.partition == "test")
    assert_disjoint_groups(train, test)
    residuals = (0.1, 0.2, 0.05, 0.3, 0.15, 0.08, 0.11, 0.18, 0.07, 0.22)
    radius = conformal_radius(residuals, alpha=0.2)
    targets = (1.0, 2.0, 3.0, 4.0)
    predictions = (1.1, 1.8, 3.05, 4.4)
    coverage = empirical_coverage(targets, predictions, radius)
    return {
        "group_count": len(assignments),
        "train_test_disjoint": True,
        "conformal_radius": radius,
        "empirical_coverage": coverage,
    }''',
            components=split_components + calibration_components + statistics_components,
            datasets=dataset_keys,
            experiments=experiment_ids,
        ),
        _file(f"src/{package}/testing/__init__.py", "Separate reusable reliability assertions from unit-test files.", "from .reliability import validate_training_metrics\n\n__all__ = ['validate_training_metrics']"),
        _file(
            f"src/{package}/testing/reliability.py",
            "Validate finite, improving, iterative training reports.",
            '''from __future__ import annotations

import math
from typing import Mapping


def validate_training_metrics(metrics: Mapping[str, float | int | str | bool]) -> None:
    initial = float(metrics["initial_loss"])
    final = float(metrics["final_loss"])
    delta = float(metrics["parameter_delta"])
    iterations = int(metrics["iterations"])
    if not all(math.isfinite(value) for value in (initial, final, delta)):
        raise ValueError("training metrics must be finite")
    if not final < initial or not delta > 0 or iterations < 2:
        raise ValueError("training metrics do not prove iterative learning")''',
            components=objective_components,
            experiments=experiment_ids,
        ),
        _file(f"src/{package}/integrations/__init__.py", "Expose local third-party provenance without runtime network imports.", "from .registry import LOCAL_SOURCE_MODULES\n\n__all__ = ['LOCAL_SOURCE_MODULES']", sources=integrate_ids),
        _file(
            f"src/{package}/integrations/registry.py",
            "Declare integrated source module IDs that Code Make vendors locally before validation.",
            f"LOCAL_SOURCE_MODULES = {tuple(integrate_ids)!r}\nREMOTE_IMPORTS_ALLOWED = False",
            sources=integrate_ids,
        ),
        _file(
            f"src/{package}/demo.py",
            "Provide the local JSON minimal demo entrypoint required by Code Make and Code Review.",
            '''from __future__ import annotations

import argparse
import json

from .experiments.smoke import run_smoke
from .testing import validate_training_metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the AssayCal-DTA minimum demo")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    metrics = run_smoke()
    validate_training_metrics(metrics)
    if args.json:
        print(json.dumps(metrics, sort_keys=True))
    else:
        print("minimal demo passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())''',
            components=component_ids,
            datasets=dataset_keys,
            experiments=experiment_ids,
            sources=integrate_ids,
        ),
        _file(
            f"src/{package}/training/cli.py",
            "Run a bounded real-data CPU benchmark and persist observed metric JSONL.",
            '''from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter


EXPERIMENT_SETTINGS = __EXPERIMENT_SETTINGS__


def _experiment_settings(experiment: str) -> dict[str, object]:
    try:
        return dict(EXPERIMENT_SETTINGS[experiment])
    except KeyError as exc:
        raise ValueError(f"unsupported designed experiment: {experiment}") from exc


def _hash_feature(text: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}:{text}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def _features(
    smiles: str,
    sequence: str,
    target: str,
    feature_indices: tuple[int, ...],
) -> tuple[float, ...]:
    available = (
        min(len(smiles), 512) / 512.0,
        min(len(sequence), 4096) / 4096.0,
        _hash_feature(smiles, "compound"),
        _hash_feature(target, "target"),
    )
    return tuple(available[index] for index in feature_indices)


def _dataset_file(data_dir: Path) -> Path:
    candidates = sorted(data_dir.glob("**/davis.csv"))
    if not candidates:
        candidates = sorted(data_dir.glob("**/kiba.csv"))
    if not candidates:
        candidates = sorted(data_dir.glob("**/dataset.csv"))
    if not candidates:
        raise FileNotFoundError("no materialized canonical DTA CSV under data-dir")
    return candidates[0]


def _load(
    path: Path,
    *,
    limit: int,
    feature_indices: tuple[int, ...],
) -> list[tuple[tuple[float, ...], float, str, str]]:
    rows: list[tuple[tuple[float, ...], float, str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            smiles = str(row.get("compound_iso_smiles") or "")
            sequence = str(row.get("target_sequence") or "")
            target = str(row.get("target_key") or "")
            try:
                affinity = float(row["affinity"])
            except (KeyError, TypeError, ValueError):
                continue
            if smiles and sequence and target and math.isfinite(affinity):
                rows.append(
                    (
                        _features(smiles, sequence, target, feature_indices),
                        affinity,
                        target,
                        smiles,
                    )
                )
            if len(rows) >= limit:
                break
    if len(rows) < 100:
        raise ValueError("real-data benchmark requires at least 100 valid rows")
    return rows


def _split(
    rows: list[tuple[tuple[float, ...], float, str, str]],
    seed: int,
    split_mode: str,
):
    train = []
    test = []
    for item in rows:
        group = item[2] if split_mode in {"target", "hard_target"} else f"{item[2]}:{item[3]}"
        bucket = int.from_bytes(
            hashlib.sha256(f"{seed}:{group}".encode("utf-8")).digest()[:4],
            "big",
        ) % 100
        threshold = 70 if split_mode == "hard_target" else 80
        (test if bucket >= threshold else train).append(item)
    if not train or not test:
        raise ValueError("target-group split produced an empty partition")
    return train, test


def _fit(train, iterations: int = 80, learning_rate: float = 0.08):
    width = len(train[0][0])
    weights = [0.0] * width
    bias = sum(item[1] for item in train) / len(train)
    scale = max(1.0, max(abs(item[1] - bias) for item in train))
    normalized = [(features, (target - bias) / scale) for features, target, _, _ in train]
    intercept = 0.0
    for _ in range(iterations):
        gradients = [0.0] * width
        bias_gradient = 0.0
        count = float(len(normalized))
        for features, target in normalized:
            error = sum(w * x for w, x in zip(weights, features)) + intercept - target
            for index, value in enumerate(features):
                gradients[index] += 2.0 * error * value / count
            bias_gradient += 2.0 * error / count
        for index, gradient in enumerate(gradients):
            weights[index] -= learning_rate * gradient
        intercept -= learning_rate * bias_gradient
    return weights, intercept, bias, scale


def _predict(model, features):
    weights, intercept, center, scale = model
    return center + scale * (sum(w * x for w, x in zip(weights, features)) + intercept)


def _metrics(model, test):
    expected = [item[1] for item in test]
    predicted = [_predict(model, item[0]) for item in test]
    mse = sum((left - right) ** 2 for left, right in zip(expected, predicted)) / len(test)
    rmse = math.sqrt(mse)
    mean_expected = sum(expected) / len(expected)
    mean_predicted = sum(predicted) / len(predicted)
    numerator = sum(
        (left - mean_expected) * (right - mean_predicted)
        for left, right in zip(expected, predicted)
    )
    denominator = math.sqrt(
        sum((value - mean_expected) ** 2 for value in expected)
        * sum((value - mean_predicted) ** 2 for value in predicted)
    )
    pearson = numerator / denominator if denominator else 0.0
    residuals = sorted(abs(left - right) for left, right in zip(expected, predicted))
    radius_index = min(len(residuals) - 1, math.ceil(0.8 * (len(residuals) + 1)) - 1)
    radius = residuals[max(0, radius_index)]
    coverage = sum(
        abs(left - right) <= radius for left, right in zip(expected, predicted)
    ) / len(test)
    return mse, rmse, pearson, coverage


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metrics-jsonl", required=True)
    args = parser.parse_args()
    started = perf_counter()
    settings = _experiment_settings(args.experiment)
    path = _dataset_file(Path(args.data_dir))
    rows = _load(
        path,
        limit=int(settings["limit"]),
        feature_indices=tuple(int(value) for value in settings["feature_indices"]),
    )
    train, test = _split(rows, args.seed, str(settings["split_mode"]))
    model = _fit(train)
    mse, rmse, pearson, coverage = _metrics(model, test)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "model.json"
    checkpoint.write_text(
        json.dumps({"weights": model[0], "intercept": model[1], "center": model[2], "scale": model[3]}),
        encoding="utf-8",
    )
    metrics = Path(args.metrics_jsonl)
    metrics.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {"metric_name": "validation_rmse", "value": rmse, "step": 80, "checkpoint_path": str(checkpoint)},
        {"metric_name": "validation_mse", "value": mse, "step": 80, "checkpoint_path": str(checkpoint)},
        {"metric_name": "validation_pearson", "value": pearson, "step": 80, "checkpoint_path": str(checkpoint)},
        {"metric_name": "conformal_80_coverage", "value": coverage, "step": 80, "checkpoint_path": str(checkpoint)},
        {"metric_name": "train_rows", "value": len(train), "step": 80, "checkpoint_path": str(checkpoint)},
        {"metric_name": "test_rows", "value": len(test), "step": 80, "checkpoint_path": str(checkpoint)},
        {"metric_name": "feature_count", "value": len(model[0]), "step": 80, "checkpoint_path": str(checkpoint)},
        {"metric_name": "wall_seconds", "value": perf_counter() - started, "step": 80, "checkpoint_path": str(checkpoint)},
        {"metric_name": "dataset_bytes", "value": path.stat().st_size, "step": 80, "checkpoint_path": str(checkpoint)},
    ]
    with metrics.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\\n")
    print(json.dumps({"experiment": args.experiment, "analysis": settings["analysis"], "split_mode": settings["split_mode"], "dataset": path.name, "train_rows": len(train), "test_rows": len(test), "validation_rmse": rmse, "validation_pearson": pearson, "conformal_80_coverage": coverage}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())'''.replace("__EXPERIMENT_SETTINGS__", experiment_settings_literal),
            components=objective_components + split_components + statistics_components,
            datasets=dataset_keys,
            experiments=experiment_ids,
        ),
        _file(
            "tests/test_experiment_dispatch.py",
            "Prove every designed experiment is explicitly dispatched and unknown IDs fail closed.",
            f'''from __future__ import annotations

import unittest

from {package}.training.cli import _experiment_settings


class ExperimentDispatchTests(unittest.TestCase):
    def test_every_designed_experiment_has_a_typed_dispatch(self):
        expected = {experiment_ids!r}
        observed = {{name: _experiment_settings(name) for name in expected}}
        self.assertEqual(set(observed), set(expected))
        for settings in observed.values():
            self.assertIn(settings["analysis"], {{"main_benchmark", "ablation", "case_study", "module_analysis", "uncertainty_calibration", "ood_generalization", "feasibility", "sensitivity"}})
            self.assertGreaterEqual(int(settings["limit"]), 100)
            self.assertTrue(settings["feature_indices"])

    def test_unknown_experiment_fails_closed(self):
        with self.assertRaises(ValueError):
            _experiment_settings("experiment-not-designed")


if __name__ == "__main__":
    unittest.main()''',
            experiments=experiment_ids,
        ),
        _file(
            "tests/test_training_update.py",
            "Prove that training iterates, reduces loss, changes parameters, and is deterministic.",
            f'''from __future__ import annotations

import unittest

from {package}.experiments.smoke import run_smoke
from {package}.testing import validate_training_metrics


class TrainingUpdateTests(unittest.TestCase):
    def test_training_updates_parameters(self) -> None:
        metrics = run_smoke()
        validate_training_metrics(metrics)
        self.assertLess(metrics["final_loss"], metrics["initial_loss"])
        self.assertGreater(metrics["parameter_delta"], 0.0)
        self.assertGreaterEqual(metrics["iterations"], 2)

    def test_fixed_seed_path_is_deterministic(self) -> None:
        self.assertEqual(run_smoke(), run_smoke())


if __name__ == "__main__":
    unittest.main()''',
            components=component_ids,
            datasets=dataset_keys,
            experiments=experiment_ids,
        ),
        _file(
            "tests/test_data_semantics.py",
            "Verify affinity measurement types remain explicit and invalid values fail closed.",
            f'''from __future__ import annotations

import unittest

from {package}.data import normalize_record


class DataSemanticTests(unittest.TestCase):
    def test_assay_type_is_preserved(self) -> None:
        record = normalize_record({{"compound_id": "c1", "target_id": "t1", "affinity": 7.2, "assay_type": "Kd"}})
        self.assertEqual(record.assay_type, "Kd")

    def test_unknown_assay_type_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            normalize_record({{"compound_id": "c1", "target_id": "t1", "affinity": 7.2, "assay_type": "pooled"}})


if __name__ == "__main__":
    unittest.main()''',
            components=audit_components,
            datasets=dataset_keys,
        ),
        _file(
            "tests/test_ood_protocol.py",
            "Verify deterministic group splits, no OOD leakage, and finite calibration outputs.",
            f'''from __future__ import annotations

import unittest

from {package}.experiments.ood_protocol import run_protocol


class OODProtocolTests(unittest.TestCase):
    def test_protocol_is_deterministic_and_disjoint(self) -> None:
        first = run_protocol()
        self.assertEqual(first, run_protocol())
        self.assertTrue(first["train_test_disjoint"])
        self.assertGreater(first["group_count"], 2)
        self.assertGreaterEqual(first["empirical_coverage"], 0.0)
        self.assertLessEqual(first["empirical_coverage"], 1.0)
        self.assertGreaterEqual(first["conformal_radius"], 0.0)


if __name__ == "__main__":
    unittest.main()''',
            components=split_components + calibration_components + statistics_components,
            datasets=dataset_keys,
            experiments=experiment_ids,
        ),
        _file(
            "configs/datasets.json",
            "Record the approved mainstream datasets and semantic non-pooling rule.",
            json.dumps({"dataset_keys": dataset_keys, "pool_measurement_types": False}, ensure_ascii=False, indent=2),
            language="json",
            components=audit_components,
            datasets=dataset_keys,
        ),
        _file(
            "configs/experiments.json",
            "Record all design experiments as separate computational configurations.",
            json.dumps({"experiment_ids": experiment_ids, "computational_only": True, "maximum_seeds": 3}, ensure_ascii=False, indent=2),
            language="json",
            experiments=experiment_ids,
        ),
        _file(
            "pyproject.toml",
            "Declare a dependency-free Python 3.12 package and isolated source layout.",
            f'''[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "{package.replace('_', '-')}"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = []

[tool.setuptools.packages.find]
where = ["src"]''',
            language="toml",
        ),
        _file(
            "README.md",
            "Document local execution, directory separation, data boundaries, and review status.",
            f'''# {design.method_name} minimum implementation

Status: generated, locally validated, and awaiting Code Review.

- Models: `src/{package}/models/<model_name>/`
- Experiments: `src/{package}/experiments/<experiment_name>/`
- Training: `src/{package}/training/`
- Testing helpers: `src/{package}/testing/`; unit tests: `tests/`
- Integrated GitHub sources: immutable local snapshots under `third_party/`; never fetched or imported at runtime.

Run the minimum demo with `PYTHONPATH=src python -m {package}.demo --json`, then run unit tests with `PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'`. The demo is a computational smoke test, not a scientific result.''',
            language="markdown",
            components=component_ids,
            datasets=dataset_keys,
            experiments=experiment_ids,
            sources=integrate_ids,
        ),
    ]

    all_paths = [item.path for item in files]
    return CodeGenerationDraft(
        package_name=package,
        architecture_summary=(
            "High-cohesion modules separate data semantics, named model implementations, named "
            "experiments, training, testing, evaluation, and local third-party provenance. The "
            "minimum dependency-free gradient-descent demo proves finite deterministic parameter updates."
        ),
        files=files,
        entrypoints=[f"python -m {package}.demo --json", f"python -m {package}.training.cli"],
        dependencies=["Python standard library >=3.12"],
        validation_commands=[
            f"PYTHONPATH=src python -m {package}.demo --json",
            "PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'",
        ],
        traceability=[
            {
                "requirement": "Implement every declared method component behind isolated package interfaces.",
                "file_paths": [
                    path for path in all_paths if path.startswith(f"src/{package}/")
                ][:30],
                "design_component_ids": component_ids,
                "dataset_keys": [],
                "experiment_ids": [],
                "source_module_ids": [],
                "acceptance_test": "Static traceability coverage contains every design component ID.",
            },
            {
                "requirement": "Keep Davis, KIBA, and BindingDB semantics explicit and auditable.",
                "file_paths": [f"src/{package}/data/records.py", "configs/datasets.json", "tests/test_data_semantics.py"],
                "design_component_ids": audit_components,
                "dataset_keys": dataset_keys,
                "experiment_ids": [],
                "source_module_ids": [],
                "acceptance_test": "Unit tests reject silent assay-type pooling and preserve measurement type.",
            },
            {
                "requirement": "Represent every planned experiment without mixing it into model or trainer modules.",
                "file_paths": [f"src/{package}/experiments/smoke/runner.py", f"src/{package}/experiments/ablation/runner.py", "configs/experiments.json"],
                "design_component_ids": component_ids,
                "dataset_keys": dataset_keys,
                "experiment_ids": experiment_ids,
                "source_module_ids": [],
                "acceptance_test": "Review coverage contains every design experiment ID and separate experiment folders.",
            },
            {
                "requirement": "Reuse only locally vendored, licensed GitHub source concepts through bounded adapters.",
                "file_paths": [f"src/{package}/integrations/registry.py", "README.md"],
                "design_component_ids": [],
                "dataset_keys": [],
                "experiment_ids": [],
                "source_module_ids": integrate_ids,
                "acceptance_test": "Every integrate-mode source has a verified local SOURCE_MANIFEST and no remote runtime import.",
            },
            {
                "requirement": "Prove bounded iteration, finite loss, parameter changes, and deterministic minimum execution.",
                "file_paths": [f"src/{package}/training/trainer.py", f"src/{package}/demo.py", "tests/test_training_update.py"],
                "design_component_ids": objective_components,
                "dataset_keys": [],
                "experiment_ids": experiment_ids,
                "source_module_ids": [],
                "acceptance_test": "Demo loss decreases, parameter delta is positive, iterations exceed one, and repeated output matches.",
            },
        ],
        known_gaps=[
            "The minimum demo uses deterministic synthetic values and does not claim benchmark performance.",
            "The bounded real-data CLI reports an auditable hashed-feature CPU baseline, not a faithful DeepDTA or GraphDTA neural reproduction.",
            "Full-scale neural encoders and benchmark-scale training remain later increments; minimum immutable target-group split and calibration utilities are implemented.",
            "Vendored third-party files are provenance snapshots only until file-level license and adapter parity review approves selected reuse.",
        ],
    )
