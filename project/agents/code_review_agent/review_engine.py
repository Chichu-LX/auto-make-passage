from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

from research_agents.schemas import (
    ArtifactRef,
    CodeImplementationManifest,
    CodeReviewCheck,
    CodeReviewExecutionRecord,
    ResearchDesign,
)
from research_agents.storage import ArtifactStore, StoragePolicyError

from .policy import (
    EXECUTION_TIMEOUT_SECONDS,
    MAX_CODE_FILE_BYTES,
    MAX_INPUT_BYTES,
    MAX_SOURCE_FILE_BYTES,
    MAX_SOURCE_TOTAL_BYTES,
    OUTPUT_TAIL_CHARS,
)


BANNED_IMPORT_ROOTS = {
    "asyncio",
    "ctypes",
    "http",
    "httpx",
    "multiprocessing",
    "requests",
    "resource",
    "shutil",
    "signal",
    "socket",
    "subprocess",
    "urllib",
}
BANNED_CALL_NAMES = {"compile", "eval", "exec", "__import__"}
BANNED_ATTRIBUTE_CALLS = {
    ("pathlib", "Path"),
    ("os", "popen"),
    ("os", "remove"),
    ("os", "removedirs"),
    ("os", "rmdir"),
    ("os", "system"),
    ("os", "unlink"),
}
REMOTE_MARKERS = ("http://", "https://", "git+", "pip install", "curl ", "wget ")


def _unsafe_cross_dataset_aggregation(source_text: str) -> bool:
    """Flag direct aggregation of datasets without an executable semantics guard."""

    normalized = source_text.casefold()
    suspicious_pooling = re.search(
        r"\b(?:pairs|rows|records)\.extend\s*\(", normalized
    )
    direct_dataset_pooling = re.search(
        r"\b(?:pairs|rows|records)\.extend\s*\(\s*_?load_(?:dataset_)?pairs\b",
        normalized,
    )
    has_measurement_guard = (
        "pool_measurement_types" in normalized
        and "false" in normalized
        and any(
            marker in normalized
            for marker in ("assay_type", "dataset_key", "measurement_unit")
        )
    )
    return bool(
        direct_dataset_pooling or (suspicious_pooling and not has_measurement_guard)
    )


# Backward-compatible import for existing callers while the public name becomes
# domain-neutral.
_unsafe_cross_dataset_pooling = _unsafe_cross_dataset_aggregation


def _check(
    check_id: str,
    category: str,
    failures: list[str],
    successes: list[str],
    *,
    files: list[str] | None = None,
    warnings: list[str] | None = None,
) -> CodeReviewCheck:
    warning_values = warnings or []
    if failures:
        status = "failed"
        summary = "; ".join(failures)[:3000]
        evidence = [*successes, *failures]
    elif warning_values:
        status = "warning"
        summary = "; ".join(warning_values)[:3000]
        evidence = [*successes, *warning_values]
    else:
        status = "passed"
        summary = "; ".join(successes)[:3000]
        evidence = successes
    return CodeReviewCheck(
        check_id=check_id,
        category=category,
        status=status,
        summary=summary or "Review check completed without additional evidence.",
        evidence=[value[:1000] for value in evidence[:50]],
        file_paths=(files or [])[:50],
    )


def architecture_check(manifest: CodeImplementationManifest) -> CodeReviewCheck:
    package_root = f"src/{manifest.package_name}/"
    paths = {item.path for item in manifest.files}
    failures: list[str] = []
    required = {
        "models": f"{package_root}models/",
        "experiments": f"{package_root}experiments/",
        "training": f"{package_root}training/",
        "testing": f"{package_root}testing/",
        "tests": "tests/",
        "configs": "configs/",
    }
    for name, prefix in required.items():
        if not any(path.startswith(prefix) for path in paths):
            failures.append(f"missing isolated {name} layer at {prefix}")
    model_relatives = [
        path.removeprefix(required["models"])
        for path in paths
        if path.startswith(required["models"]) and path.endswith(".py")
    ]
    experiment_relatives = [
        path.removeprefix(required["experiments"])
        for path in paths
        if path.startswith(required["experiments"]) and path.endswith(".py")
    ]
    if not any("/" in value for value in model_relatives):
        failures.append("model implementations are not separated into named model folders")
    if not any("/" in value for value in experiment_relatives):
        failures.append("experiments are not separated into named experiment folders")
    demo = f"{package_root}demo.py"
    if demo not in paths:
        failures.append("minimal demo entrypoint is missing")
    successes = [
        f"implementation revision is isolated at {manifest.implementation_root}",
        f"reviewed {len(paths)} manifest-owned files",
        "models, experiments, training, and testing have independent package boundaries",
    ]
    return _check("review-architecture-layout", "architecture", failures, successes, files=sorted(paths))


def traceability_check(
    manifest: CodeImplementationManifest, design: ResearchDesign
) -> tuple[CodeReviewCheck, dict[str, list[str]]]:
    covered = {
        "components": sorted({value for item in manifest.files for value in item.design_component_ids}),
        "datasets": sorted({value for item in manifest.files for value in item.dataset_keys}),
        "experiments": sorted({value for item in manifest.files for value in item.experiment_ids}),
        "sources": sorted({value for item in manifest.files for value in item.source_module_ids}),
    }
    expected = {
        "components": sorted(item.component_id for item in design.method_components),
        "datasets": sorted(item.dataset_key for item in design.dataset_requirements),
        "experiments": sorted(item.experiment_id for item in design.experiment_plans),
        "sources": sorted(
            item.module_id for item in design.reusable_code_modules if item.reuse_mode == "integrate"
        ),
    }
    failures = [
        f"{name} coverage differs: expected={expected[name]}, covered={covered[name]}"
        for name in expected
        if covered[name] != expected[name]
    ]
    successes = [
        f"component coverage={len(covered['components'])}",
        f"dataset coverage={len(covered['datasets'])}",
        f"experiment coverage={len(covered['experiments'])}",
        f"integrated source coverage={len(covered['sources'])}",
    ]
    return (
        _check("review-design-traceability", "traceability", failures, successes),
        covered,
    )


def innovation_alignment_check(
    manifest: CodeImplementationManifest, design: ResearchDesign
) -> CodeReviewCheck:
    """Require concrete modules for core design mechanisms, not traceability labels alone."""

    paths = {item.path for item in manifest.files}
    package = f"src/{manifest.package_name}/"
    design_text = " ".join(
        " ".join((item.name, item.purpose, item.innovation)).casefold()
        for item in design.method_components
    )
    requirements: list[tuple[str, tuple[str, ...]]] = []
    if "split" in design_text or "ood" in design_text:
        requirements.append(
            (
                "pre-registered OOD split generator",
                (f"{package}data/splits.py", f"{package}data/split_manager.py"),
            )
        )
    if "audit" in design_text or "leakage" in design_text:
        requirements.append(
            (
                "data audit and leakage gates",
                (f"{package}data/audit.py", f"{package}data/split_manager.py"),
            )
        )
    if "adapter" in design_text or "model-agnostic" in design_text:
        requirements.append(
            (
                "model adapter contract",
                (f"{package}models/protocol/adapter.py", f"{package}models/base.py"),
            )
        )
    if "deepdta" in design_text:
        requirements.append(
            (
                "DeepDTA baseline boundary",
                (
                    f"{package}models/deepdta/model.py",
                    f"{package}models/deepdta_style/model.py",
                ),
            )
        )
    if "graphdta" in design_text:
        requirements.append(
            (
                "GraphDTA baseline boundary",
                (
                    f"{package}models/graphdta/model.py",
                    f"{package}models/graphdta_style/model.py",
                ),
            )
        )
    if "calibration" in design_text or "conformal" in design_text:
        requirements.append(
            (
                "calibration evaluator",
                (
                    f"{package}evaluation/calibration.py",
                    f"{package}calibration/conformal.py",
                ),
            )
        )
    if "statistical" in design_text or "inference" in design_text:
        requirements.append(("statistical inference module", (f"{package}evaluation/statistics.py",)))
    failures = [
        f"missing concrete {label}: expected one of {list(expected)}"
        for label, expected in requirements
        if not any(path in paths for path in expected)
    ]
    successes = [
        f"concrete innovation modules present for {len(requirements) - len(failures)}/{len(requirements)} required mechanisms",
        "alignment is derived from ResearchDesign component semantics and manifest-owned paths",
    ]
    return _check(
        "review-innovation-alignment",
        "design_alignment",
        failures,
        successes,
        files=sorted(path for path in paths if path.startswith(package)),
    )


def scientific_protocol_check(
    manifest: CodeImplementationManifest,
    design: ResearchDesign,
    artifacts: ArtifactStore,
) -> CodeReviewCheck:
    """Reject path-complete implementations that violate the research protocol."""

    package_prefix = f"src/{manifest.package_name}/"
    source: dict[str, str] = {}
    failures: list[str] = []
    for item in manifest.files:
        if not item.path.endswith((".py", ".json", ".toml", ".yaml", ".yml")):
            continue
        try:
            source[item.path] = artifacts.read_artifact_bytes(
                item.artifact, max_bytes=MAX_CODE_FILE_BYTES
            ).decode("utf-8")
        except (StoragePolicyError, UnicodeDecodeError):
            failures.append(f"{item.path}: source could not be read for protocol review")

    def text_under(relative_prefix: str) -> str:
        prefix = f"{package_prefix}{relative_prefix}"
        return "\n".join(
            value for path, value in source.items() if path.startswith(prefix)
        ).casefold()

    design_text = " ".join(
        " ".join((item.name, item.purpose, item.innovation))
        for item in design.method_components
    ).casefold()
    protocol_design_text = " ".join(
        " ".join((item.objective, *item.variables, *item.protocol))
        for item in design.experiment_plans
    ).casefold()
    data_design_text = " ".join(
        " ".join(
            (
                item.dataset_key,
                *item.candidate_names,
                *item.required_fields,
                item.split_strategy,
            )
        )
        for item in design.dataset_requirements
    ).casefold()
    audit_design_text = " ".join(
        " ".join(
            (
                *item.schema_checks,
                *item.leakage_checks,
                item.duplicate_conflict_policy,
                *item.acceptance_gates,
            )
        )
        for item in design.data_audit_plans
    ).casefold()
    training_text = text_under("training/")
    split_text = text_under("data/")
    evaluation_text = text_under("evaluation/")
    experiment_text = text_under("experiments/")
    deepdta_text = "\n".join(
        value.casefold()
        for path, value in source.items()
        if f"{package_prefix}models/deepdta" in path
    )
    graphdta_text = "\n".join(
        value.casefold()
        for path, value in source.items()
        if f"{package_prefix}models/graphdta" in path
    )
    model_feature_text = text_under("models/features")
    calibration_text = text_under("calibration/")
    cli_text = source.get(f"{package_prefix}cli.py", "").casefold()
    configuration_text = "\n".join(
        value for path, value in source.items() if path.startswith("configs/")
    ).casefold()

    declared_feature_hashing = any(
        marker in f"{design_text} {data_design_text}"
        for marker in ("feature hashing", "hashed feature", "hashing trick")
    )
    undeclared_hash_proxy = (
        ("_hash_feature" in training_text or "hash_feature(" in training_text)
        and not declared_feature_hashing
    )
    if undeclared_hash_proxy:
        failures.append(
            "training substitutes undeclared hashed identity features for the representations required by the design"
        )
    if "deepdta" in design_text and "linearbaseline" in deepdta_text:
        failures.append("DeepDTA is implemented as LinearBaseline rather than a sequence model")
    if "graphdta" in design_text and "linearbaseline" in graphdta_text:
        failures.append("GraphDTA is implemented as LinearBaseline rather than a graph model")
    if "deepdta" in design_text and all(
        marker in deepdta_text
        for marker in ("_make_kernels", "self.mlp", "train_mlp(self.mlp")
    ):
        failures.append(
            "DeepDTA convolution kernels are fixed random features while only the MLP head is optimized"
        )
    if "graphdta" in design_text and all(
        marker in graphdta_text
        for marker in ("graph_weights", "self.mlp", "train_mlp(self.mlp")
    ):
        failures.append(
            "GraphDTA graph encoder weights are fixed random features while only the MLP head is optimized"
        )
    if (
        "graphdta" in design_text
        and "graph_conv_pool" in graphdta_text
        and "rdkit" not in f"{graphdta_text}\n{model_feature_text}"
        and "torch_geometric" not in f"{graphdta_text}\n{model_feature_text}"
    ):
        failures.append(
            "GraphDTA uses a SMILES character-chain graph without an RDKit or equivalent chemical graph builder"
        )

    shift_claimed = any(
        token in f"{design_text} {data_design_text} {protocol_design_text}"
        for token in (
            "strict cold",
            "cold-start",
            "cold start",
            "ood",
            "out-of-distribution",
            "unseen",
            "held-out",
            "held out",
            "domain shift",
            "temporal split",
            "site split",
            "geographic split",
        )
    )
    if shift_claimed:
        required_split_markers = ["train", "test"]
        if "conformal" in design_text or "calibration" in design_text:
            required_split_markers.append("calibration")
        missing = [value for value in required_split_markers if value not in split_text]
        if missing:
            failures.append(
                "shifted/held-out split code omits required partition semantics: "
                f"semantics: missing={missing}"
            )
        executable_overlap_audit = (
            "isdisjoint" in split_text
            or "intersection" in split_text
            or (
                "leakage_audit" in split_text
                and "leakage_free" in split_text
                and " & " in split_text
            )
        )
        if not executable_overlap_audit:
            failures.append(
                "shifted/held-out split code has no executable overlap or disjointness assertion"
            )
        if not any(
            marker in split_text
            for marker in (
                "group_id",
                "cluster",
                "representation_hash",
                "feature_group",
                "subject_id",
                "patient_id",
                "site_id",
                "time_block",
                "scaffold",
                "sequence_hash",
                "geographic",
            )
        ):
            failures.append(
                "shifted/held-out split code does not encode a domain-appropriate group or representation boundary"
            )
        target_representation_required = (
            "sequence" in data_design_text or "sequence" in audit_design_text
        )
        if target_representation_required and not (
            any(
                marker in split_text
                for marker in (
                    "target_sequence",
                    "sequence_hash",
                    "protein_cluster",
                    "sequence_cluster",
                )
            )
            and any(marker in split_text for marker in ("isdisjoint", "intersection", "overlap"))
        ):
            failures.append(
                "the declared sequence-level audit is absent from executable split code"
            )
        compound_representation_required = any(
            marker in audit_design_text
            for marker in ("scaffold", "canonical smiles", "chemical similarity", "tanimoto")
        )
        if compound_representation_required and not any(
            marker in split_text
            for marker in (
                "canonical_smiles",
                "murcko",
                "scaffold",
                "tanimoto",
                "chemical_similarity",
            )
        ):
            failures.append(
                "the declared chemical-representation audit is absent from executable split code"
            )
        if "conflict" in audit_design_text and not (
            "conflict" in split_text
            and any(
                marker in split_text
                for marker in (
                    "input",
                    "feature",
                    "representation",
                    "duplicate",
                    "smiles",
                    "sequence",
                )
            )
        ):
            failures.append(
                "data audit code does not reject or quarantine repeated model inputs with conflicting outcomes"
            )

    multi_dataset_experiment = any(
        len(set(item.dataset_keys)) > 1 for item in design.experiment_plans
    )
    if multi_dataset_experiment:
        source_text = "\n".join(source.values()).casefold()
        if _unsafe_cross_dataset_aggregation(source_text):
            failures.append(
                "multi-dataset code aggregates records without an executable measurement/label-semantics isolation or harmonization guard"
            )

    if "conformal" in design_text or "calibration" in design_text:
        if "calibration" not in training_text:
            failures.append("training entrypoint does not consume an independent calibration split")
        if "residuals" in training_text and "_metrics(model, test" in training_text:
            failures.append(
                "training computes conformal residuals inside a test-metric path; this is test leakage"
            )
        mapping_text = f"{calibration_text}\n{evaluation_text}"
        runtime_level_text = f"{training_text}\n{experiment_text}\n{cli_text}"
        configured_level_loop = (
            "nominal_confidence_levels" in runtime_level_text
            or "confidence_levels" in runtime_level_text
        )
        for nominal in ("0.8", "0.9", "0.95"):
            if nominal not in mapping_text:
                failures.append(f"nominal conformal level {nominal} has no calibration mapping")
            if nominal not in runtime_level_text and not configured_level_loop:
                failures.append(
                    f"nominal conformal level {nominal} is not executed by the experiment path"
                )

    combined_runtime = (
        f"{training_text}\n{evaluation_text}\n{experiment_text}\n"
        f"{configuration_text}\n{cli_text}"
    )
    for seed in (str(value) for value in design.statistical_analysis.random_seeds):
        if seed not in combined_runtime:
            failures.append(f"fixed computational seed {seed} is not encoded in the implementation")
    if "negative-control" in " ".join(
        item.experiment_id for item in design.experiment_plans
    ).casefold():
        if "permut" not in combined_runtime or "retrain" not in combined_runtime:
            failures.append(
                "negative control does not visibly permute training labels and retrain from scratch"
            )
    statistical_text = " ".join(
        (
            design.statistical_analysis.primary_metric,
            *design.statistical_analysis.secondary_metrics,
        )
    ).casefold()
    metric_aliases = {
        "rmse": ("rmse", "root mean squared"),
        "mse": ("mse", "mean squared"),
        "mae": ("mae", "mean absolute"),
        "accuracy": ("accuracy",),
        "f1": ("f1", "f-score", "f score"),
        "auroc": ("auroc", "roc auc", "auc-roc"),
        "precision": ("precision",),
        "recall": ("recall", "sensitivity"),
        "spearman": ("spearman",),
        "pearson": ("pearson",),
        "coverage": ("coverage",),
        "interval_width": ("interval width", "mean width", "median width"),
        "nll": ("nll", "negative log likelihood", "negative log-likelihood"),
        "c_index": ("c-index", "concordance index"),
    }
    required_metrics = {
        code_name
        for code_name, aliases in metric_aliases.items()
        if any(alias in statistical_text for alias in aliases)
    }
    for metric in sorted(required_metrics):
        if metric not in combined_runtime:
            failures.append(f"required observed metric {metric} is not emitted")

    successes = [
        "reviewed manifest-owned training, split, model, evaluation, experiment, and config source",
        "checked declared shift boundaries, measurement semantics, fixed seeds, controls, and planned metrics",
    ]
    return _check(
        "review-scientific-protocol",
        "scientific_protocol",
        failures,
        successes,
        files=sorted(source),
    )


def _safe_source_path(root: Path, relative: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts or not relative:
        raise StoragePolicyError("source manifest contains an unsafe path")
    target = (root / raw).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise StoragePolicyError("source file escapes its local root") from exc
    if target.is_symlink() or not target.is_file():
        raise StoragePolicyError("source file is missing or symbolic")
    return target


def vendored_source_check(
    manifest: CodeImplementationManifest,
    design: ResearchDesign,
    artifacts: ArtifactStore,
) -> CodeReviewCheck:
    expected = {
        item.module_id: item for item in design.reusable_code_modules if item.reuse_mode == "integrate"
    }
    actual = {item.module_id: item for item in manifest.vendored_sources}
    failures: list[str] = []
    successes: list[str] = []
    if set(actual) != set(expected):
        failures.append(
            f"vendored module IDs differ: expected={sorted(expected)}, actual={sorted(actual)}"
        )
    for module_id, record in actual.items():
        module = expected.get(module_id)
        if module is None:
            continue
        if record.repository_url.rstrip("/") != module.repository_url.rstrip("/"):
            failures.append(f"{module_id} repository provenance differs from the design")
        if record.license_id != module.license_id:
            failures.append(f"{module_id} license differs from the design")
        try:
            payload = artifacts.read_artifact_json(record.source_manifest, max_bytes=MAX_INPUT_BYTES)
            local_root = (artifacts.root / record.local_root).resolve()
            workspace_root = artifacts.root.resolve()
            local_root.relative_to(workspace_root)
            if not local_root.is_dir() or local_root.is_symlink():
                raise StoragePolicyError("vendored source root is missing or symbolic")
            file_entries = payload.get("files")
            if not isinstance(file_entries, list) or not file_entries:
                raise StoragePolicyError("source manifest has no file inventory")
            tree_digest = hashlib.sha256()
            total_bytes = 0
            for item in file_entries:
                if not isinstance(item, dict):
                    raise StoragePolicyError("source inventory entry is not an object")
                relative = str(item.get("path", ""))
                path = _safe_source_path(local_root, relative)
                size = path.stat().st_size
                if size <= 0 or size > MAX_SOURCE_FILE_BYTES:
                    raise StoragePolicyError("source file violates size policy")
                total_bytes += size
                if total_bytes > MAX_SOURCE_TOTAL_BYTES:
                    raise StoragePolicyError("source snapshot violates total size policy")
                digest = hashlib.sha256()
                with path.open("rb") as handle:
                    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(chunk)
                file_sha = digest.hexdigest()
                if size != item.get("size_bytes") or file_sha != item.get("sha256"):
                    raise StoragePolicyError("source file differs from the local manifest")
                tree_digest.update(relative.encode("utf-8"))
                tree_digest.update(b"\0")
                tree_digest.update(str(size).encode("ascii"))
                tree_digest.update(b"\0")
                tree_digest.update(file_sha.encode("ascii"))
            if (
                len(file_entries) != record.file_count
                or total_bytes != record.total_bytes
                or tree_digest.hexdigest() != record.tree_sha256
                or payload.get("execution_performed") is not False
            ):
                raise StoragePolicyError("source snapshot totals or non-execution claim differ")
            successes.append(
                f"{module_id}@{record.commit_sha[:12]} verified locally: "
                f"{record.file_count} files, {record.total_bytes} bytes, {record.license_id}"
            )
        except (StoragePolicyError, OSError, ValueError, TypeError) as exc:
            failures.append(f"{module_id} local source verification failed: {type(exc).__name__}")
    return _check(
        "review-vendored-source",
        "vendored_source",
        failures,
        successes or ["No integrate-mode source was declared."],
        files=[item.local_root for item in manifest.vendored_sources],
    )


def _module_name(package_name: str, path: str) -> str | None:
    prefix = f"src/{package_name}/"
    if not path.startswith(prefix) or not path.endswith(".py"):
        return None
    relative = path.removeprefix("src/").removesuffix(".py")
    parts = relative.split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _import_target(current: str, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module
    current_parts = current.split(".")
    package_parts = current_parts if current.endswith(".__init__") else current_parts[:-1]
    trim = node.level - 1
    if trim > len(package_parts):
        return None
    base = package_parts[: len(package_parts) - trim]
    if node.module:
        base.extend(node.module.split("."))
    return ".".join(base)


def _has_cycle(graph: dict[str, set[str]]) -> bool:
    active: set[str] = set()
    done: set[str] = set()

    def visit(node: str) -> bool:
        if node in active:
            return True
        if node in done:
            return False
        active.add(node)
        for neighbor in graph.get(node, set()):
            if visit(neighbor):
                return True
        active.remove(node)
        done.add(node)
        return False

    return any(visit(node) for node in graph)


def static_analysis_check(
    manifest: CodeImplementationManifest, artifacts: ArtifactStore
) -> CodeReviewCheck:
    failures: list[str] = []
    successes: list[str] = []
    code_root = (artifacts.root / manifest.implementation_root).resolve()
    try:
        code_root.relative_to(artifacts.root.resolve())
        actual_paths = {
            path.relative_to(code_root).as_posix()
            for path in code_root.rglob("*")
            if path.is_file()
        }
        declared_paths = {item.path for item in manifest.files}
        if actual_paths != declared_paths:
            failures.append(
                "implementation tree differs from manifest: "
                f"undeclared={sorted(actual_paths - declared_paths)}, "
                f"missing={sorted(declared_paths - actual_paths)}"
            )
    except (OSError, ValueError):
        failures.append("implementation root is missing or escapes the workspace")
    graph: dict[str, set[str]] = {}
    known_modules = {
        module
        for item in manifest.files
        if (module := _module_name(manifest.package_name, item.path)) is not None
    }
    python_count = 0
    for item in manifest.files:
        try:
            measured = artifacts.verify_artifact_file(item.artifact, max_bytes=MAX_CODE_FILE_BYTES)
            if measured != item.size_bytes or item.artifact.sha256 != item.sha256:
                raise StoragePolicyError("generated file size or digest differs from manifest")
            payload = artifacts.read_artifact_bytes(item.artifact, max_bytes=MAX_CODE_FILE_BYTES)
        except StoragePolicyError as exc:
            failures.append(f"{item.path}: artifact integrity failure ({type(exc).__name__})")
            continue
        text = payload.decode("utf-8", errors="strict")
        lowered = text.lower()
        markers = [marker for marker in REMOTE_MARKERS if marker in lowered]
        if markers:
            failures.append(f"{item.path}: remote-link marker(s) {markers}")
        if not item.path.endswith(".py"):
            continue
        python_count += 1
        try:
            tree = ast.parse(text, filename=item.path)
        except SyntaxError:
            failures.append(f"{item.path}: invalid Python syntax")
            continue
        module = _module_name(manifest.package_name, item.path)
        if module:
            graph.setdefault(module, set())
        for node in ast.walk(tree):
            if isinstance(node, ast.While) and isinstance(node.test, ast.Constant) and node.test.value is True:
                failures.append(f"{item.path}:{node.lineno}: unbounded while True loop")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".", 1)[0]
                    if root in BANNED_IMPORT_ROOTS:
                        failures.append(f"{item.path}:{node.lineno}: forbidden import {root}")
                    if module and alias.name in known_modules:
                        graph[module].add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".", 1)[0]
                if root in BANNED_IMPORT_ROOTS:
                    failures.append(f"{item.path}:{node.lineno}: forbidden import {root}")
                if module:
                    target = _import_target(module, node)
                    if target in known_modules:
                        graph[module].add(target)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in BANNED_CALL_NAMES:
                    failures.append(f"{item.path}:{node.lineno}: forbidden call {node.func.id}")
                if isinstance(node.func, ast.Attribute):
                    owner = node.func.value.id if isinstance(node.func.value, ast.Name) else None
                    if (owner, node.func.attr) in BANNED_ATTRIBUTE_CALLS:
                        failures.append(f"{item.path}:{node.lineno}: forbidden call {owner}.{node.func.attr}")
    if _has_cycle(graph):
        failures.append("internal Python import graph contains a cycle")
    successes.extend(
        [
            f"SHA-256 and size verified for {len(manifest.files)} generated files",
            "implementation tree contains no undeclared files",
            f"AST parsed {python_count} Python files",
            "no network, shell, dynamic execution, unbounded-loop, or import-cycle pattern found",
        ]
    )
    return _check(
        "review-static-safety",
        "static_analysis",
        failures,
        successes,
        files=[item.path for item in manifest.files if item.path.endswith(".py")],
    )


def _environment(code_root: Path) -> dict[str, str]:
    return {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str((code_root / "src").resolve()),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }


def _parse_demo(stdout: str) -> dict[str, float | int | str | bool]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("demo produced no JSON")
    payload = json.loads(lines[-1])
    if not isinstance(payload, dict):
        raise ValueError("demo output is not an object")
    cleaned: dict[str, float | int | str | bool] = {}
    for key, value in payload.items():
        if isinstance(value, (str, bool, int, float)):
            cleaned[str(key)] = value
    return cleaned


def _execute(
    code_root: Path,
    execution_id: str,
    argv: list[str],
    *,
    demo: bool,
) -> CodeReviewExecutionRecord:
    started = perf_counter()
    try:
        completed = subprocess.run(
            argv,
            cwd=code_root,
            env=_environment(code_root),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=EXECUTION_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return CodeReviewExecutionRecord(
            execution_id=execution_id,
            argv=argv,
            status="timed_out",
            return_code=None,
            duration_ms=round((perf_counter() - started) * 1000, 3),
            timeout_seconds=EXECUTION_TIMEOUT_SECONDS,
            stdout_tail=(exc.stdout or "")[-OUTPUT_TAIL_CHARS:] if isinstance(exc.stdout, str) else "",
            stderr_tail=(exc.stderr or "")[-OUTPUT_TAIL_CHARS:] if isinstance(exc.stderr, str) else "",
        )
    metrics: dict[str, float | int | str | bool] = {}
    status = "passed" if completed.returncode == 0 else "failed"
    stderr = completed.stderr[-OUTPUT_TAIL_CHARS:]
    if status == "passed" and demo:
        try:
            metrics = _parse_demo(completed.stdout)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            status = "failed"
            stderr = f"{stderr}\n{type(exc).__name__}: invalid demo evidence".strip()
    return CodeReviewExecutionRecord(
        execution_id=execution_id,
        argv=argv,
        status=status,
        return_code=completed.returncode,
        duration_ms=round((perf_counter() - started) * 1000, 3),
        timeout_seconds=EXECUTION_TIMEOUT_SECONDS,
        stdout_tail=completed.stdout[-OUTPUT_TAIL_CHARS:],
        stderr_tail=stderr[-OUTPUT_TAIL_CHARS:],
        metrics=metrics,
    )


def execute_review_commands(
    manifest: CodeImplementationManifest, artifacts: ArtifactStore
) -> list[CodeReviewExecutionRecord]:
    code_root = (artifacts.root / manifest.implementation_root).resolve()
    workspace = artifacts.root.resolve()
    code_root.relative_to(workspace)
    demo_argv = [sys.executable, "-m", f"{manifest.package_name}.demo", "--json"]
    test_argv = [
        sys.executable,
        "-m",
        "unittest",
        "discover",
        "-s",
        "tests",
        "-p",
        "test_*.py",
    ]
    return [
        _execute(code_root, "minimal_demo_first", demo_argv, demo=True),
        _execute(code_root, "minimal_demo_repeat", demo_argv, demo=True),
        _execute(code_root, "unit_tests", test_argv, demo=False),
    ]


def execution_check(executions: list[CodeReviewExecutionRecord]) -> CodeReviewCheck:
    failures = [
        f"{item.execution_id} status={item.status} return_code={item.return_code}"
        for item in executions
        if item.status != "passed"
    ]
    successes = [
        f"{item.execution_id} passed in {item.duration_ms:.3f} ms"
        for item in executions
        if item.status == "passed"
    ]
    return _check("review-fixed-execution", "execution", failures, successes)


def training_update_check(executions: list[CodeReviewExecutionRecord]) -> CodeReviewCheck:
    first = next(item for item in executions if item.execution_id == "minimal_demo_first")
    failures: list[str] = []
    successes: list[str] = []
    try:
        initial = float(first.metrics["initial_loss"])
        final = float(first.metrics["final_loss"])
        delta = float(first.metrics["parameter_delta"])
        iterations = int(first.metrics["iterations"])
        if not all(math.isfinite(value) for value in (initial, final, delta)):
            failures.append("demo training metrics contain non-finite values")
        if not final < initial:
            failures.append(f"model loss did not decrease: {initial} -> {final}")
        if not delta > 0:
            failures.append("model parameter delta is not positive")
        if iterations < 2:
            failures.append("model did not perform at least two update iterations")
        successes.append(
            f"loss {initial:.8g}->{final:.8g}, parameter_delta={delta:.8g}, iterations={iterations}"
        )
    except (KeyError, TypeError, ValueError):
        failures.append("demo omitted parseable model-update evidence")
    return _check("review-model-update", "training_update", failures, successes)


def reliability_check(executions: list[CodeReviewExecutionRecord]) -> CodeReviewCheck:
    first = next(item for item in executions if item.execution_id == "minimal_demo_first")
    repeat = next(item for item in executions if item.execution_id == "minimal_demo_repeat")
    failures: list[str] = []
    if first.status != "passed" or repeat.status != "passed":
        failures.append("both demo executions must pass before reliability can be assessed")
    elif first.metrics != repeat.metrics:
        failures.append("fixed-seed demo results differ across repeated execution")
    successes = [
        "repeated fixed-environment demo outputs are identical",
        "all reported training values are finite and deterministic",
    ]
    return _check("review-runtime-reliability", "reliability", failures, successes)


def recommendations_for(checks: list[CodeReviewCheck]) -> list[str]:
    mapping = {
        "review-architecture-layout": "Move every model and experiment into a named subfolder; keep training and testing APIs in their own packages.",
        "review-design-traceability": "Add file-level component, dataset, experiment, and local-source IDs until every ResearchDesign item is covered exactly.",
        "review-innovation-alignment": "Implement concrete OOD split, audit, model-adapter, named baseline, calibration, and statistical modules required by the ResearchDesign; traceability labels alone are insufficient.",
        "review-scientific-protocol": "Remove hash/linear model impersonation; implement disjoint drug-and-target train/calibration/test manifests, calibration-only conformal quantiles, fixed seeds 7/17/27, a retrained label-permutation control, and RMSE/Spearman/coverage/interval-width outputs.",
        "review-vendored-source": "Re-download each integrate-mode repository to a fixed local commit, verify its license, and regenerate the per-file SOURCE_MANIFEST.",
        "review-static-safety": "Remove remote access, shell/dynamic execution, unbounded loops, unsafe imports, and circular internal imports; then regenerate hashes.",
        "review-fixed-execution": "Repair the fixed local Demo or unittest failures and keep each command below the 30-second timeout.",
        "review-model-update": "Correct the training loop so at least two bounded iterations reduce finite loss and produce a positive parameter delta.",
        "review-runtime-reliability": "Fix seeds and deterministic code paths so two sanitized Demo executions return identical finite JSON metrics.",
    }
    return [mapping[item.check_id] for item in checks if item.status == "failed"]
