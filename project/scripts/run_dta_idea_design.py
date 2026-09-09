#!/usr/bin/env python3
"""Run Make Idea -> Idea Design against the verified local 100-paper DTA corpus.

This runner is deliberately offline and deterministic. It exercises the real Agent contracts and
artifact policies without claiming to be a DeepSeek API call. Set a rotated DEEPSEEK_API_KEY and use
the live runtime when an online model-generated revision is required.
"""

from __future__ import annotations

import hashlib
import json
import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

from agents.idea_design_agent import IdeaDesignAgent
from agents.lead_agent import RuleBasedLeadAgent
from agents.make_idea_agent import MakeIdeaAgent
from research_agents.registry import build_mock_registry
from research_agents.runtime import ResearchRuntime
from research_agents.schemas import AgentName, ArtifactRef, RunSnapshot, TaskEnvelope, WorkflowState
from research_agents.providers import DeepSeekModel
from research_agents.state_machine import ResearchStateMachine
from research_agents.storage import ArtifactStore


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SLUG = "assaycal-dta"
CORPUS_PATH = WORKSPACE_ROOT / "reference" / PROJECT_SLUG / "literature" / "papers.json"


def candidate(
    idea_id: str,
    hypothesis_id: str,
    title: str,
    research_question: str,
    novelty_claim: str,
    innovation_type: str,
    evidence_refs: list[str],
    differentiation: list[str],
    feasibility: float,
    novelty: float,
    risk: str,
) -> dict[str, Any]:
    return {
        "idea": {
            "idea_id": idea_id,
            "title": title,
            "research_question": research_question,
            "novelty_claim": novelty_claim,
            "rationale": (
                "The cited DTA papers jointly expose label semantics, evaluation leakage, data bias, "
                "and calibration as separable and computationally testable gaps."
            ),
            "evidence_refs": evidence_refs,
        },
        "hypothesis": {
            "hypothesis_id": hypothesis_id,
            "statement": (
                "The proposed change will improve strict out-of-domain affinity prediction without "
                "sacrificing in-domain accuracy beyond the predeclared tolerance."
            ),
            "expected_direction": "OOD error and calibration error decrease; CI or rank correlation increases.",
            "falsification_criterion": (
                "Reject when paired confidence intervals across fixed seeds include no improvement "
                "on both the primary OOD and calibration criteria."
            ),
            "evidence_refs": evidence_refs,
        },
        "innovation_type": innovation_type,
        "baseline_gap": (
            "Most benchmark pipelines optimize one affinity value while overlooking assay type, "
            "missing conditions, duplicate entities, and split-dependent optimism."
        ),
        "expected_contribution": (
            "A reproducible computational method and audit protocol with explicit failure thresholds."
        ),
        "differentiation": differentiation,
        "feasibility_score": feasibility,
        "novelty_score": novelty,
        "implementation_risk": risk,
    }


def idea_response() -> dict[str, Any]:
    return {
        "candidates": [
            candidate(
                "idea-assaycal-dta",
                "hypothesis-assaycal",
                "AssayCal-DTA: assay-semantic and uncertainty-calibrated affinity prediction",
                "Can assay-type conditioning, missingness-aware context, and heteroscedastic learning improve DTA generalization and calibration?",
                "Joint absolute regression, within-assay ranking, and uncertainty calibration model measurement semantics instead of treating all affinity labels as exchangeable.",
                "uncertainty_calibration",
                ["DTA-001", "DTA-045", "DTA-078", "DTA-091", "DTA-093"],
                [
                    "assay-type and missingness tokens rather than unconditional label pooling",
                    "absolute affinity plus within-assay ranking objective",
                    "paired OOD accuracy and predictive-interval calibration gate",
                ],
                0.84,
                0.86,
                "medium",
            ),
            candidate(
                "idea-ood-audit",
                "hypothesis-ood-audit",
                "DTA-OOD-Audit: leakage-resistant multi-axis affinity benchmark",
                "Do scaffold, target-family, pair-cluster, and correction-aware splits change DTA model rankings?",
                "A versioned split and leakage manifest makes random-split optimism measurable and turns generalization into a reproducible benchmark artifact.",
                "evaluation_protocol",
                ["DTA-001", "DTA-003", "DTA-078", "DTA-091", "DTA-093"],
                [
                    "four independent OOD axes with immutable split hashes",
                    "duplicate and similarity leakage report before training",
                    "model ranking stability rather than a single best score",
                ],
                0.93,
                0.77,
                "low",
            ),
            candidate(
                "idea-confidence-fusion",
                "hypothesis-confidence-fusion",
                "Confidence-gated multimodal DTA under missing structure",
                "Can evidence-confidence gates exploit structural inputs without degrading samples that only have sequence and SMILES?",
                "A confidence-conditioned mixture of sequence, graph, and optional structure experts treats modality absence and low-quality structure as first-class inputs.",
                "model_architecture",
                ["DTA-003", "DTA-010", "DTA-012", "DTA-040", "DTA-043"],
                [
                    "explicit missing-modality mask and confidence features",
                    "sequence-only fallback trained jointly with multimodal experts",
                    "coverage-stratified robustness and efficiency reporting",
                ],
                0.71,
                0.80,
                "high",
            ),
        ],
        "recommended_idea_id": "idea-assaycal-dta",
        "selection_rationale": (
            "AssayCal-DTA is recommended because it addresses a central label-semantics problem, "
            "has a sequence/2D minimum implementation, admits strict falsification, and can absorb "
            "DTA-OOD-Audit as its evaluation protocol without requiring predicted 3D structures."
        ),
        "diversity_note": (
            "The portfolio separates learning objective/calibration, benchmark protocol, and "
            "multimodal architecture so novelty is not reduced to adding another encoder."
        ),
        "limitations": [
            "Full text must be checked before making priority or first-use novelty claims.",
            "Assay context coverage in public data remains a pre-download verification gate.",
            "No performance result is observed at the ideation stage.",
        ],
    }


def advisor_response(role: str) -> dict[str, Any]:
    focus = {
        "method-architecture-advisor": (
            "Model assay type, domain, and missingness explicitly; isolate each contribution behind typed interfaces.",
            "Use a sequence/2D minimum path with a regression-ranking uncertainty objective.",
        ),
        "code-reuse-architecture-advisor": (
            "Wrap licensed paper repositories behind local dataset, model, trainer, and metric adapters.",
            "Require parity tests and a clean-room fallback before any third-party integration.",
        ),
        "data-decision-advisor": (
            "Gate Davis, KIBA, and BindingDB on license, provenance, schema, duplicates, missingness, and leakage audits.",
            "Use immutable random, scaffold, cold-entity, family, temporal, and cross-dataset split manifests.",
        ),
        "experiment-decision-advisor": (
            "Run data validation, baseline reproduction, equal-budget tuning, method, ablation, and stress tests in order.",
            "Use fixed seeds, paired bootstrap intervals, effect sizes, Holm correction, and bounded attempt budgets.",
        ),
        "critical-risk-advisor": (
            "Do not treat Kd, Ki, IC50, and integrated KIBA labels as exchangeable physical measurements.",
            "Constrain claims when assay coverage, baseline parity, split independence, or practical effect gates fail.",
        ),
    }
    recommendations = focus[role]
    return {
        "advisor": role,
        "recommendations": list(recommendations),
        "decisions": [recommendations[1]],
        "blocking_concerns": (
            ["BindingDB assay-context coverage and reuse terms must pass the pre-download gate."]
            if role in {"data-decision-advisor", "critical-risk-advisor"}
            else []
        ),
        "evidence_refs": ["DTA-001", "DTA-045", "DTA-078", "DTA-091", "DTA-093"],
        "assumptions": [
            "Paper metadata supports planning, while exact full-text claims and downloaded data profiles remain unverified."
        ],
        "issue_codes": (
            ["ASSAY_CONTEXT_COVERAGE_UNVERIFIED"]
            if role in {"data-decision-advisor", "critical-risk-advisor"}
            else []
        ),
    }


def experiment(
    experiment_id: str,
    experiment_type: str,
    objective: str,
    datasets: list[str],
    variables: list[str],
    metrics: list[str],
    success: list[str],
    evidence: list[str],
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "experiment_type": experiment_type,
        "objective": objective,
        "dataset_keys": datasets,
        "baselines": [
            "DeepDTA-style sequence baseline",
            "NHGNN-DTA-compatible graph baseline",
            "same encoder with ordinary MSE regression",
        ],
        "variables": variables,
        "controls": [
            "identical cleaned records and immutable split hashes",
            "same seeds, tuning budget, early stopping, and metric implementation",
            "no test-set selection or outcome-driven case selection",
        ],
        "metrics": metrics,
        "random_seeds": [17, 42, 73],
        "protocol": [
            "validate data and leakage report before training",
            "run every baseline and proposed configuration on the same seed matrix",
            "report mean, standard deviation, and paired bootstrap confidence intervals",
        ],
        "success_criteria": success,
        "evidence_refs": evidence,
        "computational_only": True,
    }


def dataset_requirement(
    key: str,
    names: list[str],
    priority: str,
    query: str,
    contents: list[str],
    fields: list[str],
    scale: str,
    distribution: str,
    split: str,
    basis: str,
    refs: list[str],
    alternatives: list[str],
) -> dict[str, Any]:
    return {
        "dataset_key": key,
        "source_kind": "external_dataset",
        "candidate_names": names,
        "priority": priority,
        "acquisition_query": query,
        "official_source_urls": [],
        "contents": contents,
        "required_fields": fields,
        "reported_scale": scale,
        "expected_distribution": distribution,
        "distribution_checks": [
            "row count, unique compounds, unique targets, and unique pairs",
            "label quantiles, histogram, tails, per-assay distribution, and missingness",
            "duplicate conflicts, scaffold frequency, target-family frequency, and pair leakage",
        ],
        "split_strategy": split,
        "reliability_basis": basis,
        "license_requirement": (
            "Download only from an official or paper-linked source with explicit reuse terms; "
            "record URL, license evidence, SHA-256, and snapshot date."
        ),
        "paper_refs": refs,
        "alternatives": alternatives,
    }


def design_response() -> dict[str, Any]:
    refs = ["DTA-001", "DTA-045", "DTA-078", "DTA-091", "DTA-093"]
    return {
        "selected_idea_id": "idea-assaycal-dta",
        "method_name": "AssayCal-DTA",
        "executive_summary": (
            "AssayCal-DTA is a sequence/2D-first affinity model that retains assay semantics instead "
            "of unconditionally merging Kd, Ki, IC50, and corrected benchmark labels. It combines "
            "absolute regression, within-assay ranking, and a heteroscedastic predictive head, then "
            "tests the method on immutable random, scaffold, cold-drug, cold-target, family, and "
            "cross-dataset splits. The minimum version is entirely computational and does not depend "
            "on docking, wet-lab work, or an experimental structure."
        ),
        "method_components": [
            {
                "component_id": "component-audit",
                "name": "Assay-semantic data audit",
                "purpose": "Normalize units while preserving measurement type, inequality relation, provenance, and missing conditions.",
                "inputs": ["raw affinity records", "assay metadata", "entity identifiers"],
                "outputs": ["typed records", "missingness masks", "leakage report"],
                "innovation": "Label semantics and missingness become explicit inputs and audit artifacts rather than hidden preprocessing decisions.",
                "evidence_refs": ["DTA-045", "DTA-091", "DTA-093"],
                "dependency_ids": [],
            },
            {
                "component_id": "component-encoder",
                "name": "Sequence and molecular-graph encoder",
                "purpose": "Create drug and target representations with a bounded, reproducible sequence/2D baseline.",
                "inputs": ["canonical SMILES graph", "protein sequence"],
                "outputs": ["drug embedding", "target embedding", "pair embedding"],
                "innovation": "The encoder is intentionally modular so the contribution of assay semantics is isolated from encoder novelty.",
                "evidence_refs": ["DTA-001", "DTA-003", "DTA-012"],
                "dependency_ids": ["component-audit"],
            },
            {
                "component_id": "component-context",
                "name": "Assay context conditioner",
                "purpose": "Condition pair representations on measurement type, source domain, and missingness without assuming unavailable fields.",
                "inputs": ["pair embedding", "assay token", "domain token", "missingness mask"],
                "outputs": ["context-conditioned pair embedding"],
                "innovation": "Context is represented with masked embeddings and a no-context fallback for fair coverage-stratified evaluation.",
                "evidence_refs": ["DTA-045", "DTA-078", "DTA-093"],
                "dependency_ids": ["component-encoder"],
            },
            {
                "component_id": "component-objective",
                "name": "Regression-ranking uncertainty objective",
                "purpose": "Predict affinity location and scale while preserving within-assay order.",
                "inputs": ["conditioned embedding", "affinity value", "assay-local pair order"],
                "outputs": ["predicted mean", "predicted scale", "ranking score"],
                "innovation": "Heteroscedastic likelihood and within-assay ranking reduce dependence on a falsely universal label scale.",
                "evidence_refs": ["DTA-045", "DTA-078", "DTA-091"],
                "dependency_ids": ["component-context"],
            },
        ],
        "reusable_code_modules": [
            {
                "module_id": "reuse-nhgnn",
                "repository_url": "https://github.com/hehh77/NHGNN-DTA",
                "source_paper_id": "DTA-003",
                "repository_status": "verified_open_source",
                "license_id": "Apache-2.0",
                "reuse_mode": "integrate",
                "reusable_parts": ["Davis/KIBA preprocessing interfaces", "graph batching", "affinity metrics"],
                "integration_strategy": "Vendor no files blindly; wrap reviewed functions behind local DatasetAdapter and MetricSuite parity tests.",
                "maturity_evidence": ["paper-linked repository", "SPDX license verified in local corpus"],
                "fallback_implementation": "Implement clean-room RDKit graph conversion and CI/MSE metrics from published definitions.",
            },
            {
                "module_id": "reuse-ssm",
                "repository_url": "https://github.com/QizhiPei/SSM-DTA",
                "source_paper_id": "DTA-008",
                "repository_status": "verified_open_source",
                "license_id": "MIT",
                "reuse_mode": "integrate",
                "reusable_parts": ["low-data split logic", "training-loop structure", "configuration pattern"],
                "integration_strategy": "Extract only reviewed split/config concepts and adapt them to immutable local manifests.",
                "maturity_evidence": ["paper-linked repository", "SPDX license verified in local corpus"],
                "fallback_implementation": "Use scikit-learn group splits plus a local typed trainer with fixed-seed tests.",
            },
            {
                "module_id": "reuse-mdf",
                "repository_url": "https://github.com/Amitranjan71/MDF-DTA-A-Multi-Dimensional-Fusion-Approach-for-Drug-Target-Binding-Affinity-Prediction",
                "source_paper_id": "DTA-012",
                "repository_status": "verified_open_source",
                "license_id": "MIT",
                "reuse_mode": "reference_only",
                "reusable_parts": ["feature fusion boundary", "baseline configuration", "evaluation organization"],
                "integration_strategy": "Use the architecture only as a paper baseline reference until dependency and code review pass.",
                "maturity_evidence": ["paper-linked repository", "SPDX license verified in local corpus"],
                "fallback_implementation": "Build a minimal concatenation baseline with explicit tensor-shape tests.",
            },
        ],
        "code_blueprint": {
            "pipeline": [
                "resolve versioned dataset sources and verify licenses/digests",
                "profile schema, entities, affinity distributions, missingness, and conflicts",
                "normalize units without collapsing assay type or inequality relations",
                "generate immutable random, scaffold, cold-drug, cold-target, family, and temporal splits",
                "train classical and deep baselines with the same tuning budget",
                "train AssayCal-DTA absolute, ranking, and uncertainty objectives",
                "evaluate accuracy, ranking, calibration, OOD gap, and efficiency across seeds",
                "write result manifests, predictions, confidence intervals, and provenance",
            ],
            "packages": [
                {
                    "path": "code/src/assaycal/data",
                    "responsibility": "Download-independent dataset adapters, semantic normalization, profiles, and leakage checks.",
                    "key_interfaces": ["DatasetAdapter.load", "normalize_affinity", "profile_dataset", "build_splits"],
                    "reference_module_ids": ["reuse-nhgnn", "reuse-ssm"],
                },
                {
                    "path": "code/src/assaycal/models",
                    "responsibility": "Sequence/graph encoders, context conditioner, and predictive distribution head.",
                    "key_interfaces": ["AffinityModel.forward", "AssayContext.encode", "PredictionDistribution"],
                    "reference_module_ids": ["reuse-nhgnn", "reuse-mdf"],
                },
                {
                    "path": "code/src/assaycal/training",
                    "responsibility": "Fixed-budget trainers for baselines, multi-objective learning, checkpoints, and seeds.",
                    "key_interfaces": ["Trainer.fit", "compute_objective", "save_checkpoint"],
                    "reference_module_ids": ["reuse-ssm"],
                },
                {
                    "path": "code/src/assaycal/evaluation",
                    "responsibility": "Metrics, calibration, OOD gaps, paired bootstrap intervals, and case selection.",
                    "key_interfaces": ["MetricSuite.compute", "calibration_report", "paired_bootstrap", "select_cases"],
                    "reference_module_ids": ["reuse-nhgnn"],
                },
                {
                    "path": "code/configs",
                    "responsibility": "Versioned dataset, split, model, objective, seed, and compute configurations.",
                    "key_interfaces": ["dataset.yaml", "model.yaml", "experiment.yaml"],
                    "reference_module_ids": [],
                },
            ],
            "entrypoints": [
                "python -m assaycal.profile --config code/configs/davis.yaml",
                "python -m assaycal.split --config code/configs/strict_splits.yaml",
                "python -m assaycal.train --config code/configs/assaycal.yaml --seed 17",
                "python -m assaycal.evaluate --run-manifest <artifact>",
            ],
            "config_contracts": [
                "source URI, license evidence, SHA-256, and snapshot date",
                "label unit/type/relation policy and missing-field behavior",
                "split algorithm, thresholds, seed, and split hash",
                "encoder/objective hyperparameters and tuning budget",
                "result path, environment fingerprint, checkpoint digest, and metric version",
            ],
            "tests": [
                "unit conversion and inequality-relation tests",
                "duplicate conflict and split leakage tests",
                "graph/sequence tensor shape and missing-context tests",
                "CI, MSE, RMSE, Spearman, NLL, CRPS, and coverage parity tests",
                "one-batch forward/backward and deterministic seed smoke tests",
                "third-party adapter parity tests before integration",
            ],
            "dependencies": ["python", "pytorch", "rdkit", "pandas", "numpy", "scikit-learn", "scipy", "pyyaml"],
            "minimum_viable_implementation": [
                "Davis and KIBA adapters with immutable random/cold/scaffold splits",
                "DeepDTA-style sequence baseline and simple molecular graph baseline",
                "assay/domain token plus missingness mask",
                "mean/scale head with regression and within-domain ranking loss",
                "three-seed accuracy, ranking, and calibration report",
            ],
        },
        "dataset_requirements": [
            dataset_requirement(
                "dataset-davis-corrected",
                ["Davis", "modification-aware DAVIS"],
                "required",
                "official correction-aware DAVIS affinity matrix compound SMILES protein sequence license download",
                ["kinase inhibitors", "kinase protein sequences", "Kd-derived affinity matrix", "correction metadata when available"],
                ["drug_id", "target_id", "canonical_smiles", "protein_sequence", "affinity", "affinity_type", "correction_version"],
                "The classic benchmark is commonly represented as a dense 68-compound by 442-target matrix; verify the correction-aware snapshot, usable pair count, and all removals against DTA-093.",
                "Expect repeated drugs and kinases, a continuous transformed affinity target, target-family imbalance, and possible corrected identifiers; measure rather than assume exact quantiles.",
                "Publish random, Bemis-Murcko scaffold, cold-drug, cold-target, and kinase-family splits with zero group overlap and file hashes.",
                "Davis is the classic DTA comparison set in DTA-001, while DTA-093 motivates correction-aware validation.",
                ["DTA-001", "DTA-093"],
                ["KIBA", "BindingDB curated Kd subset"],
            ),
            dataset_requirement(
                "dataset-kiba",
                ["KIBA"],
                "preferred",
                "official KIBA drug target affinity benchmark SMILES protein sequence license download",
                ["drug identifiers and SMILES", "protein targets and sequences", "integrated KIBA affinity scores"],
                ["drug_id", "target_id", "canonical_smiles", "protein_sequence", "kiba_score"],
                "The widely used benchmark is often reported with about 2,111 drugs, 229 targets, and 118,254 observed interactions; verify exact post-cleaning counts and provenance locally.",
                "Expect a sparse pair matrix and nonuniform entity frequency; profile score range, tails, scaffold concentration, and target-family concentration.",
                "Use the same random/cold/scaffold split generator as Davis and retain split hashes for cross-paper comparability.",
                "KIBA is a mainstream comparison set used across DTA model papers including the classic and graph baselines in the supplied corpus.",
                ["DTA-001", "DTA-003", "DTA-012"],
                ["Davis", "BindingDB curated subset"],
            ),
            dataset_requirement(
                "dataset-bindingdb-assay",
                ["BindingDB"],
                "preferred",
                "BindingDB official bulk affinity data Kd Ki IC50 assay metadata license download",
                ["compound-target affinity measurements", "Kd/Ki/IC50 type", "relation symbols", "units", "assay and source metadata"],
                ["compound_id", "target_id", "smiles", "protein_sequence", "affinity_value", "affinity_unit", "affinity_type", "relation", "source", "measurement_date"],
                "Scale depends on the licensed snapshot and filtering policy; record raw rows, unique pairs, repeated measurements, and retained assay-context coverage instead of copying a historical count.",
                "Expect heavy entity-frequency imbalance, mixed measurement types/units, censored inequalities, repeated-pair disagreement, and substantial context missingness.",
                "Create temporal, cold-drug, cold-target, target-family, and pair-cluster splits after canonicalization; fit all transforms on training data only.",
                "The supplied bioassay, uncertainty, and data-bias papers motivate a large assay-aware source, subject to license and field-coverage gates.",
                ["DTA-045", "DTA-078", "DTA-091"],
                ["Davis correction-aware snapshot", "KIBA"],
            ),
        ],
        "experiment_plans": [
            experiment(
                "experiment-main",
                "main_benchmark",
                "Compare AssayCal-DTA against reproducible sequence, graph, and plain-regression baselines.",
                ["dataset-davis-corrected", "dataset-kiba"],
                ["model family", "assay context", "learning objective"],
                ["MSE", "RMSE", "CI", "Spearman"],
                ["paired OOD MSE improves while in-domain MSE degradation stays below 2 percent"],
                ["DTA-001", "DTA-003", "DTA-012", "DTA-093"],
            ),
            experiment(
                "experiment-ood",
                "ood_generalization",
                "Quantify generalization under scaffold, cold-drug, cold-target, family, temporal, and cross-dataset shifts.",
                ["dataset-davis-corrected", "dataset-kiba", "dataset-bindingdb-assay"],
                ["split axis", "similarity threshold", "source domain"],
                ["MSE", "CI", "Spearman", "OOD gap", "split leakage rate"],
                ["leakage rate is zero by declared rules and at least two strict axes improve over plain regression"],
                ["DTA-001", "DTA-003", "DTA-091", "DTA-093"],
            ),
            experiment(
                "experiment-calibration",
                "uncertainty_calibration",
                "Test predictive likelihood, interval coverage, and selective prediction in-domain and OOD.",
                ["dataset-davis-corrected", "dataset-bindingdb-assay"],
                ["heteroscedastic head", "post-hoc scaling", "assay missingness"],
                ["NLL", "CRPS", "90 percent coverage", "interval width", "risk-coverage AUC"],
                ["coverage lies within 85 to 95 percent and selective risk decreases with abstention"],
                ["DTA-078", "DTA-091"],
            ),
            experiment(
                "experiment-ablation",
                "ablation",
                "Isolate assay tokens, missingness masks, ranking loss, and heteroscedastic likelihood.",
                ["dataset-davis-corrected", "dataset-bindingdb-assay"],
                ["remove assay token", "remove missingness mask", "remove ranking", "fixed variance"],
                ["MSE", "CI", "NLL", "coverage error"],
                ["no single component is claimed useful unless its paired interval excludes zero"],
                refs,
            ),
            experiment(
                "experiment-module",
                "module_analysis",
                "Measure context coverage, ranking-pair construction, and objective-gradient compatibility.",
                ["dataset-bindingdb-assay"],
                ["context field coverage", "ranking neighborhood", "loss weight"],
                ["usable-row rate", "gradient cosine similarity", "MSE", "NLL"],
                ["the proposed module remains stable across three predeclared loss weights"],
                ["DTA-045", "DTA-078", "DTA-091"],
            ),
            experiment(
                "experiment-case",
                "case_study",
                "Inspect preselected high-error, high-uncertainty, censored, and repeated-measurement pairs.",
                ["dataset-bindingdb-assay"],
                ["case category", "assay type", "target family"],
                ["absolute error", "predictive interval", "repeat-measurement spread", "attention-free feature attribution"],
                ["case conclusions are seed-stable and selected without test outcome cherry-picking"],
                ["DTA-045", "DTA-078", "DTA-091"],
            ),
            experiment(
                "experiment-feasibility",
                "feasibility",
                "Profile runtime, memory, preprocessing throughput, and context-field availability before full training.",
                ["dataset-davis-corrected", "dataset-kiba", "dataset-bindingdb-assay"],
                ["batch size", "sequence length", "dataset scale"],
                ["samples per second", "peak GPU memory", "wall time", "context coverage"],
                ["minimum version fits one declared GPU and completes a one-epoch smoke run without data loss"],
                ["DTA-001", "DTA-003", "DTA-045"],
            ),
        ],
        "expected_results": [
            {
                "result_id": "result-main-mse",
                "experiment_id": "experiment-main",
                "metric": "strict-split MSE delta versus the same encoder with ordinary MSE",
                "expected_direction": "lower",
                "target_delta": {"low": -0.05, "high": -0.01},
                "expectation": "A modest, repeatable reduction is expected; encoder-only gains are not sufficient evidence for the assay-semantic claim.",
                "evidence_refs": refs,
                "failure_threshold": "Fail if the paired 95 percent interval crosses zero on every strict split.",
                "claim_status": "predicted_not_observed",
            },
            {
                "result_id": "result-calibration",
                "experiment_id": "experiment-calibration",
                "metric": "absolute 90 percent interval coverage error",
                "expected_direction": "lower",
                "target_delta": {"low": -0.08, "high": -0.02},
                "expectation": "Heteroscedastic modeling is expected to improve coverage relative to an uncalibrated point head.",
                "evidence_refs": ["DTA-078", "DTA-091"],
                "failure_threshold": "Fail if coverage error or NLL does not improve after a training-only calibration fit.",
                "claim_status": "predicted_not_observed",
            },
            {
                "result_id": "result-ablation",
                "experiment_id": "experiment-ablation",
                "metric": "paired primary-metric contribution",
                "expected_direction": "higher",
                "target_delta": {"low": 0.005, "high": 0.03},
                "expectation": "Assay context and ranking should contribute mainly on cross-domain or repeated-assay subsets, not necessarily every random split.",
                "evidence_refs": ["DTA-045", "DTA-078", "DTA-091"],
                "failure_threshold": "Revise the method if every component contribution is indistinguishable from zero.",
                "claim_status": "predicted_not_observed",
            },
        ],
        "paper_comparisons": [
            {
                "paper_id": "DTA-001",
                "method_name": "DeepDTA",
                "shared_elements": ["SMILES and protein sequence inputs", "Davis/KIBA comparison"],
                "proposed_differences": ["assay context", "ranking objective", "predictive uncertainty", "strict split audit"],
                "reusable_elements": ["baseline definition and classical benchmark protocol"],
                "fair_comparison_conditions": ["same corrected records", "same split hashes", "same seeds and tuning budget"],
            },
            {
                "paper_id": "DTA-045",
                "method_name": "Multi-task bioassay pre-training",
                "shared_elements": ["bioassay information", "affinity prediction"],
                "proposed_differences": ["measurement-type semantics", "within-assay ranking", "calibration gate"],
                "reusable_elements": ["bioassay-aware comparison framing"],
                "fair_comparison_conditions": ["same raw assay snapshot", "same field coverage strata", "same downstream encoder budget"],
            },
            {
                "paper_id": "DTA-078",
                "method_name": "UAMRL",
                "shared_elements": ["uncertainty-aware DTA", "multimodal representation"],
                "proposed_differences": ["assay-measurement uncertainty", "strict probabilistic scoring", "sequence/2D minimum path"],
                "reusable_elements": ["uncertainty-aware baseline concept"],
                "fair_comparison_conditions": ["same available modalities", "same calibration set", "report NLL, CRPS, and coverage"],
            },
            {
                "paper_id": "DTA-091",
                "method_name": "Data-bias-resolved affinity prediction",
                "shared_elements": ["generalization under data bias", "strict evaluation"],
                "proposed_differences": ["assay tokens and ranking", "predictive distributions", "explicit missingness"],
                "reusable_elements": ["bias audit categories"],
                "fair_comparison_conditions": ["same deduplication", "same OOD groups", "same leakage report"],
            },
            {
                "paper_id": "DTA-093",
                "method_name": "Modification-aware DAVIS benchmark",
                "shared_elements": ["Davis correction and precision evaluation"],
                "proposed_differences": ["a learning method layered on the corrected benchmark"],
                "reusable_elements": ["correction-aware dataset validation"],
                "fair_comparison_conditions": ["use the exact corrected version and publish record-level provenance"],
            },
        ],
        "design_decisions": [
            {
                "decision_id": "decision-minimum-modality",
                "question": "Which modalities define the minimum publishable and reproducible method?",
                "options": ["sequence and 2D molecular graph", "sequence, graph, and predicted 3D structure"],
                "selected_option": "sequence and 2D molecular graph",
                "rationale": "This preserves broad public-data coverage and isolates assay semantics from uncertain structure quality.",
                "tradeoffs": ["lower structural expressivity", "lower preprocessing cost", "clearer attribution"],
                "evidence_refs": ["DTA-001", "DTA-003", "DTA-012"],
                "revisit_trigger": "Add 3D only after structure coverage, quality, compute, and leakage gates are predeclared and pass.",
            },
            {
                "decision_id": "decision-label-semantics",
                "question": "How should heterogeneous Kd, Ki, IC50, and integrated labels be represented?",
                "options": ["merge all normalized values", "retain measurement type and domain with masked context"],
                "selected_option": "retain measurement type and domain with masked context",
                "rationale": "These labels are not exchangeable physical quantities, so unconditional pooling weakens construct validity.",
                "tradeoffs": ["context sparsity", "more faithful label semantics", "requires coverage-stratified reporting"],
                "evidence_refs": ["DTA-045", "DTA-078", "DTA-091"],
                "revisit_trigger": "Remove a context field if train-only coverage is below the frozen usability threshold.",
            },
            {
                "decision_id": "decision-objective",
                "question": "Which learning objective jointly evaluates accuracy, ordering, and uncertainty?",
                "options": ["ordinary MSE", "heteroscedastic regression plus within-assay ranking"],
                "selected_option": "heteroscedastic regression plus within-assay ranking",
                "rationale": "The combined objective makes assay-local order and predictive uncertainty directly falsifiable.",
                "tradeoffs": ["loss-weight tuning", "possible gradient conflict", "richer calibration evaluation"],
                "evidence_refs": ["DTA-045", "DTA-078", "DTA-091"],
                "revisit_trigger": "Fall back to heteroscedastic regression if ranking pairs are too sparse or gradients remain conflicting.",
            },
            {
                "decision_id": "decision-primary-evaluation",
                "question": "Which evaluation family governs the main generalization claim?",
                "options": ["random split only", "cold-target, scaffold, family, and cross-dataset strict splits"],
                "selected_option": "cold-target, scaffold, family, and cross-dataset strict splits",
                "rationale": "Strict axes expose entity and domain leakage that random splits can conceal.",
                "tradeoffs": ["higher variance", "lower comparability with weak protocols", "stronger external-validity evidence"],
                "evidence_refs": ["DTA-001", "DTA-003", "DTA-091", "DTA-093"],
                "revisit_trigger": "Drop an axis only when its frozen test group lacks enough independent entities for analysis.",
            },
            {
                "decision_id": "decision-repository-reuse",
                "question": "How should paper-linked GitHub code enter the local implementation?",
                "options": ["copy repository pipelines", "typed adapters with parity tests and clean-room fallbacks"],
                "selected_option": "typed adapters with parity tests and clean-room fallbacks",
                "rationale": "Adapters preserve license and provenance review while controlling preprocessing and metric differences.",
                "tradeoffs": ["integration work", "reduced hidden coupling", "reproducible replacement path"],
                "evidence_refs": ["DTA-003", "DTA-008", "DTA-012"],
                "revisit_trigger": "Switch to clean-room reimplementation when license, commit, dependency, or parity checks fail.",
            },
        ],
        "data_audit_plans": [
            {
                "dataset_key": "dataset-davis-corrected",
                "schema_checks": ["drug, target, SMILES, sequence, affinity, unit, and correction provenance", "canonical identifier and numeric-range validation"],
                "leakage_checks": ["duplicate pair and entity overlap by split", "scaffold, sequence-similarity, target-family, and correction-version leakage"],
                "duplicate_conflict_policy": "Preserve raw provenance, group exact pairs, quantify disagreement, and fit any reconciliation rule on training records only.",
                "missingness_policy": "Report field-level missingness; reject rows only through frozen rules independent of test outcomes.",
                "distribution_shift_checks": ["label and entity-frequency drift by split", "scaffold and target-family coverage drift"],
                "acceptance_gates": ["official source, license evidence, digest, and correction version recorded", "zero declared leakage and reproducible split hashes"],
            },
            {
                "dataset_key": "dataset-kiba",
                "schema_checks": ["drug, target, SMILES, sequence, KIBA score, and source provenance", "entity count and sparse-matrix consistency"],
                "leakage_checks": ["duplicate pair and entity overlap by split", "scaffold and target-family overlap"],
                "duplicate_conflict_policy": "Retain the official integrated score and reject silent replacement by raw heterogeneous assays.",
                "missingness_policy": "Quantify absent matrix cells separately from invalid required entity fields.",
                "distribution_shift_checks": ["score quantiles and tail drift", "drug, target, scaffold, and family frequency drift"],
                "acceptance_gates": ["source terms and snapshot digest recorded", "strict split groups meet minimum independent-entity counts"],
            },
            {
                "dataset_key": "dataset-bindingdb-assay",
                "schema_checks": ["measurement value, unit, type, relation, assay, source, compound, and target fields", "unit conversion and inequality relation validation"],
                "leakage_checks": ["canonical pair duplicates across splits", "temporal, scaffold, target-family, and source-domain overlap"],
                "duplicate_conflict_policy": "Keep repeated measurements as grouped evidence, report disagreement, and prevent the same canonical pair from crossing split boundaries.",
                "missingness_policy": "Use explicit context masks, report coverage strata, and disable context-specific claims below the frozen usable-row threshold.",
                "distribution_shift_checks": ["affinity type, source, date, label, and missingness drift", "entity, scaffold, and target-family concentration"],
                "acceptance_gates": ["bulk snapshot reuse terms and provenance verified", "assay-context coverage and independent OOD groups exceed frozen thresholds"],
            },
        ],
        "statistical_analysis": {
            "primary_metric": "paired strict-split MSE delta versus the same encoder with ordinary MSE",
            "secondary_metrics": ["CI", "Spearman", "NLL", "CRPS", "90 percent coverage error", "risk-coverage AUC", "runtime", "peak GPU memory"],
            "analysis_unit": "matched observations clustered by predeclared entities within each immutable dataset split",
            "effect_size": "paired absolute and relative metric delta with per-split standardized effect where meaningful",
            "confidence_interval": "paired entity-stratified bootstrap 95 percent interval with the split assignment held fixed",
            "alpha": 0.05,
            "multiple_testing_correction": "Holm correction across confirmatory strict-split and ablation hypotheses",
            "seed_aggregation": "publish every seed plus descriptive mean and standard deviation; random seeds are not the inference unit",
            "random_seeds": [7, 17, 27],
            "minimum_random_seeds": 3,
            "minimum_successful_seeds": 2,
            "precision_or_power_analysis": "Use simulation-based precision analysis to require enough held-out entity clusters for the prespecified detectable effect and confidence-interval width.",
            "assumption_checks": ["Check dependence, residual heterogeneity, and support overlap by frozen split."],
            "robustness_checks": ["Repeat inference with pair-level and entity-cluster bootstrap schemes."],
            "interval_width_normalization": "dimensionless interval width divided by the frozen per-dataset training-label IQR",
            "negative_control_policy": "Conformal coverage alone cannot judge shuffled labels; require worse normalized interval efficiency, higher point error, and uninformative width-error Spearman tracking.",
            "practical_significance_threshold": "at least 1 percent relative strict-split MSE reduction without over 2 percent in-domain degradation",
            "failure_decision_rule": "Reject the main claim if corrected intervals cross zero on all strict axes or the practical threshold is missed.",
        },
        "experiment_stages": [
            {
                "stage_id": "stage-data-validation",
                "stage_type": "data_validation",
                "objective": "Verify licenses, schemas, distributions, duplicates, context coverage, and leakage before training.",
                "attempt_budget": 2,
                "entry_gate": ["official or paper-linked sources resolved"],
                "exit_gate": ["all dataset audit acceptance gates pass", "split manifests and hashes are frozen"],
                "stop_conditions": ["license remains ambiguous", "declared leakage cannot be removed"],
                "escalation": "Remove the failing dataset or context field and rescope claims before any model run.",
            },
            {
                "stage_id": "stage-baseline-reproduction",
                "stage_type": "baseline_reproduction",
                "objective": "Reproduce classical, DeepDTA-style, and graph baselines through the local evaluation stack.",
                "attempt_budget": 2,
                "entry_gate": ["data validation and metric unit tests pass"],
                "exit_gate": ["one-seed smoke runs and metric parity tests pass", "differences from paper settings are documented"],
                "stop_conditions": ["baseline parity remains unexplained after two attempts"],
                "escalation": "Route to code review and use a clean-room same-capacity baseline when repository code is unreliable.",
            },
            {
                "stage_id": "stage-equal-budget-tuning",
                "stage_type": "tuning",
                "objective": "Tune baselines and AssayCal-DTA with equal validation-only compute budgets.",
                "attempt_budget": 3,
                "entry_gate": ["baseline reproduction passes"],
                "exit_gate": ["search spaces, trial counts, selections, and environment are frozen"],
                "stop_conditions": ["tuning budget is exhausted", "validation instability exceeds the frozen threshold"],
                "escalation": "Reduce all methods to the same smaller search budget without inspecting test results.",
            },
            {
                "stage_id": "stage-assaycal-method",
                "stage_type": "proposed_method",
                "objective": "Run the frozen proposed method across all declared seeds and primary splits.",
                "attempt_budget": 2,
                "entry_gate": ["tuning manifest is frozen", "one-epoch feasibility run passes"],
                "exit_gate": ["all seed-level result manifests and prediction digests validate"],
                "stop_conditions": ["non-finite loss persists", "compute exceeds the declared feasibility ceiling"],
                "escalation": "Use the sequence-only context-and-uncertainty fallback without changing the primary split.",
            },
            {
                "stage_id": "stage-component-ablation",
                "stage_type": "ablation",
                "objective": "Measure assay tokens, missingness masks, ranking loss, and heteroscedastic head contributions.",
                "attempt_budget": 1,
                "entry_gate": ["the proposed method completes at least the primary seeds"],
                "exit_gate": ["paired effects, corrected intervals, and negative findings are reported"],
                "stop_conditions": ["the primary method fails its feasibility or data gate"],
                "escalation": "Narrow or reject the mechanism claim rather than adding post-hoc modules.",
            },
            {
                "stage_id": "stage-ood-stress",
                "stage_type": "stress_test",
                "objective": "Evaluate calibration, missing-context sensitivity, OOD axes, efficiency, and preselected cases.",
                "attempt_budget": 1,
                "entry_gate": ["main and ablation result manifests are immutable"],
                "exit_gate": ["all prespecified stress tables and failure thresholds are published"],
                "stop_conditions": ["case selection or calibration fit uses test outcomes"],
                "escalation": "Invalidate the affected analysis and rerun only from a frozen pre-test manifest.",
            },
        ],
        "unresolved_questions": [
            "What fraction of the licensed BindingDB snapshot has usable assay type, relation, source, and date fields?",
            "Can corrected Davis records be linked to assay semantics without introducing provenance ambiguity?",
            "Which paper baselines remain reproducible under one normalized local preprocessing and metric stack?",
            "Does within-assay ranking provide enough independent pairs after strict grouping?",
        ],
        "claim_boundaries": [
            "The design supports computational prediction, calibration, robustness, and reproducibility claims only.",
            "It does not establish biochemical mechanism, clinical utility, binding causality, or wet-lab validity.",
            "Dataset counts and distributions remain requirements until local artifacts record measured profiles.",
            "Expected metric deltas are falsifiable planning targets, never observed results.",
            "Novelty priority requires full-text and contemporaneous literature review beyond metadata summaries.",
        ],
        "implementation_roadmap": {
            "milestones": [
                {
                    "milestone_id": "milestone-data-gate",
                    "deliverable": "Licensed snapshots, typed profiles, and zero-leakage split manifests.",
                    "prerequisites": [],
                    "acceptance_criteria": ["digests and license evidence recorded", "distribution and conflict report generated", "split tests pass"],
                    "fallback": "Start with correction-aware Davis and KIBA while BindingDB provenance is reviewed.",
                },
                {
                    "milestone_id": "milestone-baselines",
                    "deliverable": "Classical, DeepDTA-style, and graph baselines on identical splits.",
                    "prerequisites": ["milestone-data-gate"],
                    "acceptance_criteria": ["one-seed smoke tests pass", "metric parity tests pass", "result manifests are reproducible"],
                    "fallback": "Use sequence-only and simple graph baselines before adding paper repositories.",
                },
                {
                    "milestone_id": "milestone-assaycal",
                    "deliverable": "Context conditioner and regression-ranking uncertainty head.",
                    "prerequisites": ["milestone-baselines"],
                    "acceptance_criteria": ["missing-context fallback works", "losses remain finite", "one-epoch feasibility run completes"],
                    "fallback": "Remove graph complexity and retain sequence encoder plus context/uncertainty head.",
                },
                {
                    "milestone_id": "milestone-evaluation",
                    "deliverable": "Three-seed main, OOD, calibration, ablation, module, case, and feasibility reports.",
                    "prerequisites": ["milestone-assaycal"],
                    "acceptance_criteria": ["paired intervals and all failure thresholds recorded", "no measured claim exists without a result artifact"],
                    "fallback": "Reduce the dataset/encoder size while preserving fixed splits and all primary gates.",
                },
            ],
            "estimated_compute": (
                "Minimum path: one modern GPU with 16-24 GB memory for sequence/2D models; run a "
                "single-seed profile first, then schedule three seeds only after memory and time gates pass."
            ),
            "stop_conditions": [
                "required dataset license or provenance cannot be verified",
                "assay fields have insufficient coverage for the predeclared context analysis",
                "split leakage remains after canonicalization",
                "baseline parity or deterministic smoke tests fail",
            ],
        },
        "risks": [
            {
                "risk_id": "risk-context-missing",
                "category": "data",
                "description": "BindingDB assay context may be incomplete or inconsistently encoded.",
                "likelihood": "high",
                "impact": "high",
                "mitigation": "Predeclare coverage thresholds, use missingness masks, and report coverage-stratified results.",
            },
            {
                "risk_id": "risk-label-semantics",
                "category": "method",
                "description": "Kd, Ki, IC50, and integrated scores are not interchangeable physical quantities.",
                "likelihood": "high",
                "impact": "high",
                "mitigation": "Never merge unconditionally; use type/domain tokens, within-domain ranking, and sensitivity analyses.",
            },
            {
                "risk_id": "risk-baseline-fairness",
                "category": "evaluation",
                "description": "Repository preprocessing or tuning differences may create an unfair paper comparison.",
                "likelihood": "medium",
                "impact": "high",
                "mitigation": "Route every method through local adapters, fixed splits, metric parity tests, and an equal tuning budget.",
            },
            {
                "risk_id": "risk-repository",
                "category": "reproducibility",
                "description": "Paper repositories may have stale dependencies or undocumented preprocessing.",
                "likelihood": "medium",
                "impact": "medium",
                "mitigation": "Review license and code, pin commits, run parity tests, and keep clean-room fallbacks.",
            },
        ],
        "limitations": [
            "Dataset counts and distributions are pre-download expectations and must be replaced by measured profiles.",
            "The proposed performance deltas are falsifiable planning targets, not observed results.",
            "The initial method focuses on sequence and 2D graphs; optional 3D structure is intentionally deferred.",
            "No wet-lab, biological, animal, or clinical experiment is part of this design.",
        ],
    }


class OfflineDtaPlanningModel:
    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        del user_prompt
        if "read-only auxiliary Agent" in system_prompt:
            role = system_prompt.split(",", 1)[0].removeprefix("You are ")
            return deepcopy(advisor_response(role))
        if system_prompt.startswith("You are make-idea-agent"):
            return deepcopy(idea_response())
        if system_prompt.startswith("You are idea-design-agent"):
            return deepcopy(design_response())
        raise RuntimeError("offline DTA planner received an unsupported Agent prompt")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Make Idea and Idea Design on the local verified DTA corpus"
    )
    parser.add_argument(
        "--deepseek",
        action="store_true",
        help="use DEEPSEEK_API_KEY from the environment instead of the offline planner",
    )
    args = parser.parse_args()
    corpus_payload = CORPUS_PATH.read_bytes()
    corpus_ref = ArtifactRef(
        uri=f"artifact://reference/{PROJECT_SLUG}/literature/papers.json",
        sha256=hashlib.sha256(corpus_payload).hexdigest(),
        kind="paper_corpus",
    )
    store = ArtifactStore(WORKSPACE_ROOT)
    model = DeepSeekModel() if args.deepseek else OfflineDtaPlanningModel()
    run_id = "run_dta_100_idea_design_v3"
    objective = (
        "Design a reliable and fully computational drug-target/protein-ligand binding affinity "
        "prediction paper method grounded in the verified local 100-paper corpus."
    )
    idea_result = MakeIdeaAgent(model, store).run(
        TaskEnvelope(
            run_id=run_id,
            task_id="task_dta_make_idea_v2",
            agent=AgentName.MAKE_IDEA,
            objective=objective,
            inputs=[corpus_ref],
            write_scope=[f"design/{PROJECT_SLUG}/ideas/"],
            context={"project_slug": PROJECT_SLUG},
        )
    )
    if idea_result.status != "succeeded":
        print(idea_result.model_dump_json(indent=2))
        return 1
    machine = ResearchStateMachine()
    registry = build_mock_registry()
    registry[AgentName.IDEA_DESIGN] = IdeaDesignAgent(model, store)
    runtime = ResearchRuntime(
        WORKSPACE_ROOT,
        RuleBasedLeadAgent(machine),
        registry,
        feedback_after_each_step=False,
        execution_mode=("live" if args.deepseek else "mock"),
    )
    pre_design = RunSnapshot(
        run_id=run_id,
        project_slug=PROJECT_SLUG,
        objective=objective,
        execution_mode=("live" if args.deepseek else "mock"),
        state=WorkflowState.DESIGNING_IDEA,
        version=3,
        completed_agents=[AgentName.DOWNLOAD_PAPER, AgentName.MAKE_IDEA],
        artifacts=[corpus_ref, *idea_result.artifacts],
    )
    runtime.events.save_snapshot(pre_design)
    paused = runtime.run_step(pre_design)
    review = paused.pending_user_review
    if review is None:
        raise RuntimeError("Idea Design run did not reach the required user-review hook")
    result_path = (
        WORKSPACE_ROOT
        / "project"
        / "var"
        / "runs"
        / run_id
        / "agent_results"
        / f"{review.source_task_id}.json"
    )
    design_result = json.loads(result_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "execution_mode": (
                    "deepseek_live_agent_contract_run"
                    if args.deepseek
                    else "offline_deterministic_agent_contract_run"
                ),
                "deepseek_called": args.deepseek,
                "make_idea": idea_result.model_dump(mode="json"),
                "idea_design": design_result,
                "workflow": {
                    "state": paused.state.value,
                    "version": paused.version,
                    "review_id": review.review_id,
                    "proposed_state": review.proposed_state.value,
                    "allowed_revision_targets": [
                        agent.value for agent in review.allowed_revision_targets
                    ],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if design_result["status"] == "succeeded" else 1


if __name__ == "__main__":
    raise SystemExit(main())
