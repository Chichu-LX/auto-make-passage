def _candidate(
    idea_id: str,
    hypothesis_id: str,
    title: str,
    innovation_type: str,
    novelty: str,
):
    return {
        "idea": {
            "idea_id": idea_id,
            "title": title,
            "research_question": f"Can {title.lower()} improve strict affinity prediction evaluation?",
            "novelty_claim": novelty,
            "rationale": "The supplied papers expose complementary accuracy and robustness gaps.",
            "evidence_refs": ["paper-one", "paper-two"],
        },
        "hypothesis": {
            "hypothesis_id": hypothesis_id,
            "statement": f"{title} will improve a predeclared computational metric.",
            "expected_direction": "Generalization quality should improve under strict controls.",
            "falsification_criterion": "Reject when paired confidence intervals include no improvement.",
            "evidence_refs": ["paper-one", "paper-two"],
        },
        "innovation_type": innovation_type,
        "baseline_gap": "Current baselines omit this factor under strict out-of-domain evaluation.",
        "expected_contribution": "A bounded and falsifiable computational affinity contribution.",
        "differentiation": ["explicit conditioning", "predeclared strict evaluation"],
        "feasibility_score": 0.8,
        "novelty_score": 0.75,
        "implementation_risk": "medium",
    }


def valid_idea_response():
    return {
        "candidates": [
            _candidate(
                "idea-one",
                "hypothesis-one",
                "Assay-aware calibrated affinity prediction",
                "uncertainty_calibration",
                "Conditioned uncertainty separates assay noise from model uncertainty.",
            ),
            _candidate(
                "idea-two",
                "hypothesis-two",
                "Leakage-resistant affinity benchmark",
                "evaluation_protocol",
                "A multi-axis split audit exposes hidden generalization failures.",
            ),
            _candidate(
                "idea-three",
                "hypothesis-three",
                "Adaptive multimodal affinity fusion",
                "model_architecture",
                "Confidence-aware gates avoid forcing unreliable structural inputs.",
            ),
        ],
        "recommended_idea_id": "idea-one",
        "selection_rationale": "Idea one balances an important label-semantics gap with a bounded implementation path.",
        "diversity_note": "Candidates span calibration, evaluation protocol, and architecture innovations.",
        "limitations": ["Full-text novelty review is still required."],
    }


def _experiment(experiment_id: str, experiment_type: str, objective: str):
    return {
        "experiment_id": experiment_id,
        "experiment_type": experiment_type,
        "objective": objective,
        "dataset_keys": ["dataset-davis"],
        "baselines": ["sequence baseline", "graph baseline"],
        "variables": ["model component"],
        "controls": ["same split", "same seeds", "same metric implementation"],
        "metrics": ["MSE", "CI", "NLL"],
        "random_seeds": [7, 17, 27],
        "protocol": ["run fixed configurations", "report paired confidence intervals"],
        "success_criteria": ["predeclared primary criterion is met"],
        "evidence_refs": ["paper-one", "paper-two"],
        "computational_only": True,
    }


def advisor_response(role: str):
    return {
        "advisor": role,
        "recommendations": [
            "Use paper-grounded interfaces and immutable experiment manifests.",
            "Predeclare evidence gates, fallbacks, and falsification thresholds.",
        ],
        "decisions": ["Retain a sequence and 2D minimum implementation path."],
        "blocking_concerns": [],
        "evidence_refs": ["paper-one", "paper-two"],
        "assumptions": ["Dataset fields and licenses remain subject to local verification."],
        "issue_codes": [],
    }


def valid_design_response():
    return {
        "selected_idea_id": "idea-one",
        "method_name": "AssayCal-DTA",
        "executive_summary": "A paper-grounded computational design that conditions affinity prediction on assay context and reports calibrated uncertainty.",
        "method_components": [
            {
                "component_id": "component-encoder",
                "name": "Drug-target encoder",
                "purpose": "Encode molecular graphs and protein sequences into pair representations.",
                "inputs": ["molecular graph", "protein sequence"],
                "outputs": ["pair embedding"],
                "innovation": "The encoder exposes stable interfaces for strict reuse comparisons.",
                "evidence_refs": ["paper-one", "paper-two"],
                "dependency_ids": [],
            },
            {
                "component_id": "component-calibration",
                "name": "Assay-aware uncertainty head",
                "purpose": "Predict affinity mean and variance conditioned on assay metadata.",
                "inputs": ["pair embedding", "assay type", "missingness mask"],
                "outputs": ["affinity mean", "predictive variance"],
                "innovation": "Heteroscedastic prediction treats assay context as measured noise evidence.",
                "evidence_refs": ["paper-one", "paper-two"],
                "dependency_ids": ["component-encoder"],
            },
        ],
        "reusable_code_modules": [
            {
                "module_id": "reuse-paper-one",
                "repository_url": "https://github.com/example/affinity",
                "source_paper_id": "paper-one",
                "repository_status": "verified_open_source",
                "license_id": "MIT",
                "reuse_mode": "integrate",
                "reusable_parts": ["dataset parser", "metric calculation"],
                "integration_strategy": "Wrap the parser and metrics behind local typed adapters.",
                "maturity_evidence": ["paper-linked repository", "explicit license"],
                "fallback_implementation": "Implement clean-room parsers and tested metric functions.",
            }
        ],
        "code_blueprint": {
            "pipeline": [
                "validate configuration",
                "load and audit raw records",
                "build leakage-safe splits",
                "train baseline and proposed models",
                "evaluate metrics and calibration",
                "write immutable result manifest",
            ],
            "packages": [
                {
                    "path": "code/src/data",
                    "responsibility": "Dataset parsing and leakage audit.",
                    "key_interfaces": ["load_records", "make_splits"],
                    "reference_module_ids": ["reuse-paper-one"],
                },
                {
                    "path": "code/src/models",
                    "responsibility": "Encoders and assay-aware prediction heads.",
                    "key_interfaces": ["build_model", "forward"],
                    "reference_module_ids": [],
                },
                {
                    "path": "code/src/evaluation",
                    "responsibility": "Metrics, calibration, and confidence intervals.",
                    "key_interfaces": ["evaluate", "bootstrap_ci"],
                    "reference_module_ids": ["reuse-paper-one"],
                },
            ],
            "entrypoints": ["python -m affinity.train", "python -m affinity.evaluate"],
            "config_contracts": ["dataset", "split", "model", "seed", "output directory"],
            "tests": ["parser unit tests", "split leakage tests", "metric parity tests"],
            "dependencies": ["pytorch", "rdkit", "pandas", "scikit-learn"],
            "minimum_viable_implementation": ["Davis loader", "sequence baseline", "assay-aware head"],
        },
        "dataset_requirements": [
            {
                "dataset_key": "dataset-davis",
                "source_kind": "external_dataset",
                "candidate_names": ["Davis"],
                "priority": "required",
                "acquisition_query": "Davis drug target affinity official dataset license download",
                "official_source_urls": [],
                "contents": ["drug SMILES", "protein sequence", "affinity matrix"],
                "required_fields": ["drug_id", "target_id", "smiles", "sequence", "affinity"],
                "reported_scale": "Verify paper-reported entity and pair counts after download.",
                "expected_distribution": "Continuous values with repeated entities; verify skew and missing cells.",
                "distribution_checks": ["label histogram and quantiles", "duplicate-pair counts"],
                "split_strategy": "Freeze random, cold-drug, cold-target, and scaffold splits with hashes.",
                "reliability_basis": "A mainstream benchmark used by both supplied affinity papers.",
                "license_requirement": "A traceable redistributable or research-use license is required.",
                "paper_refs": ["paper-one", "paper-two"],
                "alternatives": ["KIBA", "BindingDB"],
            }
        ],
        "experiment_plans": [
            _experiment(
                "experiment-main",
                "main_benchmark",
                "Compare the proposed method with paper baselines on fixed splits.",
            ),
            _experiment(
                "experiment-ablation",
                "ablation",
                "Measure the contribution of context and uncertainty components.",
            ),
            _experiment(
                "experiment-case",
                "case_study",
                "Inspect calibrated predictions for held-out target families.",
            ),
        ],
        "expected_results": [
            {
                "result_id": "result-main",
                "experiment_id": "experiment-main",
                "metric": "MSE",
                "expected_direction": "lower",
                "target_delta": {"low": -0.04, "high": -0.01},
                "expectation": "A small reproducible error reduction is expected under strict controls.",
                "evidence_refs": ["paper-one", "paper-two"],
                "failure_threshold": "The hypothesis fails when paired intervals include no improvement.",
                "claim_status": "predicted_not_observed",
            }
        ],
        "paper_comparisons": [
            {
                "paper_id": "paper-one",
                "method_name": "Sequence baseline",
                "shared_elements": ["same benchmark"],
                "proposed_differences": ["assay context"],
                "reusable_elements": ["parser"],
                "fair_comparison_conditions": ["same split and metric"],
            },
            {
                "paper_id": "paper-two",
                "method_name": "Graph baseline",
                "shared_elements": ["affinity regression"],
                "proposed_differences": ["uncertainty head"],
                "reusable_elements": [],
                "fair_comparison_conditions": ["same seeds and tuning budget"],
            },
        ],
        "design_decisions": [
            {
                "decision_id": "decision-inputs",
                "question": "Which minimum input modalities keep the contribution testable?",
                "options": ["sequence and 2D graph", "sequence, graph, and predicted structure"],
                "selected_option": "sequence and 2D graph",
                "rationale": "This isolates assay conditioning without depending on structure coverage.",
                "tradeoffs": ["less structural expressivity", "higher dataset coverage"],
                "evidence_refs": ["paper-one", "paper-two"],
                "revisit_trigger": "Revisit only if verified structure coverage exceeds the predeclared gate.",
            },
            {
                "decision_id": "decision-objective",
                "question": "Which objective separates accuracy from calibration?",
                "options": ["ordinary MSE", "heteroscedastic regression plus calibration"],
                "selected_option": "heteroscedastic regression plus calibration",
                "rationale": "It directly tests the selected uncertainty-calibration idea.",
                "tradeoffs": ["additional likelihood tuning", "explicit uncertainty scoring"],
                "evidence_refs": ["paper-one", "paper-two"],
                "revisit_trigger": "Revisit if the variance head becomes numerically unstable after two attempts.",
            },
            {
                "decision_id": "decision-splits",
                "question": "Which evaluation split should govern the primary claim?",
                "options": ["random split", "cold-target and scaffold splits"],
                "selected_option": "cold-target and scaffold splits",
                "rationale": "Strict splits provide a falsifiable generalization comparison.",
                "tradeoffs": ["higher variance", "lower leakage risk"],
                "evidence_refs": ["paper-one", "paper-two"],
                "revisit_trigger": "Revisit if a split has insufficient entities for three seeded runs.",
            },
        ],
        "data_audit_plans": [
            {
                "dataset_key": "dataset-davis",
                "schema_checks": ["required fields and dtypes", "per-dataset assay unit and identifier normalization with no pooling across measurement scales"],
                "leakage_checks": ["exact sequence hash and target identity overlap by split", "canonical SMILES scaffold and chemical-similarity overlap"],
                "duplicate_conflict_policy": "Retain provenance, flag each duplicate pair or identical model input with conflicting labels, and fit no reconciliation rule on test data.",
                "missingness_policy": "Report every field's missingness and use explicit masks without outcome-based row removal.",
                "distribution_shift_checks": ["label quantiles by split", "entity frequency and scaffold drift"],
                "acceptance_gates": ["license and digest recorded", "declared leakage rate equals zero"],
            }
        ],
        "statistical_analysis": {
            "primary_metric": "cold-target MSE",
            "secondary_metrics": ["CI", "NLL", "90 percent coverage error"],
            "analysis_unit": "matched affinity pairs clustered by drug and target within each frozen split",
            "effect_size": "coverage must stay above the lower 0.85 and below the upper 0.95 bound, with paired mean metric delta against the same-encoder baseline",
            "confidence_interval": "paired stratified bootstrap 95 percent interval over affinity pairs clustered by drug and target",
            "alpha": 0.05,
            "multiple_testing_correction": "Holm correction across confirmatory strict splits",
            "seed_aggregation": "report per-seed values plus descriptive mean and standard deviation; random seeds are not the inference unit",
            "random_seeds": [7, 17, 27],
            "minimum_random_seeds": 3,
            "minimum_successful_seeds": 2,
            "precision_or_power_analysis": "Use simulation-based precision analysis to require enough held-out entity clusters for a prespecified maximum coverage confidence-interval width and detectable paired effect.",
            "assumption_checks": [
                "Inspect dependence within drug and target clusters before inference.",
                "Check residual and interval-efficiency distributions by split.",
            ],
            "robustness_checks": [
                "Repeat inference with pair-level and entity-cluster bootstrap schemes.",
                "Report sensitivity to the prespecified practical-effect threshold.",
            ],
            "interval_width_normalization": "dimensionless interval width divided by the frozen per-dataset training-label IQR",
            "negative_control_policy": "Conformal coverage alone cannot judge shuffled labels; require worse normalized interval efficiency, higher point error, and uninformative width-error Spearman tracking.",
            "practical_significance_threshold": "coverage must remain between the lower 0.85 and upper 0.95 bounds and mean interval width must not exceed the baseline by more than ten percent",
            "failure_decision_rule": "Reject calibration only when the entity-cluster bootstrap confidence interval for coverage lies outside the exact-binomial-calibrated lower and upper tolerance; reject when normalized mean interval width exceeds the efficiency gate or when the corrected paired interval includes zero. The seed quorum gates execution completeness only.",
        },
        "experiment_stages": [
            {
                "stage_id": "stage-baseline",
                "stage_type": "baseline_reproduction",
                "objective": "Reproduce declared baselines before evaluating the proposed method.",
                "attempt_budget": 2,
                "entry_gate": ["data audit passes"],
                "exit_gate": ["metric parity and smoke tests pass"],
                "stop_conditions": ["baseline remains irreproducible after two attempts"],
                "escalation": "Route repository discrepancies to code review and use a clean-room baseline.",
            },
            {
                "stage_id": "stage-tuning",
                "stage_type": "tuning",
                "objective": "Allocate equal validation-only tuning budgets to all methods.",
                "attempt_budget": 3,
                "entry_gate": ["baseline reproduction passes"],
                "exit_gate": ["all configurations and budgets are frozen"],
                "stop_conditions": ["validation budget is exhausted"],
                "escalation": "Freeze the best valid configuration without inspecting test outcomes.",
            },
            {
                "stage_id": "stage-method",
                "stage_type": "proposed_method",
                "objective": "Run the selected assay-aware method under frozen settings.",
                "attempt_budget": 2,
                "entry_gate": ["tuning manifest is frozen"],
                "exit_gate": ["three-seed result manifests validate"],
                "stop_conditions": ["loss is non-finite after two bounded repairs"],
                "escalation": "Use the sequence-only fallback while retaining evaluation controls.",
            },
            {
                "stage_id": "stage-ablation",
                "stage_type": "ablation",
                "objective": "Isolate the assay context and uncertainty contributions.",
                "attempt_budget": 1,
                "entry_gate": ["proposed method completes all seeds"],
                "exit_gate": ["paired component effects and intervals are reported"],
                "stop_conditions": ["primary method fails its feasibility gate"],
                "escalation": "Revise the claim scope instead of adding unplanned experiments.",
            },
        ],
        "unresolved_questions": ["Does the official dataset expose usable assay context fields?"],
        "claim_boundaries": [
            "Only computational benchmark and calibration claims are permitted.",
            "Expected deltas are planning targets and not observed performance.",
        ],
        "implementation_roadmap": {
            "milestones": [
                {
                    "milestone_id": "milestone-data",
                    "deliverable": "Validated dataset adapter and split audit.",
                    "prerequisites": [],
                    "acceptance_criteria": ["schema and leakage tests pass"],
                    "fallback": "Use a smaller verified benchmark snapshot.",
                },
                {
                    "milestone_id": "milestone-model",
                    "deliverable": "Baseline and proposed model training path.",
                    "prerequisites": ["milestone-data"],
                    "acceptance_criteria": ["one-seed smoke run passes"],
                    "fallback": "Use the sequence-only encoder.",
                },
                {
                    "milestone_id": "milestone-eval",
                    "deliverable": "Reproducible evaluation and result manifest.",
                    "prerequisites": ["milestone-model"],
                    "acceptance_criteria": ["three-seed results and intervals exist"],
                    "fallback": "Reduce model size without changing the split.",
                },
            ],
            "estimated_compute": "Start with one modern GPU and profile before full seeds.",
            "stop_conditions": ["data license cannot be verified", "split leakage persists"],
        },
        "risks": [
            {
                "risk_id": "risk-one",
                "category": "data",
                "description": "Assay fields may be unavailable in the selected benchmark.",
                "likelihood": "medium",
                "impact": "high",
                "mitigation": "Use missingness masks and a sequence-only fallback.",
            }
        ],
        "limitations": ["Pre-download scales and distributions must be verified locally."],
    }
