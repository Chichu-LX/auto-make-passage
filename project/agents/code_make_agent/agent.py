from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Protocol

from pydantic import ValidationError

from research_agents.agent_package import load_agent_package
from research_agents.project_layout import artifact_prefix, scoped_path, task_project_slug
from research_agents.providers import JsonModel, ProviderRequestError
from research_agents.schemas import (
    AgentName,
    AgentResult,
    ArtifactRef,
    CodeGenerationDraft,
    CodeImplementationManifest,
    CodeRevisionPatchDraft,
    CodeTraceabilityItem,
    DatasetManifest,
    GeneratedCodeFileDraft,
    GeneratedCodeFileRecord,
    ResearchDesign,
    TaskEnvelope,
    VendoredSourceRecord,
)
from research_agents.storage import ArtifactStore, StoragePolicyError

from .policy import (
    MAX_CODE_FILE_BYTES,
    MAX_CODE_TOTAL_BYTES,
    MAX_DATA_FILE_BYTES,
    MAX_INPUT_BYTES,
    file_token,
)
from .local_validation import LocalValidationError, run_local_validations
from .source_vendor import GitHubSourceVendor, SourceVendorError
from .assaycal_scaffold import build_assaycal_scaffold, supports_assaycal_scaffold


PACKAGE = load_agent_package(__file__)
if PACKAGE.config.name != AgentName.CODE_MAKE:
    raise RuntimeError("Code Make Agent package name mismatch")


MAX_GENERATION_ATTEMPTS = 2
PATCH_SYSTEM_PROMPT = (Path(__file__).resolve().parent / "prompts" / "revision.md").read_text(
    encoding="utf-8"
)


class CodeGenerationContractError(ValueError):
    def __init__(
        self,
        cause_type: str,
        attempt_errors: list[dict[str, Any]],
    ) -> None:
        super().__init__(f"generated response remained invalid after {len(attempt_errors)} attempts")
        self.cause_type = cause_type
        self.attempt_errors = attempt_errors


class SourceVendor(Protocol):
    def vendor(self, module, *, project_slug: str) -> VendoredSourceRecord: ...


class CodeMakeAgent:
    """Generate, isolate, locally validate, and stage one implementation revision."""

    name = AgentName.CODE_MAKE
    allowed_tools = tuple(PACKAGE.config.allowed_tools)

    def __init__(
        self,
        model: JsonModel,
        artifacts: ArtifactStore,
        *,
        source_vendor: SourceVendor | None = None,
    ) -> None:
        self.model = model
        self.artifacts = artifacts
        self.source_vendor = source_vendor or GitHubSourceVendor(artifacts)

    def _read_inputs(
        self, task: TaskEnvelope
    ) -> tuple[ResearchDesign | None, ArtifactRef | None, DatasetManifest | None, ArtifactRef | None]:
        slug = task_project_slug(task)
        design: ResearchDesign | None = None
        design_ref: ArtifactRef | None = None
        data: DatasetManifest | None = None
        data_ref: ArtifactRef | None = None
        for artifact in task.inputs:
            try:
                if artifact.kind == "research_design":
                    if not artifact.uri.startswith(
                        artifact_prefix("design", slug, "research-designs")
                    ):
                        raise StoragePolicyError("research design escaped its project namespace")
                    candidate = ResearchDesign.model_validate(
                        self.artifacts.read_artifact_json(artifact, max_bytes=MAX_INPUT_BYTES)
                    )
                    if candidate.run_id != task.run_id:
                        raise StoragePolicyError("research design belongs to another run")
                    design, design_ref = candidate, artifact
                elif artifact.kind == "data_manifest":
                    if not artifact.uri.startswith(artifact_prefix("data", slug, "manifests")):
                        raise StoragePolicyError("data manifest escaped its project namespace")
                    candidate = DatasetManifest.model_validate(
                        self.artifacts.read_artifact_json(artifact, max_bytes=MAX_INPUT_BYTES)
                    )
                    if candidate.run_id != task.run_id or candidate.project_slug != slug:
                        raise StoragePolicyError("data manifest belongs to another run or project")
                    raw_prefix = artifact_prefix("data", slug, "raw")
                    dataset_prefix = artifact_prefix("data", slug, "datasets")
                    for record in candidate.records:
                        if record.status != "downloaded" or record.local_artifact is None:
                            continue
                        if not record.local_artifact.uri.startswith(raw_prefix):
                            raise StoragePolicyError("raw dataset escaped its project namespace")
                        raw_size = self.artifacts.verify_artifact_file(
                            record.local_artifact, max_bytes=MAX_DATA_FILE_BYTES
                        )
                        if raw_size != record.size_bytes:
                            raise StoragePolicyError("raw dataset size differs from its manifest")
                        for item in record.materialized_files:
                            if not item.artifact.uri.startswith(dataset_prefix):
                                raise StoragePolicyError(
                                    "materialized dataset escaped its project namespace"
                                )
                            measured_size = self.artifacts.verify_artifact_file(
                                item.artifact, max_bytes=MAX_DATA_FILE_BYTES
                            )
                            if measured_size != item.size_bytes:
                                raise StoragePolicyError(
                                    "materialized dataset size differs from its manifest"
                                )
                    data, data_ref = candidate, artifact
            except (StoragePolicyError, ValidationError, ValueError):
                continue
        return design, design_ref, data, data_ref

    def _prior_provider_failure(self, task: TaskEnvelope) -> bool:
        revision = task.context.get("user_revision")
        if isinstance(revision, dict) and str(revision.get("comments") or "").strip():
            # A concrete user repair request must receive one fresh generation
            # attempt instead of being trapped forever behind an old provider gap.
            return False
        prefix = artifact_prefix("code", task_project_slug(task), "manifests")
        for artifact in reversed(task.inputs):
            if artifact.kind != "code_generation_gap" or not artifact.uri.startswith(prefix):
                continue
            try:
                payload = self.artifacts.read_artifact_json(
                    artifact, max_bytes=MAX_INPUT_BYTES
                )
            except StoragePolicyError:
                continue
            return "ProviderRequestError" in str(payload.get("reason") or "")
        return False

    @staticmethod
    def _context(
        task: TaskEnvelope,
        design: ResearchDesign,
        data: DatasetManifest,
        vendored_sources: list[VendoredSourceRecord],
    ) -> dict[str, Any]:
        dataset_profiles = []
        for record in data.records:
            if record.status != "downloaded" or record.profile is None:
                continue
            profile = record.profile
            dataset_profiles.append(
                {
                    "dataset_id": record.dataset_id,
                    "requirement_key": record.requirement_key,
                    "format": record.format,
                    "version": record.version,
                    "size_bytes": record.size_bytes,
                    "sha256": record.sha256,
                    "raw_artifact_uri": (
                        record.local_artifact.uri if record.local_artifact else None
                    ),
                    "materialized_root": record.materialized_root,
                    "materialized_files": [
                        {
                            "path": item.relative_path,
                            "size_bytes": item.size_bytes,
                            "sha256": item.sha256,
                            "artifact_uri": item.artifact.uri,
                        }
                        for item in record.materialized_files
                    ],
                    "profile": {
                        "source_table": profile.source_table,
                        "row_count": profile.row_count,
                        "column_count": profile.column_count,
                        "columns": profile.columns,
                        "unique_compounds": profile.unique_compounds,
                        "unique_targets": profile.unique_targets,
                        "unique_pairs": profile.unique_pairs,
                        "entity_representation_profiles": [
                            item.model_dump(mode="json")
                            for item in profile.entity_representation_profiles
                        ],
                        "repeated_exact_input_group_count": profile.repeated_exact_input_group_count,
                        "conflicting_exact_input_label_group_count": profile.conflicting_exact_input_label_group_count,
                        "conflicting_exact_input_label_row_count": profile.conflicting_exact_input_label_row_count,
                        "representation_audit_complete": profile.representation_audit_complete,
                        "missing_fraction": profile.missing_fraction,
                        "key_field_coverage": profile.key_field_coverage,
                        "label_statistics": [
                            item.model_dump(mode="json") for item in profile.label_statistics
                        ],
                    },
                }
            )
        return {
            "objective": task.objective,
            "project_slug": task_project_slug(task),
            "user_revision": task.context.get("user_revision"),
            "research_design": design.model_dump(mode="json"),
            "measured_dataset_profiles": dataset_profiles,
            "local_vendored_sources": [
                {
                    "module_id": item.module_id,
                    "commit_sha": item.commit_sha,
                    "license_id": item.license_id,
                    "local_root": item.local_root,
                    "file_count": item.file_count,
                    "total_bytes": item.total_bytes,
                    "tree_sha256": item.tree_sha256,
                }
                for item in vendored_sources
            ],
            "generation_policy": {
                "write_root": f"code/{task_project_slug(task)}/",
                "execute_generated_code": "only fixed local demo and unit-test commands after static preflight",
                "third_party_copying": "integrate-mode sources are already downloaded locally and must never be fetched at runtime",
                "required_tree": [
                    "src/<package>/models/<model_name>/",
                    "src/<package>/experiments/<experiment_name>/",
                    "src/<package>/training/",
                    "src/<package>/testing/",
                    "tests/",
                    "configs/",
                    "README.md",
                    "pyproject.toml",
                ],
                "forbidden": [
                    "runtime network access",
                    "remote imports",
                    "shell=True",
                    "eval or exec",
                    "while True",
                    "embedded credentials",
                ],
                "next_gate": "code-review-agent",
            },
            "response_contract": CodeGenerationDraft.model_json_schema(),
        }

    @staticmethod
    def _language_for_path(path: str) -> str:
        suffix = Path(path).suffix.casefold()
        return {
            ".py": "python",
            ".toml": "toml",
            ".yaml": "yaml",
            ".yml": "yaml",
            ".json": "json",
            ".md": "markdown",
            ".txt": "text",
        }[suffix]

    def _revision_baseline(
        self,
        task: TaskEnvelope,
        design: ResearchDesign,
    ) -> tuple[str, CodeGenerationDraft] | None:
        """Reconstruct only the failed candidate named by the current user review."""

        revision = task.context.get("user_revision")
        if not isinstance(revision, dict):
            return None
        review_id = str(revision.get("review_id") or "")
        if not review_id.startswith("review_task_"):
            return None
        reviewed_task_id = review_id.removeprefix("review_")

        def read_gap(candidate_task_id: str) -> dict[str, Any] | None:
            if re.fullmatch(r"task_[A-Za-z0-9_-]{3,120}", candidate_task_id) is None:
                return None
            relative = scoped_path(
                "code",
                task_project_slug(task),
                "manifests",
                f"{file_token(candidate_task_id)}.gap.json",
            )
            path = (self.artifacts.root / relative).resolve()
            try:
                path.relative_to(self.artifacts.root.resolve())
            except ValueError:
                return None
            if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_INPUT_BYTES:
                return None
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                return None
            if not isinstance(payload, dict):
                return None
            if (
                payload.get("run_id") != task.run_id
                or payload.get("project_slug") != task_project_slug(task)
                or payload.get("task_id") != candidate_task_id
            ):
                return None
            return payload

        def read_implementation(
            candidate_task_id: str,
        ) -> CodeImplementationManifest | None:
            if re.fullmatch(r"task_[A-Za-z0-9_-]{3,120}", candidate_task_id) is None:
                return None
            relative = scoped_path(
                "code",
                task_project_slug(task),
                "manifests",
                f"{file_token(candidate_task_id)}.json",
            )
            path = (self.artifacts.root / relative).resolve()
            try:
                path.relative_to(self.artifacts.root.resolve())
            except ValueError:
                return None
            if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_INPUT_BYTES:
                return None
            try:
                manifest = CodeImplementationManifest.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            except (OSError, UnicodeDecodeError, ValidationError, ValueError):
                return None
            if (
                manifest.run_id != task.run_id
                or manifest.project_slug != task_project_slug(task)
                or manifest.task_id != candidate_task_id
            ):
                return None
            expected_root = scoped_path(
                "code",
                task_project_slug(task),
                "implementations",
                file_token(candidate_task_id),
            )
            if manifest.implementation_root != expected_root:
                return None
            return manifest

        def draft_from_implementation(
            manifest: CodeImplementationManifest,
        ) -> CodeGenerationDraft:
            files: list[GeneratedCodeFileDraft] = []
            for item in manifest.files:
                measured = self.artifacts.verify_artifact_file(
                    item.artifact,
                    max_bytes=MAX_CODE_FILE_BYTES,
                )
                if measured != item.size_bytes or item.artifact.sha256 != item.sha256:
                    raise StoragePolicyError(
                        "revision implementation file differs from its manifest"
                    )
                try:
                    content = self.artifacts.read_artifact_bytes(
                        item.artifact,
                        max_bytes=MAX_CODE_FILE_BYTES,
                    ).decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise StoragePolicyError(
                        "revision implementation is not valid UTF-8 text"
                    ) from exc
                files.append(
                    GeneratedCodeFileDraft(
                        path=item.path,
                        purpose=item.purpose,
                        language=item.language,
                        content=content,
                        design_component_ids=item.design_component_ids,
                        dataset_keys=item.dataset_keys,
                        experiment_ids=item.experiment_ids,
                        source_module_ids=item.source_module_ids,
                    )
                )
            draft = CodeGenerationDraft(
                package_name=manifest.package_name,
                architecture_summary=manifest.architecture_summary,
                files=files,
                entrypoints=manifest.entrypoints,
                dependencies=manifest.dependencies,
                validation_commands=manifest.validation_commands,
                traceability=manifest.traceability,
                known_gaps=[
                    *manifest.known_gaps,
                    (
                        f"Revising reviewed implementation {manifest.task_id}; no result "
                        "claim is accepted before a new Code Review."
                    ),
                ],
            )
            self._validate_traceability(draft, design)
            return draft

        comments = str(revision.get("comments") or "")
        # A user/professional review may explicitly identify the last known-good
        # implementation after the immediately reviewed candidate regressed.  Give
        # those run-scoped, manifest-verified task IDs precedence; every candidate
        # is still checked below for run, project, root and content integrity.
        candidate_ids: list[str] = re.findall(
            r"\btask_[0-9]{3}_[0-9a-f]{8}\b", comments, flags=re.IGNORECASE
        )
        candidate_ids.append(reviewed_task_id)
        reviewed_gap = read_gap(reviewed_task_id)
        if reviewed_gap is not None:
            details = reviewed_gap.get("details")
            if isinstance(details, dict):
                recorded_baseline = str(details.get("baseline_task_id") or "")
                if recorded_baseline:
                    candidate_ids.append(recorded_baseline)

        source_task_id = ""
        implementation_baseline: CodeImplementationManifest | None = None
        seen: set[str] = set()
        for candidate_task_id in candidate_ids:
            if candidate_task_id in seen:
                continue
            seen.add(candidate_task_id)
            payload = read_gap(candidate_task_id)
            if payload is not None:
                reason = str(payload.get("reason") or "").casefold()
                if (
                    "local execution preflight" in reason
                    or "fixed local demo and unit-test" in reason
                ):
                    candidate_root = (
                        self.artifacts.root
                        / scoped_path(
                            "code",
                            task_project_slug(task),
                            "implementations",
                            file_token(candidate_task_id),
                        )
                    )
                    if candidate_root.is_dir() and not candidate_root.is_symlink():
                        source_task_id = candidate_task_id
                        break
            candidate_manifest = read_implementation(candidate_task_id)
            if candidate_manifest is not None:
                source_task_id = candidate_task_id
                implementation_baseline = candidate_manifest
                break
        if not source_task_id:
            return None
        if implementation_baseline is not None:
            return source_task_id, draft_from_implementation(implementation_baseline)

        relative_root = scoped_path(
            "code",
            task_project_slug(task),
            "implementations",
            file_token(source_task_id),
        )
        root = (self.artifacts.root / relative_root).resolve()
        try:
            root.relative_to(self.artifacts.root.resolve())
        except ValueError as exc:
            raise StoragePolicyError("revision candidate escaped the artifact root") from exc
        if not root.is_dir() or root.is_symlink():
            raise StoragePolicyError("revision candidate directory is unavailable")

        files: list[GeneratedCodeFileDraft] = []
        total_bytes = 0
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                raise StoragePolicyError("revision candidate contains a symbolic link")
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            try:
                language = self._language_for_path(relative)
            except KeyError as exc:
                raise StoragePolicyError(
                    "revision candidate contains a non-allowlisted file"
                ) from exc
            size = path.stat().st_size
            if size <= 0 or size > MAX_CODE_FILE_BYTES:
                raise StoragePolicyError("revision candidate file violates size policy")
            total_bytes += size
            if total_bytes > MAX_CODE_TOTAL_BYTES:
                raise StoragePolicyError("revision candidate violates total size policy")
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise StoragePolicyError("revision candidate is not valid UTF-8 text") from exc
            files.append(
                GeneratedCodeFileDraft(
                    path=relative,
                    purpose=f"Verified failed-candidate source retained for bounded revision: {relative}",
                    language=language,
                    content=content,
                )
            )
        if not files:
            raise StoragePolicyError("revision candidate is empty")

        src_root = root / "src"
        package_dirs = sorted(
            item.name
            for item in src_root.iterdir()
            if item.is_dir() and not item.is_symlink()
        ) if src_root.is_dir() else []
        if len(package_dirs) != 1:
            raise StoragePolicyError("revision candidate must contain exactly one src package")
        package_name = package_dirs[0]
        paths = {item.path for item in files}
        anchor = "README.md" if "README.md" in paths else sorted(paths)[0]
        demo = f"src/{package_name}/demo.py"
        if demo not in paths:
            raise StoragePolicyError("revision candidate omitted the required demo")

        component_ids = [item.component_id for item in design.method_components]
        dataset_keys = [item.dataset_key for item in design.dataset_requirements]
        experiment_ids = [item.experiment_id for item in design.experiment_plans]
        source_module_ids = [
            item.module_id
            for item in design.reusable_code_modules
            if item.reuse_mode == "integrate"
        ]
        annotated = [
            item.model_copy(
                update={
                    "design_component_ids": component_ids if item.path == anchor else [],
                    "dataset_keys": dataset_keys if item.path == anchor else [],
                    "experiment_ids": experiment_ids if item.path == anchor else [],
                    "source_module_ids": source_module_ids if item.path == anchor else [],
                }
            )
            for item in files
        ]
        draft = CodeGenerationDraft(
            package_name=package_name,
            architecture_summary=(
                "Bounded revision of a verified failed Code Make candidate. The complete "
                "tree is copied into a new immutable implementation directory, patched by "
                "DeepSeek, and rechecked by the fixed local safety and execution gates."
            ),
            files=annotated,
            entrypoints=[f"{package_name}.demo", f"{package_name}.cli"],
            dependencies=["python>=3.12"],
            validation_commands=[
                f"python -m {package_name}.demo --json",
                "python -m unittest discover -s tests -p test_*.py",
            ],
            traceability=[
                CodeTraceabilityItem(
                    requirement="Preserve the approved method, data, and experiment contracts.",
                    file_paths=[anchor],
                    design_component_ids=component_ids,
                    dataset_keys=dataset_keys,
                    experiment_ids=experiment_ids,
                    source_module_ids=source_module_ids,
                    acceptance_test="Code Review confirms exact ResearchDesign coverage.",
                ),
                CodeTraceabilityItem(
                    requirement="Pass the dependency-free bounded local execution gate.",
                    file_paths=[demo],
                    acceptance_test=f"python -m {package_name}.demo --json emits update evidence.",
                ),
            ],
            known_gaps=[
                f"Recovered failed candidate {source_task_id}; no result claim is accepted before Code Review."
            ],
        )
        self._validate_traceability(draft, design)
        return source_task_id, draft

    def _patch_context(
        self,
        task: TaskEnvelope,
        design: ResearchDesign,
        data: DatasetManifest,
        vendored_sources: list[VendoredSourceRecord],
        baseline_task_id: str,
        baseline: CodeGenerationDraft,
        *,
        validation_errors: list[dict[str, Any]] | None = None,
        attempt: int = 1,
    ) -> dict[str, Any]:
        context = self._context(task, design, data, vendored_sources)
        revision = task.context.get("user_revision")
        comments = ""
        if isinstance(revision, dict):
            comments = str(revision.get("comments") or "")
        baseline_files = self._select_revision_context_files(baseline, comments)
        context.update(
            {
                "revision_mode": "replace_existing_files_only",
                "baseline_task_id": baseline_task_id,
                "baseline_files": [
                    {"path": item.path, "language": item.language, "content": item.content}
                    for item in baseline_files
                ],
                "baseline_file_inventory": [item.path for item in baseline.files],
                "revision_context_file_count": len(baseline_files),
                "response_contract": CodeRevisionPatchDraft.model_json_schema(),
            }
        )
        if validation_errors:
            context["contract_repair"] = {
                "attempt": attempt,
                "instruction": (
                    "Return a fresh complete CodeRevisionPatchDraft and fix every listed "
                    "contract error without adding, deleting, or renaming baseline files."
                ),
                "validation_errors": validation_errors,
            }
        return context

    @staticmethod
    def _select_revision_context_files(
        baseline: CodeGenerationDraft,
        comments: str,
        *,
        limit: int = 12,
    ) -> list[GeneratedCodeFileDraft]:
        """Bound revision prompts without hiding the immutable tree inventory.

        Full implementation bodies can be large enough to make a streaming model
        produce an effectively unbounded patch.  Rank files by the concrete user
        repair text, keep the CLI/README contracts visible, and send only a small
        dependency-oriented body set.  The complete path inventory is supplied
        separately and patch validation still rejects additions or renames.
        """

        words = {
            token
            for token in re.findall(r"[a-z0-9_]{4,}", comments.casefold())
            if token not in {"agent", "code", "file", "with", "from", "that", "this"}
        }
        lowered_comments = comments.casefold()
        explicit = [
            item for item in baseline.files if item.path.casefold() in lowered_comments
        ]
        if explicit:
            # A review that names concrete baseline paths authorizes a narrower
            # and more reliable edit surface than heuristic relevance ranking.
            return sorted(explicit, key=lambda item: item.path)[:3]

        def score(item: GeneratedCodeFileDraft) -> tuple[int, int, str]:
            path = item.path.casefold()
            content = item.content.casefold()
            value = 0
            if path == "readme.md":
                value += 400
            if path.endswith("/cli.py") or path == "cli.py":
                value += 1000
            if "/experiments/" in path or "/training/" in path:
                value += 180
            if "/data/" in path or "/models/" in path:
                value += 100
            if path.startswith("tests/"):
                value += 80
            for word in words:
                if word in path:
                    value += 120
                elif word in content:
                    value += 12
            return (-value, len(item.content), item.path)

        selected = sorted(baseline.files, key=score)[: max(1, limit)]
        return sorted(selected, key=lambda item: item.path)

    def _generate_patch_with_contract_repair(
        self,
        task: TaskEnvelope,
        design: ResearchDesign,
        data: DatasetManifest,
        vendored_sources: list[VendoredSourceRecord],
        baseline_task_id: str,
        baseline: CodeGenerationDraft,
    ) -> tuple[CodeGenerationDraft, list[dict[str, Any]]]:
        attempt_errors: list[dict[str, Any]] = []
        validation_errors: list[dict[str, Any]] | None = None
        baseline_by_path = {item.path: item for item in baseline.files}
        for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
            response = self.model.complete_json(
                PATCH_SYSTEM_PROMPT,
                json.dumps(
                    self._patch_context(
                        task,
                        design,
                        data,
                        vendored_sources,
                        baseline_task_id,
                        baseline,
                        validation_errors=validation_errors,
                        attempt=attempt,
                    ),
                    ensure_ascii=False,
                ),
            )
            try:
                patch = CodeRevisionPatchDraft.model_validate(response)
                if patch.baseline_task_id != baseline_task_id:
                    raise ValueError("revision patch names a different baseline task")
                unknown = {item.path for item in patch.files} - set(baseline_by_path)
                if unknown:
                    raise ValueError(f"revision patch adds unknown files: {sorted(unknown)}")
                replacements = {item.path: item for item in patch.files}
                merged = []
                for item in baseline.files:
                    replacement = replacements.get(item.path)
                    if replacement is None:
                        merged.append(item)
                        continue
                    if replacement.language != item.language:
                        raise ValueError(
                            f"revision patch changes file language: {item.path}"
                        )
                    merged.append(
                        item.model_copy(
                            update={
                                "purpose": replacement.purpose,
                                "content": replacement.content,
                            }
                        )
                    )
                draft = baseline.model_copy(
                    update={
                        "architecture_summary": patch.architecture_summary,
                        "files": merged,
                        "known_gaps": [*baseline.known_gaps, *patch.known_gaps],
                    }
                )
                draft = CodeGenerationDraft.model_validate(draft.model_dump(mode="json"))
                self._validate_traceability(draft, design)
                return draft, attempt_errors
            except (ValidationError, ValueError, TypeError) as exc:
                validation_errors = self._safe_contract_errors(exc)
                attempt_errors.append({"attempt": attempt, "errors": validation_errors})
                if attempt >= MAX_GENERATION_ATTEMPTS:
                    raise CodeGenerationContractError(
                        type(exc).__name__,
                        attempt_errors,
                    ) from exc
        raise RuntimeError("unreachable patch repair state")

    @staticmethod
    def _validate_traceability(draft: CodeGenerationDraft, design: ResearchDesign) -> None:
        component_ids = {item.component_id for item in design.method_components}
        dataset_keys = {item.dataset_key for item in design.dataset_requirements}
        experiment_ids = {item.experiment_id for item in design.experiment_plans}
        source_module_ids = {item.module_id for item in design.reusable_code_modules}
        integrated_module_ids = {
            item.module_id for item in design.reusable_code_modules if item.reuse_mode == "integrate"
        }
        for file in draft.files:
            if not set(file.design_component_ids).issubset(component_ids):
                raise ValueError("generated file references an unknown design component")
            if not set(file.dataset_keys).issubset(dataset_keys):
                raise ValueError("generated file references an unknown dataset key")
            if not set(file.experiment_ids).issubset(experiment_ids):
                raise ValueError("generated file references an unknown experiment")
            if not set(file.source_module_ids).issubset(source_module_ids):
                raise ValueError("generated file references an unknown reusable source module")
            lowered = file.content.lower()
            if "http://" in lowered or "https://" in lowered or "git+" in lowered:
                raise ValueError("generated files cannot contain runtime remote links")
        covered_components = {value for item in draft.files for value in item.design_component_ids}
        covered_datasets = {value for item in draft.files for value in item.dataset_keys}
        covered_experiments = {value for item in draft.files for value in item.experiment_ids}
        covered_sources = {value for item in draft.files for value in item.source_module_ids}
        if covered_components != component_ids:
            raise ValueError("generated files do not cover every design component")
        if covered_datasets != dataset_keys:
            raise ValueError("generated files do not cover every dataset requirement")
        if covered_experiments != experiment_ids:
            raise ValueError("generated files do not cover every experiment")
        if not integrated_module_ids.issubset(covered_sources):
            raise ValueError("generated files do not trace every integrated source module")
        total = sum(len(item.content.encode("utf-8")) for item in draft.files)
        if total > MAX_CODE_TOTAL_BYTES:
            raise ValueError("generated code exceeds the bounded total size")

    def _write_gap(
        self, task: TaskEnvelope, reason: str, *, details: dict[str, Any] | None = None
    ) -> ArtifactRef:
        return self.artifacts.write_agent_json(
            self.name,
            scoped_path(
                "code",
                task_project_slug(task),
                "manifests",
                f"{file_token(task.task_id)}.gap.json",
            ),
            {
                "schema_version": "1.0",
                "run_id": task.run_id,
                "task_id": task.task_id,
                "project_slug": task_project_slug(task),
                "status": "blocked",
                "reason": reason,
                "details": details or {},
            },
        ).model_copy(update={"kind": "code_generation_gap"})

    @staticmethod
    def _safe_contract_errors(exc: Exception) -> list[dict[str, Any]]:
        """Return bounded diagnostics without echoing model-produced source bodies."""

        if isinstance(exc, ValidationError):
            return [
                {
                    "location": [str(part) for part in error.get("loc", ())],
                    "type": str(error.get("type", "validation_error"))[:120],
                    "message": str(error.get("msg", "invalid value"))[:500],
                }
                for error in exc.errors(include_input=False)[:30]
            ]
        return [
            {
                "location": [],
                "type": type(exc).__name__,
                "message": str(exc)[:500],
            }
        ]

    def _generate_with_contract_repair(
        self,
        task: TaskEnvelope,
        design: ResearchDesign,
        data: DatasetManifest,
        vendored_sources: list[VendoredSourceRecord],
    ) -> tuple[CodeGenerationDraft, list[dict[str, Any]]]:
        """Give the provider one bounded chance to repair a schema-invalid response."""

        request = self._context(task, design, data, vendored_sources)
        attempt_errors: list[dict[str, Any]] = []
        for attempt in range(1, MAX_GENERATION_ATTEMPTS + 1):
            response = self.model.complete_json(
                PACKAGE.system_prompt,
                json.dumps(request, ensure_ascii=False),
            )
            try:
                draft = CodeGenerationDraft.model_validate(response)
                self._validate_traceability(draft, design)
                return draft, attempt_errors
            except (ValidationError, ValueError, TypeError) as exc:
                safe_errors = self._safe_contract_errors(exc)
                attempt_errors.append(
                    {
                        "attempt": attempt,
                        "errors": safe_errors,
                    }
                )
                if attempt >= MAX_GENERATION_ATTEMPTS:
                    raise CodeGenerationContractError(
                        type(exc).__name__,
                        attempt_errors,
                    ) from exc
                request = {
                    **self._context(task, design, data, vendored_sources),
                    "contract_repair": {
                        "attempt": attempt + 1,
                        "instruction": (
                            "Return a fresh complete CodeGenerationDraft that fixes every "
                            "listed validation error. Do not return a patch, prose, markdown, "
                            "or omit unchanged files. All original scientific and safety "
                            "requirements still apply."
                        ),
                        "validation_errors": safe_errors,
                    },
                }
        raise RuntimeError("unreachable generation repair state")

    def run(self, task: TaskEnvelope) -> AgentResult:
        if task.agent != self.name:
            raise ValueError("task was routed to the wrong agent")
        design, design_ref, data, data_ref = self._read_inputs(task)
        if (
            design is None
            or design_ref is None
            or data is None
            or data_ref is None
            or data.downloaded_count == 0
            or data.materialized_count != data.downloaded_count
            or data.server_script_count != 0
        ):
            reason = (
                "Approved ResearchDesign and fully materialized DatasetManifest inputs "
                "are required."
            )
            gap = self._write_gap(task, reason)
            return AgentResult(
                run_id=task.run_id,
                task_id=task.task_id,
                agent=self.name,
                status="failed",
                summary=reason,
                artifacts=[gap],
                issues=[
                    {
                        "severity": "blocking",
                        "code": "CODE_INPUT_CONTRACT_MISSING",
                        "message": reason,
                        "root_cause_key": "G4:CODE_INPUT_CONTRACT_MISSING:inputs",
                    }
                ],
                route_hint=AgentName.DOWNLOAD_DATA,
                needs_human=False,
            )
        vendored_sources: list[VendoredSourceRecord] = []
        try:
            for module in design.reusable_code_modules:
                if module.reuse_mode == "integrate":
                    vendored_sources.append(
                        self.source_vendor.vendor(module, project_slug=task_project_slug(task))
                    )
        except (SourceVendorError, StoragePolicyError, ValueError) as exc:
            reason = f"Local GitHub source materialization failed: {type(exc).__name__}."
            gap = self._write_gap(task, reason)
            return AgentResult(
                run_id=task.run_id,
                task_id=task.task_id,
                agent=self.name,
                status="failed",
                summary=reason,
                artifacts=[gap],
                issues=[
                    {
                        "severity": "blocking",
                        "code": "CODE_SOURCE_VENDOR_FAILED",
                        "message": reason,
                        "root_cause_key": "G4:CODE_SOURCE_VENDOR_FAILED:third_party",
                    }
                ],
                route_hint=AgentName.CODE_MAKE,
                needs_human=False,
            )

        provider_fallback_used = False
        contract_errors: list[dict[str, Any]] = []
        revision_baseline_task_id: str | None = None
        try:
            if self._prior_provider_failure(task) and supports_assaycal_scaffold(design):
                provider_fallback_used = True
                draft = build_assaycal_scaffold(design)
                draft = draft.model_copy(
                    update={
                        "known_gaps": [
                            *draft.known_gaps,
                            (
                                "A prior DeepSeek full-project generation request failed; "
                                "this retry used the trusted offline minimum scaffold and "
                                "requires Code Review."
                            ),
                        ]
                    }
                )
                self._validate_traceability(draft, design)
            else:
                revision_baseline = self._revision_baseline(task, design)
                if revision_baseline is None:
                    draft, contract_errors = self._generate_with_contract_repair(
                        task,
                        design,
                        data,
                        vendored_sources,
                    )
                else:
                    baseline_task_id, baseline = revision_baseline
                    revision_baseline_task_id = baseline_task_id
                    draft, contract_errors = self._generate_patch_with_contract_repair(
                        task,
                        design,
                        data,
                        vendored_sources,
                        baseline_task_id,
                        baseline,
                    )
        except ProviderRequestError as exc:
            if revision_baseline_task_id is not None:
                reason = "Code revision provider failed before returning a complete patch."
                gap = self._write_gap(
                    task,
                    reason,
                    details={
                        "baseline_task_id": revision_baseline_task_id,
                        "generation_attempt_count": 1,
                        "contract_errors": [
                            {
                                "attempt": 1,
                                "errors": self._safe_contract_errors(exc),
                            }
                        ],
                    },
                )
                return AgentResult(
                    run_id=task.run_id,
                    task_id=task.task_id,
                    agent=self.name,
                    status="failed",
                    summary=reason,
                    artifacts=[gap],
                    issues=[
                        {
                            "severity": "blocking",
                            "code": "CODE_REVISION_PROVIDER_FAILED",
                            "message": reason,
                            "root_cause_key": "G4:CODE_REVISION_PROVIDER_FAILED:output",
                        }
                    ],
                    route_hint=AgentName.CODE_MAKE,
                    needs_human=False,
                )
            if not supports_assaycal_scaffold(design):
                reason = (
                    "Code generation provider failed and no domain-matched trusted offline "
                    "scaffold is available; refusing to substitute a DTA implementation."
                )
                gap = self._write_gap(
                    task,
                    reason,
                    details={
                        "generation_attempt_count": 1,
                        "contract_errors": [
                            {
                                "attempt": 1,
                                "errors": self._safe_contract_errors(exc),
                            }
                        ],
                    },
                )
                return AgentResult(
                    run_id=task.run_id,
                    task_id=task.task_id,
                    agent=self.name,
                    status="failed",
                    summary=reason,
                    artifacts=[gap],
                    issues=[
                        {
                            "severity": "blocking",
                            "code": "CODE_PROVIDER_FAILED_NO_DOMAIN_FALLBACK",
                            "message": reason,
                            "root_cause_key": "G4:CODE_PROVIDER_FAILED_NO_DOMAIN_FALLBACK:output",
                        }
                    ],
                    route_hint=AgentName.CODE_MAKE,
                    needs_human=False,
                )
            provider_fallback_used = True
            draft = build_assaycal_scaffold(design)
            draft = draft.model_copy(
                update={
                    "known_gaps": [
                        *draft.known_gaps,
                        (
                            "DeepSeek full-project generation was unavailable; this revision "
                            "uses the trusted offline minimum scaffold and requires Code Review."
                        ),
                    ]
                }
            )
            self._validate_traceability(draft, design)
        except Exception as exc:
            failure_type = type(exc).__name__
            if isinstance(exc, CodeGenerationContractError):
                failure_type = exc.cause_type
                contract_errors = exc.attempt_errors
            reason = f"Code generation failed its bounded contract: {failure_type}."
            if not contract_errors:
                contract_errors = [
                    {
                        "attempt": 1,
                        "errors": self._safe_contract_errors(exc),
                    }
                ]
            gap = self._write_gap(
                task,
                reason,
                details={
                    "baseline_task_id": revision_baseline_task_id,
                    "generation_attempt_count": len(contract_errors),
                    "contract_errors": contract_errors,
                },
            )
            return AgentResult(
                run_id=task.run_id,
                task_id=task.task_id,
                agent=self.name,
                status="failed",
                summary=reason,
                artifacts=[gap],
                issues=[
                    {
                        "severity": "blocking",
                        "code": "CODE_GENERATION_SCHEMA_INVALID",
                        "message": reason,
                        "root_cause_key": "G4:CODE_GENERATION_SCHEMA_INVALID:output",
                    }
                ],
                route_hint=AgentName.CODE_MAKE,
                needs_human=False,
            )

        records: list[GeneratedCodeFileRecord] = []
        implementation_root = scoped_path(
            "code",
            task_project_slug(task),
            "implementations",
            file_token(task.task_id),
        )
        for item in draft.files:
            payload = item.content.encode("utf-8")
            artifact = self.artifacts.write_agent_bytes(
                self.name,
                f"{implementation_root}/{item.path}",
                payload,
                max_bytes=MAX_CODE_FILE_BYTES,
                kind="generated_code_file",
            )
            records.append(
                GeneratedCodeFileRecord(
                    path=item.path,
                    purpose=item.purpose,
                    language=item.language,
                    size_bytes=len(payload),
                    sha256=artifact.sha256,
                    artifact=artifact,
                    design_component_ids=item.design_component_ids,
                    dataset_keys=item.dataset_keys,
                    experiment_ids=item.experiment_ids,
                    source_module_ids=item.source_module_ids,
                )
            )
        try:
            local_validations = run_local_validations(
                (self.artifacts.root / implementation_root).resolve(),
                draft.package_name,
            )
        except (LocalValidationError, OSError, ValueError) as exc:
            reason = f"Generated code failed local execution preflight: {type(exc).__name__}."
            gap = self._write_gap(
                task,
                reason,
                details={
                    "baseline_task_id": revision_baseline_task_id,
                    "preflight_error": str(exc)[:1000],
                },
            )
            return AgentResult(
                run_id=task.run_id,
                task_id=task.task_id,
                agent=self.name,
                status="failed",
                summary=reason,
                artifacts=[gap],
                issues=[
                    {
                        "severity": "blocking",
                        "code": "CODE_LOCAL_PREFLIGHT_FAILED",
                        "message": reason,
                        "root_cause_key": "G4:CODE_LOCAL_PREFLIGHT_FAILED:execution",
                    }
                ],
                route_hint=AgentName.CODE_MAKE,
                needs_human=False,
            )
        failed_validations = [item for item in local_validations if item.status != "passed"]
        if failed_validations:
            reason = "Generated code did not pass the fixed local demo and unit-test gate."
            gap = self._write_gap(
                task,
                reason,
                details={
                    "baseline_task_id": revision_baseline_task_id,
                    "validations": [item.model_dump(mode="json") for item in local_validations]
                },
            )
            return AgentResult(
                run_id=task.run_id,
                task_id=task.task_id,
                agent=self.name,
                status="failed",
                summary=reason,
                artifacts=[gap],
                issues=[
                    {
                        "severity": "blocking",
                        "code": "CODE_LOCAL_VALIDATION_FAILED",
                        "message": reason,
                        "root_cause_key": "G4:CODE_LOCAL_VALIDATION_FAILED:execution",
                    }
                ],
                route_hint=AgentName.CODE_MAKE,
                needs_human=False,
            )
        manifest = CodeImplementationManifest(
            run_id=task.run_id,
            task_id=task.task_id,
            project_slug=task_project_slug(task),
            package_name=draft.package_name,
            implementation_root=implementation_root,
            research_design_uri=design_ref.uri,
            data_manifest_uri=data_ref.uri,
            architecture_summary=draft.architecture_summary,
            files=records,
            total_bytes=sum(item.size_bytes for item in records),
            entrypoints=draft.entrypoints,
            dependencies=draft.dependencies,
            validation_commands=draft.validation_commands,
            traceability=draft.traceability,
            vendored_sources=vendored_sources,
            local_validations=local_validations,
            known_gaps=draft.known_gaps,
        )
        manifest_ref = self.artifacts.write_agent_json(
            self.name,
            scoped_path(
                "code",
                task_project_slug(task),
                "manifests",
                f"{file_token(task.task_id)}.json",
            ),
            manifest.model_dump(mode="json"),
        ).model_copy(update={"kind": "code_implementation_manifest"})
        return AgentResult(
            run_id=task.run_id,
            task_id=task.task_id,
            agent=self.name,
            status="succeeded",
            summary=(
                f"已在 {manifest.implementation_root}/ 生成并隔离 {len(records)} 个文件，"
                f"共 {manifest.total_bytes:,} bytes；本地 Demo 与单元测试均通过，"
                f"固化 GitHub 源码 {len(vendored_sources)} 份，等待独立 Code Review。"
            ),
            artifacts=[manifest_ref],
            evidence=[
                {
                    "check": "project-scoped-code-generation",
                    "file_count": len(records),
                    "total_bytes": manifest.total_bytes,
                    "dataset_count": data.downloaded_count,
                    "vendored_source_count": len(vendored_sources),
                    "provider_fallback_used": provider_fallback_used,
                    "validation_statuses": {
                        item.validation_id: item.status for item in local_validations
                    },
                    "execution_scope": "fixed local minimal demo and unit tests only",
                }
            ],
            needs_human=False,
        )
