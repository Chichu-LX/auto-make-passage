from __future__ import annotations

import csv
import hashlib
import io
import math
import re
import statistics
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterable, TextIO

import py7zr

from .schemas import (
    DatasetLabelStatistics,
    DatasetProfile,
    DatasetRepresentationProfile,
)


MAX_UNIQUE_VALUES = 1_000_000
MAX_LABEL_SAMPLE = 250_000
MISSING_VALUES = {"", "na", "nan", "null", "none", "n/a"}
SAFE_TABLE_SUFFIXES = {".csv", ".tsv", ".txt"}


def canonicalize_headerless_dta_text(payload: bytes) -> bytes | None:
    """Convert the common five-column, headerless DTA text format to CSV.

    Several archival DTA deposits publish rows as whitespace-delimited
    ``drug_id target_id SMILES sequence affinity``.  Keeping the downloaded
    bytes as the raw artifact while materializing a named CSV makes the data
    usable by downstream code without silently guessing arbitrary schemas.
    The conversion is deliberately fail-closed: every non-empty row must have
    exactly five fields and a finite numeric affinity.
    """

    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        return None
    rows: list[list[str]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            return None
        try:
            affinity = float(fields[4])
        except ValueError:
            return None
        if not math.isfinite(affinity):
            return None
        rows.append(fields)
    if not rows:
        return None
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(
        ["drug_id", "target_id", "compound_iso_smiles", "target_sequence", "affinity"]
    )
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.strip().lower()).strip()


def _find_field(columns: list[str], candidates: Iterable[str]) -> str | None:
    normalized = {_normalize_header(column): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    for candidate in candidates:
        for key, original in normalized.items():
            if candidate in key:
                return original
    return None


def _parse_number(value: str) -> float | None:
    candidate = value.strip().replace(",", "")
    candidate = re.sub(r"^[<>=~]+", "", candidate).strip()
    if not candidate or candidate.lower() in MISSING_VALUES:
        return None
    try:
        number = float(candidate)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _normalize_sequence(value: str) -> str:
    return re.sub(r"\s+", "", value).upper()


def _normalize_smiles_text(value: str) -> str:
    """Normalize transport whitespace without pretending to canonicalize chemistry."""

    return re.sub(r"\s+", "", value)


def _profile_delimited(
    handle: TextIO,
    *,
    source_table: str,
    delimiter: str,
    compressed_size: int,
    uncompressed_size: int,
    archive_member_count: int,
) -> DatasetProfile:
    reader = csv.DictReader(handle, delimiter=delimiter)
    columns = [str(item) for item in (reader.fieldnames or [])]
    compound_field = _find_field(
        columns,
        (
            "compound iso smiles",
            "ligand smiles",
            "smiles",
            "canonical smiles",
            "monomerid",
        ),
    )
    target_identifier_field = _find_field(
        columns,
        (
            "target key",
            "target id",
            "target name",
            "uniprot swissprot primary id of target chain",
        ),
    )
    target_sequence_field = _find_field(
        columns,
        (
            "target sequence",
            "bindingdb target chain sequence",
            "protein sequence",
            "amino acid sequence",
        ),
    )
    target_field = target_identifier_field or target_sequence_field
    primary_affinity = _find_field(
        columns,
        (
            "affinity",
            "kiba score",
            "label",
            "outcome",
            "response",
            "target value",
            "class label",
            "endpoint",
            "score",
        ),
    )
    label_fields = (
        [primary_affinity]
        if primary_affinity
        else [
            field
            for name in ("kd nm", "ki nm", "ic50 nm", "ec50 nm")
            if (field := _find_field(columns, (name,))) is not None
        ]
    )
    representation_specs: list[tuple[str, str, str]] = []
    for role, identifier_candidates, representation_candidates in (
        (
            "target",
            ("target id", "target key", "target name", "protein id"),
            ("target sequence", "protein sequence", "amino acid sequence"),
        ),
        (
            "subject",
            ("patient id", "subject id", "participant id", "speaker id"),
            (
                "feature representation",
                "feature vector",
                "embedding",
                "image path",
                "image",
                "signal",
                "transcript",
                "document",
            ),
        ),
        (
            "entity",
            ("entity id", "sample id", "item id", "record id"),
            (
                "representation",
                "feature vector",
                "embedding",
                "input text",
                "text",
                "image",
            ),
        ),
    ):
        identifier_field = _find_field(columns, identifier_candidates)
        representation_field = _find_field(columns, representation_candidates)
        if identifier_field and representation_field:
            spec = (role, identifier_field, representation_field)
            if spec not in representation_specs:
                representation_specs.append(spec)

    normalized_label_fields = {_normalize_header(field) for field in label_fields}
    generic_feature_fields = [
        field
        for field in columns
        if _normalize_header(field) not in normalized_label_fields
        and not re.search(
            r"(?:^| )(?:id|key|index|row|uuid|name)(?:$| )",
            _normalize_header(field),
        )
    ][:40]

    compounds: set[str] = set()
    targets: set[str] = set()
    target_identifiers: set[str] = set()
    target_sequences: set[str] = set()
    pairs: set[tuple[str, str]] = set()
    representation_identifiers: dict[
        tuple[str, str, str], dict[str, set[str]]
    ] = {spec: {} for spec in representation_specs}
    representation_unique_ids: dict[tuple[str, str, str], set[str]] = {
        spec: set() for spec in representation_specs
    }
    representation_unique_values: dict[tuple[str, str, str], set[str]] = {
        spec: set() for spec in representation_specs
    }
    model_input_labels: dict[str, list[float | int]] = {}
    representation_audit_truncated = False
    unique_truncated = False
    label_accumulators = {
        field: {
            "values": [],
            "count": 0,
            "mean": 0.0,
            "m2": 0.0,
            "minimum": None,
            "maximum": None,
        }
        for field in label_fields
    }
    field_observed = {
        field: 0
        for field in [
            compound_field,
            target_identifier_field,
            target_sequence_field,
            *label_fields,
        ]
        if field is not None
    }
    rows = 0
    missing_cells = 0
    for record in reader:
        rows += 1
        missing_cells += sum(
            str(record.get(column) or "").strip().lower() in MISSING_VALUES
            for column in columns
        )
        compound = str(record.get(compound_field) or "").strip() if compound_field else ""
        target = str(record.get(target_field) or "").strip() if target_field else ""
        target_identifier = (
            str(record.get(target_identifier_field) or "").strip()
            if target_identifier_field
            else ""
        )
        target_sequence = (
            _normalize_sequence(str(record.get(target_sequence_field) or ""))
            if target_sequence_field
            else ""
        )
        for spec in representation_specs:
            _, identifier_field, representation_field = spec
            identifier = str(record.get(identifier_field) or "").strip()
            representation = str(record.get(representation_field) or "").strip()
            if not identifier or not representation:
                continue
            normalized_representation = re.sub(r"\s+", "", representation).casefold()
            if len(representation_unique_ids[spec]) < MAX_UNIQUE_VALUES:
                representation_unique_ids[spec].add(identifier)
            elif identifier not in representation_unique_ids[spec]:
                representation_audit_truncated = True
            if len(representation_unique_values[spec]) < MAX_UNIQUE_VALUES:
                representation_unique_values[spec].add(normalized_representation)
            elif normalized_representation not in representation_unique_values[spec]:
                representation_audit_truncated = True
            identifiers = representation_identifiers[spec].get(normalized_representation)
            if identifiers is not None:
                if len(identifiers) < MAX_UNIQUE_VALUES:
                    identifiers.add(identifier)
                elif identifier not in identifiers:
                    representation_audit_truncated = True
            elif len(representation_identifiers[spec]) < MAX_UNIQUE_VALUES:
                representation_identifiers[spec][normalized_representation] = {identifier}
            else:
                representation_audit_truncated = True
        for field in field_observed:
            if str(record.get(field) or "").strip().lower() not in MISSING_VALUES:
                field_observed[field] += 1
        if len(compounds) < MAX_UNIQUE_VALUES and compound:
            compounds.add(compound)
        elif compound and compound not in compounds:
            unique_truncated = True
        if len(targets) < MAX_UNIQUE_VALUES and target:
            targets.add(target)
        elif target and target not in targets:
            unique_truncated = True
        if compound and target:
            pair = (compound, target)
            if len(pairs) < MAX_UNIQUE_VALUES:
                pairs.add(pair)
            elif pair not in pairs:
                unique_truncated = True
        if target_identifier:
            if len(target_identifiers) < MAX_UNIQUE_VALUES:
                target_identifiers.add(target_identifier)
            elif target_identifier not in target_identifiers:
                representation_audit_truncated = True
        if target_sequence:
            if len(target_sequences) < MAX_UNIQUE_VALUES:
                target_sequences.add(target_sequence)
            elif target_sequence not in target_sequences:
                representation_audit_truncated = True
        for label_field, accumulator in label_accumulators.items():
            value = _parse_number(str(record.get(label_field) or ""))
            if value is not None:
                accumulator["count"] += 1
                delta = value - accumulator["mean"]
                accumulator["mean"] += delta / accumulator["count"]
                accumulator["m2"] += delta * (value - accumulator["mean"])
                accumulator["minimum"] = (
                    value
                    if accumulator["minimum"] is None
                    else min(accumulator["minimum"], value)
                )
                accumulator["maximum"] = (
                    value
                    if accumulator["maximum"] is None
                    else max(accumulator["maximum"], value)
                )
                if len(accumulator["values"]) < MAX_LABEL_SAMPLE:
                    accumulator["values"].append(value)
                if label_field == label_fields[0]:
                    if compound and target_sequence:
                        input_values = (
                            _normalize_smiles_text(compound),
                            target_sequence,
                        )
                    else:
                        input_values = tuple(
                            str(record.get(field) or "").strip()
                            for field in generic_feature_fields
                        )
                    if not input_values or not any(input_values):
                        continue
                    input_key = hashlib.sha256(
                        "\x1f".join(input_values).encode("utf-8")
                    ).hexdigest()
                    group = model_input_labels.get(input_key)
                    if group is not None:
                        group[0] = min(float(group[0]), value)
                        group[1] = max(float(group[1]), value)
                        group[2] = int(group[2]) + 1
                    elif len(model_input_labels) < MAX_UNIQUE_VALUES:
                        model_input_labels[input_key] = [value, value, 1]
                    else:
                        representation_audit_truncated = True

    label_statistics: list[DatasetLabelStatistics] = []
    for label_field, accumulator in label_accumulators.items():
        quantiles: dict[str, float] = {}
        if accumulator["values"]:
            ordered = sorted(accumulator["values"])
            for name, fraction in (("q05", 0.05), ("q25", 0.25), ("q50", 0.50), ("q75", 0.75), ("q95", 0.95)):
                index = min(len(ordered) - 1, round((len(ordered) - 1) * fraction))
                quantiles[name] = ordered[index]
        count = int(accumulator["count"])
        label_statistics.append(
            DatasetLabelStatistics(
                field=label_field,
                observed_count=count,
                minimum=accumulator["minimum"],
                maximum=accumulator["maximum"],
                mean=accumulator["mean"] if count else None,
                standard_deviation=(
                    math.sqrt(accumulator["m2"] / (count - 1)) if count > 1 else None
                ),
                quantiles=quantiles,
            )
        )
    notes: list[str] = []
    if unique_truncated:
        notes.append("Unique entity and pair counts are lower bounds because the safety cap was reached.")
    if any(
        int(item["count"]) > len(item["values"])
        for item in label_accumulators.values()
    ):
        notes.append("Label quantiles use the first bounded numeric sample; mean and standard deviation use all rows.")
    if compound_field is None:
        notes.append("No compound identifier field was recognized automatically.")
    if target_field is None:
        notes.append("No target identifier field was recognized automatically.")
    if not label_fields:
        notes.append("No numeric outcome/label field was recognized automatically.")
    entity_profiles = [
        DatasetRepresentationProfile(
            entity_role=role,
            identifier_field=identifier_field,
            representation_field=representation_field,
            unique_identifier_count=len(representation_unique_ids[spec]),
            unique_representation_count=len(representation_unique_values[spec]),
            shared_representation_group_count=sum(
                len(identifiers) > 1
                for identifiers in representation_identifiers[spec].values()
            ),
        )
        for spec in representation_specs
        for role, identifier_field, representation_field in (spec,)
    ]
    repeated_model_inputs = sum(
        int(group[2]) > 1 for group in model_input_labels.values()
    )
    conflicting_groups = [
        group
        for group in model_input_labels.values()
        if int(group[2]) > 1
        and not math.isclose(float(group[0]), float(group[1]), rel_tol=1e-9, abs_tol=1e-12)
    ]
    conflicting_rows = sum(int(group[2]) for group in conflicting_groups)
    for profile in entity_profiles:
        if profile.shared_representation_group_count:
            notes.append(
                f"{profile.shared_representation_group_count} exact {profile.entity_role} representation groups map to multiple identifiers; identifier-disjoint splits are not representation-disjoint by construction."
            )
    if conflicting_groups:
        notes.append(
            f"{len(conflicting_groups)} repeated exact model-input groups contain conflicting numeric outcomes across {conflicting_rows} rows."
        )
    if representation_audit_truncated:
        notes.append(
            "Representation-level sequence/input audits are lower bounds because the safety cap was reached."
        )
    total_cells = rows * len(columns)
    return DatasetProfile(
        profile_status="partial" if notes else "complete",
        source_table=source_table,
        archive_member_count=archive_member_count,
        compressed_size_bytes=compressed_size,
        uncompressed_size_bytes=uncompressed_size,
        row_count=rows,
        column_count=len(columns),
        columns=columns[:300],
        unique_compounds=len(compounds) if compound_field else None,
        unique_targets=len(targets) if target_field else None,
        unique_pairs=len(pairs) if compound_field and target_field else None,
        entity_representation_profiles=entity_profiles,
        repeated_exact_input_group_count=repeated_model_inputs,
        conflicting_exact_input_label_group_count=len(conflicting_groups),
        conflicting_exact_input_label_row_count=conflicting_rows,
        representation_audit_complete=bool(
            label_fields
            and model_input_labels
            and not representation_audit_truncated
        ),
        missing_cell_count=missing_cells,
        missing_fraction=(missing_cells / total_cells if total_cells else 0.0),
        key_field_coverage={
            field: (count / rows if rows else 0.0)
            for field, count in field_observed.items()
        },
        label_statistics=label_statistics,
        notes=notes,
    )


def _delimiter_for(name: str) -> str:
    return "\t" if PurePosixPath(name).suffix.lower() in {".tsv", ".txt"} else ","


def _profile_binary_table(
    handle: BinaryIO,
    *,
    name: str,
    compressed_size: int,
    uncompressed_size: int,
    archive_member_count: int,
) -> DatasetProfile:
    text = io.TextIOWrapper(handle, encoding="utf-8-sig", errors="replace", newline="")
    try:
        return _profile_delimited(
            text,
            source_table=name,
            delimiter=_delimiter_for(name),
            compressed_size=compressed_size,
            uncompressed_size=uncompressed_size,
            archive_member_count=archive_member_count,
        )
    finally:
        text.detach()


def profile_dataset_payload(payload: bytes, dataset_format: str) -> DatasetProfile:
    compressed_size = len(payload)
    if dataset_format in {"csv", "tsv", "txt"}:
        return _profile_binary_table(
            io.BytesIO(payload),
            name=f"dataset.{dataset_format}",
            compressed_size=compressed_size,
            uncompressed_size=compressed_size,
            archive_member_count=1,
        )
    if dataset_format == "zip":
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = [
                item
                for item in archive.infolist()
                if not item.is_dir() and PurePosixPath(item.filename).suffix.lower() in SAFE_TABLE_SUFFIXES
            ]
            if not members:
                raise ValueError("validated ZIP contains no profileable delimited table")
            selected = max(members, key=lambda item: item.file_size)
            with archive.open(selected) as handle:
                return _profile_binary_table(
                    handle,
                    name=selected.filename,
                    compressed_size=compressed_size,
                    uncompressed_size=sum(item.file_size for item in archive.infolist()),
                    archive_member_count=len(archive.infolist()),
                )
    if dataset_format == "7z":
        with tempfile.TemporaryDirectory(prefix="assaycal-profile-") as directory:
            with py7zr.SevenZipFile(io.BytesIO(payload), mode="r") as archive:
                members = [
                    item
                    for item in archive.list()
                    if not item.is_directory
                    and PurePosixPath(item.filename).suffix.lower() in SAFE_TABLE_SUFFIXES
                ]
                if not members:
                    raise ValueError("validated 7z contains no profileable delimited table")
                selected = max(members, key=lambda item: int(item.uncompressed or 0))
                total_uncompressed = sum(int(item.uncompressed or 0) for item in archive.list())
                archive.extract(path=directory, targets=[selected.filename])
            target = Path(directory, selected.filename)
            with target.open("rb") as handle:
                return _profile_binary_table(
                    handle,
                    name=selected.filename,
                    compressed_size=compressed_size,
                    uncompressed_size=total_uncompressed,
                    archive_member_count=len(members),
                )
    raise ValueError(f"profiling is not implemented for dataset format: {dataset_format}")
