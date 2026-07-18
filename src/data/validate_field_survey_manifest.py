from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PERSONAL_FIELDS = {
    "_id",
    "_uuid",
    "_submission_time",
    "start",
    "end",
    "Name_of_the_Surveyer",
    "Team_Number",
    "College_name",
    "_Farmer_Name",
}

MANIFEST_SCHEMA_VERSION = "1.0"
TRAINING_MANIFEST_SCHEMA_VERSION = "1.0"
TRAINING_MANIFEST_TYPE = "field_survey_training"
FIELD_SURVEY_DATASET_SOURCE = "field_survey"

TRAINING_TOP_LEVEL_ALLOWED_FIELDS = {
    "schema_version",
    "manifest_type",
    "dataset_name",
    "dataset_source",
    "review_manifest_sha256",
    "statistics",
    "class_names",
    "records",
}
TRAINING_TOP_LEVEL_REQUIRED_FIELDS = TRAINING_TOP_LEVEL_ALLOWED_FIELDS
TRAINING_STATISTICS_ALLOWED_FIELDS = {
    "total_source_records",
    "accepted_records",
    "replaced_records",
    "rejected_records",
    "pending_records",
    "exported_records",
    "excluded_invalid_records",
}
TRAINING_STATISTICS_REQUIRED_FIELDS = TRAINING_STATISTICS_ALLOWED_FIELDS
TRAINING_RECORD_ALLOWED_FIELDS = {
    "record_id",
    "dataset_source",
    "image_path",
    "image_sha256",
    "canonical_class",
    "review_status",
    "review_decision",
    "replacement_class",
}
TRAINING_RECORD_REQUIRED_FIELDS = {
    "record_id",
    "dataset_source",
    "image_path",
    "image_sha256",
    "canonical_class",
    "review_status",
    "review_decision",
}
TRAINING_REVIEW_DECISIONS = {"accept", "replace"}
TRAINING_REVIEW_STATUSES = {"validated"}
SENSITIVE_FIELD_KEYWORDS = {
    "address",
    "college_name",
    "coordinate",
    "coordinates",
    "email",
    "farmer_name",
    "lat",
    "latitude",
    "lon",
    "long",
    "longitude",
    "metadata",
    "phone",
    "raw_metadata",
    "source_metadata",
    "surveyor",
    "surveyor_name",
    "team_number",
}
FORBIDDEN_TRAINING_FIELD_KEYS = {
    re.sub(r"[^a-z0-9]+", "_", field.casefold()).strip("_")
    for field in PERSONAL_FIELDS
} | SENSITIVE_FIELD_KEYWORDS
WINDOWS_ABSOLUTE_PATH = re.compile(r"^[a-zA-Z]:[\\/]")
SHA256_HEX = re.compile(r"^[0-9a-fA-F]{64}$")


class PrivacyAuditError(Exception):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalised_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).casefold()).strip("_")


def _is_forbidden_training_field(key: Any) -> bool:
    normalised = _normalised_key(key)
    if normalised in FORBIDDEN_TRAINING_FIELD_KEYS:
        return True
    return normalised.endswith("_email") or normalised.endswith("_phone")


def _walk_json(value: Any, path: str = "$"):
    yield path, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from _walk_json(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_json(child, f"{path}[{index}]")


def _is_absolute_path_string(value: str) -> bool:
    text = value.strip()
    return (
        Path(text).is_absolute()
        or bool(WINDOWS_ABSOLUTE_PATH.match(text))
        or text.startswith("\\\\")
        or text.startswith("//")
        or "://" in text
    )


def _has_path_traversal(value: str) -> bool:
    parts = value.replace("\\", "/").split("/")
    return any(part == ".." for part in parts)


def _safe_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    return not _is_absolute_path_string(value) and not _has_path_traversal(value)


def _path_under_root(root: Path, relative_path: str) -> Path:
    resolved_root = root.resolve()
    candidate = (resolved_root / relative_path).resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise PrivacyAuditError(f"Training manifest path escapes root: {relative_path}")
    return candidate


def _field_failures(payload: Any) -> list[dict[str, Any]]:
    forbidden = sorted(
        {str(key) for _path, node in _walk_json(payload) if isinstance(node, dict) for key in node if _is_forbidden_training_field(key)}
    )
    if not forbidden:
        return []
    return [{
        "check": "no_forbidden_personal_fields",
        "message": f"Forbidden personal/source field names found: {', '.join(forbidden)}",
        "fields": forbidden,
    }]


def _string_path_failures(payload: Any) -> tuple[list[dict[str, Any]], int, int]:
    absolute_paths = 0
    traversal_paths = 0
    for _path, node in _walk_json(payload):
        if not isinstance(node, str) or not node.strip():
            continue
        if _is_absolute_path_string(node):
            absolute_paths += 1
        if _has_path_traversal(node):
            traversal_paths += 1

    failures = []
    if absolute_paths:
        failures.append({
            "check": "no_absolute_paths",
            "message": f"{absolute_paths} absolute path-like values found in training manifest",
        })
    if traversal_paths:
        failures.append({
            "check": "no_path_traversal",
            "message": f"{traversal_paths} path traversal values found in training manifest",
        })
    return failures, absolute_paths, traversal_paths


def audit_privacy(manifest_path: str | Path) -> dict[str, Any]:
    """Audit a field-survey manifest for privacy risks and training eligibility."""
    path = Path(manifest_path)
    payload = _load_json(path)
    records = payload.get("records", [])
    validation_events = payload.get("audit_history", [])

    personal_field_counts: dict[str, int] = defaultdict(int)
    personal_field_examples: dict[str, list[str]] = defaultdict(list)
    records_with_personal_metadata = 0
    absolute_image_paths = 0
    null_image_paths = 0
    missing_images = 0
    invalid_labels = 0
    eligible_count = 0
    pending_count = 0
    rejected_count = 0
    duplicate_hash_groups = 0
    duplicate_reference_groups = 0

    for record in records:
        metadata = record.get("metadata", {})
        has_personal = False
        for field in PERSONAL_FIELDS:
            value = metadata.get(field)
            if value is not None and str(value).strip():
                personal_field_counts[field] += 1
                if len(personal_field_examples[field]) < 3:
                    personal_field_examples[field].append(str(value))
                has_personal = True
        if has_personal:
            records_with_personal_metadata += 1

        image_path = record.get("image_path")
        if image_path is None or str(image_path).strip() == "":
            null_image_paths += 1
        elif Path(str(image_path)).is_absolute():
            absolute_image_paths += 1

        validation = record.get("validation", {})
        status = validation.get("status")
        if validation.get("eligible_for_training") is True:
            eligible_count += 1
        elif status == "pending" or not validation:
            pending_count += 1

    source_issues = payload.get("source_issues", {})
    duplicate_hash_groups = len(source_issues.get("duplicate_image_hashes", []))
    duplicate_reference_groups = len(source_issues.get("duplicate_image_reference_groups", []))

    stats = payload.get("statistics", {})
    missing_images = stats.get("missing_images", 0)
    invalid_labels = stats.get("invalid_labels", 0)

    report: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "audited_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "manifest_path": str(path.resolve()),
        "manifest_sha256": _sha256_file(path),
        "summary": {
            "total_records": len(records),
            "records_with_personal_metadata": records_with_personal_metadata,
            "absolute_image_paths": absolute_image_paths,
            "null_image_paths": null_image_paths,
            "missing_images": missing_images,
            "invalid_labels": invalid_labels,
            "eligible_for_training": eligible_count,
            "pending_review": pending_count,
            "duplicate_hash_groups": duplicate_hash_groups,
            "duplicate_reference_groups": duplicate_reference_groups,
        },
        "personal_fields": {
            "counts": dict(sorted(personal_field_counts.items())),
            "examples": dict(personal_field_examples),
        },
        "review_progress": {
            "total_groups": len(validation_events),
            "accepted": sum(1 for e in validation_events if e.get("action") == "accept"),
            "replaced": sum(1 for e in validation_events if e.get("action") == "replace"),
            "rejected": sum(1 for e in validation_events if e.get("action") == "reject"),
        },
        "passes": [],
        "failures": [],
    }

    if eligible_count > 0:
        report["failures"].append({
            "check": "no_eligible_records_without_review",
            "message": f"{eligible_count} records are marked eligible_for_training without human review",
        })
    else:
        report["passes"].append({
            "check": "no_eligible_records_without_review",
            "message": "No records have been approved for training without review",
        })

    if records_with_personal_metadata > 0:
        report["failures"].append({
            "check": "no_personal_metadata_in_training_manifests",
            "message": f"{records_with_personal_metadata} records contain personal metadata fields",
            "fields": list(personal_field_counts.keys()),
        })
    else:
        report["passes"].append({
            "check": "no_personal_metadata_in_training_manifests",
            "message": "No personal metadata fields found in records",
        })

    if absolute_image_paths > 0:
        report["failures"].append({
            "check": "portable_image_paths",
            "message": f"{absolute_image_paths} records use absolute image paths",
        })
    else:
        report["passes"].append({
            "check": "portable_image_paths",
            "message": "All image paths are relative or null",
        })

    if missing_images > 0:
        report["failures"].append({
            "check": "no_missing_images",
            "message": f"{missing_images} records reference missing images",
        })
    else:
        report["passes"].append({
            "check": "no_missing_images",
            "message": "No missing images referenced",
        })

    if duplicate_hash_groups > 0:
        report["failures"].append({
            "check": "no_duplicate_groups",
            "message": f"{duplicate_hash_groups} duplicate image hash groups exist",
        })
    else:
        report["passes"].append({
            "check": "no_duplicate_groups",
            "message": "No duplicate image hash groups",
        })

    report["overall_status"] = "pass" if not report["failures"] else "fail"
    return report


def audit_training_manifest(
    manifest_path: str | Path,
    path_root: str | Path | None = None,
) -> dict[str, Any]:
    """Audit a sanitized field-survey training manifest against the allowlisted schema."""
    path = Path(manifest_path)
    payload = _load_json(path)
    records = payload.get("records") if isinstance(payload, dict) else None
    statistics = payload.get("statistics") if isinstance(payload, dict) else None
    class_names = payload.get("class_names") if isinstance(payload, dict) else None

    failures: list[dict[str, Any]] = []
    passes: list[dict[str, Any]] = []
    duplicate_record_ids = 0
    invalid_records = 0
    missing_files = 0

    if not isinstance(payload, dict):
        failures.append({
            "check": "manifest_object",
            "message": "Training manifest must be a JSON object",
        })
        payload = {}
        records = []
        statistics = {}
        class_names = []

    unexpected_top = sorted(set(payload) - TRAINING_TOP_LEVEL_ALLOWED_FIELDS) if isinstance(payload, dict) else []
    missing_top = sorted(TRAINING_TOP_LEVEL_REQUIRED_FIELDS - set(payload)) if isinstance(payload, dict) else []
    if unexpected_top:
        failures.append({
            "check": "top_level_allowlist",
            "message": f"Unexpected top-level fields: {', '.join(unexpected_top)}",
            "fields": unexpected_top,
        })
    if missing_top:
        failures.append({
            "check": "top_level_required_fields",
            "message": f"Missing top-level fields: {', '.join(missing_top)}",
            "fields": missing_top,
        })

    if payload.get("schema_version") != TRAINING_MANIFEST_SCHEMA_VERSION:
        failures.append({
            "check": "schema_version",
            "message": f"schema_version must be {TRAINING_MANIFEST_SCHEMA_VERSION}",
        })
    if payload.get("manifest_type") != TRAINING_MANIFEST_TYPE:
        failures.append({
            "check": "manifest_type",
            "message": f"manifest_type must be {TRAINING_MANIFEST_TYPE}",
        })
    if payload.get("dataset_source") != FIELD_SURVEY_DATASET_SOURCE:
        failures.append({
            "check": "dataset_source",
            "message": f"dataset_source must be {FIELD_SURVEY_DATASET_SOURCE}",
        })

    if not isinstance(statistics, dict):
        failures.append({
            "check": "statistics_object",
            "message": "statistics must be an object",
        })
        statistics = {}
    else:
        unexpected_stats = sorted(set(statistics) - TRAINING_STATISTICS_ALLOWED_FIELDS)
        missing_stats = sorted(TRAINING_STATISTICS_REQUIRED_FIELDS - set(statistics))
        if unexpected_stats:
            failures.append({
                "check": "statistics_allowlist",
                "message": f"Unexpected statistics fields: {', '.join(unexpected_stats)}",
                "fields": unexpected_stats,
            })
        if missing_stats:
            failures.append({
                "check": "statistics_required_fields",
                "message": f"Missing statistics fields: {', '.join(missing_stats)}",
                "fields": missing_stats,
            })

    if not isinstance(class_names, list) or not all(isinstance(name, str) and name.strip() for name in class_names):
        failures.append({
            "check": "class_names",
            "message": "class_names must be a list of non-empty canonical class strings",
        })
        canonical_classes: set[str] = set()
    else:
        canonical_classes = set(class_names)
        if len(canonical_classes) != len(class_names):
            failures.append({
                "check": "duplicate_class_names",
                "message": "class_names contains duplicate canonical classes",
            })

    failures.extend(_field_failures(payload))
    path_failures, absolute_paths, traversal_paths = _string_path_failures(payload)
    failures.extend(path_failures)

    if not isinstance(records, list):
        failures.append({
            "check": "records_array",
            "message": "records must be an array",
        })
        records = []

    path_root_path = Path(path_root) if path_root is not None else None
    seen_ids: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            failures.append({
                "check": "record_object",
                "message": f"Record at index {index} must be an object",
            })
            invalid_records += 1
            continue

        unexpected_fields = sorted(set(record) - TRAINING_RECORD_ALLOWED_FIELDS)
        missing_fields = sorted(TRAINING_RECORD_REQUIRED_FIELDS - set(record))
        if unexpected_fields:
            failures.append({
                "check": "record_allowlist",
                "message": f"Record at index {index} has unexpected fields: {', '.join(unexpected_fields)}",
                "record_id": record.get("record_id"),
                "fields": unexpected_fields,
            })
            invalid_records += 1
        if missing_fields:
            failures.append({
                "check": "record_required_fields",
                "message": f"Record at index {index} is missing fields: {', '.join(missing_fields)}",
                "record_id": record.get("record_id"),
                "fields": missing_fields,
            })
            invalid_records += 1

        record_id = record.get("record_id")
        if not isinstance(record_id, str) or not record_id.strip():
            failures.append({
                "check": "stable_record_id",
                "message": f"Record at index {index} has an invalid record_id",
            })
            invalid_records += 1
        elif record_id in seen_ids:
            duplicate_record_ids += 1
            failures.append({
                "check": "unique_record_ids",
                "message": f"Duplicate record_id found: {record_id}",
                "record_id": record_id,
            })
            invalid_records += 1
        else:
            seen_ids.add(record_id)

        if record.get("dataset_source") != FIELD_SURVEY_DATASET_SOURCE:
            failures.append({
                "check": "record_dataset_source",
                "message": f"Record {record_id or index} has an invalid dataset_source",
                "record_id": record_id,
            })
            invalid_records += 1

        image_path = record.get("image_path")
        if not _safe_relative_path(image_path):
            failures.append({
                "check": "safe_relative_image_path",
                "message": f"Record {record_id or index} has an unsafe image_path",
                "record_id": record_id,
            })
            invalid_records += 1
        elif path_root_path is not None:
            candidate = _path_under_root(path_root_path, image_path)
            if not candidate.is_file():
                missing_files += 1
                failures.append({
                    "check": "existing_image_file",
                    "message": f"Record {record_id or index} references a missing image",
                    "record_id": record_id,
                })
                invalid_records += 1

        image_sha256 = record.get("image_sha256")
        if not isinstance(image_sha256, str) or not SHA256_HEX.match(image_sha256):
            failures.append({
                "check": "image_hash_required",
                "message": f"Record {record_id or index} is missing a valid image_sha256",
                "record_id": record_id,
            })
            invalid_records += 1

        decision = record.get("review_decision")
        status = record.get("review_status")
        if decision not in TRAINING_REVIEW_DECISIONS:
            failures.append({
                "check": "reviewed_records_only",
                "message": f"Record {record_id or index} is not accepted or replaced",
                "record_id": record_id,
            })
            invalid_records += 1
        if status not in TRAINING_REVIEW_STATUSES:
            failures.append({
                "check": "validated_records_only",
                "message": f"Record {record_id or index} is not in validated review status",
                "record_id": record_id,
            })
            invalid_records += 1

        canonical_class = record.get("canonical_class")
        if not isinstance(canonical_class, str) or canonical_class not in canonical_classes:
            failures.append({
                "check": "canonical_class_names_only",
                "message": f"Record {record_id or index} has a non-canonical class",
                "record_id": record_id,
            })
            invalid_records += 1

        replacement = record.get("replacement_class")
        if decision == "replace":
            if replacement != canonical_class:
                failures.append({
                    "check": "replacement_class_matches_final_class",
                    "message": f"Record {record_id or index} has an invalid replacement_class",
                    "record_id": record_id,
                })
                invalid_records += 1
        elif replacement not in {None, ""}:
            failures.append({
                "check": "replacement_class_only_for_replacements",
                "message": f"Record {record_id or index} has replacement_class without a replacement decision",
                "record_id": record_id,
            })
            invalid_records += 1

    if not any(f["check"] == "no_forbidden_personal_fields" for f in failures):
        passes.append({
            "check": "no_forbidden_personal_fields",
            "message": "No forbidden personal/source field names found",
        })
    if absolute_paths == 0:
        passes.append({
            "check": "no_absolute_paths",
            "message": "No absolute path-like values found",
        })
    if traversal_paths == 0:
        passes.append({
            "check": "no_path_traversal",
            "message": "No path traversal values found",
        })
    if duplicate_record_ids == 0:
        passes.append({
            "check": "unique_record_ids",
            "message": "No duplicate record IDs found",
        })
    if not any(f["check"] in {"reviewed_records_only", "validated_records_only"} for f in failures):
        passes.append({
            "check": "reviewed_records_only",
            "message": "All exported records are accepted or replaced human-review records",
        })
    if not any(f["check"] == "canonical_class_names_only" for f in failures):
        passes.append({
            "check": "canonical_class_names_only",
            "message": "All exported records use canonical class names",
        })
    if path_root_path is not None and missing_files == 0:
        passes.append({
            "check": "existing_image_file",
            "message": "All exported image files exist under the configured root",
        })

    report: dict[str, Any] = {
        "schema_version": TRAINING_MANIFEST_SCHEMA_VERSION,
        "audited_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "manifest_path": str(path.resolve()),
        "manifest_sha256": _sha256_file(path),
        "summary": {
            "total_records": len(records),
            "absolute_paths": absolute_paths,
            "path_traversal_values": traversal_paths,
            "duplicate_record_ids": duplicate_record_ids,
            "invalid_records": invalid_records,
            "missing_files": missing_files,
        },
        "passes": passes,
        "failures": failures,
    }
    report["overall_status"] = "pass" if not failures else "fail"
    return report


def validate_privacy_policy(manifest_path: str | Path) -> dict[str, Any]:
    """Run privacy and eligibility validation. Raises on hard failures."""
    report = audit_privacy(manifest_path)
    hard_failures = [
        f for f in report["failures"]
        if f["check"] in {
            "no_eligible_records_without_review",
            "no_personal_metadata_in_training_manifests",
        }
    ]
    if hard_failures:
        raise PrivacyAuditError(
            "Privacy/eligibility validation failed: " +
            "; ".join(f["message"] for f in hard_failures)
        )
    return report


def validate_training_manifest(
    manifest_path: str | Path,
    path_root: str | Path | None = None,
) -> dict[str, Any]:
    """Validate a sanitized field-survey training manifest. Raises on any failure."""
    report = audit_training_manifest(manifest_path, path_root=path_root)
    if report["failures"]:
        raise PrivacyAuditError(
            "Training manifest validation failed: "
            + "; ".join(f["message"] for f in report["failures"])
        )
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Validate field-survey manifest privacy and eligibility.")
    parser.add_argument("--manifest", default="data/manifests/field_survey/validated_manifest.json")
    parser.add_argument("--training", action="store_true", help="Validate a sanitized training manifest instead of the detailed review manifest.")
    parser.add_argument("--path-root", default=None, help="Root used to resolve training manifest image_path values.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    report = (
        audit_training_manifest(args.manifest, path_root=args.path_root)
        if args.training
        else audit_privacy(args.manifest)
    )
    output_path = Path(args.output) if args.output else None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Report written to {output_path}")
    else:
        print(json.dumps(report, indent=2))

    return 0 if report["overall_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
