from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from src.data.validate_field_survey_manifest import (
    FIELD_SURVEY_DATASET_SOURCE,
    TRAINING_MANIFEST_SCHEMA_VERSION,
    TRAINING_MANIFEST_TYPE,
    validate_training_manifest,
)


class TrainingManifestExportError(ValueError):
    pass


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Review manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TrainingManifestExportError("Review manifest must be a JSON object")
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _canonical_classes(review_manifest: dict[str, Any]) -> set[str]:
    labels: set[str] = set()
    taxonomy = review_manifest.get("canonical_taxonomy", {}).get("crops", {})
    if isinstance(taxonomy, dict):
        for diseases in taxonomy.values():
            if isinstance(diseases, list):
                labels.update(_text(label) for label in diseases if _text(label))
    for record in review_manifest.get("records", []):
        if not isinstance(record, dict):
            continue
        canonical = record.get("normalization", {}).get("disease", {}).get("canonical")
        if _text(canonical):
            labels.add(_text(canonical))
    return labels


def _review_events(review_manifest: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_event_id: dict[str, dict[str, Any]] = {}
    by_group_id: dict[str, dict[str, Any]] = {}
    for event in review_manifest.get("audit_history", []):
        if not isinstance(event, dict):
            continue
        event_id = _text(event.get("event_id"))
        group_id = _text(event.get("group_id"))
        if event_id:
            by_event_id[event_id] = event
        if group_id:
            by_group_id[group_id] = event
    return by_event_id, by_group_id


def _matching_event(
    record: dict[str, Any],
    by_event_id: dict[str, dict[str, Any]],
    by_group_id: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    validation = record.get("validation", {})
    event_id = _text(validation.get("latest_decision_event_id"))
    group_id = _text(validation.get("group_id"))
    if event_id:
        return by_event_id.get(event_id)
    if group_id:
        return by_group_id.get(group_id)
    return None


def _resolve_source_image(record: dict[str, Any], image_root: Path) -> Path:
    raw_path = record.get("image_path")
    if not _text(raw_path):
        raise TrainingManifestExportError(f"Approved record {record.get('record_id')} has no image_path")
    root = image_root.resolve()
    source_path = Path(str(raw_path))
    candidate = source_path.resolve() if source_path.is_absolute() else (root / source_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise TrainingManifestExportError(
            f"Approved record {record.get('record_id')} image_path escapes image root"
        ) from exc
    if not candidate.is_file():
        raise TrainingManifestExportError(
            f"Approved record {record.get('record_id')} references a missing image"
        )
    return candidate


def _safe_output_path(candidate: Path, path_base: Path, record_id: str) -> str:
    base = path_base.resolve()
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(base)
    except ValueError as exc:
        raise TrainingManifestExportError(
            f"Approved record {record_id} cannot be represented without an absolute or escaping path"
        ) from exc
    value = relative.as_posix()
    if value == "." or value.startswith("../") or "/../" in value or Path(value).is_absolute():
        raise TrainingManifestExportError(f"Approved record {record_id} has an unsafe output image path")
    return value


def _review_action(record: dict[str, Any], event: dict[str, Any] | None) -> str | None:
    if event is None:
        return None
    action = _text(event.get("action")).casefold()
    record_id = _text(record.get("record_id"))
    event_record_ids = {_text(value) for value in event.get("record_ids", [])}
    if event_record_ids and record_id not in event_record_ids:
        raise TrainingManifestExportError(
            f"Approved record {record_id} does not belong to its latest review event"
        )
    return action


def _class_for_record(record: dict[str, Any], event: dict[str, Any], canonical_classes: set[str]) -> str:
    validation = record.get("validation", {})
    validation_class = _text(validation.get("canonical_disease"))
    event_class = _text(event.get("canonical_disease"))
    final_class = validation_class or event_class
    record_id = _text(record.get("record_id"))
    if not final_class:
        raise TrainingManifestExportError(f"Approved record {record_id} has no canonical class")
    if event_class and validation_class and validation_class != event_class:
        raise TrainingManifestExportError(
            f"Approved record {record_id} disagrees with its latest review event"
        )
    if final_class not in canonical_classes:
        raise TrainingManifestExportError(
            f"Approved record {record_id} has non-canonical class {final_class!r}"
        )
    return final_class


def _approved_record(
    record: dict[str, Any],
    event: dict[str, Any],
    action: str,
    canonical_classes: set[str],
    image_root: Path,
    path_base: Path,
) -> dict[str, Any]:
    record_id = _text(record.get("record_id"))
    if not record_id:
        raise TrainingManifestExportError("Approved record is missing record_id")
    source_image = _resolve_source_image(record, image_root)
    image_path = _safe_output_path(source_image, path_base, record_id)
    image_sha256 = _text(record.get("sha256") or record.get("image_sha256"))
    if not image_sha256:
        raise TrainingManifestExportError(f"Approved record {record_id} is missing image hash")
    final_class = _class_for_record(record, event, canonical_classes)
    output = {
        "record_id": record_id,
        "dataset_source": FIELD_SURVEY_DATASET_SOURCE,
        "image_path": image_path,
        "image_sha256": image_sha256,
        "canonical_class": final_class,
        "review_status": "validated",
        "review_decision": action,
    }
    if action == "replace":
        output["replacement_class"] = final_class
    return output


def export_training_manifest(
    review_manifest_path: str | Path,
    output_path: str | Path,
    image_root: str | Path,
    path_base: str | Path | None = None,
) -> dict[str, Any]:
    """Export a sanitized training manifest from a detailed human-review manifest."""
    review_path = Path(review_manifest_path)
    output = Path(output_path)
    root = Path(image_root)
    base = Path(path_base) if path_base is not None else Path.cwd()
    review_manifest = _load_json(review_path)
    records = review_manifest.get("records", [])
    if not isinstance(records, list):
        raise TrainingManifestExportError("Review manifest records must be an array")

    canonical = _canonical_classes(review_manifest)
    if not canonical:
        raise TrainingManifestExportError("Review manifest does not define any canonical classes")

    ids = [_text(record.get("record_id")) for record in records if isinstance(record, dict)]
    duplicate_ids = [record_id for record_id, count in Counter(ids).items() if record_id and count > 1]
    if duplicate_ids:
        raise TrainingManifestExportError(
            "Review manifest contains duplicate record_id values: " + ", ".join(sorted(duplicate_ids))
        )

    by_event_id, by_group_id = _review_events(review_manifest)
    accepted_records = 0
    replaced_records = 0
    rejected_records = 0
    pending_records = 0
    exported_records: list[dict[str, Any]] = []

    for record in records:
        if not isinstance(record, dict):
            continue
        validation = record.get("validation", {})
        event = _matching_event(record, by_event_id, by_group_id)
        action = _review_action(record, event)
        eligible = validation.get("eligible_for_training") is True
        status = _text(validation.get("status")).casefold()

        if not eligible:
            if status in {"reject", "rejected"} or action == "reject":
                rejected_records += 1
            else:
                pending_records += 1
            continue

        if status != "validated":
            raise TrainingManifestExportError(
                f"Approved record {record.get('record_id')} is not in validated status"
            )
        if action not in {"accept", "replace"} or event is None:
            raise TrainingManifestExportError(
                f"Approved record {record.get('record_id')} lacks an accept/replace human-review event"
            )
        if action == "accept":
            accepted_records += 1
        else:
            replaced_records += 1
        exported_records.append(
            _approved_record(record, event, action, canonical, root, base)
        )

    exported_records.sort(key=lambda item: item["record_id"])
    manifest = {
        "schema_version": TRAINING_MANIFEST_SCHEMA_VERSION,
        "manifest_type": TRAINING_MANIFEST_TYPE,
        "dataset_name": FIELD_SURVEY_DATASET_SOURCE,
        "dataset_source": FIELD_SURVEY_DATASET_SOURCE,
        "review_manifest_sha256": _file_sha256(review_path),
        "statistics": {
            "total_source_records": len(records),
            "accepted_records": accepted_records,
            "replaced_records": replaced_records,
            "rejected_records": rejected_records,
            "pending_records": pending_records,
            "exported_records": len(exported_records),
            "excluded_invalid_records": 0,
        },
        "class_names": sorted(canonical),
        "records": exported_records,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".tmp")
    temp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    validate_training_manifest(temp, path_root=base)
    os.replace(temp, output)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export a sanitized field-survey training manifest.")
    parser.add_argument("--review-manifest", type=Path, default=Path("data/manifests/field_survey/validated_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("data/manifests/field_survey/training_manifest.json"))
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--path-base", type=Path, default=Path("."))
    args = parser.parse_args(argv)

    manifest = export_training_manifest(
        review_manifest_path=args.review_manifest,
        output_path=args.output,
        image_root=args.image_root,
        path_base=args.path_base,
    )
    stats = manifest["statistics"]
    print(f"Wrote sanitized training manifest: {args.output}")
    print(f"Source records: {stats['total_source_records']}")
    print(f"Accepted records: {stats['accepted_records']}")
    print(f"Replaced records: {stats['replaced_records']}")
    print(f"Rejected records: {stats['rejected_records']}")
    print(f"Pending records: {stats['pending_records']}")
    print(f"Exported records: {stats['exported_records']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
