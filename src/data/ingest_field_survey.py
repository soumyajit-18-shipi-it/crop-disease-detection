from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, UnidentifiedImageError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_IMAGE_COLUMN_MARKERS = ("upload", "image", "photo", "attachment")
DEFAULT_METADATA_COLUMNS = (
    "_id",
    "_uuid",
    "_submission_time",
    "start",
    "end",
    "Name_of_the_Surveyer",
    "Team_Number",
    "College_name",
    "_Farmer_Name",
    "_Crop",
    "_Other_Crops",
    "_Disease",
    "_Symptoms",
    "_Causes",
)


def _clean_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_label(value: str) -> str:
    return " ".join(value.strip().split()).lower()


def _read_table(path: Path, sheet: str | int | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path, sheet_name=0 if sheet is None else sheet, dtype=str)
    if suffix == ".csv":
        return pd.read_csv(path, sep=None, engine="python", dtype=str)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", dtype=str)
    raise ValueError(f"Unsupported survey file type: {path.suffix}")


def _load_allowed_labels(path: Path | None) -> set[str]:
    if path is None:
        return set()
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if "class_to_idx" in payload:
        labels = payload["class_to_idx"].keys()
    elif "idx_to_class" in payload:
        labels = payload["idx_to_class"].values()
    elif isinstance(payload, list):
        labels = payload
    else:
        labels = payload.keys()
    return {_normalize_label(str(label)) for label in labels}


def _detect_image_columns(columns: list[str]) -> list[str]:
    image_columns = []
    for column in columns:
        lower = column.lower()
        if lower.endswith("_url") or lower.endswith("url"):
            continue
        if any(marker in lower for marker in DEFAULT_IMAGE_COLUMN_MARKERS):
            image_columns.append(column)
    return image_columns


def _index_images(image_root: Path) -> tuple[dict[str, list[Path]], dict[str, Path]]:
    by_name: dict[str, list[Path]] = defaultdict(list)
    by_relative: dict[str, Path] = {}
    if not image_root.exists():
        return by_name, by_relative

    for path in image_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        by_name[path.name.lower()].append(path)
        by_relative[path.relative_to(image_root).as_posix().lower()] = path
    return by_name, by_relative


def _resolve_image(value: str, image_root: Path, by_name: dict[str, list[Path]], by_relative: dict[str, Path]) -> tuple[Path | None, str]:
    normalized = value.replace("\\", "/").strip()
    if not normalized:
        return None, "empty"

    direct = Path(normalized)
    candidates = [
        direct if direct.is_absolute() else image_root / direct,
        by_relative.get(normalized.lower()),
    ]
    for candidate in candidates:
        if candidate and candidate.exists() and candidate.is_file():
            return candidate, "found"

    matches = by_name.get(Path(normalized).name.lower(), [])
    if len(matches) == 1:
        return matches[0], "found"
    if len(matches) > 1:
        return None, "ambiguous"
    return None, "missing"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_metadata(path: Path) -> tuple[dict[str, Any], str | None]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            return {
                "width": width,
                "height": height,
                "format": image.format,
                "mode": image.mode,
            }, None
    except (UnidentifiedImageError, OSError) as exc:
        return {}, str(exc)


def _record_metadata(row: pd.Series) -> dict[str, str]:
    return {column: _clean_value(row[column]) for column in DEFAULT_METADATA_COLUMNS if column in row.index}


def build_manifest(
    survey_file: str | Path,
    image_root: str | Path,
    output_path: str | Path,
    label_column: str = "_Disease",
    crop_column: str = "_Crop",
    other_crop_column: str = "_Other_Crops",
    image_columns: list[str] | None = None,
    allowed_labels_path: str | Path | None = None,
    sheet: str | int | None = None,
) -> dict[str, Any]:
    survey_path = Path(survey_file)
    image_root_path = Path(image_root)
    output = Path(output_path)
    allowed_labels = _load_allowed_labels(Path(allowed_labels_path) if allowed_labels_path else None)

    dataframe = _read_table(survey_path, sheet)
    dataframe.columns = [str(column).strip() for column in dataframe.columns]
    selected_image_columns = image_columns or _detect_image_columns(list(dataframe.columns))
    if not selected_image_columns:
        raise ValueError("No image columns found. Pass --image-columns with comma-separated column names.")
    missing_required = [column for column in [label_column, *selected_image_columns] if column not in dataframe.columns]
    if missing_required:
        raise ValueError(f"Missing expected columns: {missing_required}")

    by_name, by_relative = _index_images(image_root_path)
    records: list[dict[str, Any]] = []
    issues = {
        "missing_images": [],
        "ambiguous_images": [],
        "invalid_images": [],
        "invalid_labels": [],
        "duplicate_image_hashes": [],
        "duplicate_image_references": [],
    }
    hash_to_records: dict[str, list[str]] = defaultdict(list)
    path_to_records: dict[str, list[str]] = defaultdict(list)

    for row_index, row in dataframe.iterrows():
        raw_label = _clean_value(row.get(label_column, ""))
        normalized_label = _normalize_label(raw_label)
        crop = _clean_value(row.get(crop_column, "")) or _clean_value(row.get(other_crop_column, ""))
        submission_id = _clean_value(row.get("_id", "")) or _clean_value(row.get("_uuid", "")) or str(row_index + 1)
        label_valid = bool(normalized_label) and (not allowed_labels or normalized_label in allowed_labels)
        if not label_valid:
            issues["invalid_labels"].append(
                {
                    "row_index": int(row_index),
                    "submission_id": submission_id,
                    "label": raw_label,
                    "reason": "blank" if not normalized_label else "not_in_allowed_labels",
                }
            )

        for image_column in selected_image_columns:
            image_value = _clean_value(row.get(image_column, ""))
            if not image_value:
                continue

            resolved_path, status = _resolve_image(image_value, image_root_path, by_name, by_relative)
            record_id = f"{submission_id}:{image_column}"
            record = {
                "record_id": record_id,
                "row_index": int(row_index),
                "submission_id": submission_id,
                "image_column": image_column,
                "source_image_value": image_value,
                "image_path": None,
                "image_exists": resolved_path is not None,
                "image_status": status,
                "sha256": None,
                "label": raw_label,
                "label_normalized": normalized_label,
                "label_valid": label_valid,
                "crop": crop,
                "metadata": _record_metadata(row),
                "image_metadata": {},
            }

            if resolved_path is None:
                issue = {
                    "record_id": record_id,
                    "row_index": int(row_index),
                    "submission_id": submission_id,
                    "image_column": image_column,
                    "image": image_value,
                }
                if status == "ambiguous":
                    issues["ambiguous_images"].append(issue)
                else:
                    issues["missing_images"].append(issue)
            else:
                record["image_path"] = resolved_path.as_posix()
                record["sha256"] = _file_sha256(resolved_path)
                metadata, error = _image_metadata(resolved_path)
                record["image_metadata"] = metadata
                if error:
                    record["image_status"] = "invalid"
                    issues["invalid_images"].append(
                        {
                            "record_id": record_id,
                            "path": resolved_path.as_posix(),
                            "error": error,
                        }
                    )
                hash_to_records[str(record["sha256"])].append(record_id)
                path_to_records[resolved_path.as_posix()].append(record_id)

            records.append(record)

    for sha256, record_ids in sorted(hash_to_records.items()):
        if sha256 and len(record_ids) > 1:
            issues["duplicate_image_hashes"].append({"sha256": sha256, "record_ids": record_ids})
    for path, record_ids in sorted(path_to_records.items()):
        if len(record_ids) > 1:
            issues["duplicate_image_references"].append({"image_path": path, "record_ids": record_ids})

    valid_records = [record for record in records if record["image_exists"] and record["image_status"] == "found" and record["label_valid"]]
    manifest = {
        "schema_version": "1.0",
        "dataset_name": "field_survey",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "survey_file": survey_path.as_posix(),
            "image_root": image_root_path.as_posix(),
            "label_column": label_column,
            "crop_column": crop_column,
            "other_crop_column": other_crop_column,
            "image_columns": selected_image_columns,
            "allowed_labels_path": str(allowed_labels_path) if allowed_labels_path else None,
        },
        "statistics": {
            "survey_rows": int(len(dataframe)),
            "image_records": int(len(records)),
            "valid_records": int(len(valid_records)),
            "missing_images": int(len(issues["missing_images"])),
            "ambiguous_images": int(len(issues["ambiguous_images"])),
            "invalid_images": int(len(issues["invalid_images"])),
            "invalid_labels": int(len(issues["invalid_labels"])),
            "duplicate_image_hash_groups": int(len(issues["duplicate_image_hashes"])),
            "duplicate_image_reference_groups": int(len(issues["duplicate_image_references"])),
            "labels": dict(sorted(Counter(record["label_normalized"] or "<blank>" for record in records).items())),
            "valid_labels": dict(sorted(Counter(record["label_normalized"] for record in valid_records).items())),
            "crops": dict(sorted(Counter(record["crop"] or "<blank>" for record in records).items())),
            "image_extensions": dict(sorted(Counter(Path(record["image_path"]).suffix.lower() for record in records if record["image_path"]).items())),
        },
        "issues": issues,
        "records": records,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
        file.write("\n")
    return manifest


def _parse_columns(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a validated manifest for a field survey image dataset.")
    parser.add_argument("--survey-file", required=True, help="Path to survey Excel/CSV/TSV file.")
    parser.add_argument("--image-root", required=True, help="Root directory containing downloaded survey images.")
    parser.add_argument("--output", default="data/manifests/field_survey/manifest.json", help="Manifest output path.")
    parser.add_argument("--label-column", default="_Disease")
    parser.add_argument("--crop-column", default="_Crop")
    parser.add_argument("--other-crop-column", default="_Other_Crops")
    parser.add_argument("--image-columns", default=None, help="Comma-separated image attachment columns. Auto-detected by default.")
    parser.add_argument("--allowed-labels", default=None, help="Optional JSON list or class_mapping.json for strict label validation.")
    parser.add_argument("--sheet", default=None, help="Excel sheet name or index. Defaults to the first sheet.")
    args = parser.parse_args()

    sheet: str | int | None = args.sheet
    if isinstance(sheet, str) and sheet.isdigit():
        sheet = int(sheet)

    manifest = build_manifest(
        survey_file=args.survey_file,
        image_root=args.image_root,
        output_path=args.output,
        label_column=args.label_column,
        crop_column=args.crop_column,
        other_crop_column=args.other_crop_column,
        image_columns=_parse_columns(args.image_columns),
        allowed_labels_path=args.allowed_labels,
        sheet=sheet,
    )
    stats = manifest["statistics"]
    print(f"Wrote manifest: {args.output}")
    print(f"Rows: {stats['survey_rows']}")
    print(f"Image records: {stats['image_records']}")
    print(f"Valid records: {stats['valid_records']}")
    print(f"Missing images: {stats['missing_images']}")
    print(f"Invalid labels: {stats['invalid_labels']}")
    print(f"Duplicate image hash groups: {stats['duplicate_image_hash_groups']}")


if __name__ == "__main__":
    main()
