from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


UNKNOWN_VALUES = {
    "", "na", "n/a", "yes", "no", "crop", "dip", "sugar", "chemical medicine",
    "plant problem", "related to crop growth", "1000", "5000", "60000",
}

CROP_ALIASES = {
    "bengal gram": "Bengal Gram", "beans": "Beans", "beens": "Beans",
    "bottle gourd": "Bottle Gourd", "bottle guard": "Bottle Gourd", "bottlegourd": "Bottle Gourd",
    "brinjal": "Brinjal", "corn": "Corn", "cotton": "Cotton", "cucumber": "Cucumber",
    "drum stick": "Drumstick", "drum sticks": "Drumstick", "drumstick": "Drumstick",
    "drumsticks": "Drumstick", "gauva": "Guava", "gova": "Guava", "guava": "Guava",
    "guava leaf": "Guava", "guva": "Guava", "ladies finger": "Okra", "ladys finger": "Okra",
    "ladysfinger": "Okra", "leman": "Lemon", "lemon": "Lemon", "papaya": "Papaya",
    "red gram": "Red Gram", "rice": "Rice", "sugar cane": "Sugar Cane", "tomato": "Tomato",
    "chilli": "Chilli", "broad bean": "Broad Beans", "broad beans": "Broad Beans",
    "broad been": "Broad Beans", "green gram": "Green Gram", "green peas": "Green Peas",
}

DISEASE_ALIASES = {
    "anthracnose": "Anthracnose", "anthrancnose": "Anthracnose", "antracnose": "Anthracnose",
    "alternaria leaf spot": "Alternaria Leaf Spot", "alternia leafspot": "Alternaria Leaf Spot",
    "bacteria wilt": "Bacterial Wilt", "bacterial wilt": "Bacterial Wilt",
    "bacterial blight": "Bacterial Blight", "bacterial leaf blight": "Bacterial Leaf Blight",
    "bactetial leaf blight": "Bacterial Leaf Blight", "bacterial leaf streak": "Bacterial Leaf Streak",
    "baterial leaf streak": "Bacterial Leaf Streak", "bacterial stalk rot": "Bacterial Stalk Rot",
    "brown spot": "Brown Spot", "brown spots": "Brown Spot", "common blight": "Common Blight",
    "downy mildew": "Downy Mildew", "fire blight": "Fire Blight", "fire blite comman": "Fire Blight",
    "foot root of papaya": "Papaya Foot Rot", "foot rot of papaya": "Papaya Foot Rot",
    "foot rot of papya": "Papaya Foot Rot", "fruit rot": "Fruit Rot", "fruit rots": "Fruit Rot",
    "fruits rot": "Fruit Rot", "fusarium": "Fusarium Wilt", "fusarium eilt": "Fusarium Wilt",
    "fusarium wilt": "Fusarium Wilt", "gray leaf spot": "Gray Leaf Spot",
    "grey leaf spot": "Gray Leaf Spot", "halo blight": "Halo Blight", "leaf blight": "Leaf Blight",
    "leaf spot": "Leaf Spot", "leaf streak": "Leaf Streak", "powdery mildew": "Powdery Mildew",
    "pythium": "Pythium", "rice blast": "Rice Blast", "ringspot": "Ring Spot",
    "papaya ring spot diseases": "Papaya Ring Spot", "septoria leaf spot": "Septoria Leaf Spot",
    "stripe rust": "Stripe Rust", "twig canker": "Twig Canker", "verticillium wilt": "Verticillium Wilt",
    "no disease yet": "Healthy", "none": "Healthy",
}

PEST_OR_TREATMENT_TERMS = {
    "aphid", "bird", "caterpillar", "fungicide", "insect", "mealybug", "mosquito",
    "pest", "purugu", "thrip", "worm",
}
MULTILINGUAL_MARKERS = {
    "aaku", "aggi", "dhoma", "doma", "gobbarogam", "kaatu", "katu", "macha", "mudatha",
    "nalli", "penu", "purugu", "rogam", "tegilu", "tegulu", "thegulu", "telugu", "yendhipovuta",
}


def _key(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text).strip().casefold()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .")


def _display(value: str) -> str:
    return " ".join(value.strip().split())


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z]+", value.casefold()))


def _is_multilingual(value: str) -> bool:
    if any(ord(char) > 127 and char.isalpha() for char in value):
        return True
    return bool(_tokens(value) & MULTILINGUAL_MARKERS)


def _is_compound(value: str) -> bool:
    return bool(re.search(r"[,;/]|\band\b", value, flags=re.IGNORECASE))


def _normalize_crop(raw: str) -> dict[str, Any]:
    original = raw or ""
    encoded = re.findall(r"__([a-z_]+)", original.casefold())
    if encoded:
        candidates = [CROP_ALIASES.get(item.replace("_", " "), item.replace("_", " ").title()) for item in encoded]
        if len(candidates) == 1:
            return _result(original, candidates[0], "normalized", "encoded_crop_value", 1.0, [])
        return _result(original, None, "manual_review", "multiple_crop_values", None, ["multiple_crops"], candidates)

    key = _key(original)
    if not key:
        return _result(original, None, "manual_review", "blank_crop", None, ["missing_crop"])
    if _is_compound(key):
        candidates = [CROP_ALIASES.get(_key(part)) for part in re.split(r"[,;/]|\band\b", key) if _key(part)]
        return _result(original, None, "manual_review", "multiple_crop_values", None, ["multiple_crops"], [x for x in candidates if x])
    canonical = CROP_ALIASES.get(key)
    if canonical:
        rule = "exact_canonical" if original == canonical else "crop_alias"
        return _result(original, canonical, "unchanged" if rule == "exact_canonical" else "normalized", rule, 1.0, [])
    return _result(original, None, "manual_review", "unknown_crop", None, ["unknown_crop"])


def _normalize_disease(raw: str) -> dict[str, Any]:
    original = raw or ""
    key = _key(original)
    multilingual = _is_multilingual(original)
    reasons = ["multilingual_or_transliterated"] if multilingual else []
    if key in UNKNOWN_VALUES:
        reasons.append("missing_disease" if not key else "unknown_value")
        return _result(original, None, "manual_review", "unknown_value", None, reasons, multilingual=multilingual)
    if _is_compound(key):
        reasons.append("compound_or_ambiguous_label")
        return _result(original, None, "manual_review", "compound_label", None, reasons, multilingual=multilingual)
    if _tokens(key) & PEST_OR_TREATMENT_TERMS:
        reasons.append("pest_treatment_or_non_disease")
        return _result(original, None, "manual_review", "non_disease_label", None, reasons, multilingual=multilingual)
    canonical = DISEASE_ALIASES.get(key)
    if canonical:
        rule = "exact_canonical" if original == canonical else "disease_alias_or_spelling"
        return _result(original, canonical, "unchanged" if rule == "exact_canonical" else "normalized", rule, 1.0, reasons, multilingual=multilingual)
    reasons.append("ambiguous_label")
    return _result(original, None, "manual_review", "unmapped_label", None, reasons, multilingual=multilingual)


def _result(
    original: str,
    canonical: str | None,
    status: str,
    rule: str,
    confidence: float | None,
    reasons: list[str],
    candidates: list[str] | None = None,
    multilingual: bool = False,
) -> dict[str, Any]:
    return {
        "original": original,
        "canonical": canonical,
        "status": status,
        "rule": rule,
        "confidence": confidence,
        "review_reasons": reasons,
        "candidates": candidates or [],
        "multilingual": multilingual,
        "modified": canonical is not None and canonical != original,
    }


def _variation_groups(values: Counter[str], threshold: float = 0.86) -> list[dict[str, Any]]:
    keys = sorted(value for value in values if value)
    groups: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, left in enumerate(keys):
        if left in seen:
            continue
        matches = [left]
        for right in keys[index + 1:]:
            if right in seen or abs(len(left) - len(right)) > 5:
                continue
            if SequenceMatcher(None, left, right).ratio() >= threshold:
                matches.append(right)
        if len(matches) > 1:
            seen.update(matches)
            groups.append({"values": matches, "total_records": sum(values[item] for item in matches)})
    return groups


def _mapping_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        disease = record["normalization"]["disease"]
        crop = record["normalization"]["crop"]
        key = (record.get("label", ""), record.get("crop", ""))
        if key not in grouped:
            grouped[key] = {
                "original_disease": key[0], "original_crop": key[1], "record_count": 0,
                "canonical_disease": disease["canonical"] or "", "canonical_crop": crop["canonical"] or "",
                "disease_status": disease["status"], "crop_status": crop["status"],
                "disease_rule": disease["rule"], "crop_rule": crop["rule"],
                "multilingual": disease["multilingual"],
                "review_reasons": ";".join(sorted(set(disease["review_reasons"] + crop["review_reasons"]))),
                "disease_candidates": ";".join(disease["candidates"]), "crop_candidates": ";".join(crop["candidates"]),
            }
        grouped[key]["record_count"] += 1
    return sorted(grouped.values(), key=lambda row: (-row["record_count"], row["original_crop"].casefold(), row["original_disease"].casefold()))


def _write_mapping(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else ["original_disease"])
        writer.writeheader()
        writer.writerows(rows)


def _report(manifest: dict[str, Any], records: list[dict[str, Any]], mapping: list[dict[str, Any]], analysis: dict[str, Any]) -> str:
    disease_counts = Counter(record.get("label", "") or "<blank>" for record in records)
    crop_counts = Counter(record.get("crop", "") or "<blank>" for record in records)
    taxonomy: dict[str, set[str]] = defaultdict(set)
    for record in records:
        crop = record["normalization"]["crop"]["canonical"]
        disease = record["normalization"]["disease"]["canonical"]
        if crop and disease:
            taxonomy[crop].add(disease)

    lines = [
        "# Field Survey Dataset Label Report", "",
        f"Generated from `{manifest.get('source_manifest', manifest.get('source', {}).get('survey_file', 'manifest.json'))}`.", "",
        "## Summary", "",
        f"- Records: {len(records)}",
        f"- Unique original crop names: {len(crop_counts)}",
        f"- Unique original disease names: {len(disease_counts)}",
        f"- Missing disease records: {analysis['missing_disease_records']}",
        f"- Unknown-value records: {analysis['unknown_value_records']}",
        f"- Multilingual/transliterated disease records: {analysis['multilingual_records']}",
        f"- Records requiring manual review: {analysis['manual_review_records']}",
        f"- Automatically normalized records: {analysis['normalized_records']}", "",
        "No records were discarded. Canonical values are additions; original values remain unchanged.", "",
        "## Canonical Taxonomy", "",
    ]
    if taxonomy:
        for crop in sorted(taxonomy):
            lines.append(f"### {crop}")
            lines.append("")
            lines.extend(f"- {disease}" for disease in sorted(taxonomy[crop]))
            lines.append("")
    else:
        lines.extend(["No high-confidence crop/disease pairs were found.", ""])

    def frequency_section(title: str, counts: Counter[str]) -> None:
        lines.extend([f"## {title}", "", "| Original value | Records |", "|---|---:|"])
        for value, count in sorted(counts.items(), key=lambda item: (-item[1], item[0].casefold())):
            escaped = value.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {escaped} | {count} |")
        lines.append("")

    frequency_section("Crop Names and Frequencies", crop_counts)
    frequency_section("Disease Names and Frequencies", disease_counts)
    lines.extend(["## Missing and Unknown Disease Values", "", "| Value | Records | Category |", "|---|---:|---|"])
    for value, count in sorted(disease_counts.items(), key=lambda item: (-item[1], item[0])):
        key = "" if value == "<blank>" else _key(value)
        if key in UNKNOWN_VALUES:
            lines.append(f"| {value} | {count} | {'missing' if not key else 'unknown'} |")
    lines.append("")
    lines.extend(["## Duplicate Labels", "", "Case, whitespace, and punctuation variants sharing one normalized key:", ""])
    duplicate_groups = analysis["duplicate_labels"]
    if duplicate_groups:
        for item in duplicate_groups:
            lines.append(f"- `{item['normalized']}`: " + ", ".join(f"`{value}`" for value in item["original_values"]))
    else:
        lines.append("- None detected")
    lines.extend(["", "## Spelling Variations", ""])
    if analysis["spelling_variations"]:
        for group in analysis["spelling_variations"]:
            lines.append("- " + ", ".join(f"`{value}`" for value in group["values"]))
    else:
        lines.append("- None detected")
    lines.extend(["", "## Multilingual or Transliterated Labels", "", "| Value | Records |", "|---|---:|"])
    for value, count in analysis["multilingual_labels"].items():
        lines.append(f"| {value.replace('|', '\\|')} | {count} |")
    lines.extend(["", "## Manual Review", "", "Ambiguous, compound, unknown, multilingual, pest, treatment, and symptom labels are preserved and flagged in `label_mapping.csv`.", ""])
    return "\n".join(lines)


def clean_manifest(input_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    source_path = Path(input_path)
    output = Path(output_dir)
    with source_path.open("r", encoding="utf-8") as file:
        source = json.load(file)
    records = []
    for source_record in source.get("records", []):
        record = dict(source_record)
        record["normalization"] = {
            "disease": _normalize_disease(str(record.get("label", "") or "")),
            "crop": _normalize_crop(str(record.get("crop", "") or "")),
        }
        records.append(record)

    raw_labels = Counter(str(record.get("label", "") or "") for record in records)
    normalized_to_originals: dict[str, set[str]] = defaultdict(set)
    for value in raw_labels:
        normalized_to_originals[_key(value)].add(value)
    duplicate_labels = [
        {"normalized": key, "original_values": sorted(values)}
        for key, values in sorted(normalized_to_originals.items()) if len(values) > 1
    ]
    multilingual = Counter(record.get("label", "") for record in records if record["normalization"]["disease"]["multilingual"])
    taxonomy: dict[str, set[str]] = defaultdict(set)
    for record in records:
        crop = record["normalization"]["crop"]["canonical"]
        disease = record["normalization"]["disease"]["canonical"]
        if crop and disease:
            taxonomy[crop].add(disease)
    analysis = {
        "missing_disease_records": sum(1 for record in records if not _key(record.get("label", ""))),
        "unknown_value_records": sum(1 for record in records if _key(record.get("label", "")) in UNKNOWN_VALUES),
        "multilingual_records": sum(multilingual.values()),
        "manual_review_records": sum(1 for record in records if record["normalization"]["disease"]["status"] == "manual_review" or record["normalization"]["crop"]["status"] == "manual_review"),
        "normalized_records": sum(1 for record in records if record["normalization"]["disease"]["status"] == "normalized" or record["normalization"]["crop"]["status"] == "normalized"),
        "duplicate_labels": duplicate_labels,
        "spelling_variations": _variation_groups(Counter({_key(k): v for k, v in raw_labels.items() if _key(k)})),
        "multilingual_labels": dict(sorted(multilingual.items(), key=lambda item: (-item[1], item[0].casefold()))),
    }
    mapping = _mapping_rows(records)
    cleaned = {
        "schema_version": "1.1",
        "dataset_name": source.get("dataset_name", "field_survey"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_manifest": source_path.as_posix(),
        "normalization_policy": {
            "records_discarded": False,
            "automatic_changes": "Only curated, high-confidence spelling, case, and encoding aliases.",
            "ambiguous_values": "Preserved and flagged for manual review.",
        },
        "canonical_taxonomy": {
            "status": "provisional_source_observed",
            "requires_domain_validation": True,
            "note": "Canonical names are normalized, but crop-disease associations have not been biologically validated.",
            "crops": {crop: sorted(diseases) for crop, diseases in sorted(taxonomy.items())},
        },
        "statistics": {**source.get("statistics", {}), **{key: value for key, value in analysis.items() if isinstance(value, int)}},
        "analysis": analysis,
        "source_issues": source.get("issues", {}),
        "records": records,
    }
    output.mkdir(parents=True, exist_ok=True)
    cleaned_path = output / "cleaned_manifest.json"
    with cleaned_path.open("w", encoding="utf-8") as file:
        json.dump(cleaned, file, indent=2, ensure_ascii=False)
        file.write("\n")
    _write_mapping(output / "label_mapping.csv", mapping)
    (output / "dataset_report.md").write_text(_report(cleaned, records, mapping, analysis), encoding="utf-8")
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize field survey crop and disease labels without dropping records.")
    parser.add_argument("--manifest", default="data/manifests/field_survey/manifest.json")
    parser.add_argument("--output-dir", default="data/manifests/field_survey")
    args = parser.parse_args()
    cleaned = clean_manifest(args.manifest, args.output_dir)
    stats = cleaned["statistics"]
    print(f"Wrote cleaned manifest and reports to: {args.output_dir}")
    print(f"Records preserved: {len(cleaned['records'])}")
    print(f"Automatically normalized records: {stats['normalized_records']}")
    print(f"Records requiring manual review: {stats['manual_review_records']}")


if __name__ == "__main__":
    main()
