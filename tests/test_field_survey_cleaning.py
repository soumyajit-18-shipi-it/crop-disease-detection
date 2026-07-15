import csv
import json

from src.data.clean_field_survey_labels import clean_manifest


def test_cleaning_preserves_records_and_traces_changes(tmp_path):
    source = {
        "dataset_name": "field_survey",
        "statistics": {"image_records": 4},
        "issues": {"missing_images": []},
        "records": [
            {"record_id": "1", "label": "Anthrancnose", "crop": "TOMATO"},
            {"record_id": "2", "label": "Brown spot, rice blast", "crop": "__rice"},
            {"record_id": "3", "label": "", "crop": "__rice __cotton"},
            {"record_id": "4", "label": "NALLI(TELUGU)", "crop": "Rice"},
        ],
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(source), encoding="utf-8")

    cleaned = clean_manifest(manifest, tmp_path / "out")

    assert len(cleaned["records"]) == 4
    first = cleaned["records"][0]
    assert first["label"] == "Anthrancnose"
    assert first["normalization"]["disease"]["canonical"] == "Anthracnose"
    assert first["normalization"]["disease"]["modified"] is True
    assert cleaned["records"][1]["normalization"]["disease"]["status"] == "manual_review"
    assert "multiple_crops" in cleaned["records"][2]["normalization"]["crop"]["review_reasons"]
    assert cleaned["records"][3]["normalization"]["disease"]["multilingual"] is True
    assert cleaned["canonical_taxonomy"]["requires_domain_validation"] is True

    with (tmp_path / "out" / "label_mapping.csv").open(encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 4
    assert (tmp_path / "out" / "dataset_report.md").exists()
