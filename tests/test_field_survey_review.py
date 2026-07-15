import json

from src.data.review_field_survey import ReviewStore, build_review_groups


def _manifest():
    return {
        "dataset_name": "field_survey",
        "canonical_taxonomy": {"crops": {"Tomato": ["Anthracnose", "Early Blight"]}},
        "records": [
            {"record_id": "1", "label": "Anthrancnose", "crop": "Tomato", "image_path": None,
             "metadata": {"_Symptoms": "dark spots"},
             "normalization": {"disease": {"canonical": "Anthracnose", "confidence": 1.0, "status": "normalized", "review_reasons": []}, "crop": {"canonical": "Tomato", "status": "unchanged", "review_reasons": []}}},
            {"record_id": "2", "label": "unknown spots", "crop": "Tomato", "image_path": None,
             "metadata": {"_Symptoms": "spots"},
             "normalization": {"disease": {"canonical": None, "confidence": None, "status": "manual_review", "review_reasons": ["ambiguous_label"]}, "crop": {"canonical": "Tomato", "status": "unchanged", "review_reasons": []}}},
        ],
    }


def test_review_decisions_are_audited_and_gate_training(tmp_path):
    manifest_path = tmp_path / "cleaned_manifest.json"
    output_path = tmp_path / "validated_manifest.json"
    decisions_path = tmp_path / "review_decisions.jsonl"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    store = ReviewStore(manifest_path, output_path, decisions_path)

    groups = build_review_groups(_manifest())
    first = next(group for group in groups if group["original_disease"] == "Anthrancnose")
    store.decide({"group_id": first["group_id"], "action": "accept", "reviewer": "tester"})
    store.decide({"group_id": first["group_id"], "action": "replace", "canonical_disease": "Early Blight", "reviewer": "tester", "note": "second review"})

    validated = json.loads(output_path.read_text(encoding="utf-8"))
    reviewed = next(record for record in validated["records"] if record["record_id"] == "1")
    pending = next(record for record in validated["records"] if record["record_id"] == "2")
    assert reviewed["validation"]["eligible_for_training"] is True
    assert reviewed["validation"]["canonical_disease"] == "Early Blight"
    assert pending["validation"]["eligible_for_training"] is False
    assert len(validated["audit_history"]) == 2
    assert len(decisions_path.read_text(encoding="utf-8").splitlines()) == 2
