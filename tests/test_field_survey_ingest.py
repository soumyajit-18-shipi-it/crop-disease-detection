from __future__ import annotations

from pathlib import Path

from PIL import Image

from src.data.ingest_field_survey import build_manifest


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=color).save(path)


def test_build_manifest_reports_missing_duplicates_and_invalid_labels(tmp_path: Path):
    survey_path = tmp_path / "survey.csv"
    image_root = tmp_path / "images"
    output_path = tmp_path / "manifest.json"
    labels_path = tmp_path / "labels.json"

    _write_image(image_root / "leaf_a.jpg", (20, 120, 20))
    _write_image(image_root / "leaf_b.jpg", (20, 120, 20))
    labels_path.write_text('["early blight"]', encoding="utf-8")
    survey_path.write_text(
        "\n".join(
            [
                '"_id";"_Crop";"_Disease";"Upload 1";"Upload 2"',
                '"1";"Tomato";"early blight";"leaf_a.jpg";"leaf_b.jpg"',
                '"2";"Tomato";"";"missing.jpg";""',
            ]
        ),
        encoding="utf-8",
    )

    manifest = build_manifest(
        survey_file=survey_path,
        image_root=image_root,
        output_path=output_path,
        label_column="_Disease",
        crop_column="_Crop",
        image_columns=["Upload 1", "Upload 2"],
        allowed_labels_path=labels_path,
    )

    assert output_path.exists()
    assert manifest["statistics"]["survey_rows"] == 2
    assert manifest["statistics"]["image_records"] == 3
    assert manifest["statistics"]["valid_records"] == 2
    assert manifest["statistics"]["missing_images"] == 1
    assert manifest["statistics"]["invalid_labels"] == 1
    assert manifest["statistics"]["duplicate_image_hash_groups"] == 1
    assert manifest["issues"]["missing_images"][0]["image"] == "missing.jpg"
