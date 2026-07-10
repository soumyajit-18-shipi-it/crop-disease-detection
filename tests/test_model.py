from app.model import CropDiseaseModel


def test_crop_disease_model_tomato() -> None:
    model = CropDiseaseModel()
    res = model.predict("my_tomato_leaf.png")
    assert res["filename"] == "my_tomato_leaf.png"
    assert res["prediction"] == "Tomato Early Blight"
    assert res["confidence"] == 0.94
    assert "leaf spots" in res["symptoms"]


def test_crop_disease_model_potato() -> None:
    model = CropDiseaseModel()
    res = model.predict("potato_leaf_disease.jpg")
    assert res["filename"] == "potato_leaf_disease.jpg"
    assert res["prediction"] == "Potato Late Blight"
    assert res["confidence"] == 0.89


def test_crop_disease_model_rice() -> None:
    model = CropDiseaseModel()
    res = model.predict("diseased_rice.png")
    assert res["filename"] == "diseased_rice.png"
    assert res["prediction"] == "Rice Blast"
    assert res["confidence"] == 0.91
