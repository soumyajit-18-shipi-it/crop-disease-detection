class CropDiseaseModel:
    """Mock Deep Learning Model for classifying crop leaf diseases."""

    def __init__(self) -> None:
        self.diseases: dict[str, dict[str, str | list[str]]] = {
            "tomato": {
                "prediction": "Tomato Early Blight",
                "confidence": "0.94",
                "symptoms": ["leaf spots", "concentric rings", "yellowing margins"],
            },
            "potato": {
                "prediction": "Potato Late Blight",
                "confidence": "0.89",
                "symptoms": ["dark lesions", "white mold on underside", "stem rotting"],
            },
            "rice": {
                "prediction": "Rice Blast",
                "confidence": "0.91",
                "symptoms": [
                    "spindle-shaped lesions",
                    "grayish centers",
                    "brown margins",
                ],
            },
        }

    def predict(self, filename: str) -> dict[str, str | float | list[str]]:
        """Classify disease based on plant image filename."""
        name_lower = filename.lower()
        selected_crop = "tomato"
        for crop in self.diseases:
            if crop in name_lower:
                selected_crop = crop
                break

        crop_info = self.diseases[selected_crop]
        return {
            "filename": filename,
            "prediction": str(crop_info["prediction"]),
            "confidence": float(crop_info["confidence"]),
            "symptoms": list(crop_info["symptoms"]),  # type: ignore
        }
