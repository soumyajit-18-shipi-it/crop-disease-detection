from src.training.config import load_config


def test_phase1_config_declares_required_architectures_and_datasets():
    config = load_config("configs/training/phase1.yaml")
    assert config.model.architectures == ["efficientnetv2_s", "convnext_tiny", "convnext_base"]
    assert {source.name for source in config.data.sources} >= {"plantvillage", "plantdoc", "field_survey"}
    field = next(source for source in config.data.sources if source.name == "field_survey")
    assert field.type == "field_survey_training_manifest"
    assert field.enabled is True
