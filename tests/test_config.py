import pytest


def test_can_read_config():
    from europa.config import DataConfig
    config = DataConfig.from_yaml("tests/artifacts/test_config.yaml")
    assert config.repo_id == "google/paligemma-3b-pt-224"
    assert config.max_length == 512
    assert config.batch_size == 2

def test_can_create_datamodule():
    from europa.config import DataConfig
    from europa.datasets import BIDProtoDataModule

    config = DataConfig.from_yaml("tests/artifacts/test_config.yaml")
    dm = BIDProtoDataModule(cfg=config)
    assert dm is not None