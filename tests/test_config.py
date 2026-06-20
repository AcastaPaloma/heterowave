from pathlib import Path

import pytest

from heterowave.config import load_config


CONFIG = Path(__file__).parents[1] / "configs" / "local_smoke.yaml"


def test_local_config_is_safe_and_tiny():
    config = load_config(CONFIG)
    assert config.precision == "fp32"
    assert config.compile is False
    assert config.data.num_workers == 0
    assert config.data.dataset == "synthetic"
    assert config.data.preprocess is False
    assert config.data.image_size == 32


def test_dotted_override_and_strict_unknown_keys():
    assert load_config(CONFIG, ["physics.num_angles=8"]).physics.num_angles == 8
    with pytest.raises(KeyError):
        load_config(CONFIG, ["physics.typo=8"])

