from pathlib import Path

import pytest

from heterowave.config import load_config


CONFIG = Path(__file__).parents[1] / "configs" / "local_smoke.yaml"
HETEROWAVE_CONFIG = Path(__file__).parents[1] / "configs" / "local_heterowave_smoke.yaml"
MASKED_UNET_CONFIG = Path(__file__).parents[1] / "configs" / "local_masked_unet_smoke.yaml"
PHASE7_CONFIG = Path(__file__).parents[1] / "configs" / "local_phase7_smoke.yaml"


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


def test_heterowave_config_enables_masked_set_model():
    config = load_config(HETEROWAVE_CONFIG)
    assert config.model.name == "heterowave"
    assert config.model.aggregation == "mean_var_count"
    assert config.physics.num_angles % config.physics.num_sectors == 0
    assert config.masking.minimum_sectors == 2


def test_masked_unet_config_matches_heterowave_mask_protocol():
    baseline = load_config(MASKED_UNET_CONFIG)
    heterowave = load_config(HETEROWAVE_CONFIG)
    assert baseline.model.name == "masked_fbp_unet"
    assert baseline.seed == heterowave.seed == 1337
    assert baseline.physics.num_sectors == heterowave.physics.num_sectors
    assert baseline.masking == heterowave.masking


def test_phase7_config_enables_physics_and_uncertainty():
    config = load_config(PHASE7_CONFIG)
    assert config.model.uncertainty is True
    assert config.loss.data_weight == pytest.approx(0.05)
    assert config.loss.data_every_n_steps == 2
    assert config.loss.data_angle_fraction == pytest.approx(0.5)
    assert config.loss.uncertainty_weight == pytest.approx(0.1)
