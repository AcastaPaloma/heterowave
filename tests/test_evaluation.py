from dataclasses import asdict
from pathlib import Path

import csv
import torch

from heterowave.config import load_config
from heterowave.evaluate import main as evaluate_main
from heterowave.evaluation import evaluate_scenario
from heterowave.training import build_model, create_datasets
from torch.utils.data import DataLoader


ROOT = Path(__file__).parents[1]


def _checkpoint(config_path: Path, output: Path):
    config = load_config(config_path, ["device=cpu"])
    model = build_model(config)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": {},
            "epoch": 0,
            "global_step": 1,
            "best_nrmse": 1.0,
            "config": asdict(config),
        },
        output,
    )
    return output


def test_scenario_metrics_include_missingness_and_runtime_fields():
    config = load_config(ROOT / "configs" / "local_heterowave_smoke.yaml", ["device=cpu"])
    _, dataset = create_datasets(config)
    loader = DataLoader(dataset, batch_size=2)
    row = evaluate_scenario(
        kind="fbp",
        label="fbp",
        model=None,
        loader=loader,
        metadata=dataset.metadata,
        scenario="random_8",
        mask=torch.tensor([1, 0] * 8, dtype=torch.bool),
        device=torch.device("cpu"),
        max_samples=2,
    )
    assert row["samples"] == 2
    assert row["observed_sectors"] == 8
    assert row["observed_fraction"] == 0.5
    assert row["nrmse"] > 0
    assert row["observed_data_residual"] >= 0
    assert row["inference_ms_per_sample"] > 0


def test_complete_phase6_suite_writes_all_artifacts(tmp_path):
    unet_checkpoint = _checkpoint(ROOT / "configs" / "local_smoke.yaml", tmp_path / "unet.pt")
    masked_unet_checkpoint = _checkpoint(
        ROOT / "configs" / "local_masked_unet_smoke.yaml", tmp_path / "masked_unet.pt"
    )
    heterowave_checkpoint = _checkpoint(
        ROOT / "configs" / "local_heterowave_smoke.yaml", tmp_path / "heterowave.pt"
    )
    output = tmp_path / "evaluation"
    rows = evaluate_main(
        [
            "--config",
            str(ROOT / "configs" / "local_heterowave_smoke.yaml"),
            "--suite",
            "--unet-checkpoint",
            str(unet_checkpoint),
            "--masked-unet-checkpoint",
            str(masked_unet_checkpoint),
            "--heterowave-checkpoint",
            str(heterowave_checkpoint),
            "--output-dir",
            str(output),
            "--max-samples",
            "2",
            "device=cpu",
            "data.batch_size=1",
        ]
    )
    assert len(rows) == 8 * 4
    expected = {
        "metrics_by_scenario.csv",
        "robustness_random.png",
        "robustness_wedge.png",
        "qualitative_grid.png",
        "architecture.png",
        "evaluation_config.json",
        "config.yaml",
        "checkpoint_provenance.json",
        "model.pt",
        "summary.json",
    }
    assert expected <= {path.name for path in output.iterdir()}
    with (output / "metrics_by_scenario.csv").open(newline="", encoding="utf-8") as handle:
        saved_rows = list(csv.DictReader(handle))
    assert len(saved_rows) == 32
    assert {row["model"] for row in saved_rows} == {
        "fbp",
        "fbp_unet",
        "masked_fbp_unet",
        "heterowave_mean_var_count",
    }
    assert {row["scenario"] for row in saved_rows} == {
        "all_16",
        "observed_12",
        "random_8",
        "contiguous_8",
        "random_4",
        "contiguous_4",
        "random_2",
        "contiguous_2",
    }
