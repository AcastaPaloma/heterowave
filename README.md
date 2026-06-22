# HeteroWave

Acquisition-aware sparse-view reconstruction for water-coupled ultrasonic CT.

HeteroWave reconstructs ultrasound CT speed maps from incomplete angular
measurements. The useful measured result so far is specific and positive:
latent sector fusion improves or matches a strong mask-aware FBP U-Net baseline
on distributed sparse-view cases, including settings where up to 8 of 16 angular
sectors are missing.

This repository uses a controlled straight-ray Radon proxy built from
OpenBreastUS-style speed maps. It is not a full-wave acoustic solver and does
not claim clinical validation.

## Start Here

| Item | Location |
| --- | --- |
| Curated public summary folder | https://drive.google.com/drive/folders/1i5QF6GbeHMcSdbbVv-v8ZqANvEVkmfNv |
| Public summary document | https://docs.google.com/document/d/1-J8gkt-wqqP1d7brSKGs8eFXAS43OS9zbZZC6yQCKyU |
| Agent handoff, full paths, pending runs | [AGENTS.md](AGENTS.md) |
| README figures | [docs/assets](docs/assets) |
| Safest current implementation | `heterowave-v2` branch |
| Best average experiment | `experiment/tempered-precision-pooling` branch |

The Drive folder contains presentation-safe files only: summary prose, plots,
qualitative grids, CSV metrics, and the workflow GIF. It does not include raw
caches, logs, checkpoints, or model weights.

![Workflow overview](docs/assets/workflow_overview.gif)

## Main Result

Lower NRMSE is better. The fair baseline is a masked FBP U-Net that receives the
same incomplete-acquisition information as HeteroWave.

![Average NRMSE chart](docs/assets/model_average_nrmse.png)

| Model | Validation avg NRMSE | Test avg NRMSE | Result |
| --- | ---: | ---: | --- |
| Masked FBP U-Net | 0.35220 | 0.38142 | strong fair baseline |
| HeteroWave v2 | 0.34375 | 0.38075 | safest HeteroWave model |
| HeteroWave v3 | 0.34334 | 0.38116 | best random/full sparse-view behavior |
| Tempered precision pooling | 0.34304 | 0.38068 | best average result so far |

## Where It Wins

The scenario chart shows NRMSE improvement over the masked FBP U-Net baseline.
Positive bars are better.

![Scenario improvement chart](docs/assets/scenario_test_nrmse.png)

Measured pattern:

- HeteroWave v2 and v3 improve the full, 12-sector, and random sparse-view
  settings.
- HeteroWave v3 is strongest on the distributed sparse-view cases.
- Contiguous limited-angle wedges remain the hard bottleneck.

This is why the current claim is about distributed missing sectors, not global
state of the art.

## Core Idea

HeteroWave adds acquisition-aware latent representation fusion to a
physics-informed FBP image prior.

```text
observed sinogram
  -> sector masks
  -> masked FBP prior
  -> per-sector backprojections
  -> shared sector encoder
  -> latent sector representations
  -> mask/geometry-aware pooling
  -> fused image decoder
  -> reconstructed speed map
```

Why that matters:

- The FBP prior gives the network a reconstruction that respects the measurement
  geometry.
- Each observed angular sector becomes a learned latent representation.
- The pooled latent representation lets the model reason over a variable set of
  observed sectors instead of treating the corrupted image alone as input.
- Zero-gated fusion in v2 lets the model start from the known-good masked U-Net
  behavior and learn only the sector-latent correction that helps.

## Main Discoveries

| Checkpoint | Measured discovery |
| --- | --- |
| Masked FBP U-Net | A mask-aware U-Net is a much stronger and fairer baseline than a plain image U-Net. |
| HeteroWave v1 | Sector aggregation alone was not enough; it lost to the fair baseline. |
| HeteroWave v2 | Masked-FBP trunk plus zero-gated latent sector fusion produced the safest positive result. |
| HeteroWave v3 | Precision weighting and mask geometry helped random sparse-view cases but hurt wedges. |
| Tempered precision pooling | Reliability pooling gave the best average result so far, by a small margin. |

## Checkpoints And Results

Use `best.pt` for evaluation. Use `last.pt` only to resume the same architecture
and config. Do not use `--resume` across architecture changes; use
`--initialize-from` for partial warm starts.

| Artifact | Drive path |
| --- | --- |
| Masked FBP U-Net | `/content/drive/MyDrive/heterowave/results/masked_fbp_unet` |
| HeteroWave v2 checkpoint | `/content/drive/MyDrive/heterowave/results/heterowave_v2_fusion` |
| HeteroWave v2 validation/test | `/content/drive/MyDrive/heterowave/results/heterowave_v2_validation` and `/content/drive/MyDrive/heterowave/results/heterowave_v2_test` |
| HeteroWave v3 checkpoint | `/content/drive/MyDrive/heterowave/results/heterowave_v3_precision` |
| HeteroWave v3 validation/test | `/content/drive/MyDrive/heterowave/results/heterowave_v3_validation` and `/content/drive/MyDrive/heterowave/results/heterowave_v3_test` |
| Tempered precision pooling | `/content/drive/MyDrive/heterowave/results/heterowave_tempered_precision` |
| Tempered validation/test | `/content/drive/MyDrive/heterowave/results/heterowave_tempered_precision_validation` and `/content/drive/MyDrive/heterowave/results/heterowave_tempered_precision_test` |
| External baselines | `/content/drive/MyDrive/heterowave/results/fbpconvnet_sparse` and `/content/drive/MyDrive/heterowave/results/learned_primal_dual` |

More branch and pending-run detail is in [AGENTS.md](AGENTS.md).

## Why It Is Pertinent

Sparse and limited-angle acquisition is a practical problem for water-coupled
ultrasonic CT: transducer dropout, missing views, calibration differences, and
scan geometry can all reduce usable angular coverage.

The research direction is useful because it combines three constraints that
matter in inverse imaging:

- keep a physics-informed FBP prior;
- preserve what was actually measured;
- make acquisition geometry part of the learned representation.

The next most promising step is an observed-preserving sinogram completion
branch:

```python
completed = observed_mask * observed_sinogram + missing_mask * predicted_missing
```

That branch should target the remaining wedge failure mode without overwriting
observed measurements.

## Branch Map

| Branch | Purpose |
| --- | --- |
| `main` | public README and early code |
| `heterowave-v2` | masked-FBP trunk plus sector-wise latent fusion |
| `heterowave-v3` | precision weighting and mask geometry |
| `experiment/tempered-precision-pooling` | best average measured result |
| `experiment/set-transformer-sector-pooling` | set-transformer sector aggregation |
| `experiment/fbpconvnet-baseline` | FBPConvNet-style baseline |
| `experiment/learned-primal-dual-baseline` | learned primal-dual baseline |
| `phase7-v1` | physics consistency and uncertainty experiments |
