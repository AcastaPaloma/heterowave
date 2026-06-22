# HeteroWave

Acquisition-aware sparse-view reconstruction for water-coupled ultrasonic CT.

HeteroWave is a research reconstruction stack for recovering 2D ultrasound CT
speed maps from incomplete angular acquisitions. The project combines a
physics-informed filtered-backprojection prior with learned latent-space
representations of observed angular sectors, then evaluates robustness under
fixed sparse-view and limited-angle masks.

The forward model in this repository is a straight-ray Radon approximation, not
a full-wave acoustic simulator. Results should be read as controlled sparse-view
reconstruction experiments, not clinical validation.

## Public visual summary

A curated Drive package is available here:

https://drive.google.com/drive/folders/1i5QF6GbeHMcSdbbVv-v8ZqANvEVkmfNv

It contains only presentation-safe artifacts: a short project summary,
architecture diagram, workflow overview GIF, qualitative reconstruction grids,
random/wedge robustness plots, and scenario-level metrics CSVs. Raw cache files,
checkpoints, unfinished logs, and internal experiment clutter are intentionally
excluded.

Suggested visual walkthrough:

```text
speed map -> complete sinogram -> missing-sector mask
          -> masked FBP prior
          -> sector-wise latent representations
          -> acquisition-aware fusion
          -> reconstructed speed map
```

The curated folder includes `09_workflow_overview.gif`, built from a cached
sample to show this sequence:

```text
target speed map
  -> complete sinogram
  -> remove angular sectors
  -> masked FBP baseline
  -> HeteroWave reconstruction
  -> absolute-error map
```

## What was built

The project includes:

- robust OpenBreastUS-style MATLAB loading and inspection;
- restart-safe memory-mapped cache generation at `128 x 128`;
- normalized speed-map targets and slowness-contrast inputs;
- pure PyTorch projection and backprojection operators;
- deterministic train/validation/test sparse-view evaluation masks;
- complete-view FBP and U-Net baselines;
- a fair masked FBP U-Net baseline with coverage channels;
- HeteroWave sector-fusion models;
- deterministic evaluation artifacts: scenario metrics, robustness plots,
  qualitative grids, architecture diagrams, and provenance files.

## Core idea

The main architectural idea is not simply to train a larger U-Net.

The strongest HeteroWave variants keep a physically meaningful masked-FBP image
prior, then add sector-wise latent-space reasoning. Each observed angular sector
is backprojected and encoded into a learned representation. Those
representations are pooled with acquisition-mask awareness and fused into the
decoder, so the model reconstructs using both image evidence and the geometry of
what was actually observed.

In short: the model reasons over the acquisition pattern, not just the corrupted
image.

## Most presentable result

The fair masked FBP U-Net baseline is very strong. It receives a masked FBP
image plus angular-coverage and observed-fraction channels, and it remains the
baseline every HeteroWave variant should be compared against.

Measured results so far:

| Model | Validation avg NRMSE | Test avg NRMSE | Interpretation |
| --- | ---: | ---: | --- |
| Masked FBP U-Net | 0.35220 | 0.38142 | strong fair baseline |
| HeteroWave v2 | 0.34375 | 0.38075 | safest overall HeteroWave model |
| HeteroWave v3 | 0.34334 | 0.38116 | improves random/full views, regresses wedges |
| Tempered precision pooling | 0.34304 | 0.38068 | currently best average by a very small margin |

The cleanest claim is:

> HeteroWave reconstructs ultrasound CT speed maps from incomplete acquisitions,
> including cases where up to 8 of 16 angular sectors are absent, and improves
> over a strong mask-aware FBP U-Net baseline on distributed sparse-view cases.

The honest caveat is equally important:

> Contiguous limited-angle wedges remain the hard failure mode. Distributed
> missing sectors are recoverable with latent sector fusion; contiguous missing
> wedges expose a real inverse-problem null space.

That bottleneck motivates the next architecture direction: observed-preserving
sinogram completion and data-consistency refinement, rather than another generic
image-domain network.

## Branch map

`main` is currently the clean public landing branch. The main experimental
implementations live on topic branches:

| Branch | Purpose |
| --- | --- |
| `heterowave-v2` | masked-FBP trunk plus sector-wise latent fusion |
| `heterowave-v3` | learned precision weighting, mask geometry conditioning, acquisition perturbation hooks |
| `experiment/tempered-precision-pooling` | tempered reliability pooling; best average result so far by a small margin |
| `experiment/set-transformer-sector-pooling` | set-transformer sector aggregation experiment |
| `experiment/fbpconvnet-baseline` | FBPConvNet-style external baseline |
| `experiment/learned-primal-dual-baseline` | compact unrolled learned primal-dual baseline |
| `phase7-v1` | physics-consistency and uncertainty experiments |

Use `best.pt` checkpoints for evaluation. Use `last.pt` only to resume the same
architecture/configuration. Do not use `--resume` across architecture changes;
use `--initialize-from` for warm starts when model keys/shapes partially match.

## Setup

Use Python 3.11. On a local CUDA 12.6 system:

```powershell
conda create -n heterowave python=3.11 git -y
conda activate heterowave
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e ".[dev]"
```

Run tests:

```powershell
pytest -q
```

## Preprocessing

Inspect MATLAB contents before selecting arrays:

```bash
python scripts/inspect_mat.py /path/to/breast_train_speed.mat /path/to/breast_test_speed.mat
```

Build a restart-safe memory-mapped cache using explicit paths:

```bash
python scripts/prepare_cache.py \
  --train-mat /path/to/breast_train_speed.mat \
  --test-mat /path/to/breast_test_speed.mat \
  --output-dir /path/to/cache_128 \
  --image-size 128 \
  --num-angles 64 \
  --batch-size 32 \
  --device cuda
```

Visualize a cached sample:

```bash
python -m heterowave.data.visualize \
  --cache-dir /path/to/cache_128 \
  --output outputs/cached_sample.png \
  --split train \
  --index 0
```

## Evaluation protocol

The deterministic evaluation suite uses fixed seed-1337 masks:

| Scenario | Observed sectors | Pattern |
| --- | ---: | --- |
| `all_16` | 16 | all sectors observed |
| `observed_12` | 12 | random observed sectors |
| `random_8` | 8 | random observed sectors |
| `random_4` | 4 | random observed sectors |
| `random_2` | 2 | random observed sectors |
| `contiguous_8` | 8 | contiguous angular wedge |
| `contiguous_4` | 4 | contiguous angular wedge |
| `contiguous_2` | 2 | contiguous angular wedge |

Evaluation writes:

- `metrics_by_scenario.csv`;
- `robustness_random.png`;
- `robustness_wedge.png`;
- `qualitative_grid.png`;
- `architecture.png`;
- config and checkpoint provenance files.

Use `--split test` only for frozen final evaluation after validation has already
identified the model to report.

## Scientific framing

Use this language:

- water-coupled ultrasonic CT;
- sparse-view / limited-angle reconstruction;
- physics-aware FBP prior;
- acquisition-aware latent representation fusion;
- missing-sector robustness;
- calibration/timing/channel robustness as future acquisition perturbations.

Avoid this language:

- clinical validation;
- full-wave ultrasound inversion;
- X-ray CT;
- water “noise” as if water corrupts ultrasound;
- global SOTA claims without matched task/data/protocol comparisons.

The strongest current story is measured and credible: a simple fair baseline is
hard to beat, HeteroWave improves distributed sparse-view reconstruction by
reasoning over latent sector representations, and the remaining wedge failure
mode identifies a real inverse-problem bottleneck worth attacking next.
