# HeteroWave

HeteroWave is a research prototype for sparse-view speed-of-sound reconstruction
from breast ultrasound tomography style data. The current branch,
`heterowave-v2`, evaluates whether sector-aware reconstruction features improve
over a strong mask-aware FBP U-Net baseline.

The physics module uses a differentiable straight-ray parallel-beam Radon
approximation implemented in pure PyTorch. It is not a full-wave acoustic
propagation simulator.

## Current status

The strongest model in this repository is HeteroWave v2. It starts from the
trained masked FBP U-Net baseline and adds a gated, permutation-invariant sector
fusion branch. The initial v2 prediction is exactly the masked U-Net prediction
when loaded from the masked U-Net checkpoint; any improvement is therefore
attributable to the learned sector branch.

On validation, HeteroWave v2 improves over the masked FBP U-Net on all tested
missing-view scenarios. On the held-out test split, v2 is approximately tied
with a small positive average improvement. The gain is consistent on random
sector masks and full/near-full observations, while contiguous missing wedges
remain the main failure case.

## Implemented components

- Robust MATLAB ingestion for conventional SciPy `.mat` files and HDF5-backed
  MATLAB files through `scipy.io` and `h5py`.
- Dataset inspection utilities that print candidate arrays and shapes before
  cache generation.
- Deterministic train/validation splitting with seed `1337`.
- Restart-safe memory-mapped cache generation at `128 x 128`.
- Speed-map normalization and slowness-contrast target preparation.
- GPU-batched complete-sinogram generation using the PyTorch projector.
- Cache metadata recording normalization constants and projector settings.
- Dataset visualization commands for cache sanity checks.
- Complete-view FBP and FBP U-Net baselines.
- Fair masked FBP U-Net baseline using the same missing-sector curriculum as
  HeteroWave.
- HeteroWave v1 sector aggregation model.
- Phase 7 observed-angle data consistency and heteroscedastic uncertainty.
- HeteroWave v2 gated sector-fusion model.
- Deterministic validation/test evaluation suite with scenario-level metrics,
  plots, qualitative grids, config snapshots, and checkpoint provenance.
- Synthetic MATLAB fixture tests, training/evaluation smoke tests, and
  architecture-specific regression tests.

## Main results

Metrics below are normalized RMSE. Lower is better.

### Validation split

720 validation samples, 8 fixed seed-1337 missing-sector scenarios.

| Scenario | Masked FBP U-Net | HeteroWave v2 | Relative change |
|---|---:|---:|---:|
| all_16 | 0.10591 | 0.10271 | 3.0% better |
| observed_12 | 0.16608 | 0.16382 | 1.4% better |
| random_8 | 0.25548 | 0.25107 | 1.7% better |
| random_4 | 0.41459 | 0.40991 | 1.1% better |
| random_2 | 0.54952 | 0.53958 | 1.8% better |
| contiguous_8 | 0.28206 | 0.27192 | 3.6% better |
| contiguous_4 | 0.46512 | 0.44889 | 3.5% better |
| contiguous_2 | 0.57889 | 0.56220 | 2.9% better |

Average validation NRMSE:

| Model | All scenarios | Missing-view scenarios only |
|---|---:|---:|
| Masked FBP U-Net | 0.35220 | 0.38739 |
| HeteroWave v2 | 0.34375 | 0.37817 |

### Held-out test split

800 test samples, same 8 missing-sector scenarios.

| Scenario | Masked FBP U-Net | HeteroWave v2 | Result |
|---|---:|---:|---|
| all_16 | 0.10998 | 0.10735 | v2 better |
| observed_12 | 0.17386 | 0.17222 | v2 better |
| random_8 | 0.26564 | 0.26344 | v2 better |
| random_4 | 0.45532 | 0.45340 | v2 better |
| random_2 | 0.58913 | 0.58728 | v2 better |
| contiguous_8 | 0.32624 | 0.32893 | masked U-Net slightly better |
| contiguous_4 | 0.51462 | 0.51649 | masked U-Net slightly better |
| contiguous_2 | 0.61657 | 0.61688 | effectively tied |

Average test NRMSE:

| Model | All scenarios | Missing-view scenarios only |
|---|---:|---:|
| Masked FBP U-Net | 0.38142 | 0.42020 |
| HeteroWave v2 | 0.38075 | 0.41981 |

The test-set difference is small. The correct interpretation is that v2
matches or slightly improves the fair masked U-Net baseline overall, with the
clearest gains on random missing-sector patterns and the weakest performance on
contiguous missing wedges.

### Runtime and memory

HeteroWave v2 is more expensive than the masked U-Net because it computes
per-sector backprojections and sector encodings.

Representative test-set measurements:

| Model | Inference time | Peak GPU memory |
|---|---:|---:|
| Masked FBP U-Net | about 1.2 ms/sample | about 103 MB |
| HeteroWave v2 | about 5.7 ms/sample | about 251 MB |

The current v2 branch prioritizes controlled research attribution over
deployment efficiency.

## Problem formulation

The repository reconstructs normalized speed targets from sinograms generated
from speed-of-sound maps.

Preprocessing converts speed maps into:

- normalized speed targets;
- slowness contrast maps;
- complete synthetic sinograms using the PyTorch projector;
- memory-mapped train/validation/test cache files;
- `metadata.json` containing normalization and projector settings.

Training and evaluation then apply missing-sector masks to simulate incomplete
angular acquisition.

The default geometry is:

- image size: `128 x 128`;
- projection angles: `64`;
- detector bins: `128`;
- angular sectors: `16`;
- sector width: `4` projection angles.

## Evaluation protocol

The core sparse-view scenarios are:

| Scenario | Observed sectors | Observed fraction | Pattern |
|---|---:|---:|---|
| all_16 | 16 | 1.000 | all sectors observed |
| observed_12 | 12 | 0.750 | random observed sectors |
| random_8 | 8 | 0.500 | random observed sectors |
| random_4 | 4 | 0.250 | random observed sectors |
| random_2 | 2 | 0.125 | random observed sectors |
| contiguous_8 | 8 | 0.500 | contiguous angular wedge |
| contiguous_4 | 4 | 0.250 | contiguous angular wedge |
| contiguous_2 | 2 | 0.125 | contiguous angular wedge |

Fixed validation masks are stored in:

```text
configs/validation_masks_seed1337.json
```

The primary metric is NRMSE. Evaluation also reports MAE, RMSE, PSNR, SSIM,
observed-data residual, inference time, and peak GPU memory.

## Model lineage

### Complete-view FBP

The deterministic baseline reconstructs from complete or masked sinograms using
the differentiable PyTorch backprojector. It is fast and interpretable but not
competitive under sparse views.

### Complete-view FBP U-Net

The first learned baseline receives complete FBP images and learns image-domain
artifact correction. It performs well on complete-view inputs but is not a fair
baseline for sparse-view reconstruction because it is not trained with missing
angular sectors.

### Masked FBP U-Net

The fair sparse-view baseline receives three channels:

1. masked FBP reconstruction;
2. angular coverage map;
3. scalar observed fraction broadcast over the image.

It is trained with the same masking curriculum used for HeteroWave experiments:
random-sector masks, contiguous wedge masks, and periodic masks. This baseline
is strong and is the main comparison point for v2.

### HeteroWave v1

The original HeteroWave model independently backprojects observed angular
sectors, encodes per-sector images, aggregates sector features, and predicts
the reconstruction from the aggregated representation.

Phase 6 showed that v1 underperformed the fair masked FBP U-Net. Phase 7 added
observed-angle data consistency and heteroscedastic uncertainty; it produced
meaningful uncertainty/error correlation but did not close the reconstruction
gap.

### HeteroWave v2

HeteroWave v2 keeps the winning masked FBP U-Net trunk and adds a controlled
sector-fusion path. The design goal is to test whether sector-aware features
add measurable information beyond the masked FBP input.

## HeteroWave v2 architecture

Implementation:

```text
src/heterowave/models/heterowave_v2.py
```

Configuration:

```text
configs/colab_heterowave_v2.yaml
```

### Inputs

V2 receives:

- the original sinogram;
- a Boolean sector mask of shape `[batch, 16]`;
- global masked-FBP features of shape `[batch, 3, 128, 128]`.

The three global channels are:

1. masked FBP image;
2. angular coverage image;
3. observed-fraction image.

### Global trunk

The global trunk is intentionally name-compatible with `FBPUNet`:

```text
encoders
decoders
output
```

This allows direct warm-starting from the trained masked U-Net checkpoint.

Current widths:

```yaml
channels: [16, 32, 64, 96]
```

The trunk is a residual U-Net with bilinear upsampling and skip connections.
The v2 experiment freezes this trunk:

```yaml
freeze_global_trunk: true
```

Freezing makes the ablation conservative: any improvement must come from the
sector branch rather than from additional fine-tuning of the baseline U-Net.

### Sector partial backprojections

For each observed sector, v2 computes a partial backprojection. With 16 sectors
and 64 angles, each sector contains 4 projection angles.

The sector tensor has shape:

```text
[batch, sectors, 1, height, width]
```

Unobserved sectors are masked.

### Geometry channels

When enabled, each sector image is augmented with three sector-geometry
channels. The resulting per-sector input has 4 channels:

```text
partial backprojection + 3 geometry channels
```

This is controlled by:

```yaml
geometry_channels: true
```

### Shared sector encoder

Each sector is passed through the same residual encoder at each scale. Sharing
weights across sectors preserves permutation consistency and prevents the model
from assigning unrelated parameters to individual sector indices.

### Permutation-invariant aggregation

At each scale, sector features are aggregated using:

```yaml
aggregation: mean_var_count
```

The aggregator computes masked set statistics over observed sectors:

- mean feature;
- variance feature;
- observed-sector count feature.

This produces one spatial feature map per U-Net scale, independent of the
ordering of sectors.

### Gated fusion

At each global U-Net encoder scale:

1. global feature map and aggregated sector feature map are concatenated;
2. a `1 x 1` convolution produces a sector update;
3. the update is multiplied by a learned scalar gate;
4. the gated update is added to the global feature map.

The gate is initialized to zero:

```yaml
fusion_gate_init: 0.0
```

The model applies:

```text
hidden = hidden + tanh(gate) * sector_update
```

Therefore, at initialization, the sector branch contributes exactly zero.
With copied masked U-Net trunk weights, v2 starts as the masked FBP U-Net.

This is the central attribution mechanism in v2.

### Output

The current v2 run uses residual output:

```yaml
residual_output: true
```

The model predicts a residual correction added to the masked FBP image. The
current v2 checkpoint does not use uncertainty output:

```yaml
uncertainty: false
```

### Current v2 training setup

The completed v2 experiment used:

```yaml
model:
  name: heterowave_v2
  channels: [16, 32, 64, 96]
  residual_output: true
  aggregation: mean_var_count
  geometry_channels: true
  uncertainty: false
  sector_fusion: true
  fusion_gate_init: 0.0
  freeze_global_trunk: true

optimizer:
  name: adamw
  learning_rate: 0.0003
  weight_decay: 0.0001

loss:
  image_weight: 1.0
  gradient_weight: 0.1
  data_weight: 0.0
  uncertainty_weight: 0.0

training:
  epochs: 15
  checkpoint_metric: nrmse
  checkpoint_mode: min
```

## Reproduction

### Environment

Python 3.11 or 3.12 is supported by the package metadata. A CUDA-enabled PyTorch
install is recommended for training.

Example local setup:

```powershell
conda create -n heterowave python=3.11 git -y
conda activate heterowave
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e ".[dev]"
```

Run tests:

```bash
pytest -q
```

### Inspect MATLAB data

```bash
python scripts/inspect_mat.py /path/to/breast_train_speed.mat /path/to/breast_test_speed.mat
```

The inspector prints candidate MATLAB arrays and shapes. If multiple plausible
arrays exist, pass explicit keys to the cache builder.

### Build the cache

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

If needed:

```bash
python scripts/prepare_cache.py \
  --train-mat /path/to/breast_train_speed.mat \
  --test-mat /path/to/breast_test_speed.mat \
  --train-key TRAIN_ARRAY_KEY \
  --test-key TEST_ARRAY_KEY \
  --output-dir /path/to/cache_128 \
  --image-size 128 \
  --num-angles 64 \
  --batch-size 32 \
  --device cuda
```

Visualize one cached sample:

```bash
python -m heterowave.data.visualize \
  --cache-dir /path/to/cache_128 \
  --output outputs/cached_sample.png \
  --split train \
  --index 0
```

### Train masked FBP U-Net baseline

```bash
python -m heterowave.train --config configs/colab_masked_baseline.yaml
```

Resume:

```bash
python -m heterowave.train \
  --config configs/colab_masked_baseline.yaml \
  --resume /content/drive/MyDrive/heterowave/results/masked_fbp_unet/last.pt
```

### Train HeteroWave v2

```bash
python -m heterowave.train \
  --config configs/colab_heterowave_v2.yaml \
  --initialize-from /content/drive/MyDrive/heterowave/results/masked_fbp_unet/best.pt
```

Resume:

```bash
python -m heterowave.train \
  --config configs/colab_heterowave_v2.yaml \
  --resume /content/drive/MyDrive/heterowave/results/heterowave_v2_fusion/last.pt
```

### Evaluate validation

```bash
python -m heterowave.evaluate \
  --config configs/colab_heterowave_v2.yaml \
  --suite \
  --unet-checkpoint /content/drive/MyDrive/heterowave/results/fbp_unet_baseline/best.pt \
  --masked-unet-checkpoint /content/drive/MyDrive/heterowave/results/masked_fbp_unet/best.pt \
  --heterowave-checkpoint /content/drive/MyDrive/heterowave/results/heterowave_v2_fusion/best.pt \
  --output-dir /content/drive/MyDrive/heterowave/results/heterowave_v2_validation
```

### Evaluate held-out test

```bash
python -m heterowave.evaluate \
  --config configs/colab_heterowave_v2.yaml \
  --suite \
  --split test \
  --unet-checkpoint /content/drive/MyDrive/heterowave/results/fbp_unet_baseline/best.pt \
  --masked-unet-checkpoint /content/drive/MyDrive/heterowave/results/masked_fbp_unet/best.pt \
  --heterowave-checkpoint /content/drive/MyDrive/heterowave/results/heterowave_v2_fusion/best.pt \
  --output-dir /content/drive/MyDrive/heterowave/results/heterowave_v2_test
```

Evaluation writes:

- `metrics_by_scenario.csv`;
- random-mask and wedge-mask robustness plots;
- qualitative reconstruction grid;
- architecture diagram;
- evaluation config JSON/YAML;
- checkpoint provenance;
- copied model checkpoint;
- summary JSON.

## Important artifacts

Expected result directories:

```text
/content/drive/MyDrive/heterowave/results/fbp_unet_baseline
/content/drive/MyDrive/heterowave/results/masked_fbp_unet
/content/drive/MyDrive/heterowave/results/heterowave_mean_var_count
/content/drive/MyDrive/heterowave/results/heterowave_phase7
/content/drive/MyDrive/heterowave/results/heterowave_v2_fusion
/content/drive/MyDrive/heterowave/results/heterowave_v2_validation
/content/drive/MyDrive/heterowave/results/heterowave_v2_test
```

For training continuation, use `last.pt`. For model selection and evaluation,
use `best.pt`.

## Limitations

- The forward model is straight-ray parallel-beam tomography, not full-wave
  ultrasound.
- V2's held-out test improvement over masked FBP U-Net is small.
- V2 is slower and uses more memory than the masked U-Net baseline.
- Contiguous missing-wedge scenarios remain the weakest setting for v2.
- Current evaluation reports aggregate image metrics; local structure-specific
  analysis may reveal differences not captured by whole-image NRMSE.

## Recommended next experiments

1. Unfreeze the global trunk after the sector gates have warmed up.
2. Add observed-angle data consistency to v2 after the fusion branch has proven
   stable.
3. Increase wedge-focused training probability or add a wedge-specific
   validation checkpoint criterion.
4. Evaluate whether a stronger sector-context module helps contiguous missing
   wedges.
5. Add region-specific metrics around high-gradient tissue boundaries and
   lesion-like structures instead of relying only on whole-image averages.

The present branch establishes that sector-aware features can be added to a
strong masked FBP U-Net without degrading the baseline and with small positive
average validation/test gains.
