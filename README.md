# HeteroWave

This repository currently implements Phases 1 and 2 of the HeteroWave plan:
configuration, synthetic speed-of-sound phantoms, and differentiable
parallel-beam projection/backprojection in pure PyTorch.

The forward model is a straight-ray Radon approximation. It is not a full-wave
ultrasound propagation model.

## Local setup

Use Python 3.11. On the GTX 1080, install the CUDA 12.6 PyTorch wheel before the
editable package:

```powershell
conda create -n heterowave python=3.11 git -y
conda activate heterowave
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e ".[dev]"
```

## Phase 1–2 checks

```powershell
pytest -q
python scripts/verify_environment.py --config configs/local_smoke.yaml
python scripts/visualize_physics.py --config configs/local_smoke.yaml
```

## Phase 3 preprocessing

Inspect MATLAB contents before selecting arrays:

```bash
python scripts/inspect_mat.py /path/to/breast_train_speed.mat /path/to/breast_test_speed.mat
```

Build the restart-safe memory-mapped cache using explicit locations:

```bash
python scripts/prepare_cache.py \
  --train-mat /path/to/breast_train_speed.mat \
  --test-mat /path/to/breast_test_speed.mat \
  --output-dir /path/to/cache_128 \
  --image-size 128 --num-angles 64 --batch-size 32 --device cuda
```

If inspection finds multiple plausible arrays, pass `--train-key` and
`--test-key`. If sample layout is ambiguous, also pass the corresponding
`--train-sample-axis` or `--test-sample-axis`.

Visualize one cached sample:

```bash
python -m heterowave.data.visualize \
  --cache-dir /path/to/cache_128 \
  --output outputs/cached_sample.png \
  --split train --index 0
```

## Phase 4 baselines

After copying the cache to fast local storage, measure the complete-sinogram
FBP baseline and train the fixed-input U-Net:

```bash
python -m heterowave.evaluate --config configs/colab_baseline.yaml --baseline fbp --split val
python -m heterowave.train --config configs/colab_baseline.yaml
python -m heterowave.evaluate --config configs/colab_baseline.yaml \
  --baseline unet \
  --checkpoint /content/drive/MyDrive/heterowave/results/fbp_unet_baseline/best.pt \
  --split val
```

Resume an interrupted training runtime with:

```bash
python -m heterowave.train --config configs/colab_baseline.yaml \
  --resume /content/drive/MyDrive/heterowave/results/fbp_unet_baseline/last.pt
```

Phase 4 uses all cached angles. Missing-sector scenarios and acquisition
masking are introduced in Phase 5.

## Phase 5 HeteroWave

The Phase 5 model trains with mixed random-sector, contiguous-wedge, and
periodic masks. Fixed seed-1337 validation masks are stored in
`configs/validation_masks_seed1337.json`.

Run the tiny local smoke configuration:

```bash
python -m heterowave.train --config configs/local_heterowave_smoke.yaml
```

Train on the Colab-local cache while persisting checkpoints to Drive:

```bash
python -m heterowave.train --config configs/colab_heterowave.yaml
```

Resume after a runtime interruption:

```bash
python -m heterowave.train \
  --config configs/colab_heterowave.yaml \
  --resume /content/drive/MyDrive/heterowave/results/heterowave_mean_var_count/last.pt
```

Regenerate the committed fixed masks when the validation protocol changes:

```bash
python scripts/generate_validation_masks.py \
  --output configs/validation_masks_seed1337.json \
  --seed 1337
```

## Phase 6 deterministic evaluation

Run the full validation suite with the two best learned checkpoints:

```bash
python -m heterowave.evaluate \
  --config configs/research_benchmark.yaml \
  --suite \
  --unet-checkpoint /content/drive/MyDrive/heterowave/results/fbp_unet_baseline/best.pt \
  --heterowave-checkpoint /content/drive/MyDrive/heterowave/results/heterowave_mean_var_count/best.pt \
  --split val
```

This writes `metrics_by_scenario.csv`, random/wedge robustness plots, a
qualitative grid, architecture diagram, configuration, checkpoint provenance,
and a copy of the selected HeteroWave checkpoint. Use `--split test` only for
the final frozen evaluation. `--max-samples N` is available for smoke tests.

The local configuration is deliberately tiny and uses FP32, `num_workers: 0`,
and `torch.compile: false`. It generates synthetic data in memory and performs
no dataset download or preprocessing.
