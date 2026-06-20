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

The local configuration is deliberately tiny and uses FP32, `num_workers: 0`,
and `torch.compile: false`. It generates synthetic data in memory and performs
no dataset download or preprocessing.
