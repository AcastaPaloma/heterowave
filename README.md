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

The local configuration is deliberately tiny and uses FP32, `num_workers: 0`,
and `torch.compile: false`. It generates synthetic data in memory and performs
no dataset download or preprocessing.

