# HeteroWave: Codex Architecture and Implementation Plan

## 0. Mission

Build a weekend-scale research prototype for **acquisition-subset robust tomographic reconstruction**.

The model receives an arbitrary subset of tomographic angular sectors and reconstructs a quantitative 2D speed-of-sound map. It should degrade gracefully when measurements are missing and should expose uncertainty where reconstruction is unreliable.

This is inspired by HeMIS, but the "modalities" are acquisition sectors rather than MRI sequences.

### Research question

Can a single geometry-aware, permutation-invariant reconstruction model handle arbitrary and previously unseen acquisition subsets better than:

1. filtered backprojection;
2. a fixed-input FBP + U-Net baseline;
3. a mean-only HeMIS-style model?

### Weekend claim

Do not claim sparse-view tomography is novel.

Use this claim:

> We investigate whether HeMIS-style latent statistics can turn tomographic reconstruction into a variable-set inverse problem, allowing one geometry-conditioned model to reconstruct from arbitrary angular subsets and unseen sensor-failure patterns.

---

# 1. Constraints and priorities

## Hardware

- Primary training: Google Colab GPU.
- Local smoke testing: Windows PC with GTX 1080.
- Local CPU is slow; local workflows must not require full preprocessing.
- Avoid H100-only kernels, FlashAttention requirements, custom CUDA compilation, or large CPU-bound preprocessing.
- `torch.compile` must be disabled by default on the GTX 1080.

## Weekend priorities

1. Correct physics simulator and visual sanity checks.
2. Reproducible data preprocessing and caching.
3. FBP baseline.
4. Fixed-input U-Net baseline.
5. HeteroWave mean-only model.
6. HeteroWave mean + variance model.
7. Robustness evaluation.
8. Physics-consistency loss.
9. Uncertainty head.
10. Autoresearch loop only after the normal training pipeline is stable.

Never sacrifice the baselines and evaluation for a more exotic architecture.

---

# 2. System overview

```mermaid
flowchart LR
    X[Ground-truth speed map c(x)] --> S[Convert to slowness contrast]
    S --> A[Parallel-beam forward projector]
    A --> Y[Complete sinogram]
    Y --> M[Random angular-sector mask]
    M --> YO[Observed sinogram]
    YO --> BP[Per-sector partial backprojection]
    BP --> E[Shared sector encoder]
    E --> AGG[Masked mean + variance aggregation at each scale]
    AGG --> D[U-Net decoder]
    D --> P[Predicted speed map]
    D --> U[Predicted uncertainty]
    P --> L1[Image and edge losses]
    P --> DC[Observed-angle data-consistency loss]
```

## Mathematical formulation

Let:

- `c(x)` be the ground-truth speed-of-sound map.
- `s(x) = 1 / c(x)` be slowness.
- `s_water = 1 / c_water`.
- `delta_s(x) = s(x) - s_water` be slowness contrast.
- `A` be a parallel-beam Radon-style forward operator.
- `y = A(delta_s)` be the complete sinogram.
- `M` be an angular-sector observation mask.
- `y_obs = M ⊙ y`.

The model predicts:

\[
(\hat c, \log \hat \sigma^2) = f_\theta(y_{\mathrm{obs}}, M, G)
\]

where `G` contains angular geometry.

The model is an approximate inverse operator conditioned on the observed acquisition subset.

---

# 3. Dataset

## Primary dataset

Use the OpenBreastUS speed maps:

- `breast_train_speed.mat`
- `breast_test_speed.mat`

Do not download or generate the full multi-terabyte wavefield dataset.

## Expected preprocessing

1. Read the `.mat` arrays.
2. Detect array layout and convert to `[N, H, W]`.
3. Remove NaNs and invalid values.
4. Resize maps to `128 × 128` for main training.
5. Split the official training collection into train and validation sets with a fixed seed.
6. Preserve the official test collection for final testing.
7. Store dataset statistics in `metadata.json`.
8. Generate complete 64-angle sinograms using the GPU projector.
9. Cache targets and sinograms.

## Cache layout

```text
data/cache_128/
├── train_targets.npy       # float32 or float16 [N, 128, 128]
├── train_sinograms.npy     # float16 [N, 64, 128]
├── val_targets.npy
├── val_sinograms.npy
├── test_targets.npy
├── test_sinograms.npy
└── metadata.json
```

Use `.npy` arrays so they can be memory-mapped. Avoid thousands of tiny files.

## Loader robustness

Implement `.mat` loading with both:

- `scipy.io.loadmat` for classic MATLAB files;
- `h5py.File` for MATLAB v7.3/HDF5 files.

When the variable name is unknown, print all numeric arrays and their shapes. Select the array only after validating that it contains plausible spatial maps. Do not silently select the largest array without logging the choice.

## Local development data

Create a synthetic phantom generator for unit tests:

- circles;
- ellipses;
- rectangles;
- multiple sound-speed regions;
- optional Gaussian noise.

Local smoke tests must run without downloading OpenBreastUS.

---

# 4. Physics implementation

## 4.1 Parallel-beam forward projector

Implement a differentiable Radon-style operator using only PyTorch:

1. Rotate an image for each projection angle using `affine_grid` and `grid_sample`.
2. Sum along one spatial dimension.
3. Return `[B, A, D]`, where:
   - `B` = batch;
   - `A` = number of angles;
   - `D` = detector bins.

Use 64 angles over `[0, 180)` for the main model.

Cache affine sampling grids by:

- image size;
- angle count;
- device;
- dtype.

The forward projector must support batches and gradients.

## 4.2 Backprojector

For each sinogram projection:

1. Expand the detector vector across one image dimension.
2. Rotate it back by the corresponding angle.
3. Sum over angles.
4. Apply a normalization factor.

Implement:

- unfiltered backprojection;
- filtered backprojection with a ramp filter in detector-frequency space.

## 4.3 Validation tests

Required tests:

1. Constant image produces expected smooth projections.
2. A centered disk produces a symmetric sinogram.
3. Backprojection output has the correct shape.
4. FBP from all 64 angles roughly reconstructs a simple disk.
5. Gradients propagate through the forward projector.
6. Gradients propagate through the backprojector.
7. CPU and GPU outputs are numerically close on tiny inputs.

Do not continue to the neural architecture until these tests and visualizations work.

---

# 5. Acquisition masking

Divide 64 angles into 16 sectors, with 4 angles per sector.

For every training example, sample one of the following mask types:

## Random-sector mask

- Sample `K` uniformly or curriculum-weighted from `[2, 16]`.
- Select `K` sectors without replacement.

## Contiguous-wedge mask

- Sample a wedge length.
- Remove one contiguous block of sectors, including wraparound.

## Mixed training distribution

Default:

- 50% random-sector masks;
- 35% contiguous missing wedges;
- 15% structured alternating or periodic masks.

The model must receive the mask explicitly.

## Fixed validation scenarios

Use deterministic masks for comparable experiments:

- all 16 sectors;
- 12 sectors;
- 8 sectors random;
- 8 sectors contiguous;
- 4 sectors random;
- 4 sectors contiguous;
- 2 sectors random;
- 2 sectors contiguous.

Store validation masks in a file generated from a fixed seed.

---

# 6. Model architectures

## 6.1 Baseline A: Filtered backprojection

Input:

- zero-filled observed sinogram.

Output:

- FBP image.

No neural network.

## 6.2 Baseline B: FBP + U-Net

Input channels:

1. zero-filled FBP reconstruction;
2. backprojected angular coverage map;
3. normalized observed-sector count.

Architecture:

- small 2D U-Net;
- channels `[16, 32, 64, 96]`;
- GroupNorm;
- SiLU;
- residual convolution blocks;
- output one reconstructed normalized speed map.

This baseline tests whether a conventional fixed-input model is already sufficient.

## 6.3 HeteroWave

### Step 1: partial backprojection per sector

For every sector `i`:

\[
b_i = B_i(y_i)
\]

where only the angles belonging to sector `i` are backprojected.

Produce:

```text
partial_bp: [B, S, 1, H, W]
sector_mask: [B, S]
```

with `S = 16`.

### Step 2: geometry channels

For sector center angle `theta_i`, provide:

- `sin(theta_i)`;
- `cos(theta_i)`;
- normalized sector width.

Broadcast them spatially and concatenate with the partial backprojection.

Per-sector input:

```text
[partial_backprojection, sin_theta, cos_theta, sector_width]
```

### Step 3: shared multiscale encoder

Apply the same encoder weights to every sector.

Recommended channels:

```text
level 0: 16
level 1: 32
level 2: 64
bottleneck: 96
```

Use:

- residual convolution blocks;
- GroupNorm;
- SiLU;
- strided convolutions for downsampling.

Reshape `[B, S, C, H, W]` to `[B*S, C, H, W]` during shared encoding.

### Step 4: masked HeMIS statistics at every scale

For encoded features `z_i` and binary mask `m_i`:

\[
\mu = \frac{\sum_i m_i z_i}{\sum_i m_i}
\]

\[
v = \frac{\sum_i m_i (z_i-\mu)^2}{\sum_i m_i}
\]

Use population variance, not unbiased sample variance.

Clamp variance to a small nonnegative minimum.

At each scale, concatenate:

```text
mean features
variance features
normalized observed-sector count
```

Then reduce channels with a `1 × 1` convolution.

The aggregation must be permutation-invariant.

### Step 5: U-Net decoder

Decode from the aggregated bottleneck and aggregated skip features.

Output two channels:

1. predicted normalized speed map mean;
2. predicted log variance.

Clamp log variance to a stable interval, initially `[-8, 3]`.

## 6.4 Ablations

Implement config switches:

- `aggregation: mean`
- `aggregation: mean_var`
- `aggregation: mean_var_count`
- `geometry_channels: true/false`
- `physics_loss.enabled: true/false`
- `uncertainty.enabled: true/false`

Do not build attention, Mamba, FNO, or transformers before the core ablations work.

---

# 7. Loss functions

## 7.1 Reconstruction loss

Use Charbonnier or Smooth L1:

\[
L_{\mathrm{image}} = \operatorname{SmoothL1}(\hat c, c)
\]

## 7.2 Gradient loss

\[
L_{\mathrm{grad}}
=
\|\nabla_x \hat c-\nabla_x c\|_1
+
\|\nabla_y \hat c-\nabla_y c\|_1
\]

Use finite differences.

## 7.3 Data-consistency loss

Convert predicted speed back to slowness contrast and forward project:

\[
L_{\mathrm{data}}
=
\|M \odot (A(\hat{\delta_s}) - y)\|_1
\]

Initially compute this:

- on a subset of observed angles; or
- every `N` training steps;

to reduce cost.

## 7.4 Uncertainty loss

When uncertainty is enabled, use heteroscedastic Gaussian NLL:

\[
L_{\mathrm{nll}}
=
\frac{1}{2}\exp(-\log \sigma^2)(\hat c-c)^2
+
\frac{1}{2}\log \sigma^2
\]

Start without uncertainty. Add it only after mean reconstruction works.

## 7.5 Default total loss

```text
loss =
    1.00 * image_loss
  + 0.10 * gradient_loss
  + 0.05 * data_consistency_loss
  + 0.10 * uncertainty_nll
```

All weights must be configurable.

---

# 8. Metrics

Report per missingness scenario:

- MAE;
- RMSE;
- normalized RMSE;
- PSNR;
- SSIM;
- observed-angle data residual;
- inference time;
- peak GPU memory.

For uncertainty:

- Spearman correlation between predicted variance and absolute error;
- error in the highest-uncertainty quartile versus lowest quartile;
- sparsification curve if time permits.

Primary robustness plot:

```text
normalized RMSE vs percentage of observed sectors
```

Plot separate curves for:

- random missing sectors;
- contiguous missing wedge.

---

# 9. Repository structure

```text
heterowave/
├── README.md
├── AGENTS.md
├── pyproject.toml
├── environment.yml
├── .gitignore
├── configs/
│   ├── local_smoke.yaml
│   ├── colab_baseline.yaml
│   ├── colab_heterowave.yaml
│   ├── colab_full.yaml
│   └── research_benchmark.yaml
├── notebooks/
│   ├── colab_setup.ipynb
│   └── visualize_physics.ipynb
├── scripts/
│   ├── download_openbreastus.py
│   ├── prepare_cache.py
│   ├── sync_results_to_drive.py
│   └── verify_environment.py
├── src/heterowave/
│   ├── __init__.py
│   ├── data/
│   │   ├── mat_loader.py
│   │   ├── dataset.py
│   │   ├── masks.py
│   │   └── phantoms.py
│   ├── physics/
│   │   ├── projector.py
│   │   ├── backprojector.py
│   │   └── filters.py
│   ├── models/
│   │   ├── blocks.py
│   │   ├── unet_baseline.py
│   │   ├── set_stats.py
│   │   └── heterowave.py
│   ├── losses.py
│   ├── metrics.py
│   ├── train.py
│   ├── evaluate.py
│   ├── visualize.py
│   └── utils.py
├── autoresearch/
│   ├── README.md
│   ├── program.md
│   ├── loop.py
│   ├── runner.py
│   ├── candidate.py
│   ├── results.tsv
│   └── allowlist.json
├── tests/
│   ├── test_projector.py
│   ├── test_backprojector.py
│   ├── test_masks.py
│   ├── test_set_stats.py
│   ├── test_model_shapes.py
│   └── test_training_step.py
└── outputs/
```

---

# 10. Local Windows setup

## 10.1 Create the environment

Run in Anaconda Prompt:

```powershell
conda create -n heterowave python=3.11 git -y
conda activate heterowave

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
```

Then, from the repository root:

```powershell
python -m pip install -e ".[dev]"
```

## 10.2 Verify the GTX 1080

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print(torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None); print(torch.cuda.get_arch_list() if torch.cuda.is_available() else None)"
```

Expected device capability for GTX 1080:

```text
(6, 1)
```

If CUDA is unavailable:

1. update the NVIDIA driver;
2. ensure the active Conda environment contains only one PyTorch installation;
3. rerun the verification command;
4. do not install a separate full CUDA toolkit unless another compiled dependency requires it.

## 10.3 Local config

Use:

```yaml
device: cuda
precision: fp32
compile: false
image_size: 32
num_angles: 16
num_sectors: 4
batch_size: 2
num_workers: 0
train_samples: 16
val_samples: 8
max_steps: 20
dataset: synthetic
```

Pascal GPUs have weak FP16 training behavior compared with newer GPUs. Prefer FP32 locally unless a specific AMP smoke test is stable.

## 10.4 Local commands

```powershell
pytest -q
python -m heterowave.visualize --config configs/local_smoke.yaml
python -m heterowave.train --config configs/local_smoke.yaml
```

Local acceptance criterion:

- all tests pass;
- one forward/backward training step runs;
- FBP reconstructs a synthetic disk;
- loss decreases on a tiny overfit set.

Do not preprocess the complete dataset locally.

---

# 11. Google Colab in VS Code

## 11.1 Install extensions

Install:

- Google Colab, publisher Google;
- Jupyter;
- Python;
- Codex.

## 11.2 Connect to Colab

1. Open `notebooks/colab_setup.ipynb`.
2. Click `Select Kernel`.
3. Select `Colab`.
4. Select `Auto Connect` or provision a desired Colab server.
5. Complete Google sign-in.
6. Confirm a GPU is attached:

```python
import torch
print(torch.cuda.get_device_name(0))
```

## 11.3 Mount Drive

Use the command palette:

```text
Colab: Mount Google Drive to Server...
```

Use Drive only for durable storage:

```text
/content/drive/MyDrive/heterowave/
├── raw/
├── cache/
├── checkpoints/
├── results/
└── autoresearch/
```

Do not train directly against many files on Drive.

At session start, copy prepared cache files to fast Colab-local storage:

```bash
mkdir -p /content/heterowave_data
rsync -ah --info=progress2 \
  /content/drive/MyDrive/heterowave/cache/cache_128/ \
  /content/heterowave_data/cache_128/
```

At experiment end, copy only checkpoints, tables, plots, and logs back to Drive.

## 11.4 VS Code server mounting

Enable the Colab extension's experimental:

- Server Mounting;
- Colab Terminal.

Then run:

```text
Colab: Mount Server To Workspace...
```

This allows VS Code and Codex to inspect and edit `/content`.

Important operating rule:

- local Git repository or GitHub is the source of truth;
- `/content` is disposable;
- commit useful remote edits frequently;
- push improvements before the runtime is released;
- refresh the mounted server view when files are modified by commands outside VS Code.

## 11.5 Recommended Codex workflow

Use a multi-root VS Code workspace:

```text
heterowave-local/       # durable local clone, primary editing target
colab-content/          # experimental mounted runtime
```

Preferred cycle:

1. Codex edits the local repository.
2. Commit and push.
3. In Colab terminal:

```bash
cd /content/heterowave
git pull
python -m pip install -e .
```

4. Execute experiments on the Colab kernel.
5. Commit and push useful results or changes.
6. Save checkpoints/results to Drive.

For very rapid edits, Codex may edit the mounted `/content/heterowave` clone directly. Treat this as temporary and immediately commit/push successful changes.

---

# 12. Colab bootstrap notebook

The notebook should contain only orchestration cells, not the project implementation.

## Cell 1: inspect hardware

```python
import os
import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

## Cell 2: mount Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

## Cell 3: clone or update repository

```bash
%%bash
if [ ! -d /content/heterowave/.git ]; then
  git clone YOUR_REPOSITORY_URL /content/heterowave
else
  cd /content/heterowave
  git fetch --all
  git pull --ff-only
fi
```

## Cell 4: install

```bash
%cd /content/heterowave
!python -m pip install --upgrade pip
!python -m pip install -e ".[colab]"
```

Do not reinstall PyTorch unless the repository has a demonstrated incompatibility with Colab's preinstalled version.

## Cell 5: copy cache locally

```bash
!mkdir -p /content/heterowave_data/cache_128
!rsync -ah \
  /content/drive/MyDrive/heterowave/cache/cache_128/ \
  /content/heterowave_data/cache_128/
```

## Cell 6: smoke test

```bash
!pytest -q
!python -m heterowave.train --config configs/local_smoke.yaml
```

## Cell 7: train

```bash
!python -m heterowave.train \
  --config configs/colab_heterowave.yaml \
  data.root=/content/heterowave_data/cache_128 \
  output.root=/content/drive/MyDrive/heterowave/results
```

---

# 13. Download and preprocessing

## 13.1 Download script

Implement `scripts/download_openbreastus.py` using `huggingface_hub.hf_hub_download`.

Download only:

```text
breast_train_speed.mat
breast_test_speed.mat
```

Default destination:

```text
/content/drive/MyDrive/heterowave/raw/openbreastus/
```

CLI:

```bash
python scripts/download_openbreastus.py \
  --output-dir /content/drive/MyDrive/heterowave/raw/openbreastus
```

The script must:

- skip valid existing files;
- use Hugging Face resumable downloads;
- print resulting paths and sizes;
- never download the sample wavefields or OpenWaves installer by default.

## 13.2 Preprocessing command

```bash
python scripts/prepare_cache.py \
  --train-mat /content/drive/MyDrive/heterowave/raw/openbreastus/breast_train_speed.mat \
  --test-mat /content/drive/MyDrive/heterowave/raw/openbreastus/breast_test_speed.mat \
  --output-dir /content/drive/MyDrive/heterowave/cache/cache_128 \
  --image-size 128 \
  --num-angles 64 \
  --batch-size 32 \
  --device cuda \
  --seed 1337
```

The preprocessing script must be restart-safe.

Write temporary files followed by atomic rename.

Save metadata:

```json
{
  "image_size": 128,
  "num_angles": 64,
  "num_sectors": 16,
  "speed_mean": 0.0,
  "speed_std": 1.0,
  "water_speed": 1500.0,
  "split_seed": 1337,
  "projector_version": "..."
}
```

Actual mean/std values must be computed, not hard-coded.

---

# 14. Training configuration

## Recommended first Colab run

```yaml
seed: 1337
device: cuda
precision: auto
compile: false

data:
  root: /content/heterowave_data/cache_128
  image_size: 128
  num_angles: 64
  num_sectors: 16
  batch_size: 8
  num_workers: 2
  pin_memory: true
  persistent_workers: true

model:
  name: heterowave
  channels: [16, 32, 64, 96]
  aggregation: mean_var_count
  geometry_channels: true
  uncertainty: false

masking:
  minimum_sectors: 2
  random_probability: 0.50
  wedge_probability: 0.35
  periodic_probability: 0.15

optimizer:
  name: adamw
  learning_rate: 0.0003
  weight_decay: 0.0001

scheduler:
  name: cosine
  warmup_steps: 200

loss:
  image_weight: 1.0
  gradient_weight: 0.1
  data_weight: 0.0
  uncertainty_weight: 0.0

training:
  epochs: 30
  gradient_clip: 1.0
  validate_every_epochs: 1
  checkpoint_metric: robust_nrmse
  checkpoint_mode: min
```

Turn on data consistency only after the basic model trains:

```yaml
loss:
  data_weight: 0.05
  data_every_n_steps: 4
  data_angle_fraction: 0.5
```

Turn on uncertainty last.

## Precision handling

Autodetect:

- T4: FP16 autocast;
- L4/A100: BF16 if supported;
- GTX 1080 local: FP32 default.

Use `torch.amp.autocast` and `GradScaler` only when appropriate.

---

# 15. Evaluation protocol

## Required models

1. FBP.
2. FBP + U-Net.
3. HeteroWave mean-only.
4. HeteroWave mean + variance + count.
5. HeteroWave full model with data consistency.
6. Full model with uncertainty, if completed.

## Fairness

All learned models must use:

- the same train/validation/test split;
- the same cached sinograms;
- the same deterministic validation masks;
- the same image normalization;
- the same test metrics.

## Final outputs

Generate:

```text
outputs/final/
├── metrics_by_scenario.csv
├── robustness_random.png
├── robustness_wedge.png
├── qualitative_grid.png
├── uncertainty_examples.png
├── architecture.png
├── config.yaml
└── model.safetensors
```

Qualitative grid columns:

1. target;
2. available sectors;
3. FBP;
4. FBP + U-Net;
5. HeteroWave;
6. absolute error;
7. predicted uncertainty.

---

# 16. Autoresearch adaptation

## Important expectation

Karpathy's autoresearch does not make a single training run faster. It improves the rate of research iteration by letting an agent repeatedly modify a constrained training system, run a fixed-budget experiment, and keep only improvements.

The upstream project is specialized to LLM training. Implement the same principles, not a direct copy of its model code.

## 16.1 Toggle

Normal mode:

```bash
AUTORESEARCH_ENABLED=0 python -m heterowave.train \
  --config configs/colab_heterowave.yaml
```

Autoresearch mode:

```bash
AUTORESEARCH_ENABLED=1 python -m autoresearch.loop \
  --max-experiments 12 \
  --budget-minutes 8
```

Autoresearch must be off by default.

## 16.2 Immutable benchmark

The agent must not modify:

- dataset split;
- cached validation data;
- validation masks;
- metric implementation;
- physics implementation;
- trial time budget;
- random seed;
- test files;
- experiment runner;
- result parser.

Create a file allowlist.

Initially permit edits only to:

```text
autoresearch/candidate.py
```

Optionally permit:

```text
autoresearch/candidate.yaml
```

The candidate file should expose:

```python
def build_model(config):
    ...

def configure_optimizer(model, config):
    ...

def configure_loss(config):
    ...
```

## 16.3 Fixed rapid benchmark

Use a smaller research benchmark:

```yaml
image_size: 96
num_angles: 32
num_sectors: 8
train_samples: 1024
val_samples: 256
batch_size: 8
budget_minutes: 8
seed: 1337
```

Validation scenarios:

- random 50% observed;
- wedge 50% observed;
- random 25% observed;
- wedge 25% observed.

## 16.4 Single scalar objective

Lower is better:

```text
robust_score =
    mean(normalized_RMSE over four validation scenarios)
  + 0.05 * normalized_observed_data_residual
```

Print one machine-readable line:

```text
RESEARCH_RESULT {"score": 0.1234, "nrmse": 0.1180, "data_residual": 0.1080, "runtime_seconds": 478}
```

Do not optimize SSIM alone.

## 16.5 Agent loop

Each iteration:

1. Read:
   - `autoresearch/program.md`;
   - current candidate;
   - recent `results.tsv`;
   - best score;
   - last failure.
2. Propose exactly one coherent change.
3. Apply patch only to allowlisted files.
4. Reject patch if it changes non-allowlisted files.
5. Run:
   - formatting;
   - unit tests;
   - model shape test;
   - one-step training test.
6. Run fixed-budget experiment in a subprocess with a hard timeout.
7. Parse `RESEARCH_RESULT`.
8. If score improves by `min_delta`:
   - commit;
   - append result;
   - push to the autoresearch branch.
9. Otherwise:
   - append result;
   - revert candidate file.
10. On crash or OOM:
   - record failure;
   - revert;
   - continue.

## 16.6 Colab persistence

Use:

```text
/content/heterowave                 # fast working clone
/content/drive/MyDrive/heterowave/autoresearch/
├── results.tsv
├── state.json
├── best_candidate.py
└── logs/
```

After every experiment:

- append result to Drive;
- save current best candidate to Drive;
- commit accepted improvements;
- push accepted improvements to branch:
  `autoresearch/colab`.

The loop must resume from `state.json`.

## 16.7 Agent backend

Support two modes:

### Manual Codex mode

- Codex in VS Code edits `candidate.py`.
- User launches trial.
- Lowest setup risk.
- Use this first.

### API autonomous mode

- `autoresearch/loop.py` calls an LLM API to propose patches.
- Read API key from environment or Colab Secrets.
- Never place API tokens in notebooks, repository files, output logs, or prompts.
- Require patch/diff output, not full repository rewrites.
- Limit retries and API spend.

Do not make autonomous API mode a prerequisite for completing the imaging project.

## 16.8 Promotion rule

Short-budget autoresearch results are hypotheses, not final results.

After the loop:

1. take the three best candidates;
2. train each with the main 128 × 128 config;
3. use at least three seeds if time permits;
4. compare against the untouched baseline;
5. report only confirmed improvements.

---

# 17. `program.md` for the research agent

```markdown
You are optimizing HeteroWave, a variable-set tomographic reconstruction model.

Your goal is to minimize `robust_score` under a fixed wall-clock training budget.

You may edit only `autoresearch/candidate.py`.

You may change:
- encoder and decoder blocks;
- channel widths;
- aggregation implementation;
- normalization;
- activation functions;
- residual connections;
- optimizer configuration;
- learning rate;
- loss weights;
- regularization.

You may not change:
- dataset;
- train/validation split;
- validation masks;
- metric;
- fixed seed;
- trial budget;
- physics operator;
- data loader;
- evaluator;
- tests;
- result parser.

Make one coherent modification per experiment.
Prefer simple, interpretable changes.
Never bypass the metric.
Never load test data.
Never hide crashes or NaNs.
If an experiment fails, explain the likely reason in the next proposal.
```

---

# 18. Codex execution order

## Phase 1: repository and environment

Acceptance criteria:

- package installs;
- tests run;
- config loading works;
- local synthetic dataset works.

## Phase 2: physics

Acceptance criteria:

- projector/backprojector unit tests pass;
- sinogram and FBP visualization looks correct;
- gradients work.

## Phase 3: cached dataset

Acceptance criteria:

- official `.mat` files load;
- cache generation is restart-safe;
- dataloader loads cached arrays without high CPU use;
- one batch is visualized.

## Phase 4: baselines

Acceptance criteria:

- FBP baseline metrics;
- tiny U-Net can overfit 8 examples;
- full U-Net baseline trains.

## Phase 5: HeteroWave

Acceptance criteria:

- arbitrary masks work;
- output is invariant to sector ordering;
- mean-only and mean+variance modes work;
- tiny overfit test passes;
- model trains on Colab.

## Phase 6: evaluation

Acceptance criteria:

- deterministic scenario metrics;
- robustness plots;
- qualitative grids;
- config and checkpoint saved.

## Phase 7: physics consistency and uncertainty

Acceptance criteria:

- data residual decreases;
- uncertainty correlates positively with absolute error;
- no NaNs.

## Phase 8: autoresearch

Acceptance criteria:

- off by default;
- fixed benchmark;
- allowlist enforcement;
- timeout and OOM recovery;
- resume state;
- accepted experiments committed;
- rejected experiments reverted.

---

# 19. Risk controls

## Risk: projector is too slow

Response:

1. cache complete sinograms once;
2. use fewer angles in development;
3. compute data consistency every few steps;
4. reduce image size to 96;
5. profile before redesigning.

## Risk: per-sector backprojection is too expensive

Response:

1. vectorize sectors as batch dimension;
2. process only the 8 or fewer observed sectors during rapid trials;
3. precompute per-sector partial backprojections for the research subset only;
4. reduce sectors from 16 to 8 for rapid experiments.

## Risk: Drive I/O stalls training

Response:

- copy cache to `/content`;
- write checkpoints infrequently;
- sync results after validation;
- never read thousands of samples directly from Drive.

## Risk: GTX 1080 incompatibility

Response:

- use pinned CUDA 12.6 PyTorch wheel;
- disable `torch.compile`;
- use FP32;
- run only tiny tests locally;
- use Colab for real training.

## Risk: autoresearch corrupts evaluation

Response:

- immutable evaluator;
- file allowlist;
- git worktree or clean branch;
- subprocess timeout;
- parse one fixed metric;
- rerun top candidates manually.

## Risk: good short-run score does not transfer

Response:

- promote top three;
- train full-size;
- repeat with multiple seeds;
- report confirmed improvements only.

---

# 20. Definition of done

The project is successful when it contains:

1. reproducible setup;
2. differentiable forward and backprojection operators;
3. OpenBreastUS speed-map preprocessing;
4. FBP and U-Net baselines;
5. HeMIS-inspired variable-set reconstruction model;
6. random and contiguous missing-sector evaluation;
7. robustness curves;
8. at least one clear ablation;
9. qualitative reconstruction demo;
10. documented limitations;
11. optional, safely isolated autoresearch harness.

The final README must clearly state that the forward model is a straight-ray approximation and not full-wave ultrasound tomography.
