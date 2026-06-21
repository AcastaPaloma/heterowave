# HeteroWave agent handoff

Last updated: 2026-06-21.

This file is the canonical handoff for future agents. Read this first before
using chat history, README prose, or Drive artifacts. The README is useful for
public-facing explanation, but this file is the operational memory: phases,
model lineage, decisions, known results, commands, and research conclusions.

## Self-maintenance rule

Every agent that changes project direction, adds a model, runs an important
experiment, changes Colab/Drive paths, or learns a durable result must update
this file in the same turn. Do not let this file become stale.

When updating:

- update `Last updated`;
- record new branches, configs, checkpoints, result folders, and key metrics;
- mark conclusions as measured, inferred, or proposed;
- preserve prior decisions unless new evidence supersedes them;
- keep commands copy-pasteable;
- do not remove failed/negative findings just because they are inconvenient;
- if a claim came from literature, keep enough citation breadcrumbs for the
  next agent to reconstruct the reasoning.

## Non-negotiable development constraints

- Implement phases from `HETEROWAVE_CODEX_IMPLEMENTATION_PLAN.md`
  sequentially unless the user explicitly changes scope.
- Keep physics operators in pure PyTorch.
- Preserve tiny GTX 1080 / local smoke configurations.
- Use `apply_patch` for local file edits.
- Prefer `rg`/`rg --files` for searching.
- Do not hard-code Google Drive paths into library modules. Drive paths are OK
  in Colab config files and README commands.
- Do not use `--resume` across architecture changes. Use `--initialize-from`
  for warm starts when model keys/shapes partially overlap.
- Local Python environment: `C:\Users\Win10\anaconda3\envs\heterowave\python.exe`.
- Local repo path: `A:\projects\heterowave`.
- Local cache may exist as untracked `/cache_128`; persistent Colab/Drive cache
  is `/content/drive/MyDrive/heterowave/cache/cache_128`; runtime copy is
  usually `/content/heterowave_data/cache_128`.

## Current branch and code state

Current working branch after the latest implementation: `heterowave-v3`.

The branch adds:

- `src/heterowave/models/heterowave_v3.py`
- `src/heterowave/acquisition.py`
- `configs/local_heterowave_v3_smoke.yaml`
- `configs/colab_heterowave_v3.yaml`
- `configs/colab_heterowave_v3_robust.yaml`
- `tests/test_heterowave_v3.py`
- README updates describing v3.

Validation performed locally after v3 implementation:

```text
python -m pytest tests/test_heterowave_v3.py tests/test_heterowave_v2.py -q
7 passed

python -m pytest -q
58 passed
```

Also ran local CPU smoke training:

```bash
python -m heterowave.train \
  --config configs/local_heterowave_v3_smoke.yaml \
  device=cpu \
  output.root=outputs/heterowave_v3_smoke_test
```

It completed 4 smoke steps and wrote a checkpoint.

## Phase history

### Phase 1 — repository/environment

Implemented earlier. The repo has package structure, configs, tests, and local
smoke support.

### Phase 2 — pure PyTorch physics

Implemented earlier. Projector/backprojector are pure PyTorch straight-ray
parallel-beam approximations. This is not full-wave acoustic propagation.

Important framing:

- the physics is intentionally simplified;
- outputs are useful for sparse-view speed-map reconstruction experiments;
- do not overclaim real ultrasound wave physics from this projector.

### Phase 3 — OpenBreastUS cache/preprocessing

Implemented earlier.

Key requirements satisfied:

- robust MATLAB loading through `scipy.io` and `h5py`;
- dataset inspection printing candidate arrays and shapes;
- deterministic train/validation split with seed `1337`;
- resize to `128 x 128`;
- speed maps converted to normalized speed targets and slowness contrast;
- GPU-batched complete sinogram generation;
- restart-safe memory-mapped cache generation;
- `metadata.json` with normalization and projector settings;
- dataset visualization command;
- tests using small synthetic MATLAB fixtures;
- explicit input/output paths for preprocessing.

Do not hard-code Drive paths in preprocessing/library modules.

### Phase 4 — baselines

Implemented earlier.

Models:

- complete-view FBP;
- complete-view FBP U-Net;
- fair masked FBP U-Net.

Important conclusion:

The fair masked FBP U-Net is very strong. It receives:

1. masked FBP image;
2. angular coverage map;
3. observed-fraction channel.

It is trained with the same missing-sector curriculum used for HeteroWave. It
became the main baseline.

### Phase 5 — original HeteroWave / v1

Implemented earlier.

HeteroWave v1 independently backprojects observed angular sectors, encodes
per-sector images, aggregates with masked set statistics, and decodes a speed
map.

Measured conclusion:

- v1 was interesting architecturally but underperformed the fair masked FBP
  U-Net;
- this showed that a purely sector-set model was not enough;
- the strong masked FBP trunk should not be thrown away.

### Phase 6 — deterministic evaluation suite

Implemented earlier.

Evaluation uses fixed validation/test scenarios:

| Scenario | Observed sectors | Pattern |
|---|---:|---|
| `all_16` | 16 | all sectors observed |
| `observed_12` | 12 | random observed sectors |
| `random_8` | 8 | random observed sectors |
| `random_4` | 4 | random observed sectors |
| `random_2` | 2 | random observed sectors |
| `contiguous_8` | 8 | contiguous angular wedge |
| `contiguous_4` | 4 | contiguous angular wedge |
| `contiguous_2` | 2 | contiguous angular wedge |

Fixed masks are in `configs/validation_masks_seed1337.json`.

Evaluation artifacts:

- `metrics_by_scenario.csv`
- `robustness_random.png`
- `robustness_wedge.png`
- `qualitative_grid.png`
- `architecture.png`
- config/provenance JSON/YAML
- copied `model.pt`
- `summary.json`

### Phase 7 — physics consistency and uncertainty

Implemented earlier for HeteroWave v1/v1-style work.

Added:

- observed-angle data-consistency loss;
- heteroscedastic uncertainty output;
- uncertainty metrics such as Spearman correlation.

Conclusion:

- uncertainty was meaningful and correlated with error;
- it did not by itself make v1 beat the fair masked U-Net.

## Model lineage and decisions

### Masked FBP U-Net

This is the central fair baseline.

Why it is strong:

- FBP gives a physically meaningful initial reconstruction;
- coverage and observed-fraction channels tell the network what is missing;
- the model is cheap and stable.

Do not dismiss this baseline. Any HeteroWave result must be compared against it.

### HeteroWave v2

Implementation: `src/heterowave/models/heterowave_v2.py`

Config: `configs/colab_heterowave_v2.yaml`

Result folder:

```text
/content/drive/MyDrive/heterowave/results/heterowave_v2_fusion
```

Design:

- keep the masked FBP U-Net trunk;
- add per-sector partial backprojection branch;
- aggregate sector features with masked mean/variance/count;
- fuse sector features into each global U-Net scale through zero-initialized
  scalar gates;
- optionally freeze global trunk.

Important checkpoint behavior:

- v2 can be initialized from masked FBP U-Net with `--initialize-from`;
- matching trunk weights load by name/shape;
- output head may adapt from 1 to 2 channels when uncertainty is enabled;
- new sector branch starts fresh.

Measured conclusion:

- v2 is the safest overall HeteroWave model so far;
- it slightly improves the masked U-Net on average validation/test metrics;
- gains are strongest on random/full/near-full observations;
- contiguous missing wedges remain hard and often favor masked U-Net.

### HeteroWave v3

Implementation: `src/heterowave/models/heterowave_v3.py`

Clean config: `configs/colab_heterowave_v3.yaml`

Optional robustness config: `configs/colab_heterowave_v3_robust.yaml`

Result folders:

```text
/content/drive/MyDrive/heterowave/results/heterowave_v3_precision
/content/drive/MyDrive/heterowave/results/heterowave_v3_validation
/content/drive/MyDrive/heterowave/results/heterowave_v3_test
```

No `heterowave_v3_robust` result folder had been found as of 2026-06-21.

Design:

- keep the same warm-startable masked FBP U-Net trunk;
- replace equal HeMIS-style aggregation with learned positive sector precision;
- per sector/pixel, compute reliability/precision maps;
- aggregate weighted mean, weighted variance, effective count,
  observed-fraction, and log-precision summaries;
- add higher-frequency sector angle encoding;
- add mask-geometry encoder and zero-initialized FiLM decoder conditioning;
- add optional projection-space acquisition perturbations.

Why v3 builds on v2/baseline:

- the masked FBP trunk is the best known reconstruction backbone;
- v1 proved that replacing it entirely was worse;
- zero-gated additive modules give clean attribution;
- if v3 improves, the gain is due to reliability/mask geometry modules, not a
  random architecture reset.

Correct warm start:

```bash
python -m heterowave.train \
  --config configs/colab_heterowave_v3.yaml \
  --initialize-from /content/drive/MyDrive/heterowave/results/masked_fbp_unet/best.pt
```

Do not resume v3 from v2:

```bash
# Incorrect
python -m heterowave.train \
  --config configs/colab_heterowave_v3.yaml \
  --resume /content/drive/MyDrive/heterowave/results/heterowave_v2_fusion/best.pt
```

Reason: `--resume` expects the same architecture and optimizer state.

## Measured results

### V2 validation

720 validation samples. Average NRMSE:

| Model | All scenarios | Missing-view only |
|---|---:|---:|
| Masked FBP U-Net | 0.35220 | 0.38739 |
| HeteroWave v2 | 0.34375 | 0.37817 |

V2 improved all 8 validation scenarios over masked U-Net.

### V2 held-out test

800 test samples. Average NRMSE:

| Model | All scenarios | Missing-view only |
|---|---:|---:|
| Masked FBP U-Net | 0.38142 | 0.42020 |
| HeteroWave v2 | 0.38075 | 0.41981 |

Scenario-level test conclusion:

- v2 beats masked U-Net on `all_16`, `observed_12`, `random_8`, `random_4`,
  `random_2`;
- masked U-Net slightly beats v2 on `contiguous_8`, `contiguous_4`;
- `contiguous_2` is effectively tied.

### V3 validation

Result folder:

```text
/content/drive/MyDrive/heterowave/results/heterowave_v3_validation
```

Average validation NRMSE:

| Model | All scenarios |
|---|---:|
| Masked FBP U-Net | 0.35220 |
| HeteroWave v2 | 0.34375 |
| HeteroWave v3 | 0.34334 |

V3 validation conclusion:

- v3 beats masked U-Net on all validation scenarios;
- v3 slightly beats v2 on all validation scenarios;
- improvement over v2 is very small, about 0.12% average NRMSE.

### V3 held-out test

Result folder:

```text
/content/drive/MyDrive/heterowave/results/heterowave_v3_test
```

Average test NRMSE:

| Model | All scenarios | Missing-view only |
|---|---:|---:|
| Masked FBP U-Net | 0.38142 | 0.42020 |
| HeteroWave v2 | 0.38075 | 0.41981 |
| HeteroWave v3 | 0.38116 | 0.42046 |

Scenario-level test NRMSE:

| Scenario | Masked U-Net | v2 | v3 | Current read |
|---|---:|---:|---:|---|
| `all_16` | 0.10998 | 0.10735 | 0.10600 | v3 best |
| `observed_12` | 0.17386 | 0.17222 | 0.17163 | v3 best |
| `random_8` | 0.26564 | 0.26344 | 0.26308 | v3 best |
| `random_4` | 0.45532 | 0.45340 | 0.45294 | v3 best |
| `random_2` | 0.58913 | 0.58728 | 0.58809 | v2 best, v3 beats masked |
| `contiguous_8` | 0.32624 | 0.32893 | 0.33094 | masked best |
| `contiguous_4` | 0.51462 | 0.51649 | 0.51796 | masked best |
| `contiguous_2` | 0.61657 | 0.61688 | 0.61860 | masked best |

V3 test conclusion:

- v3 is directionally useful for full/random-sector sparse views;
- v3 does not solve contiguous missing wedges;
- on held-out test, v3 is slightly better than masked U-Net overall but worse
  than v2 overall;
- v2 remains the safest best current model;
- v3 is valuable evidence that reliability weighting helps distributed
  missingness, but wedge extrapolation needs a different module.

Runtime/memory:

- masked U-Net test inference: about 1.1-1.3 ms/sample, about 104 MB peak GPU;
- v2 test inference: about 5.7 ms/sample, about 251 MB peak GPU;
- v3 test inference: about 7.3-7.5 ms/sample, about 317 MB peak GPU.

## Current best interpretation

The scientifically honest story is:

1. A simple fair masked FBP U-Net is extremely strong.
2. Pure sector-set HeteroWave v1 was not enough.
3. V2 found the right foundation: keep the strong masked FBP trunk and add
   zero-gated sector reasoning.
4. V2 is the best overall model so far by held-out test average.
5. V3 improved random/full scenarios but slightly worsened contiguous wedges.
6. The unsolved bottleneck is limited-angle contiguous missing-wedge
   extrapolation, not generic sparse-view denoising.

Do not claim v3 is globally SOTA or globally better than v2. It is a useful
ablation and architecture direction, but not the new best overall checkpoint.

## Recommended next architecture work

Do not run robustness training as the next primary experiment. First fix the
wedge problem.

Most promising next steps, in order:

1. Observed-preserving sinogram completion/residual branch.
   - Work in sinogram domain.
   - Never overwrite observed measurements.
   - Use synthesized/missing sinogram only as extra feature or FBP channel.
   - Evaluate specifically on contiguous wedges.
2. Tiny unrolled observed-data consistency refiner.
   - Use existing PyTorch projector/backprojector.
   - One step for GTX 1080 smoke, maybe two for Colab.
   - Zero-initialize learned residual correction.
3. Wedge-focused mask geometry/directional artifact module.
   - Current mask FiLM is too weak to solve wedges alone.
   - Consider explicit longest-missing-gap features or directional filters.
4. Wavelet/frequency detail head.
   - Useful for edges/SSIM.
   - Not expected to fix gross wedge hallucination alone.

## 2026-06-21 SOTA/input-invariance research addendum

Status: proposed, not implemented.

User specifically asked whether a surface-code-decoder-style idea could apply:
learn a model that extrapolates across acquisition sizes/geometries instead of
memorizing one fixed input layout. Four subagents reviewed UST, sparse/limited
angle CT, neural operators, set/graph models, and repo implementation risk.

Consensus:

1. True ultrasound tomography SOTA is still full-waveform inversion (FWI) and
   frequency-domain/time-domain wave-equation inversion. That solves a richer
   problem than this repo's straight-ray cache, so do not compare NRMSEs
   directly.
2. For the current repo task, the closest high-performing family is sparse-view
   CT / projection tomography: FBP prior + learned image refiner, dual-domain
   sinogram/image networks, learned primal-dual/unrolled reconstruction, and
   data consistency.
3. The current bottleneck is contiguous limited-angle wedges. The literature
   says this is a missing-data/null-space problem, not just a generic denoising
   problem. More image-domain attention alone is unlikely to be the best next
   move.
4. The best immediate positive experiment is `HeteroWaveV4DualDomainLite`:
   v2-style masked-FBP trunk plus observed-preserving sinogram completion.
5. The most novel/presentable follow-up is an acquisition-invariant model:
   represent observed rays/sectors as variable-size geometry tokens and train
   on families of acquisition geometries, then test zero-shot on unseen sector
   counts, angular patterns, detector counts, and/or resolutions.

### Proposed v4: DualDomainLite

Goal: improve `contiguous_8`, `contiguous_4`, and ideally `contiguous_2`
without sacrificing random/full-view scenarios.

Design:

```text
full cached sinogram
  -> apply training/eval mask internally
  -> observed sinogram y_obs and mask M
  -> small sinogram completion network predicts missing views only
  -> completed = M * y_obs + (1 - M) * y_pred
  -> FBP/backproject completed/missing sinogram into image-like features
  -> v2 masked-FBP U-Net trunk with zero-gated dual-domain fusion
  -> speed map
```

Hard safety rule:

```python
completed = observed_mask * observed_sinogram + missing_mask * predicted_missing
```

Never allow the completion branch to overwrite observed measurements.

Suggested losses:

```text
image_loss(pred_speed, target_speed)
+ lambda_missing * L1((1 - M) * (completed_sinogram - full_sinogram))
+ lambda_observed * L1(M * (project(pred_speed) - observed_sinogram))
```

Implementation notes:

- add `src/heterowave/models/heterowave_v4.py`;
- warm-start from `/content/drive/MyDrive/heterowave/results/heterowave_v2_fusion/best.pt`
  with `--initialize-from`, not `--resume`;
- add `configs/local_heterowave_v4_smoke.yaml`;
- add `configs/colab_heterowave_v4_dual_domain.yaml`;
- keep gates zero-initialized so the model begins as v2 when warm-started;
- make missing-sinogram auxiliary loss robust to `all_16` divide-by-zero;
- evaluate validation before held-out test.

Suggested Colab train command:

```bash
python -m heterowave.train \
  --config configs/colab_heterowave_v4_dual_domain.yaml \
  --initialize-from /content/drive/MyDrive/heterowave/results/heterowave_v2_fusion/best.pt
```

Suggested success criteria:

- primary: beat v2 and masked U-Net on `contiguous_8` and `contiguous_4`;
- secondary: do not regress materially on `all_16`, `observed_12`,
  `random_8`, `random_4`, `random_2`;
- report observed-projection residual separately so sinogram completion cannot
  silently hallucinate observed data;
- only run test-set evaluation after validation wedge average improves.

### Proposed v5/parallel novelty: Acquisition-Token HeteroWave

This is the strongest analogy to variable-size surface-code decoding.

Core idea:

```text
measurement/ray/sector token_i = [
  observed flag,
  sin(angle),
  cos(angle),
  angle width or angular bin bounds,
  detector coordinate / detector spacing,
  detector count or normalized sampling density,
  optional calibration/noise metadata
]
```

Encode variable-size token sets with Deep Sets, Set Transformer, Perceiver IO,
or a lightweight message-passing block. Fuse the resulting acquisition code
into the reconstruction trunk, and optionally query image coordinates with
Fourier/coordinate features for resolution transfer.

Why this is worth doing:

- fixed 16-sector conditioning will not convincingly extrapolate to 8, 12, 24,
  or irregular sector layouts;
- a token/set model can be trained/evaluated on acquisition families rather
  than one fixed tensor shape;
- the evaluation story can be "zero-shot generalization to unseen acquisition
  geometry," which is more impressive than another same-distribution NRMSE
  tweak.

Minimum credible experiment:

```text
train geometries:
  num_sectors: 8, 16, 24
  masks: random, periodic, contiguous, clustered dropout
  detector_bins/image_size: keep 128 for first pass

held-out geometries:
  num_sectors: 12 or 32
  irregular/non-divisible angle sets
  contiguous wedges with unseen widths/centers
```

Only claim acquisition invariance if evaluation includes held-out geometries.
Do not claim size/resolution invariance from a single 128x128 cache.

### SOTA-related citations from this pass

UST / USCT:

- Chenevert et al. 1984, early breast ultrasonic CT, DOI
  `10.1148/radiology.152.1.6729107`.
- Li, Duric, Littrup, Huang 2009, in-vivo breast sound-speed UST with
  TV-regularized bent-ray tomography, DOI `10.1016/j.ultrasmedbio.2009.05.011`.
- Perez-Liva et al. 2017, sound speed and attenuation in USCT with FWI,
  DOI `10.1121/1.4976688`.
- Javaherian, Lucka, Cox 2020, refraction-corrected ray-based inversion,
  DOI `10.1088/1361-6420/abc0fc`.
- Lucka et al. 2022, high-resolution 3D breast TD-FWI, arXiv `2102.00755`.
- Ali et al. 2024, open-source frequency-domain waveform inversion for ring
  array UST, DOI `10.1109/TMI.2024.3383816`.
- Zhao et al. 2020, fully convolutional learned UST reconstruction,
  DOI `10.1088/1361-6560/abb5c3`.
- Qu et al. 2022, U-Net with Tikhonov pseudo-inverse prior for sound-speed
  tomography, DOI `10.1016/j.ultrasmedbio.2022.05.033`.
- Fang et al. 2022, singular-value-threshold completion for missing TOF
  matrices, DOI `10.3934/mbe.2022476`.
- Jeong et al. 2024, traveltime + reflection tomography to high-resolution SOS,
  DOI `10.1109/TUFFC.2024.3459391`.
- OpenBreastUS 2025, large frequency-domain USCT dataset, arXiv `2507.15035`.
- Diff-ANO 2025/2026, conditional consistency/adjoin neural operators for
  sparse/partial-view USCT, arXiv `2507.16344`.

Sparse/limited-angle CT and inverse problems:

- FBPConvNet, arXiv `1611.03679`.
- LEARN, arXiv `1707.09636`.
- Learned Primal-Dual Reconstruction, arXiv `1707.06474`.
- CNN-PGD, arXiv `1709.01809`.
- MoDL, arXiv `1712.02862`.
- Deep sinogram synthesis, arXiv `1803.00694`.
- Lose the Views / CTNet, CVPR 2018.
- DuDoDR-Net, DOI `10.1016/j.media.2021.102289`.
- DuDoTrans, arXiv `2111.10790`.
- CTTR, DOI `10.1016/j.ejmp.2022.07.001`.
- ProCT, arXiv `2312.07846`.
- FreeSeed, arXiv `2307.05890`.
- TD-Net, arXiv `2311.15369`.
- DOLCE diffusion for limited-angle CT, arXiv `2211.12340`.
- Diffusion Posterior Sampling, arXiv `2209.14687`.
- Sinogram-domain score model, arXiv `2211.13926`.
- Computed Tomography Neural Operator, arXiv `2512.12236`.

Input-invariant / variable-geometry models:

- Deep Sets, arXiv `1703.06114`.
- Set Transformer, arXiv `1810.00825`.
- Perceiver IO, arXiv `2107.14795`.
- Conditional Neural Processes, arXiv `1807.01613`.
- Attentive Neural Processes, arXiv `1901.05761`.
- MeshGraphNets, arXiv `2010.03409`.
- Fourier Neural Operator, arXiv `2010.08895`.
- DeepONet, arXiv `1910.03193`.
- Geo-FNO, JMLR 2023.
- SIREN, arXiv `2006.09661`.
- PINER sparse-view CT, WACV 2023.
- Recurrent Stacked Back Projection, arXiv `2112.04998`.

## Acquisition robustness conclusions

Do not call the water bath "water noise" or imply water arbitrarily disturbs
ultrasound. Water is the coupling medium and is much better than air for
ultrasound.

Defensible robustness perturbations for current straight-ray sinogram cache:

- calibration / delay baseline drift;
- per-angle/per-detector gain-like projection scaling;
- low-level measurement noise / TOF jitter in projection space;
- detector-axis fractional shifts as a surrogate for timing, geometry, or
  simple motion;
- sensor/sector dropout.

Do not claim true attenuation, bandwidth loss, refraction, or full-wave effects
unless a richer forward model or waveform/amplitude cache is implemented.

Implemented v3 robustness layer:

```text
src/heterowave/acquisition.py
```

It is optional and disabled in the clean v3 config.

## Multimodal conclusion

The user asked whether combining CT + another modality is interesting.

Conclusion:

- true CT/MRI/PA multimodal reconstruction is academically credible but out of
  scope for this repo right now;
- current cache is single-modality synthetic transmission UST from speed maps;
- no paired CT, MRI, PA, echo/reflection, attenuation, or raw RF wavefield data
  exists here;
- do not pivot v3 to multimodal claims.

Most credible future multimodal direction:

1. transmission ultrasound + reflection/echo or structural prior;
2. PA + US as a later external v4 benchmark;
3. CT/MRI + US only as registration/prior concepts, not current
   reconstruction.

If adding a second modality later, start with a controlled synthetic
transmission + reflection/edge-prior ablation, not clinical CT/MRI fusion.

## Midjourney Medical / water-coupled UST context

Official pages consulted:

- `https://www.midjourney.com/medical`
- `https://www.midjourney.com/medical/blogpost`

Observed official framing:

- Midjourney describes a full-body ultrasound / Ultrasonic CT concept;
- patient descends into water;
- ring of underwater sensors sends ultrasonic waves from many angles;
- stated target is roughly 60-second scan;
- water is the medium/coupler;
- waves change through body tissues due to density/stiffness changes.

Video itself was not programmatically inspected; conclusions came from the
official page text/technical breakdown.

Use this framing:

- "water-coupled ultrasonic CT";
- "acquisition-aware reconstruction";
- "calibration/timing/channel robustness".

Avoid this framing:

- "water disturbance noise";
- "water corrupts ultrasound";
- "CT-style X-ray reconstruction".

## Important commands

### Train masked FBP U-Net

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

Resume v2:

```bash
python -m heterowave.train \
  --config configs/colab_heterowave_v2.yaml \
  --resume /content/drive/MyDrive/heterowave/results/heterowave_v2_fusion/last.pt
```

### Train clean HeteroWave v3

```bash
python -m heterowave.train \
  --config configs/colab_heterowave_v3.yaml \
  --initialize-from /content/drive/MyDrive/heterowave/results/masked_fbp_unet/best.pt
```

Resume v3:

```bash
python -m heterowave.train \
  --config configs/colab_heterowave_v3.yaml \
  --resume /content/drive/MyDrive/heterowave/results/heterowave_v3_precision/last.pt
```

### Evaluate v3 validation

```bash
python -m heterowave.evaluate \
  --config configs/colab_heterowave_v3.yaml \
  --suite \
  --unet-checkpoint /content/drive/MyDrive/heterowave/results/fbp_unet_baseline/best.pt \
  --masked-unet-checkpoint /content/drive/MyDrive/heterowave/results/masked_fbp_unet/best.pt \
  --heterowave-checkpoint /content/drive/MyDrive/heterowave/results/heterowave_v3_precision/best.pt \
  --output-dir /content/drive/MyDrive/heterowave/results/heterowave_v3_validation
```

### Evaluate v3 held-out test

```bash
python -m heterowave.evaluate \
  --config configs/colab_heterowave_v3.yaml \
  --suite \
  --split test \
  --unet-checkpoint /content/drive/MyDrive/heterowave/results/fbp_unet_baseline/best.pt \
  --masked-unet-checkpoint /content/drive/MyDrive/heterowave/results/masked_fbp_unet/best.pt \
  --heterowave-checkpoint /content/drive/MyDrive/heterowave/results/heterowave_v3_precision/best.pt \
  --output-dir /content/drive/MyDrive/heterowave/results/heterowave_v3_test
```

## Drive artifact map

Expected important folders:

```text
/content/drive/MyDrive/heterowave/results/fbp_unet_baseline
/content/drive/MyDrive/heterowave/results/masked_fbp_unet
/content/drive/MyDrive/heterowave/results/heterowave_mean_var_count
/content/drive/MyDrive/heterowave/results/heterowave_phase7
/content/drive/MyDrive/heterowave/results/heterowave_v2_fusion
/content/drive/MyDrive/heterowave/results/heterowave_v2_validation
/content/drive/MyDrive/heterowave/results/heterowave_v2_test
/content/drive/MyDrive/heterowave/results/heterowave_v3_precision
/content/drive/MyDrive/heterowave/results/heterowave_v3_validation
/content/drive/MyDrive/heterowave/results/heterowave_v3_test
```

Use `best.pt` for evaluation and model selection. Use `last.pt` only for
continuing the same training run with the same architecture/config.

## Literature and resources consulted

The following resources shaped the current conclusions. Future agents do not
need to re-read all of them before routine coding, but should preserve these
breadcrumbs when explaining design decisions.

### Dataset / UST context

- OpenBreastUS, arXiv `2507.15035`.
- Zhao et al., "Ultrasound transmission tomography image reconstruction with a
  fully convolutional neural network," DOI `10.1088/1361-6560/abb5c3`.
- Sandhu et al., breast waveform tomography work.
- Duric / Li breast sound-speed UST work.

### Missing-modality / set fusion

- HeMIS, arXiv `1607.05194`.
- U-HVED, arXiv `1907.11150`.
- HyperDenseNet, DOI `10.1109/TMI.2018.2878669`.
- Deep Sets.
- Set Transformer, arXiv `1810.00825`.
- Perceiver IO, arXiv `2107.14795`.
- Conditional Neural Processes / Attentive Neural Processes.
- Product-of-Experts missing-modality fusion.

### Sparse-view CT / inverse problems

- FBPConvNet, arXiv `1611.03679`, DOI `10.1109/TIP.2017.2713099`.
- Learned Primal-Dual Reconstruction, arXiv `1707.06474`,
  DOI `10.1109/TMI.2018.2799231`.
- LEARN for sparse-data CT, arXiv `1707.09636`.
- CNN-PGD for consistent CT reconstruction, arXiv `1709.01809`,
  DOI `10.1109/TMI.2018.2832656`.
- MoDL, arXiv `1712.02862`, DOI `10.1109/TMI.2018.2865356`.
- Deep learning CT projection-domain weights,
  DOI `10.1109/TMI.2018.2833499`.
- DuDoNet, DOI `10.1109/CVPR.2019.01076`.
- DuDoDR-Net, DOI `10.1016/j.media.2021.102289`.
- CTTR dual-domain transformer, DOI `10.1016/j.ejmp.2022.07.001`.
- DuDoTrans, arXiv `2111.10790`.
- Deep sinogram synthesis, DOI `10.1109/TRPMS.2018.2867611`.
- Quinto, limited-data artifacts / visible singularities,
  DOI `10.1007/s11220-017-0158-7`.

### Operator/frequency/detail modules

- Fourier Neural Operator.
- U-NO.
- Wavelet Neural Operator.
- DeepONet.
- SIREN / Fourier features.
- LIIF.
- FFC.
- Framelet U-Net for sparse-view CT, arXiv `1708.08333`.
- Wavelet residual CT network, DOI `10.1109/TMI.2018.2823756`.
- TD-Net tri-domain sparse-view CT, arXiv `2311.15369`.

### UST acquisition robustness

- Li et al., in vivo breast sound-speed imaging with UST,
  DOI `10.1016/j.ultrasmedbio.2009.05.011`.
- Li et al., automatic time-of-flight picker,
  DOI `10.1016/j.ultras.2008.05.005`.
- Filipik et al., UCT calibration with TOF positioning,
  DOI `10.1109/IEMBS.2007.4352747`.
- Tan, Steiner, Ruiter, self-calibration accuracy for 3D UST,
  handle/DOI `10.5445/IR/1000079792`.
- Cueto et al., spatial response identification for robust experimental UCT,
  arXiv `2103.10722`.
- van Neer et al., reflector-based phase calibration,
  DOI `10.1016/j.ultras.2010.05.001`.
- Ruiter et al., patient movement during 3D USCT,
  DOI `10.1117/12.2216680`.
- Del Grosso and Mader, speed of sound in pure water,
  DOI `10.1121/1.1913258`.
- Li et al., refraction-corrected transmission USCT,
  DOI `10.1118/1.3360180`.
- "Analysis of the Refraction Effect in Ultrasound Breast Tomography,"
  DOI `10.3390/app12073578`.
- Hormati et al., robust travel-time tomography with bent-ray model,
  DOI `10.1117/12.844693`.
- Pratt et al., sound-speed and attenuation waveform tomography,
  DOI `10.1117/12.708789`.
- Perez-Liva et al., sound speed and attenuation in USCT with FWI,
  DOI `10.1121/1.4976688`.

### Multimodal / future directions

- Yao et al., structural echo prior for sound-speed reconstruction,
  DOI `10.1109/ACCESS.2020.3000062`.
- Ziegler et al., FWI with structural similarity EIT prior,
  DOI `10.3934/ipi.2023023`.
- Lin et al., transmission-reflection USCT,
  DOI `10.3390/s23073701`.
- Korta Martiartu et al., joint reflection-transmission breast US design,
  DOI `10.1121/1.5122291`.
- Forte et al., multimodal ultrasound tomography breast feasibility,
  DOI `10.1186/s41747-017-0029-y`.
- Kelly et al., photoacoustic tomography in automated breast US,
  DOI `10.1117/1.JBO.25.11.116010`.
- Zhang et al., US-guided sound-speed correction for PACT,
  DOI `10.1016/j.pacs.2026.100804`.
- Huang/Lin et al., high-speed PA + ultrasonic CT breast tumor imaging,
  DOI `10.1126/sciadv.adz2046`; Dryad dataset DOI `10.5061/dryad.vmcvdnd55`.
- Banerjee et al., CT-to-US registration, DOI `10.1016/j.media.2019.02.003`.
- Haskins et al., MR-TRUS registration, DOI `10.1007/s11548-018-1875-7`.

## What not to forget

- The most important negative result is that contiguous missing wedges remain
  unsolved.
- The most important positive result is that sector-aware additions can help
  without destroying the strong masked FBP baseline.
- V2 is currently the safest overall result.
- V3 is a useful architecture probe but not the overall best checkpoint.
- A future "v4" should probably be dual-domain or unrolled, not just more
  attention.
- Keep this file current.

## Branch experiment: tempered precision pooling

Branch: `experiment/tempered-precision-pooling`.

Status: implemented as a trainable branch, not yet benchmarked.

Purpose:

- keep v3's learned positive sector precision maps;
- cap and temper precision before normalization to avoid overconfident
  Product-of-Experts behavior from correlated angular sectors;
- expose `model.precision_temperature` and `model.precision_max` in config.

Colab train command:

```bash
python -m heterowave.train \
  --config configs/colab_heterowave_tempered_precision.yaml \
  --initialize-from /content/drive/MyDrive/heterowave/results/heterowave_v2_fusion/best.pt
```

Local smoke:

```bash
python -m heterowave.train \
  --config configs/local_heterowave_tempered_precision_smoke.yaml \
  device=cpu
```
