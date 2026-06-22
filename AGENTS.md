# HeteroWave agent handoff

Last updated: 2026-06-22.

This is the canonical operational handoff for future agents. Read this before
chat history or public README prose.

## Self-maintenance rule

Every agent that changes project direction, adds a model, runs or interprets an
important experiment, changes Colab/Drive paths, creates public artifacts, or
learns a durable result must update this file in the same turn.

When updating:

- update `Last updated`;
- record new branches, configs, checkpoints, result folders, and key metrics;
- mark conclusions as measured, inferred, or proposed;
- preserve failed/negative findings;
- keep commands copy-pasteable;
- do not hard-code Google Drive paths into library modules.

## Non-negotiable constraints

- Implement phases from `HETEROWAVE_CODEX_IMPLEMENTATION_PLAN.md` sequentially
  unless the user explicitly changes scope.
- Keep physics operators in pure PyTorch.
- Preserve tiny GTX 1080 / local smoke configurations.
- Use `apply_patch` for local file edits.
- Prefer `rg` / `rg --files` for searching.
- Do not use `--resume` across architecture changes. Use `--initialize-from`
  for partial warm starts.
- Local Python: `C:\Users\Win10\anaconda3\envs\heterowave\python.exe`.
- Local repo: `A:\projects\heterowave`.
- Persistent Colab cache:
  `/content/drive/MyDrive/heterowave/cache/cache_128`.
- Colab runtime cache usually:
  `/content/heterowave_data/cache_128`.

## Current public-facing state

Current branch at the time of this update: `main`.

`README.md` on `main` has been rewritten as the external-facing first-read
document. It should stay concise, approachable, and objective: explain the
medical/imaging context, the latent-sector architecture, the measured model
comparisons, checkpoint/result locations, and the next wedge-focused research
direction. It is intentionally honest that `main` is a landing branch and the
latest model implementations live on topic branches.

Repo-local README figures:

- `docs/assets/workflow_overview.gif`
- `docs/assets/model_average_nrmse.png`
- `docs/assets/scenario_test_nrmse.png`

Curated public-summary Drive folder:

```text
https://drive.google.com/drive/folders/1i5QF6GbeHMcSdbbVv-v8ZqANvEVkmfNv
```

Folder contents created on 2026-06-22:

- `HeteroWave public summary` Google Doc:
  `https://docs.google.com/document/d/1-J8gkt-wqqP1d7brSKGs8eFXAS43OS9zbZZC6yQCKyU`
- `01_architecture_heterowave_v2.png`
- `02_qualitative_grid_v2_validation.png`
- `03_random_sparse_view_robustness_v2_validation.png`
- `04_contiguous_wedge_robustness_v2_validation.png`
- `05_metrics_by_scenario_v2_validation.csv`
- `06_metrics_by_scenario_v2_test.csv`
- `07_metrics_by_scenario_tempered_precision_test.csv`
- `08_qualitative_grid_tempered_precision_test.png`
- `09_workflow_overview.gif`:
  `https://drive.google.com/file/d/1704ZlvnHA83xGgpba7dvxegwt3PQ5dhZ/view?usp=drivesdk`
- `10_model_average_nrmse.png`:
  `https://drive.google.com/file/d/17G7-p85F3lPr3gau_8aD_edtW9nBREr3/view?usp=drivesdk`
- `11_scenario_test_nrmse.png`:
  `https://drive.google.com/file/d/1JPKcxDegDI7P6nNTiehWBwByuA3-B9Gl/view?usp=drivesdk`

Important: the Google Drive connector could create/copy the folder and files,
but it could not set true "anyone with the link" public sharing from this
account. The user must manually open the folder and set Share -> General access
-> Anyone with the link -> Viewer before sending it externally.

No checkpoints, raw cache files, unfinished logs, or model weights were placed
in the curated folder.

## Branch map

| Branch | Purpose |
| --- | --- |
| `main` | public landing README and early code |
| `heterowave-v2` | masked-FBP trunk plus sector-wise latent fusion |
| `heterowave-v3` | precision weighting, mask geometry conditioning, acquisition hooks |
| `experiment/tempered-precision-pooling` | tempered reliability pooling; best average result so far by a small margin |
| `experiment/set-transformer-sector-pooling` | set-transformer sector aggregation experiment |
| `experiment/fbpconvnet-baseline` | FBPConvNet-style external baseline |
| `experiment/learned-primal-dual-baseline` | compact learned primal-dual baseline |
| `phase7-v1` | physics-consistency and uncertainty experiments |

## Measured results to preserve

Fair masked FBP U-Net is the main fair baseline.

| Model | Validation avg NRMSE | Test avg NRMSE | Status |
| --- | ---: | ---: | --- |
| Masked FBP U-Net | 0.35220 | 0.38142 | strong baseline |
| HeteroWave v2 | 0.34375 | 0.38075 | safest HeteroWave model |
| HeteroWave v3 | 0.34334 | 0.38116 | improves random/full views, worse wedges |
| Tempered precision pooling | 0.34304 | 0.38068 | best average by a tiny margin |

Measured interpretation:

1. HeteroWave improves distributed sparse-view reconstruction by adding
   acquisition-aware latent representation fusion to a masked-FBP trunk.
2. Contiguous limited-angle wedges remain the hard bottleneck.
3. Do not claim global SOTA or clinical validation.
4. Use "water-coupled ultrasonic CT", not "water noise" or "X-ray CT".

## Current pending runs as of 2026-06-22

Drive checkpoints inspected:

- `learned_primal_dual/last.pt`: `epoch=38`, `global_step=126360`,
  `best_nrmse=0.4133045164550751`, configured for 40 epochs. Not yet finished
  at last check; around epoch 39/40 because epoch is zero-indexed.
- `heterowave_set_transformer_pooling/last.pt`: `epoch=9`,
  `global_step=16200`, `best_nrmse=0.37368453673071206`, configured for
  20 epochs. Not finished at last check.

Do not use those pending runs as headline claims until validation/test suite
results exist.

## Best external-facing wording

Safe headline:

> HeteroWave reconstructs ultrasound CT speed maps from incomplete
> acquisitions, including cases where up to 8 of 16 angular sectors are absent,
> and improves over a strong mask-aware FBP U-Net baseline on distributed
> sparse-view cases.

Core innovation wording:

> The main innovation is acquisition-aware latent-space fusion: each observed
> angular sector is encoded into a learned representation, then pooled with
> mask/geometry awareness and fused with a physics-informed FBP prior.

Important caveat:

> Contiguous limited-angle wedges expose a real inverse-problem null-space and
> remain the next bottleneck.

## Recommended next architecture work

Most promising next experiment:

1. Observed-preserving sinogram completion branch.
2. FBP/backproject completed-missing sinogram as an extra feature.
3. Never overwrite observed measurements:

```python
completed = observed_mask * observed_sinogram + missing_mask * predicted_missing
```

Success criteria:

- improve `contiguous_8` and `contiguous_4`;
- do not materially regress `all_16`, `observed_12`, or random sparse-view
  cases;
- only run held-out test after validation wedge average improves.

## Key Drive artifact roots

```text
/content/drive/MyDrive/heterowave/results/fbp_unet_baseline
/content/drive/MyDrive/heterowave/results/masked_fbp_unet
/content/drive/MyDrive/heterowave/results/heterowave_v2_fusion
/content/drive/MyDrive/heterowave/results/heterowave_v2_validation
/content/drive/MyDrive/heterowave/results/heterowave_v2_test
/content/drive/MyDrive/heterowave/results/heterowave_v3_precision
/content/drive/MyDrive/heterowave/results/heterowave_v3_validation
/content/drive/MyDrive/heterowave/results/heterowave_v3_test
/content/drive/MyDrive/heterowave/results/heterowave_tempered_precision
/content/drive/MyDrive/heterowave/results/heterowave_tempered_precision_validation
/content/drive/MyDrive/heterowave/results/heterowave_tempered_precision_test
/content/drive/MyDrive/heterowave/results/fbpconvnet_sparse
/content/drive/MyDrive/heterowave/results/learned_primal_dual
/content/drive/MyDrive/heterowave/results/heterowave_set_transformer_pooling
```

## Literature breadcrumbs

Keep these categories in mind:

- Ultrasound tomography / USCT: OpenBreastUS, Li/Duric breast sound-speed UST,
  full-waveform inversion papers, refraction-corrected ray tomography.
- Sparse/limited-angle CT: FBPConvNet, Learned Primal-Dual, LEARN, MoDL,
  DuDoNet/DuDoDR-Net, CTTR/DuDoTrans, diffusion posterior sampling.
- Missing-modality / set fusion: HeMIS, Deep Sets, Set Transformer, Perceiver
  IO, product-of-experts fusion.
- Operator and geometry generalization: Fourier Neural Operator, DeepONet,
  neural processes, graph/message-passing models.

Do not compare this repo's NRMSE directly to full-wave clinical UST or CT SOTA
unless the data, forward model, metrics, and acquisition protocol are matched.
