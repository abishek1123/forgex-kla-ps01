# AI-Based Restoration of Degraded Images — KLA PS01

Restores degraded semiconductor inspection images: removes signal-dependent
noise and upscales 2× (128×128 → 256×256, or 256×256 → 512×512).

**Team ForgeX** — Abishek SR · Anmol BA · Hardik — VIT Vellore
SEMICON India Hackathon 2026 · Problem Statement PS01 · **Round 2**

> Round-1 archive, including full commit history: `github.com/abishek1123/forgex`

---

## Quick start

```bash
pip install -r requirements.txt

python run.py <input-dir> <output-dir>
```

That is the whole thing. `run.py` reads every `.npy` file in the input
directory, restores each at 2× resolution, and writes one `.npy` of the same
name into the output directory (created automatically). It auto-detects CUDA
and falls back to CPU.

**Input:** `.npy`, grayscale, shape `(H, W)` or `(H, W, 1)`, float32. Values may
fall outside `[0, 1]` — expected for this degradation, and handled.

**Output:** `.npy`, float32, shape `(2H, 2W)` — or `(2H, 2W, 1)` if the input
carried a trailing channel axis. Guaranteed finite and clipped to `[0, 1]`.

**Requirements:** `torch` and `numpy` only. No internet, no API keys, no model
downloads, no user interaction. Weights ship in `models/model.pt`.

| Flag | Effect |
|---|---|
| `--weights PATH` | use a different checkpoint (default `models/model.pt`) |
| `--device cpu` | force CPU |
| `--no-fp16` | disable half precision on CUDA |
| `--tta` | 8× self-ensemble: small gain, ~7× slower. **Off by default.** |
| `--batch N` | images per forward pass (default 32 on GPU, 1 on CPU) |
| `--profile` | print a stage-by-stage wall-clock breakdown |

---

## Results

Held-out split of 200 real KLA round-2 pairs (`make_split(seed=0)`), never seen
during training. Scores are **final-5-epoch means**, not best-epoch — see
*How we decide what is real* below for why that distinction matters.

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|
| Bicubic ×2 (no denoising) | 20.22 | — | — |
| **Ours — 1.37 M params** | **23.27** | **0.612** | **0.348** |

**+3.05 dB over bicubic.** A spectral decomposition of what is recoverable
without hallucinating puts the ceiling near 26.6 dB, so this captures roughly
48% of the recoverable range.

**Round 1 for context:** 28.43 dB / 0.764 SSIM, +5.19 dB over bicubic. The gap
between rounds is the problem, not the model — see below.

Reproduce with
`python src/validate.py --data <data> --ckpt models/model.pt --baseline`.

---

## Approach

### 1. What we measured about the data

Round 2 is a **different problem** from round 1, not a harder version of it. We
re-derived every property rather than assuming any carried over:

| | Round 1 | Round 2 |
|---|---|---|
| Content | natural photographs | SEM micrographs |
| Training pairs | 3,200 | 4,785 |
| Sizes present | mixed | all 256→128; **no 512×512 anywhere** |
| f90 — structure fineness | 0.0945 | **0.2913** (3.1× finer) |
| Normalisation | — | per-image min–max, exact `[0,1]` |
| Downsampling operator | exact 2×2 box average | exact 2×2 box average |
| Bicubic ×2 baseline | 23.23 dB | 20.22 dB |

`f90` is the radial frequency below which 90% of non-DC energy sits. Structure
3.1× finer means 3.1× more of the signal lives above the Nyquist limit of the
downsampled image — information genuinely destroyed, not merely obscured.

Dataset ids are **content-ordered**: `f90` runs from 0.3677 (ids 478–955) down
to 0.1233 (ids 2868–3345). That makes contiguous id blocks usable as genuine
out-of-distribution splits, which `--holdout-range` exploits.

### 2. The degradation, reverse-engineered

Fitting the residual variance of KLA's own pairs against the ground truth gives
a three-term law (R² = 0.948), where `m` is the 2×2 block mean and `v` the
within-block variance:

```
var(residual)  =  σ_add²  +  σ_mul²·m²  +  c·v

σ_add ∈ [0.000, 0.065]   additive floor
σ_mul ∈ [0.155, 0.258]   speckle — multiplicative, grows with brightness
c     ∈ [0.026, 0.234]   detail-dependent term
```

Three changes from the round-1 generator, each driven by a measurement:

* **Added the `c·v` term.** Detail-dependent noise is 4.4% of total variance on
  SEM against 1.8% on photographs — the noise tracks edges here in a way it did
  not before.
* **Removed the blur.** Measured harmful: residual error rose 0.01074 → 0.01235
  with blur in the pipeline. It was in round 1's generator on reasoning, never
  on evidence.
* **Gaussian → Gamma (k = 14).** The real residual is spatially white (lag-1
  correlation −0.055 / −0.053) but decidedly not Gaussian: skew +0.43, excess
  kurtosis +0.62.

Calibration check — refitting our own synthetic output with the same estimator:

| | σ_add | σ_mul |
|---|---|---|
| KLA's real pairs | 0.0414 | 0.1863 |
| Our rebuilt generator | 0.0415 | 0.1986 |

`tools/calibrate.py` reproduces this fit.

### 3. Why we train on real pairs, not synthetic ones

This is the result we did not expect, and it is reported as we found it.

| Run | Training data | PSNR | SSIM |
|---|---|---|---|
| **r2-preal1** | **100% real pairs** | **23.2672** | **0.6124** |
| r2-newgen | 70% rebuilt synthetic | 23.2178 | 0.5925 |
| r2-long120 | 70% old (round-1) synthetic | 23.1342 | 0.5930 |

The rebuild **worked as a rebuild** — +0.084 dB over the old generator, eight
times our noise floor. It **still lost to real data** — −0.049 dB and −0.020
SSIM. We ship `p_real = 1.0`.

Stated plainly: we built a demonstrably better forward model of the degradation
and it still could not beat 4,585 real examples. Round 1's conclusion (synthesise
unlimited pairs, keep 30% real as insurance) **does not survive** on round-2 data,
and we changed the pipeline rather than the story.

### 4. Architecture (`src/model.py`)

A residual CNN that does all its work at low resolution and upsamples only in
the final block via PixelShuffle — 4× cheaper than upsample-first designs, which
matters because inference time is scored.

* **Global bicubic skip** — the network predicts only the *correction* to a
  cheap baseline, so it starts at the bicubic score and converges fast.
* **Variance-stabilising stem** — a parameter-free layer feeding the first
  convolution raw, `√x` and `log(1+x)` views, so it can pick the representation
  in which signal-dependent noise is closest to uniform.
* **No BatchNorm** — well established to hurt super-resolution.

1.37 M parameters, 5.5 MB. `tools/capacity_sweep.py` maps the accuracy/size
frontier across a 78× span of parameter count.

### 5. Loss (`src/losses.py`)

Charbonnier + SSIM + gradient. No GAN. A differentiable LPIPS term is available
(`--w-lpips`) but off by default: perceptual and adversarial losses work by
synthesising plausible texture, and invented texture on an inspection image is a
fabricated defect. The gradient term sharpens edges only where the ground truth
has edges.

---

## How we decide what is real: the measured noise floor

Before trusting any result we measured what *nothing* looks like. Two runs
differing only in random seed land **0.010 dB** apart on a final-5-epoch mean.
Selecting the best epoch instead inflates that apparent spread to 0.055 dB —
which is why every number in this repo is a final-5 mean, and why anything under
0.010 dB is reported as null.

| Change tested | Effect on PSNR | Verdict |
|---|---|---|
| 9× more training data (500 → 4,585 images) | +0.006 dB | null |
| 2.7× parameters (1.37 M → 3.74 M) | +0.026 dB | null |
| 1.8× receptive field, parameters matched | +0.008 dB | null |
| Random seed alone | 0.010 dB | *the floor itself* |
| Degradation recalibration on its own | −0.016 dB | null |
| 3× training length (40 → 120 epochs) | +0.030 dB | marginal |
| **Real pairs instead of synthetic** | **+0.133 dB** | **real effect** |

One intervention in seven cleared the floor decisively, and it was a data
decision rather than a model one.

**What actually dominates the score is content**, not anything we control:

| Held-out content block | f90 | PSNR |
|---|---|---|
| Coarsest structure (ids 2868–3345) | 0.1233 | 26.664 dB |
| Finest structure (ids 478–955) | 0.3677 | 21.274 dB |
| **Spread from content alone** | | **5.39 dB** |

That spread is 40× larger than any architectural change we measured, which is
why robustness across content — not peak score — is what this pipeline targets.

Full experimental record: [`docs/ENGINEERING_LOG.md`](docs/ENGINEERING_LOG.md).

---

## Repository layout

```
run.py                  ← THE SUBMISSION SCRIPT: python run.py <in-dir> <out-dir>
models/model.pt         ← trained weights, 1.37 M params, 5.5 MB
requirements.txt        ← minimal set to run inference
requirements-full.txt   ← full training environment
src/
  degrade.py            degradation model, refitted on round-2 data
  dataset.py            pair loading, augmentation, random + block splits
  model.py              the network
  train.py              training loop
  losses.py             Charbonnier / SSIM / gradient / LPIPS
  metrics.py            PSNR, SSIM, LPIPS
  validate.py           score a checkpoint, append a row to results.csv
tools/
  check_data.py         verify dataset layout before training
  calibrate.py          does our synthetic damage match KLA's real damage?
  capacity_sweep.py     accuracy vs parameter count vs latency frontier
  real_noise_bins.py    score checkpoints on REAL data binned by noise or f90
  noise_sweep.py        robustness across synthetic noise levels
  bench.py              end-to-end timing, alternating reps
  tonight.py            unattended multi-experiment driver
  preview.py            before/after figure
docs/
  ENGINEERING_LOG.md    every experiment, including the ones that failed
  results.csv           one row per scored checkpoint
outputs/                restored test-set images
```

---

## Reproducing training

Expected data layout:

```
<data>/GT/000000.npy        256×256 float32 in [0,1]
<data>/NoisyLR/000000.npy   128×128 float32, may fall outside [0,1]
```

```bash
python tools/check_data.py --data <data>       # verify layout
python tools/calibrate.py  --data <data>       # verify degradation model
python src/train.py --data <data> --smoke      # ~5 s, CPU, checks plumbing

# the shipped configuration
python src/train.py --data <data> --out runs/r2-preal1 --amp \
                    --epochs 120 --iters 500 --batch 32 \
                    --ch 64 --nb 16 --p-real 1.0 --seed 0 --split-seed 0

python src/validate.py --data <data> --ckpt runs/r2-preal1/last.pt --baseline
```

`--seed` and `--split-seed` are deliberately separate: the validation split must
stay fixed while training randomness varies, or a seed-variance experiment
compares two models on two different validation sets and measures nothing.

Interrupted run? `--resume runs/<name>/last.pt` picks up exactly where it
stopped, optimiser and schedule included.

**Hardware:** NVIDIA RTX 4050 Laptop, 6 GB (75 W), Windows 11 · **Training
time:** 4.3 h (120 epochs, 60,000 steps) · **Peak VRAM:** 0.95 GB

---

## Notes for reviewers

* `run.py` is **fully self-contained** — the network definition is inlined, so
  it imports only `torch` and `numpy` and depends on no other file in this repo
  except the weights. It downloads nothing at runtime.
* It resolves `models/model.pt` relative to its own location, so it runs from
  any working directory.
* Outputs pass through `nan_to_num` and are clamped to `[0,1]`, so the
  finite-value and range guarantees hold regardless of input.
* Nothing hardcodes an input resolution; 256×256 → 512×512 works unchanged.
  Note the round-2 training set contains **no** 512×512 pairs, so that path is
  architecturally supported but not trained on — we have raised this with the
  organisers.
* `run.py` overlaps its fixed costs deliberately: CUDA context initialisation
  runs on a background thread, inputs are read on a thread pool, weights are
  memory-mapped, and outputs are written by a writer pool. `--profile` prints
  the breakdown. On a warm machine roughly 72% of wall-clock is process startup
  (`import torch` plus CUDA init) rather than the network, so startup cost
  amortises over the size of the test set.

## References

1. B. Lim et al., *Enhanced Deep Residual Networks for Single Image
   Super-Resolution*, CVPRW 2017. (residual SR backbone, no BatchNorm)
2. W. Shi et al., *Real-Time Single Image and Video Super-Resolution Using an
   Efficient Sub-Pixel Convolutional Neural Network*, CVPR 2016. (PixelShuffle)
3. W.-S. Lai et al., *Deep Laplacian Pyramid Networks for Fast and Accurate
   Super-Resolution*, CVPR 2017. (Charbonnier loss)
4. H. Zhao et al., *Loss Functions for Image Restoration with Neural Networks*,
   IEEE Trans. Computational Imaging, 2017. (L1 + MS-SSIM loss design)
5. R. Zhang et al., *The Unreasonable Effectiveness of Deep Features as a
   Perceptual Metric*, CVPR 2018. (LPIPS)
6. F. J. Anscombe, *The Transformation of Poisson, Binomial and
   Negative-Binomial Data*, Biometrika, 1948. (variance stabilisation)
