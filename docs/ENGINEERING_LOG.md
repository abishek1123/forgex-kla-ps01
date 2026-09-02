# Engineering log — KLA PS01, Round 2

Every experiment we ran on the round-2 data, including the ones that failed and
the conclusions we had to withdraw. Team ForgeX · VIT Vellore.

---

## 0. Method: how we decide that something is real

Nothing below is reported unless it clears a measured noise floor.

**The floor.** Two training runs differing *only* in random seed — same data,
same split, same schedule — land **0.010 dB** apart when scored as a
final-5-epoch mean. Scored by best epoch instead, the same pair appears
**0.055 dB** apart, because best-epoch selection is a max over ~120 noisy draws
and therefore rewards luck.

Consequences we hold to throughout:

1. Every number is a **final-5-epoch mean**, never a best epoch.
2. Anything under 0.010 dB is reported as **null**, regardless of sign.
3. The validation split is fixed by `--split-seed`, independent of `--seed`.
   Before this separation existed, a single `--seed` drove both, so two "seed
   variance" runs would have been scored on two *different* validation sets and
   the experiment would have measured nothing.

---

## 1. Round 1 → Round 2: what actually changed

We re-derived every data property rather than assuming it carried over. Almost
none of it did.

| Property | Round 1 | Round 2 | How measured |
|---|---|---|---|
| Content | natural photographs | SEM micrographs | contact sheet, visual |
| Pairs | 3,200 | 4,785 | inventory |
| Sizes | mixed | all 256→128, no 512×512 | shape histogram over all pairs |
| f90 | 0.0945 | 0.2913 | radial power spectrum |
| Normalisation | — | per-image min–max, exact `[0,1]` | 200/200 spans exactly `[0,1]` |
| Downsampling | 2×2 box average | 2×2 box average | fit vs bilinear/bicubic/area/blur-decimate |
| Bicubic ×2 | 23.23 dB | 20.22 dB | direct measurement |

**Dataset ids are content-ordered.** Scoring by contiguous id block gives f90
0.3677 (ids 478–955) monotonically down to 0.1233 (ids 2868–3345), while
σ_mul stays flat at ≈0.19 throughout. So content varies along the id axis and
noise does not — which makes id blocks usable as true out-of-distribution
splits (`--holdout-range`), and means a random split is *easier* than the hidden
test set will be.

---

## 2. The degradation model, refitted

### 2.1 The fit

Regressing residual variance on the ground-truth block statistics:

```
var(residual) = σ_add² + σ_mul²·m²  +  c·v          R² = 0.948
```

`m` = 2×2 block mean (the noise-free low-res reference), `v` = within-block
variance (how much detail the block destroyed).

| Parameter | Observed range | Training range (deliberately wider) |
|---|---|---|
| σ_add | 0.000 – 0.065 | 0.000 – 0.094 |
| σ_mul | 0.155 – 0.258 | 0.116 – 0.335 |
| c | 0.026 – 0.234 | 0.016 – 0.316 |

### 2.2 Three corrections to the round-1 generator

| Change | Evidence |
|---|---|
| **Added `c·v`** | detail share of variance: 4.4% on SEM vs 1.8% on photographs; `c/b` ratio 3.40 vs 2.22 |
| **Removed blur** | measured *harmful*: residual error 0.01074 → 0.01235 with blur present |
| **Gaussian → Gamma(k=14)** | residual is spatially white (lag-1 −0.055 / −0.053 / +0.009) but skew +0.43, excess kurtosis +0.62 |
| **Removed operation-order randomisation** | round 1 randomised order on the brief's "do not read into the order"; on round-2 data the order is identifiable and fixed |

### 2.3 Calibration check

Refitting our own synthetic output with the same estimator used on KLA's data:

| | σ_add | σ_mul |
|---|---|---|
| KLA's real pairs | 0.0414 | 0.1863 |
| Our rebuilt generator | 0.0415 | 0.1986 |

Reproduce: `python tools/calibrate.py --data <data>`.

---

## 3. The experiment ledger

All runs: 120 epochs unless noted, 500 iterations/epoch, batch 32, crop 64,
ch 64, nb 16, seed 0, split-seed 0, AMP on. Scores are final-5-epoch means on
the same held-out 200.

| Run | What it tested | PSNR | SSIM | Δ vs reference | Verdict |
|---|---|---|---|---|---|
| `r2-preal1` | 100% real pairs | **23.2672** | **0.6124** | — (reference) | **shipped** |
| `r2-newgen` | 70% rebuilt synthetic | 23.2178 | 0.5925 | −0.0494 | rejected |
| `r2-long120` | 70% old synthetic | 23.1342 | 0.5930 | −0.1330 | rejected |
| `r2-degfix` | degradation recalibration only | 23.1180 | 0.5932 | −0.1492 | rejected |
| `r2-full-s0` | 40-epoch baseline, seed 0 | 23.1038 | 0.5839 | — | screening |
| `r2-full-s1` | identical but seed 1 | ≈23.09 | — | 0.010 | **the noise floor** |
| `r2-sub500` | 500 training images only | 23.0893 | 0.5838 | +0.006 vs full | null |
| `r2-ctx127` | 1.8× receptive field, params matched | — | — | +0.008 | null |
| `v1` (round 1 model) | zero-shot on round-2 data | 23.1721 | 0.5596 | −0.095 | see §4.2 |

### 3.1 The headline result

**Question, pre-registered before the run:** does a correctly-calibrated
generator beat training on real pairs alone?

**Answer: no, and the rebuild still improved.** `r2-newgen` beat the old
generator by **+0.084 dB** (8× the noise floor) — the recalibration genuinely
worked. It still lost to real pairs by **0.049 dB and 0.020 SSIM**.

We ship `p_real = 1.0`. The round-1 conclusion — synthesise unlimited pairs,
keep 30% real as insurance — does not survive on round-2 data. We changed the
pipeline rather than the story.

### 3.2 The null-result table

| Change | Effect | Verdict |
|---|---|---|
| 9× training data (500 → 4,585) | +0.006 dB | null |
| 2.7× parameters (1.37 M → 3.74 M) | +0.026 dB | null |
| 1.8× receptive field, params matched | +0.008 dB | null |
| Random seed | 0.010 dB | the floor |
| Degradation recalibration alone | −0.016 dB | null |
| 3× training length (40 → 120 epochs) | +0.030 dB | marginal |
| Real pairs instead of synthetic | **+0.133 dB** | **real** |

One in seven cleared the floor. This table is the most useful artefact we own:
it tells us where *not* to spend the remaining time.

### 3.3 Content dominates everything we control

| Block | f90 | PSNR |
|---|---|---|
| Coarsest (ids 2868–3345) | 0.1233 | 26.664 dB |
| Finest (ids 478–955) | 0.3677 | 21.274 dB |
| **Spread** | | **5.39 dB** |

40× larger than any architectural change measured. Which images the hidden test
set contains will move the score more than anything left to tune.

### 3.4 Capacity frontier (in progress)

40-epoch screening runs, `p_real 1.0`, smallest first:

| Config | Parameters | PSNR | SSIM |
|---|---|---|---|
| ch12 nb4 | 17,449 | 22.7959 | 0.5804 |
| ch16 nb4 | 30,753 | 22.8300 | 0.5827 |
| ch16 nb8 | 49,313 | 22.8738 | 0.5854 |
| ch64 nb16 (indicative) | 1,368,705 | ≈23.10 | 0.5839 |

28× fewer parameters costs ≈0.23 dB. Note ch16 nb8 already **exceeds** the
reference on SSIM. Remaining configs and measured forward latency pending.

### 3.5 Out-of-distribution holdout (in progress)

Ids 2868–3345 — structure 3× coarser than the rest — withheld from training
entirely. In-distribution reference (`r2-preal1`, which trained on those images)
scores **24.5622 dB / 0.6638 SSIM / 0.3385 LPIPS** on that block. The held-out
model reached 24.3830 at epoch 42 of 120 before the run was stopped; correcting
for the 0.131 dB `r2-preal1` gains between epoch 42 and 119 projects a
generalisation cost near **0.05 dB**. Not quoted as a result until the run
finishes.

---

## 4. Corrections — claims we had to withdraw

Kept deliberately. Each is a check that caught something, and each changed how
we verify afterwards.

### 4.1 A CPU test used to certify GPU behaviour

We claimed an optimised `run.py` was bit-identical to its predecessor on the
strength of a CPU comparison. On GPU it differed by 2.4e-4 — the fp16 quantum.
The CPU check had no autocast, no `cudnn.benchmark`, and batch 1, so it
exercised none of the code paths that could differ. **This is the same failure
as round 1's `.half()`-without-autocast verification.** Equivalence is now
checked on the device that will run it, in the configuration that will run.

### 4.2 Two predictions that were simply wrong

* Predicted degradation recalibration would recover ≈1.2 dB. It recovered
  **zero** (`r2-degfix` 23.1180 vs `r2-long120` 23.1342).
* Predicted retraining on-domain would recover ≈1.2 dB PSNR. It recovered
  **0 PSNR** (though +0.023 SSIM and −0.10 LPIPS).

Both errors share a cause: treating "gain over bicubic" as a fixed quantity when
the achievable range had itself shrunk. The round-2 bicubic floor is 3.0 dB
lower than round 1's, so equal *skill* shows up as a lower *score*.

### 4.3 Optimisations that were slower

An "optimised" inference script ran **3.8× slower** on GPU (17.0 s vs 4.5 s).
Cause: `cudnn.benchmark=True` autotuning and per-batch `pin_memory()` — both
added by reasoning, neither measured. Removing them gave 3.8 s. Nothing enters
`run.py` now on the strength of an argument alone.

### 4.4 A biased benchmark, caught before it reached a slide

`tools/noise_sweep.py` builds its test inputs with *our own* generator, which
structurally favours a synthetic-trained model. It was nearly used to justify a
model-selection decision. `tools/real_noise_bins.py` replaces it: real data
only, binned by measured σ or by f90.

### 4.5 A seed that controlled two things

`--seed` originally drove both the train/validation split and training
randomness, which would have made the entire seed-variance experiment —
and therefore the noise floor — meaningless. `--split-seed` now separates them.

---

## 5. Latency

End-to-end, warm, 200 images, RTX 4050 Laptop:

| Stage | Time | Share |
|---|---|---|
| `import torch` | 1.62 s | 35% |
| CUDA context init | 1.67 s | 37% |
| Compute + disk I/O | 1.17 s | 26% |
| **Total** | **4.58 s** | |

Cold, `import torch` alone is 11.07 s (75.8% of the run). **Roughly 72% of
wall-clock is fixed process startup, not the network** — so shrinking the model
attacks at most the last quarter of the budget, and startup amortises over the
size of the test set. `run.py` therefore overlaps what it can: CUDA
initialisation on a background thread, threaded input reads, memory-mapped
weights, a writer pool.

---

## 6. Open questions

1. **512×512.** The problem statement mentions it; the round-2 training set
   contains none. The model is fully convolutional so it runs at that scale, but
   it is untrained there. Raised with the organisers; no reply yet.
2. **Metric weighting.** The organisers have stated they do not know how PSNR,
   SSIM and LPIPS will be combined. We optimise all three jointly rather than
   betting on one.
3. **Per-bin ceiling.** §3.3 gives per-content PSNR but not per-content
   *ceiling*; the two together would say whether the fine-structure blocks are
   genuinely harder or merely further from convergence.
