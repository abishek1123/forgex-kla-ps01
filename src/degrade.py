"""Degradation model for the KLA image-restoration challenge.

REBUILT 1 Sep 2026 from measurements on semicon_train_data (4,785 real pairs).
The round-1 version is preserved at the bottom for the record.

WHAT WE MEASURED
----------------
Writing m for the 2x2 block mean of the ground truth (= the noise-free low-res
reference) and v for the within-block variance, a least-squares fit over 140
two-dimensional bins covering 409,600 real low-res pixels gives

    var(residual) = sigma_add^2 + sigma_mul^2 * m^2 + c * v        R^2 = 0.948

The third term is the one the round-1 generator never had. It scales with local
detail, and it grew 2.4x moving from photographs to SEM (1.8% -> 4.4% of
residual energy) because SEM structure is ~3x finer. A generator missing it is
systematically wrong in proportion to how detailed the image is -- which is the
worst possible failure mode for this dataset.

Three further findings, each of which changed the code:

  * Downsampling is an exact 2x2 box average. Tested against bilinear, bicubic,
    area, and Gaussian-blur-then-decimate at sigma 0.3/0.5/0.7/1.0. Every blur
    setting made the residual WORSE (0.01074 -> 0.01235). The round-1 generator
    blurred 25% of samples. That was a defect; blur_p is gone.

  * The round-1 generator randomised the order of [speckle, downsample, gauss].
    Speckle applied before a 2x2 average has a quarter the variance of speckle
    applied after, so those orders are materially different processes and at most
    one matches reality. Emitting a mixture of six guarantees five are wrong.
    We now generate the measured variance structure directly at low resolution,
    which is agnostic to the mechanism and matches what the data actually shows.

  * The residual is spatially white (neighbour correlation -0.055 / -0.053 /
    +0.009) so independent per-pixel draws are faithful. It is NOT Gaussian:
    skew +0.43, excess kurtosis +0.62. We draw from a shifted Gamma, which is
    the physically correct family for speckle and reproduces both.
"""
import numpy as np

# --- per-image parameter ranges, fitted individually on 25 real pairs -------
# sigma_add  mean 0.0449   p5 0.0136  p95 0.0748   observed 0.000 - 0.081
# sigma_mul  mean 0.1618   p5 0.1200  p95 0.1995   observed 0.109 - 0.204
# c          mean 0.1606   p5 0.0519  p95 0.2307   observed 0.031 - 0.357
# Parameters solved (not scaled) so that a generated set reproduces THREE
# properties of the real pairs simultaneously: the two-parameter fit that
# tools/calibrate.py performs, and the total residual variance.
#   real       sigma_add 0.0404   sigma_mul 0.1980   total var 0.011254
#   generated  sigma_add 0.0400   sigma_mul 0.2010   total var 0.011848
#   error           1.0%               1.5%               5.3%
OBSERVED = dict(sigma_add=(0.000, 0.065), sigma_mul=(0.155, 0.258), c=(0.026, 0.234))

# Training ranges bracket the observed distribution on both ends, the posture
# that worked in round 1. Deliberately wider because the hidden test set is
# stated to contain out-of-distribution samples at varying noise levels.
TRAIN = dict(sigma_add=(0.000, 0.094), sigma_mul=(0.116, 0.335), c=(0.016, 0.316))

# Skew 2/sqrt(k), excess kurtosis 6/k. k=14 gives 0.53 / 0.43 against the
# measured 0.43 / 0.62 -- no single Gamma matches both exactly; this splits it.
GAMMA_K = 14.0


def box_down(x, f=2):
    """Exact f x f box average -- this is how KLA downsamples. Verified against
    bilinear, bicubic, area and blur-then-decimate; box wins."""
    H, W = x.shape
    H, W = (H // f) * f, (W // f) * f
    return x[:H, :W].reshape(H // f, f, W // f, f).mean(axis=(1, 3))


def block_stats(gt, f=2):
    """-> (m, v): the noise-free low-res reference and the within-block variance."""
    x = np.asarray(gt, dtype=np.float64)
    H, W = x.shape
    H, W = (H // f) * f, (W // f) * f
    b = x[:H, :W].reshape(H // f, f, W // f, f)
    m = b.mean(axis=(1, 3))
    v = (b ** 2).mean(axis=(1, 3)) - m * m
    return m, np.maximum(v, 0.0)


def _unit_skewed(rng, shape, k=GAMMA_K):
    """Mean 0, variance 1, right-skewed -- the speckle family."""
    return (rng.gamma(k, 1.0, shape) - k) / np.sqrt(k)


def degrade_at(gt, rng, sigma_add, sigma_mul, c, k=GAMMA_K):
    """Degrade with the three parameters pinned rather than sampled."""
    m, v = block_stats(gt)
    var = sigma_add ** 2 + (sigma_mul ** 2) * (m * m) + c * v
    out = m + np.sqrt(var) * _unit_skewed(rng, m.shape, k)
    return np.ascontiguousarray(out, dtype=np.float32)


def degrade(gt, rng, cfg=TRAIN):
    """Clean HR (H,W) float32 in [0,1] -> noisy LR (H/2,W/2) float32.

    The output may fall outside [0,1]; that is correct and matches KLA's data.
    `rng` must be a numpy Generator so every worker/epoch gets fresh noise.
    """
    return degrade_at(gt, rng,
                      float(rng.uniform(*cfg["sigma_add"])),
                      float(rng.uniform(*cfg["sigma_mul"])),
                      float(rng.uniform(*cfg["c"])),
                      cfg.get("gamma_k", GAMMA_K))


# --- round 1, kept for the record ------------------------------------------
# Calibrated on the photograph dataset. var = sigma_add^2 + sigma_mul^2 * I^2,
# no detail term, Gaussian noise, 25% blur, randomised operation order.
TRAIN_R1    = dict(speckle=(0.02, 0.30), gauss=(0.00, 0.18), blur_p=0.25, blur_sigma=(0.3, 0.9))
OBSERVED_R1 = dict(speckle=(0.13, 0.21), gauss=(0.00, 0.07), blur_p=0.0,  blur_sigma=(0.0, 0.0))
