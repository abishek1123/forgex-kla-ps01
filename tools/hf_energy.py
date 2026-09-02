#!/usr/bin/env python3
"""Does the model INVENT detail? Measured, not asserted.

    python tools/hf_energy.py --data <train-root> --ckpts runs/*/last.pt

We refuse GAN and perceptual losses on the grounds that invented texture on an
inspection image is a fabricated defect. That has been an argument. This makes
it a number.

A 2x downsample destroys every spatial frequency above the low-resolution
Nyquist limit. Those frequencies are GONE -- any energy the model puts back
there is inferred, not recovered. So compare, in exactly that band:

    ratio = HF energy of the OUTPUT / HF energy of the GROUND TRUTH

    ratio ~ 1.0   the model reconstructs about as much fine detail as truly
                  exists. Ideal.
    ratio < 1.0   under-sharpened: blurry, but everything it draws is real.
                  The safe failure for a fab.
    ratio > 1.0   it is generating MORE fine structure than the ground truth
                  contains. That is hallucination, and on an inspection image
                  it is a defect that was never there.

Bicubic is included as the floor: it invents nothing, so its ratio shows how
much of the band is genuinely unrecoverable.

This is what makes an LPIPS term safe to TEST rather than refuse outright -- add
the term, watch this number, and reject any weight that pushes it past 1.
"""
import argparse, os, sys
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
from model import Restorer                                    # noqa: E402
from dataset import list_ids, make_split                      # noqa: E402

_R = {}


def hf_fraction(x, cut=0.5):
    """Energy above `cut` x Nyquist, as a fraction of total non-DC energy.

    cut=0.5 is exactly the low-resolution Nyquist limit for a 2x downsample --
    the band the degradation destroyed and the model must infer.
    """
    x = np.asarray(x, dtype=np.float64)
    F2 = np.abs(np.fft.fftshift(np.fft.fft2(x - x.mean()))) ** 2
    H, W = F2.shape
    if (H, W) not in _R:
        y, xx = np.ogrid[:H, :W]
        _R[(H, W)] = np.sqrt(((y - H // 2) / (H // 2)) ** 2 + ((xx - W // 2) / (W // 2)) ** 2)
    r = _R[(H, W)]
    tot = F2.sum()
    return float(F2[r > cut].sum() / tot) if tot > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--cut", type=float, default=0.5)
    a = ap.parse_args()

    gt_dir = os.path.join(a.data, "GT")
    lr_dir = os.path.join(a.data, "NoisyLR")
    _, val_ids = make_split(gt_dir, n_val=200, seed=0)
    val_ids = val_ids[:a.n]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(val_ids)} held-out images, band above {a.cut:.2f}x Nyquist, device={dev}\n")

    gts, lrs = [], []
    for i in val_ids:
        g = np.load(os.path.join(gt_dir, i + ".npy")).astype(np.float32)
        l = np.load(os.path.join(lr_dir, i + ".npy")).astype(np.float32)
        gts.append(g[..., 0] if g.ndim == 3 else g)
        lrs.append(l[..., 0] if l.ndim == 3 else l)

    gt_hf = np.array([hf_fraction(g, a.cut) for g in gts])

    rows = []
    # bicubic floor -- invents nothing by construction
    bic = []
    for l, g in zip(lrs, gts):
        t = torch.from_numpy(l)[None, None]
        up = F.interpolate(t, size=g.shape, mode="bicubic", align_corners=False)
        bic.append(hf_fraction(up[0, 0].numpy().clip(0, 1), a.cut))
    rows.append(("bicubic (floor)", np.array(bic)))

    for ck in a.ckpts:
        tag = os.path.basename(os.path.dirname(ck)) or os.path.basename(ck)
        d = torch.load(ck, map_location="cpu", weights_only=False)
        m = Restorer(**(d.get("config", {}) or {})).eval().to(dev)
        m.load_state_dict(d.get("state_dict", d))
        out = []
        with torch.inference_mode():
            for l, g in zip(lrs, gts):
                t = torch.from_numpy(l)[None, None].to(dev)
                y = m(t).float().clamp(0, 1)[0, 0].cpu().numpy()
                out.append(hf_fraction(y, a.cut))
        rows.append((tag, np.array(out)))

    print(f"{'model':<20}{'HF(out)':>10}{'HF(gt)':>10}{'ratio':>9}{'p95':>9}   verdict")
    print("-" * 76)
    for tag, hf in rows:
        ratio = hf / np.maximum(gt_hf, 1e-12)
        m_, p95 = float(ratio.mean()), float(np.percentile(ratio, 95))
        if p95 > 1.05:
            v = "HALLUCINATING -- invents structure"
        elif m_ > 1.02:
            v = "over-sharpening, watch it"
        elif m_ < 0.55:
            v = "very conservative / blurry"
        else:
            v = "safe -- draws less than exists"
        print(f"{tag:<20}{hf.mean():>10.4f}{gt_hf.mean():>10.4f}{m_:>9.3f}{p95:>9.3f}   {v}")
    print("\nratio > 1 anywhere means energy above the LR Nyquist that the ground")
    print("truth does not contain -- i.e. detail the model made up.")


if __name__ == "__main__":
    main()
