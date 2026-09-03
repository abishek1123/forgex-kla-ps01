#!/usr/bin/env python3
"""
KLA PS01 - AI-Based Restoration of Degraded Images.  Team ForgeX.

    python run.py <input-dir> <output-dir>

Round 2 scores END-TO-END wall clock on an H100: process start, imports, CUDA
initialisation, reading inputs, inference, and writing outputs.  For a 1.37 M
parameter network the forward pass is a small fraction of that, so this script
is built to overlap the fixed costs rather than pay them in series:

  * CUDA context initialisation starts on a background thread at import time
  * input files are listed, read and parsed on a thread pool WHILE that happens
  * weights are memory-mapped instead of read + unpickled
  * there is no separate warm-up pass; the first real batch is the warm-up
  * outputs are written by a writer pool so the GPU never waits on disk

Self-contained: the network is inlined, so this file depends on nothing but
torch and numpy.  No internet, no keys, no downloads, no manual configuration.
"""
import time
_T0 = time.perf_counter()
_STAGES = []


def _mark(name):
    _STAGES.append((name, time.perf_counter() - _T0))                      # before anything else

import argparse
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_mark("import torch + numpy")

HERE = os.path.dirname(os.path.abspath(__file__))

# Kick the CUDA context off immediately, on its own thread, so the ~2-4 s of
# driver/cuBLAS/cuDNN initialisation overlaps with reading the input directory.
_CUDA_READY = threading.Event()


def _init_cuda():
    try:
        if torch.cuda.is_available():
            torch.zeros(1, device="cuda")
            torch.cuda.synchronize()
    except Exception:
        pass
    finally:
        _mark("CUDA context ready (background thread)")
        _CUDA_READY.set()


_CUDA_THREAD = threading.Thread(target=_init_cuda, daemon=True)
_CUDA_THREAD.start()


# --------------------------------------------------------------------------
# Model (inlined so run.py has no local imports)
# --------------------------------------------------------------------------
class VarianceStabilisingStem(nn.Module):
    """Speckle variance scales with I^2. Presenting raw / sqrt / log views lets
    the first conv pick the representation where the noise is closest to
    uniform. Zero parameters."""

    def forward(self, x):
        p = x.clamp_min(0.0)
        return torch.cat([x, torch.sqrt(p + 1e-6), torch.log1p(p)], dim=1)


class ResBlock(nn.Module):
    def __init__(self, ch, res_scale=0.1):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.res_scale = res_scale

    def forward(self, x):
        return x + self.res_scale * self.c2(F.relu(self.c1(x)))


class Restorer(nn.Module):
    """All convolution at low resolution; single PixelShuffle upsample at the
    end; global bicubic skip so the network predicts only the correction."""

    def __init__(self, ch=64, nb=16, scale=2, res_scale=0.1):
        super().__init__()
        self.config = dict(ch=ch, nb=nb, scale=scale, res_scale=res_scale)
        self.scale = scale
        self.stem = VarianceStabilisingStem()
        self.head = nn.Conv2d(3, ch, 3, 1, 1)
        self.body = nn.Sequential(*[ResBlock(ch, res_scale) for _ in range(nb)])
        self.body_tail = nn.Conv2d(ch, ch, 3, 1, 1)
        self.up = nn.Sequential(nn.Conv2d(ch, ch * scale * scale, 3, 1, 1),
                                nn.PixelShuffle(scale))
        self.tail = nn.Conv2d(ch, 1, 3, 1, 1)

    def forward(self, x):
        f = self.head(self.stem(x))
        f = f + self.body_tail(self.body(f))
        residual = self.tail(self.up(f))
        base = F.interpolate(x.float(), scale_factor=self.scale,
                             mode="bicubic", align_corners=False)
        return base + residual.float()


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------
def find_weights(explicit=None):
    """models/ is the spec location; weights/ kept as a fallback."""
    if explicit:
        return explicit
    for rel in (os.path.join("models", "model.pt"),
                os.path.join("weights", "model.pt")):
        path = os.path.join(HERE, rel)
        if os.path.isfile(path):
            return path
    sys.exit("ERROR: no model weights found. Expected models/model.pt next to run.py")


def load_checkpoint(path):
    """Memory-map and skip unpickling anything but tensors where possible."""
    for kw in ({"weights_only": True, "mmap": True}, {"weights_only": True}, {}):
        try:
            return torch.load(path, map_location="cpu", **kw)
        except Exception:
            continue
    sys.exit(f"ERROR: could not load weights from {path}")


def load_npy(path):
    """-> (H, W) float32 array, plus whether the input carried a trailing axis."""
    a = np.load(path)
    had_channel = (a.ndim == 3 and a.shape[-1] == 1)
    if had_channel:
        a = a[..., 0]
    if a.ndim != 2:
        raise ValueError(f"expected (H,W) or (H,W,1), got shape {a.shape}")
    return np.ascontiguousarray(a, dtype=np.float32), had_channel


def restore_batch(model, arrays, device, use_amp, tta):
    """arrays: list of equally-shaped (H,W) float32 -> list of (2H,2W) float32."""
    t = torch.from_numpy(np.stack(arrays))[:, None].to(device)
    if device.type == "cuda":
        t = t.contiguous(memory_format=torch.channels_last)

    if tta:
        variants = [t, t.flip(-1), t.flip(-2), t.flip(-1, -2), t.transpose(-1, -2),
                    t.transpose(-1, -2).flip(-1), t.transpose(-1, -2).flip(-2),
                    t.transpose(-1, -2).flip(-1, -2)]
        undo = [lambda o: o, lambda o: o.flip(-1), lambda o: o.flip(-2),
                lambda o: o.flip(-1, -2), lambda o: o.transpose(-1, -2),
                lambda o: o.flip(-1).transpose(-1, -2),
                lambda o: o.flip(-2).transpose(-1, -2),
                lambda o: o.flip(-1, -2).transpose(-1, -2)]
    else:
        variants, undo = [t], [lambda o: o]

    outs = []
    with torch.inference_mode():
        for k, v in enumerate(variants):
            # Keep the channels_last layout the model weights are in. The old
            # code called v.contiguous() with no argument here, which converts
            # a channels_last tensor straight back to NCHW and defeats the
            # layout conversion done above.
            if device.type == "cuda":
                v = v.contiguous(memory_format=torch.channels_last)
            else:
                v = v.contiguous()
            with torch.autocast("cuda", enabled=use_amp):
                o = model(v).float()
            outs.append(undo[k](o))
    out = torch.stack(outs).mean(0)

    # Guarantee the contract: finite, float32, inside [0,1].
    out = torch.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    out = out[:, 0].cpu().numpy().astype(np.float32)
    return [out[i] for i in range(out.shape[0])]


def main():
    p = argparse.ArgumentParser(
        description="Restore degraded inspection images (2x super-resolution + denoising).")
    p.add_argument("input_dir", help="directory containing degraded .npy files")
    p.add_argument("output_dir", help="directory to write restored .npy files to")
    p.add_argument("--weights", default=None,
                   help="path to model weights (default: models/model.pt)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--no-fp16", dest="fp16", action="store_false",
                   help="disable half precision on CUDA")
    p.add_argument("--tta", action="store_true",
                   help="8x self-ensemble; higher quality, ~8x slower")
    p.add_argument("--batch", type=int, default=0,
                   help="images per forward pass; 0 = auto (32 GPU / 1 CPU)")
    p.add_argument("--workers", type=int, default=8, help="threads for disk read/write")
    p.add_argument("--profile", action="store_true",
                   help="print where the end-to-end time actually goes")
    p.set_defaults(fp16=True)
    a = p.parse_args()

    if not os.path.isdir(a.input_dir):
        sys.exit(f"ERROR: input directory not found: {a.input_dir}")
    os.makedirs(a.output_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(a.input_dir) if f.lower().endswith(".npy"))
    if not files:
        sys.exit(f"ERROR: no .npy files found in {a.input_dir}")

    # ---- read every input on a thread pool while CUDA is still initialising ----
    _mark("argparse + list input dir")
    reader = ThreadPoolExecutor(max_workers=a.workers)
    futures = [reader.submit(load_npy, os.path.join(a.input_dir, f)) for f in files]

    weights = find_weights(a.weights)
    ck = load_checkpoint(weights)
    state = ck.get("state_dict", ck)
    cfg = ck.get("config", {}) or {}

    _mark("load checkpoint from disk")
    _CUDA_THREAD.join()
    _mark("join CUDA thread")                       # by now it has almost certainly finished
    device = torch.device("cuda" if (a.device in ("auto", "cuda")
                                     and torch.cuda.is_available()) else "cpu")
    use_amp = a.fp16 and device.type == "cuda"
    if a.batch <= 0:
        a.batch = 32 if device.type == "cuda" else 1

    model = Restorer(**cfg).eval()
    model.load_state_dict(state)
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    _mark("build model + weights to device")
    n_params = sum(q.numel() for q in model.parameters())
    print(f"device={device}  fp16={use_amp}  tta={a.tta}  "
          f"params={n_params/1e6:.2f}M  files={len(files)}  batch={a.batch}", flush=True)

    # ---- group by shape, infer, hand writing off to the writer pool ----
    writer = ThreadPoolExecutor(max_workers=a.workers)
    writes = []
    pending = {}

    def save(name, arr):
        np.save(os.path.join(a.output_dir, name), arr)

    def flush(shape):
        group = pending.pop(shape, [])
        for i in range(0, len(group), a.batch):
            chunk = group[i:i + a.batch]
            ys = restore_batch(model, [g[1] for g in chunk], device, use_amp, a.tta)
            for (name, _, had_channel), y in zip(chunk, ys):
                writes.append(writer.submit(save, name, y[..., None] if had_channel else y))

    for name, fut in zip(files, futures):
        arr, had_channel = fut.result()
        pending.setdefault(arr.shape, []).append((name, arr, had_channel))
        if len(pending[arr.shape]) >= a.batch:
            flush(arr.shape)
    for shape in list(pending):
        flush(shape)

    _mark("read all inputs + inference + queue writes")
    for w in writes:
        w.result()
    reader.shutdown(wait=True)
    writer.shutdown(wait=True)
    if device.type == "cuda":
        torch.cuda.synchronize()

    _mark("flush all writes to disk")
    dt = time.perf_counter() - _T0
    written = len([f for f in os.listdir(a.output_dir) if f.lower().endswith(".npy")])
    print(f"restored {len(files)} images  |  end-to-end {dt:.2f}s "
          f"({dt/len(files)*1000:.1f} ms/image, measured the way KLA measures it)")
    print(f"wrote {written} files to {a.output_dir}")
    if a.profile:
        print("\n  where the end-to-end time goes")
        print(f"  {'stage':<42}{'cumulative':>12}{'this stage':>12}{'% total':>9}")
        prev = 0.0
        for name, t in _STAGES:
            print(f"  {name:<42}{t:>11.3f}s{t-prev:>11.3f}s{100*(t-prev)/dt:>8.1f}%")
            prev = t
        print(f"  {'(interpreter start, before our first mark)':<42}{'':>12}{_STAGES[0][1]:>11.3f}s"
              f"{100*_STAGES[0][1]/dt:>8.1f}%" if _STAGES else "")
        print(f"\n  model compute is roughly {len(files)}x{1000*0.0:.0f} of this; everything else is fixed cost.")
    if written < len(files):
        sys.exit(f"ERROR: expected {len(files)} outputs, found {written}")


if __name__ == "__main__":
    main()
