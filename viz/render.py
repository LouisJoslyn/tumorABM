"""Render a run's snapshots into a glowing dark-mode MP4 (decoupled from the simulation).

Usage: python -m viz.render runs/<name> [--mode state|lineage] [--scale 3] [--fps 30] [--frames-only]
"""
import argparse
import json
import os

import numpy as np
import zarr
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

BG = np.array([0.020, 0.027, 0.047])
TISSUE = np.array([0.045, 0.055, 0.085])
VESSEL = np.array([0.85, 0.16, 0.28])
PROLIF = np.array([1.00, 0.72, 0.28])
QUIES = np.array([0.80, 0.38, 0.14])
NECRO = np.array([0.22, 0.05, 0.07])
APOP = np.array([1.00, 0.25, 0.70])
T_ACT = np.array([0.25, 0.95, 1.00])
T_EXH = np.array([0.42, 0.50, 0.66])
FLASH = np.array([1.00, 0.80, 0.95])


def _font(size):
    for p in ["/System/Library/Fonts/SFNSMono.ttf", "/System/Library/Fonts/Menlo.ttc",
              "/Library/Fonts/Arial.ttf"]:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                pass
    return ImageFont.load_default()


def _lineage_colors(lin):
    """Deterministic vivid colour per lineage id (golden-ratio hue walk)."""
    h = (lin.astype(np.float64) * 0.6180339887) % 1.0
    s = 0.75
    v = 1.0
    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], -1)


class Renderer:
    def __init__(self, run_dir, mode="state", scale=3, show_oxygen=True):
        self.run_dir = run_dir
        self.z = zarr.open_group(os.path.join(run_dir, "snap.zarr"), mode="r")
        self.n = int(self.z.attrs["n_frames"])
        self.fh = float(self.z.attrs["frame_hours"])
        self.mode = mode
        self.scale = scale
        self.show_oxygen = show_oxygen
        self.t_col = np.array([1.0, 1.0, 1.0]) if mode == "lineage" else T_ACT
        kills = np.load(os.path.join(run_dir, "kills.npy"))
        self.kills_by_frame = {}
        for f, x, y in kills:
            self.kills_by_frame.setdefault(int(f), []).append((x, y))
        with open(os.path.join(run_dir, "timeseries.json")) as fh:
            self.ts = json.load(fh)
        nx, ny = self.z["state"].shape[1:]
        self.shape = (nx, ny)
        self.trail = np.zeros((nx, ny))
        self.flash = np.zeros((nx, ny))
        rng = np.random.default_rng(0)
        self.grain = rng.random((nx, ny))
        self.fine_grain = rng.random((nx * scale, ny * scale)).astype(np.float32)
        self.font = _font(22)
        self.font_s = _font(16)

    def _up(self, a):
        return np.repeat(np.repeat(a, self.scale, 0), self.scale, 1)

    def _vignette(self, h, w):
        if getattr(self, "_vig", None) is None or self._vig.shape != (h, w):
            yy, xx = np.mgrid[0:h, 0:w]
            r = np.sqrt(((yy - h / 2) / (h / 2)) ** 2 + ((xx - w / 2) / (w / 2)) ** 2)
            self._vig = np.clip(1.15 - 0.35 * r ** 2, 0.55, 1.0)[..., None].astype(np.float32)
        return self._vig

    def frame(self, k):
        st = self.z["state"][k]
        tc = self.z["tcell"][k]
        s = self.scale
        nx, ny = self.shape
        g = self.grain[..., None]
        img = TISSUE * (0.85 + 0.3 * g)
        img = np.broadcast_to(img, (nx, ny, 3)).copy()
        if self.show_oxygen:
            o = self.z["oxygen"][k].astype(np.float64)
            hyp = np.clip((0.35 - o) / 0.35, 0, 1)  # hypoxia as violet haze
            img += hyp[..., None] * np.array([0.05, 0.0, 0.09])
        emis = np.zeros((nx, ny, 3))
        if self.mode == "lineage":
            lin = self.z["lineage"][k]
            col = _lineage_colors(lin)
            live = (st == 1) | (st == 2)
            shade = np.where(st == 1, 1.0, 0.55)[..., None]
            img[live] = (col * shade * (0.85 + 0.15 * g))[live]
            emis[st == 1] = col[st == 1] * 0.35
        else:
            m = st == 1
            img[m] = (PROLIF * (0.82 + 0.18 * g))[m]
            emis[m] = PROLIF * 0.45
            m = st == 2
            img[m] = (QUIES * (0.65 + 0.3 * g))[m]
        m = st == 3
        img[m] = (NECRO * (0.5 + 0.9 * g))[m]
        m = st == 4
        img[m] = APOP * 0.8
        emis[m] = APOP * 0.7
        m = st == 5
        img[m] = VESSEL * 0.7
        emis[m] = VESSEL * 0.35
        act = (tc == 1).astype(np.float64)
        exh = (tc == 2).astype(np.float64)
        emis += act[..., None] * self.t_col * 1.4
        # kill flashes (grid res)
        self.flash *= 0.6
        for x, y in self.kills_by_frame.get(k, []):
            self.flash[x, y] += 1.0
        emis += np.clip(self.flash, 0, 2)[..., None] * FLASH * 0.75
        # trails (grid res)
        self.trail = self.trail * 0.82 + act * 0.55

        # glow computed at grid resolution, then upsampled bilinearly
        glow = (ndimage.gaussian_filter(emis, sigma=(1.0, 1.0, 0)) * 1.4
                + ndimage.gaussian_filter(emis, sigma=(3.0, 3.0, 0)) * 0.9)
        glow_big = ndimage.uniform_filter(self._up(glow.astype(np.float32)), size=(s, s, 1))
        tr_big = ndimage.uniform_filter(
            self._up(ndimage.gaussian_filter(self.trail, 0.4).astype(np.float32)), size=(s, s))

        big = self._up(img.astype(np.float32))
        big = ndimage.uniform_filter(big, size=(2, 2, 1))  # soften cell edges
        big *= (0.93 + 0.07 * self.fine_grain[..., None])
        big += tr_big[..., None] * self.t_col * 0.35
        # crisp T-cell dots
        dots = ndimage.gaussian_filter(self._up(act + 2.0 * exh), 0.55 * s / 3)
        da = np.clip(ndimage.gaussian_filter(self._up(act), 0.55 * s / 3) * 2.2, 0, 1)[..., None]
        de = np.clip((dots - da[..., 0] / 2.2) / 2.0 * 2.0, 0, 1)[..., None]
        big = big * (1 - da) + da * self.t_col
        big = big * (1 - de) + de * T_EXH
        out = big + glow_big
        out *= self._vignette(*out.shape[:2])
        out = 1 - np.exp(-out * 1.25)  # filmic tonemap
        return (np.clip(out, 0, 1) ** (1 / 1.1) * 255).astype(np.uint8)

    def hud(self, rgb, k):
        im = Image.fromarray(rgb)
        d = ImageDraw.Draw(im, "RGBA")
        s = self.ts["series"]
        day = s["day"][k]
        d.rounded_rectangle([14, 14, 300, 118], radius=10, fill=(5, 8, 16, 170))
        d.text((28, 22), f"DAY {day:5.1f}", font=self.font, fill=(235, 240, 255, 255))
        d.text((28, 56), f"tumor   {s['tumor_live'][k]:6d}", font=self.font_s, fill=(255, 190, 90, 255))
        d.text((28, 76), f"T cells {s['t_active'][k]:6d}", font=self.font_s, fill=(90, 240, 255, 255))
        d.text((28, 96), f"kills   {s['kills_cum'][k]:6d}", font=self.font_s, fill=(255, 120, 200, 255))
        # treatment badge
        td = self.ts["meta"].get("treat_day")
        if td is not None and day >= td:
            d.rounded_rectangle([14, 128, 300, 162], radius=10, fill=(20, 90, 60, 200))
            d.text((28, 136), f"anti-PD-1 ON (day {td:g})", font=self.font_s, fill=(170, 255, 200, 255))
        elif td is not None:
            d.text((28, 132), f"anti-PD-1 starts day {td:g}", font=self.font_s, fill=(150, 160, 185, 200))
        # scale bar: 1 mm = 1000/15 sites
        L = int(1000 / 15 * self.scale)
        W, H = im.size
        d.rectangle([W - 30 - L, H - 34, W - 30, H - 30], fill=(220, 225, 240, 220))
        d.text((W - 30 - L, H - 58), "1 mm", font=self.font_s, fill=(220, 225, 240, 230))
        return np.asarray(im)


def _render_chunk(args):
    run_dir, mode, scale, k0, k1, tmp = args
    r = Renderer(run_dir, mode=mode, scale=scale)
    for kk in range(max(0, k0 - 12), k0):  # warm up trails / flashes
        r.frame(kk)
    for k in range(k0, k1):
        Image.fromarray(r.hud(r.frame(k), k)).save(os.path.join(tmp, f"f{k:05d}.png"), compress_level=1)
    return k1 - k0


def render(run_dir, mode="state", scale=3, fps=30, workers=8, out_name=None, poster_frac=0.8):
    import shutil
    import subprocess
    import tempfile
    from concurrent.futures import ProcessPoolExecutor

    import imageio_ffmpeg
    n = int(zarr.open_group(os.path.join(run_dir, "snap.zarr"), mode="r").attrs["n_frames"])
    out_name = out_name or ("video.mp4" if mode == "state" else f"video_{mode}.mp4")
    path = os.path.join(run_dir, out_name)
    tmp = tempfile.mkdtemp(prefix="render_")
    nchunks = workers * 4  # small chunks: a failure costs little, progress is visible
    bounds = np.linspace(0, n, nchunks + 1).astype(int)
    done = 0
    with ProcessPoolExecutor(workers) as ex:
        for c in ex.map(_render_chunk, [(run_dir, mode, scale, bounds[i], bounds[i + 1], tmp)
                                        for i in range(nchunks)]):
            done += c
            print(f"  {run_dir} [{mode}] {done}/{n} frames", flush=True)
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", str(fps), "-i",
                    os.path.join(tmp, "f%05d.png"), "-c:v", "libx264", "-crf", "20", "-preset",
                    "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path], check=True)
    with open(os.path.join(run_dir, "timeseries.json")) as fh:
        ser = json.load(fh)["series"]
    # poster: moment of peak killing activity (falls back to a fixed fraction)
    pk = int(np.argmax(ser["t_active"])) if max(ser["t_active"]) > 0 else int((n - 1) * poster_frac)
    poster = os.path.join(tmp, f"f{pk:05d}.png")
    Image.open(poster).convert("RGB").save(path.replace(".mp4", "_poster.jpg"), quality=88)
    shutil.rmtree(tmp)
    return path


def still(run_dir, k, mode="state", scale=3, out=None):
    r = Renderer(run_dir, mode=mode, scale=scale)
    # warm up trails/flashes over a few preceding frames
    for kk in range(max(0, k - 6), k):
        r.frame(kk)
    rgb = r.hud(r.frame(k), k)
    out = out or os.path.join(run_dir, f"still_{mode}_{k:04d}.png")
    Image.fromarray(rgb).save(out)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--mode", default="state")
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--stills", help="comma-separated days to render as PNG instead of a video")
    a = ap.parse_args()
    if a.stills:
        with open(os.path.join(a.run_dir, "timeseries.json")) as f:
            days = json.load(f)["series"]["day"]
        for dd in a.stills.split(","):
            k = int(np.argmin(np.abs(np.array(days) - float(dd))))
            print(still(a.run_dir, k, a.mode, a.scale))
    else:
        print(render(a.run_dir, a.mode, a.scale, a.fps, a.workers))
