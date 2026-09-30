"""
stitchify.py - Stage 1 prototype: photo -> subject-aware cross stitch chart.

Pipeline:
  1. Segment the main subject (rembg), optional
  2. Simplify ONLY the background (mean-shift smoothing + blur)
  3. Resize to the stitch grid
  4. K-means in Lab color space, snap to DMC threads
  5. Remove confetti (isolated stitches) with a neighborhood majority filter
  6. Write preview, symbol chart, legend, and metrics

Install:
  pip install tarraz rembg onnxruntime scikit-learn scikit-image opencv-python-headless pillow

Examples:
  python stitchify.py photo.jpg                      # full pipeline
  python stitchify.py photo.jpg --no-segment         # plain conversion (baseline)
  python stitchify.py chatgpt_output.png --no-segment --tag gpt
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.color import rgb2lab
from sklearn.cluster import KMeans
from tarraz.colors import DMC_COLORS

SYMBOLS = list("ABCDEFGHJKLMNPQRSTUVWXYZ123456789@#$%&*+=?")


# ---------- 1. segmentation ----------
def subject_mask(img_rgb, model="u2net"):
    """Returns float mask in [0,1], 1 = subject. Uses rembg (downloads a model on first run).
    Models: u2netp (5 MB, fast, rough), u2net (170 MB), isnet-general-use (170 MB, sharper)."""
    from rembg import new_session, remove
    rgba = np.array(remove(Image.fromarray(img_rgb), session=new_session(model)))
    mask = rgba[:, :, 3].astype(np.float32) / 255.0
    return cv2.GaussianBlur(mask, (0, 0), 3)  # soft edge so the transition isn't a hard cut


# ---------- 2. background simplification ----------
def simplify_background(img_rgb, mask, strength):
    """Flatten the background into a few smooth regions; keep the subject untouched."""
    bg = cv2.pyrMeanShiftFiltering(img_rgb, sp=15 * strength, sr=30 * strength)
    bg = cv2.GaussianBlur(bg, (0, 0), 4 * strength)
    m = mask[:, :, None]
    return (img_rgb * m + bg * (1 - m)).astype(np.uint8)


# ---------- 3-4. grid + palette ----------
def dmc_palette():
    cols = [c for c in DMC_COLORS if "Variegated" not in c["name"]]
    rgb = np.array([c["rgb"] for c in cols], dtype=np.float64)
    lab = rgb2lab(rgb[None] / 255.0)[0]
    return cols, rgb, lab


def quantize_to_dmc(grid_rgb, n_colors, seed=0):
    h, w, _ = grid_rgb.shape
    lab = rgb2lab(grid_rgb / 255.0).reshape(-1, 3)
    km = KMeans(n_clusters=n_colors, n_init=4, random_state=seed).fit(lab)

    cols, _, dmc_lab = dmc_palette()
    # nearest DMC thread per cluster (Euclidean in Lab ~ perceptual distance)
    d = np.linalg.norm(km.cluster_centers_[:, None] - dmc_lab[None], axis=2)
    cluster_to_dmc = d.argmin(axis=1)
    idx = cluster_to_dmc[km.labels_].reshape(h, w)  # index into DMC list
    return idx, cols  # two clusters may merge into the same thread, which is fine


# ---------- 5. confetti cleanup ----------
def remove_confetti(idx, passes=2, min_same=1):
    """A stitch with <= min_same identical 8-neighbours becomes its neighbourhood majority."""
    idx = idx.copy()
    h, w = idx.shape
    for _ in range(passes):
        p = np.pad(idx, 1, mode="edge")
        neigh = np.stack([p[1 + dy:h + 1 + dy, 1 + dx:w + 1 + dx]
                          for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)])
        same = (neigh == idx[None]).sum(0)
        ys, xs = np.where(same <= min_same)
        for y, x in zip(ys, xs):
            vals, counts = np.unique(neigh[:, y, x], return_counts=True)
            idx[y, x] = vals[counts.argmax()]
    return idx


# ---------- metrics ----------
def metrics(idx):
    h, w = idx.shape
    p = np.pad(idx, 1, mode="constant", constant_values=-1)
    neigh = np.stack([p[1 + dy:h + 1 + dy, 1 + dx:w + 1 + dx]
                      for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)])
    isolated = int(((neigh == idx[None]).sum(0) == 0).sum())
    changes_per_row = float((idx[:, 1:] != idx[:, :-1]).sum(1).mean())
    return {
        "size_stitches": f"{w} x {h}",
        "total_stitches": int(h * w),
        "colors": int(len(np.unique(idx))),
        "isolated_stitches": isolated,
        "isolated_pct": round(100 * isolated / (h * w), 2),
        "avg_color_changes_per_row": round(changes_per_row, 1),
    }


# ---------- 6. rendering ----------
def render(idx, cols, out_prefix, cell=16):
    h, w = idx.shape
    used, counts = np.unique(idx, return_counts=True)
    order = used[np.argsort(-counts)]
    sym = {c: SYMBOLS[i % len(SYMBOLS)] for i, c in enumerate(order)}
    rgb = np.array([cols[i]["rgb"] for i in range(len(cols))], dtype=np.uint8)

    # preview: what the finished piece roughly looks like
    Image.fromarray(rgb[idx]).resize((w * 8, h * 8), Image.NEAREST).save(f"{out_prefix}_preview.png")

    # chart: colored cells + symbols + grid, bold line every 10 stitches
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", int(cell * 0.7))
    except OSError:
        font = ImageFont.load_default()
    chart = Image.new("RGB", (w * cell + 1, h * cell + 1), "white")
    dr = ImageDraw.Draw(chart)
    for y in range(h):
        for x in range(w):
            c = tuple(int(v) for v in rgb[idx[y, x]])
            light = tuple(int(0.55 * v + 0.45 * 255) for v in c)  # pale fill keeps symbols readable
            dr.rectangle([x * cell, y * cell, (x + 1) * cell, (y + 1) * cell], fill=light)
            dr.text((x * cell + cell / 2, y * cell + cell / 2), sym[idx[y, x]], fill="black",
                    font=font, anchor="mm")
    for x in range(w + 1):
        dr.line([(x * cell, 0), (x * cell, h * cell)], fill="black" if x % 10 == 0 else "#bbb",
                width=2 if x % 10 == 0 else 1)
    for y in range(h + 1):
        dr.line([(0, y * cell), (w * cell, y * cell)], fill="black" if y % 10 == 0 else "#bbb",
                width=2 if y % 10 == 0 else 1)
    chart.save(f"{out_prefix}_chart.png")

    # legend
    lines = ["symbol\tDMC\tname\tstitches"]
    for c in order:
        lines.append(f"{sym[c]}\t{cols[c]['code']}\t{cols[c]['name']}\t{int(counts[used == c][0])}")
    Path(f"{out_prefix}_legend.tsv").write_text("\n".join(lines))


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--width", type=int, default=80, help="pattern width in stitches")
    ap.add_argument("--colors", type=int, default=12, help="max thread colors")
    ap.add_argument("--bg-strength", type=float, default=1.5, help="background simplification, 0.5-3")
    ap.add_argument("--seg-model", default="u2net", help="u2netp | u2net | isnet-general-use")
    ap.add_argument("--no-segment", action="store_true", help="skip subject detection (baseline)")
    ap.add_argument("--no-cleanup", action="store_true", help="skip confetti removal")
    ap.add_argument("--tag", default="", help="suffix for output files")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()

    img = np.array(Image.open(a.image).convert("RGB"))
    # work at moderate resolution; the grid is tiny anyway
    scale = 800 / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    if not a.no_segment:
        img = simplify_background(img, subject_mask(img, a.seg_model), a.bg_strength)

    h = round(img.shape[0] * a.width / img.shape[1])
    grid = cv2.resize(img, (a.width, h), interpolation=cv2.INTER_AREA).astype(np.float64)

    idx, cols = quantize_to_dmc(grid, a.colors)
    if not a.no_cleanup:
        idx = remove_confetti(idx)

    Path(a.out).mkdir(exist_ok=True)
    name = Path(a.image).stem + (f"_{a.tag}" if a.tag else "")
    prefix = str(Path(a.out) / name)
    if not a.no_segment:
        Image.fromarray(img).save(f"{prefix}_simplified.png")
    render(idx, cols, prefix)
    m = metrics(idx)
    Path(f"{prefix}_metrics.json").write_text(json.dumps(m, indent=2))
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
