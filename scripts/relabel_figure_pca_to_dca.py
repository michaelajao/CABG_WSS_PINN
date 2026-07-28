"""Retype the Figure 5 x-axis category label 'PCA' as 'DCA'.

The manuscript was unified on DCA (diseased coronary artery) in response to
Referee #5 point L3, but the ten Figure 5 panels are raster exports that still
read PCA. Only the leading glyph differs, and every panel already contains a
'D' of the same font, size, weight and colour in its 'DG' label -- so the fix
is to lift that 'D' and stamp it over the 'P' rather than to redraw text in a
font we do not have. Right edges are aligned so the D-C letterspacing is
exactly the P-C spacing of the original.

The PDFs LaTeX includes are single-image wrappers around these PNGs at 96 dpi,
so each patched PNG is rewrapped at the same resolution.

Originals are copied to ``_original_pca_labels/`` beside the figures before
anything is overwritten. Note that ``doc/`` is git-ignored, so that backup is
the only copy -- do not delete it without checking.

Usage:
    python scripts/relabel_figure_pca_to_dca.py --proof DIR  # patched PNGs to DIR
    python scripts/relabel_figure_pca_to_dca.py --apply      # overwrite PNGs+PDFs
"""
import argparse
import shutil
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

# panel PNG -> the PDF that main.tex \includegraphics
PANELS = {
    "Newtonian AWAWSS.png": "Newtonian AWAWSS (2) (1).pdf",
    "Carreau AWAWSS.png": "Carreau AWAWSS (2) (1).pdf",
    "WSS Bifurcaions Newtonain.png": "WSS Bifurcaions Newtonain (1) (1).pdf",
    "Carreau WSS Bifurcations.png": "Carreau WSS Bifurcations (1) (1).pdf",
    "Area Fraction Newtonian.png": "Area Fraction Newtonian (1) (1).pdf",
    "Carreau Area Fraction.png": "Carreau Area Fraction (1) (1).pdf",
    "Newtonian AWAWSS Greater than 40.png":
        "Newtonian AWAWSS Greater than 40 (1) (1).pdf",
    "Carreau AWAWSS Greater than 40.png":
        "Carreau AWAWSS Greater than 40 (1) (1).pdf",
    "Newtonain ICR.png": "Newtonain ICR (1) (1).pdf",
    "Carreau AWAWSS ICA.png": "Carreau AWAWSS ICA (1) (1).pdf",
}

DPI = 96.0
INK = 128  # grey level below which a pixel counts as ink


def write_image_pdf(img, path, dpi=DPI):
    """Wrap an image in a one-page PDF as a FlateDecode XObject.

    Pillow's PDF writer re-encodes RGB as JPEG, which is lossy and visibly
    rings around the sharp black text in these charts. The source PDFs were
    losslessly compressed, so we match that.
    """
    img = img.convert("RGB")
    w, h = img.size
    pw, ph = w * 72.0 / dpi, h * 72.0 / dpi
    data = zlib.compress(img.tobytes(), 9)

    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {pw:.2f} {ph:.2f}] "
         f"/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>"
         ).encode(),
        (f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} "
         f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode "
         f"/Length {len(data)} >>\nstream\n").encode() + data + b"\nendstream",
        None,  # content stream, filled below
    ]
    content = f"q {pw:.2f} 0 0 {ph:.2f} 0 0 cm /Im0 Do Q".encode()
    objs[4] = (f"<< /Length {len(content)} >>\nstream\n".encode()
               + content + b"\nendstream")

    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs)+1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objs)+1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n").encode()
    Path(path).write_bytes(bytes(out))


def runs(flags, min_gap):
    idx = np.flatnonzero(flags)
    if idx.size == 0:
        return []
    out, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev >= min_gap:
            out.append((start, prev + 1))
            start = i
        prev = i
    out.append((start, prev + 1))
    return out


def label_band(mask):
    """Lowest band of ink in the image: the category labels."""
    h = mask.shape[0]
    bands = runs(mask.sum(axis=1) > 0, min_gap=max(3, h // 200))
    for s, e in reversed(bands):
        if 8 <= (e - s) <= h // 8:
            return s, e
    raise ValueError("no plausible label band")


def glyph_box(mask, top, cols):
    """Row extent of the ink inside a column slice, in full-image coords."""
    sub = mask[top:, cols[0]:cols[1]]
    rows = np.flatnonzero(sub.sum(axis=1) > 0)
    return top + rows[0], top + rows[-1] + 1


def locate(img):
    """Return (P box, D box) as (y0, y1, x0, x1) in full-image coordinates."""
    grey = np.asarray(img.convert("L"))
    mask = grey < INK
    h, w = mask.shape
    top, bot = label_band(mask)
    band = mask[top:bot]
    groups = [g for g in runs(band.sum(axis=0) > 0, min_gap=max(20, w // 60))
              if (g[1] - g[0]) > w // 200]
    if len(groups) != 4:
        raise ValueError(f"expected 4 labels (HCA/PCA/HG/DG), found {len(groups)}")

    def first_glyph(group):
        gs, ge = group
        sub = band[:, gs:ge]
        gl = runs(sub.sum(axis=0) > 0, min_gap=max(2, w // 500))
        a, b = gl[0]
        return gs + a, gs + b

    px0, px1 = first_glyph(groups[1])   # the P of PCA
    dx0, dx1 = first_glyph(groups[3])   # the D of DG
    py0, py1 = glyph_box(mask, top, (px0, px1))
    dy0, dy1 = glyph_box(mask, top, (dx0, dx1))
    return (py0, py1, px0, px1), (dy0, dy1, dx0, dx1)


def patch(img):
    (py0, py1, px0, px1), (dy0, dy1, dx0, dx1) = locate(img)
    ph, dh = py1 - py0, dy1 - dy0
    if abs(ph - dh) > max(3, ph // 10):
        raise ValueError(f"P height {ph} and D height {dh} disagree; "
                         "glyph segmentation is probably wrong")
    arr = np.asarray(img.convert("RGB")).copy()

    # background immediately left of the label, used to clear the old glyph
    bg = arr[py0:py1, max(0, px0 - 40):px0 - 20].reshape(-1, 3)
    bg = np.median(bg, axis=0).astype(arr.dtype)

    dw = dx1 - dx0
    tx1 = px1                 # keep the original gap to the following 'C'
    tx0 = tx1 - dw
    ty1 = py1                 # sit on the original baseline
    ty0 = ty1 - dh
    if tx0 < 0 or ty0 < 0:
        raise ValueError("patch target falls outside the image")

    pad = 3
    arr[py0 - pad:py1 + pad, min(tx0, px0) - pad:px1 + pad] = bg
    arr[ty0:ty1, tx0:tx1] = np.asarray(img.convert("RGB"))[dy0:dy1, dx0:dx1]
    info = (f"P x={px0}..{px1} (w={px1-px0}, h={ph})  "
            f"D x={dx0}..{dx1} (w={dw}, h={dh})  -> stamped at x={tx0}..{tx1}")
    return Image.fromarray(arr), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--proof", type=Path)
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dir", type=Path,
                    default=Path(__file__).resolve().parents[1]
                    / "doc" / "CABG_Paper")
    args = ap.parse_args()

    if args.proof:
        args.proof.mkdir(parents=True, exist_ok=True)
    backup = args.dir / "_original_pca_labels"
    if args.apply:
        backup.mkdir(exist_ok=True)

    for png, pdf in PANELS.items():
        src = args.dir / png
        if not src.exists():
            print(f"MISSING  {png}")
            continue
        img = Image.open(src)
        try:
            out, info = patch(img)
        except ValueError as exc:
            print(f"SKIP     {png}: {exc}")
            continue
        print(f"ok       {png}\n           {info}")
        if args.proof:
            out.save(args.proof / png)
        if args.apply:
            for f in (png, pdf):
                if (args.dir / f).exists() and not (backup / f).exists():
                    shutil.copy2(args.dir / f, backup / f)
            out.save(src)
            write_image_pdf(out, args.dir / pdf)


if __name__ == "__main__":
    main()
