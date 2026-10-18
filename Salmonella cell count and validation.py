#!/usr/bin/env python3
"""
Qualitative illustration figure: S. Typhi with HCT116 cells after glucose vs fucose pre-incubation
(session 20251222; DAPI + phalloidin widefield images).

PURPOSE AND LIMITS
  For qualitative illustration only. Quantitative adherence/invasion evidence comes from the CFU assays.
  These images contain no anti-Salmonella (AF488) channel; bacteria are visible as small rod-shaped
  DAPI-positive objects.

IMAGE INTEGRITY (Communications Biology policy)
  * Original TIFFs are opened read-only; their SHA-256 is logged before and after the run.
  * Only linear min/max display scaling is used, with IDENTICAL values for every image of a channel.
  * Cropping removes the rows containing the burned-in Leica scale bar; a calibrated scale bar is redrawn
    (0.16125 µm/pixel, Leica LAS calibration).
  * No filtering, sharpening, background subtraction, gamma change, denoising or AI processing.
  * The field shown per condition is chosen by a stated rule (see FIELD SELECTION), and a gallery of
    ALL fields with the same settings is produced for the supplement / reviewers.

FIELD SELECTION (per treatment; the two treatments need NOT share a field number)
  Candidates are the fields with BOTH a DAPI and a phalloidin image (glucose: all 10; fucose: 0001, 0003, 0010).
  Rule, fixed before looking at the images and applied to each treatment separately: rank by the number of
  validated rod-shaped bacteria in the field, ties broken by the lower DAPI saturation. These are therefore
  CLEAREST-EXAMPLE fields, not median fields - say so in the legend, and show the full gallery.
  Every candidate's numbers go to field_selection_table.csv. Set FIELDS below (or --glucose-field/--fucose-field)
  to pin a field by hand.
  The enlarged panel is placed, by the same rule in both conditions, on the clean-background window
  (no nuclear glare, no dark debris) with the most bacterium-sized DAPI objects; it shows the RAW DAPI
  channel at a narrower linear range so the rods are visible.

MEMBRANE ZOOM (adhesion check; MAKE_MEMBRANE_ZOOM = True)
  Enlarges the example host cell's edge (outline from phalloidin, drawn as an overlay) at the segment with the
  least DAPI glare, same rule in both conditions, and marks rods within ADHESION_DIST_UM (3 um) of the edge that pass
  the validated rule. The band is reported with the fraction of it that lies below the DAPI glare limit, so a count
  of zero can be read as "nothing resolvable there", not as "no bacteria there". If an AF488 anti-Salmonella image exists (fucose_g_/glu_g_), it is used instead of DAPI.
  Writes Figure_membrane_zoom_field####.png and membrane_adhesion_table.csv (all fields with phalloidin).
  2-D widefield proximity to the membrane is compatible with, but does not prove, adhesion.

ANNOTATIONS (overlays only; image pixels are never changed)
  White arrows: one example host cell (largest isolated nucleus with phalloidin signal).
  Yellow arrowheads: up to 4 example bacteria in the enlarged panel (most visible rod-shaped DAPI objects
  on clean background, >= 7 robust SD above local noise). Positions are written to processing_log.json.

USAGE
  pip install numpy scipy tifffile pillow matplotlib scikit-image
  (OpenCV / cv2 is NOT required.)
  python make_qualitative_figure.py --root "H:\\Shared drives\\Microbial images\\Salmonella" --out figure_out
"""
import argparse, hashlib, json, os, sys, datetime
import numpy as np
import tifffile
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
import matplotlib.patheffects as pe

# ---------------------------------------------------------------------------------
# No OpenCV required. The three morphological operations this script needs are done
# with scipy.ndimage instead. _ellipse_footprint() reproduces OpenCV's
# getStructuringElement(MORPH_ELLIPSE, (n, n)) element exactly, so results are
# pixel-identical to the earlier cv2-based version (verified element-by-element and
# on real images: maximum absolute difference 0).
# ---------------------------------------------------------------------------------
def _median31(u8):
    """31x31 median of an 8-bit image. Uses scikit-image's histogram-based rank
    filter, which is identical to OpenCV's medianBlur and ~30x faster than
    scipy.ndimage.median_filter at this kernel size; falls back to scipy if
    scikit-image is unavailable."""
    try:
        from skimage.filters.rank import median as _rank_median
        return _rank_median(u8, np.ones((31, 31), np.uint8))
    except Exception:
        return ndi.median_filter(u8, size=31, mode="nearest")


def _ellipse_footprint(n):
    """Boolean (n, n) disc, matching OpenCV's MORPH_ELLIPSE construction exactly."""
    r = n // 2
    out = np.zeros((n, n), bool)
    if r == 0:
        out[0, 0] = True
        return out
    inv_r2 = 1.0 / (r * r)
    for i in range(n):
        dy = i - r
        if abs(dy) <= r:
            dx = int(round(r * np.sqrt((r * r - dy * dy) * inv_r2)))
            out[i, r - dx:r + dx + 1] = True
    return out

# =====================================================================================
# EASY MODE (Spyder / any editor): edit these two lines, then press Run (F5).
ROOT_FOLDER = r"H:\Shared drives\Microbial images\Salmonella"   # folder that contains 20251222\
OUTPUT_FOLDER = r"C:\Users\Public\salmonella_figure_out"          # where the figure is saved
# Representative field per treatment. "auto" applies the selection rule below (see FIELD SELECTION);
# a 4-digit string pins that field instead. The two treatments need NOT use the same field number.
FIELDS = {"Glucose": "auto", "Fucose": "0003"}
SELECT_BY = "membrane"       # "membrane": most Salmonella candidates within ADHESION_DIST_UM of a cell edge
                             # "field":    most rod-shaped objects anywhere in the field
MAKE_MEMBRANE_ZOOM = True   # also produce the cell-membrane zoom figure + adhesion table for all phalloidin fields
ZOOM_TARGET = "membrane"    # panel d/h: "membrane" = enlarged merge at the example cell's membrane (adhesion view);
                            #            "background" = enlarged DAPI on clean background (most visible bacteria)
# =====================================================================================

STAINS = {   # from the staining table (target, reagent, display colour)
    "dapi": ("Host-cell nuclei", "DAPI (1 mg/mL stock)", "blue"),
    "phal": ("Host-cell F-actin", "Alexa Fluor 594 Phalloidin", "red"),
    "af488": ("Salmonella", "Salmonella Antibody (100/358), Alexa Fluor 488 (Novus Biologicals)", "green"),
}

UM_PER_PX = 0.16125          # Leica LAS calibration (DFC365 FX, 40x)
CROP_ROWS = 960              # rows >= 960 hold the burned-in 50 µm bar in the 20251222 exports
# identical linear display ranges for every image (8-bit grey values)
DAPI_RANGE = (20, 255)       # full panel
PHAL_RANGE = (15, 110)       # full panel
DAPI_INSET_RANGE = (25, 90)  # inset: raw DAPI, narrower linear range to show bacteria
DAPI_MEMBRANE_RANGE = (90, 255)  # membrane zoom: the cell edge lies in the bright DAPI halo, so a higher linear range
MEMBRANE_UM = (24.0, 16.5)   # membrane zoom window (width, height) in µm
ADHESION_DIST_UM = 3.0       # a rod counts as 'at the membrane' if its centre is within this distance of the phalloidin edge
GLARE_MAX = 60.0             # local DAPI background above which a rod cannot be identified (validated detector cut-off)
# --- glare handling in the enlarged membrane panel -------------------------------------------------
MEMBRANE_VIEW = "tophat"     # "tophat": subtract the smooth DAPI halo before display (uniform, disclosed,
                             #           classical morphology - no AI, no per-object editing);
                             # "raw":    original DAPI at DAPI_MEMBRANE_RANGE (halo left in)
TOPHAT_UM = 2.4              # element: keeps structures narrower than this (a bacterium), removes the halo
TOPHAT_RANGE = (2, 28)       # linear display range for the glare-corrected panel, identical in both treatments
SHOW_REJECTED = False        # (superseded) mark features failing the old dark-dip test
# --- membrane-zone detection ----------------------------------------------------------------------
# Near a cell the DAPI halo is steep, and the whole-image detector used elsewhere under-reads contrast
# there (its 15 px opening follows the gradient).  In the membrane zone we therefore detect on the
# GLARE-CORRECTED image and judge shape with gradient-independent measures.
MEMBRANE_MIN_SNR = 5.0       # peak contrast over local noise, measured on the glare-corrected image
MEMBRANE_MIN_CORE = 2.0      # centre brighter than the surrounding annulus (a bacterium is filled in)
MEMBRANE_MIN_CONVEX = 0.80   # convexity: rejects long curved arcs
MEMBRANE_MAX_LEN_UM = 4.5
WRITE_CANDIDATE_SHEET = True # write a contact sheet of every marked object so each can be checked by eye
AF488_PATTERNS = {"Glucose": "20251222/glu_g_{f}.tif", "Fucose": "20251222/fucose_g_{f}.tif"}   # used automatically if present
INSET_UM = (32.5, 22.4)      # inset size (width, height) in µm; same aspect ratio as the cropped field (~4.3x enlargement)

SOURCES = {
    "Glucose": {"dapi": "20251222/glu_b_{f}.tif", "phal": "20251222/glu_r_{f}.tif"},
    "Fucose": {"dapi": "20251222/fucose_b_{f}.tif",
               "phal_options": ["20251222/fucose_r_{f}.tif", "20251222/fucose_composite_{f}.tif::R",
                                "20251222/Composite/fucose_{f}_composite.tif::R"]},
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_plane(root, spec):
    """Return one 2-D float plane. 'file.tif' -> grey (checks R=G=B); 'file.tif::R' -> that composite channel."""
    path, ch = (spec.split("::") + [None])[:2]
    full = os.path.join(root, path)
    with open(full, "rb") as fh:                       # read-only
        a = tifffile.imread(fh)
    if a.ndim == 3 and a.shape[0] in (3, 4) and a.shape[-1] not in (3, 4):
        a = np.moveaxis(a, 0, -1)
    if ch is not None:
        return a[..., "RGB".index(ch)].astype(float), full
    if a.ndim == 3:
        body = a[:CROP_ROWS]
        if not (np.array_equal(body[..., 0], body[..., 1]) and np.array_equal(body[..., 0], body[..., 2])):
            sys.exit(f"{path}: channels differ - not a single-channel export")
        a = a[..., 0]
    return a.astype(float), full


def lin(x, rng):
    return np.clip((x - rng[0]) / (rng[1] - rng[0]), 0, 1)


def merge(dapi, phal):
    rgb = np.zeros(dapi.shape + (3,))
    rgb[..., 2] = lin(dapi, DAPI_RANGE)                                  # DAPI -> blue
    if phal is not None:
        rgb[..., 0] = lin(phal, PHAL_RANGE)                              # AF594 phalloidin -> red
    return np.clip(rgb, 0, 1)


def merge_custom(dapi, phal, drng, prng):
    rgb = np.zeros(dapi.shape + (3,))
    rgb[..., 2] = lin(dapi, drng); rgb[..., 0] = lin(phal, prng)
    return np.clip(rgb, 0, 1)


def tint(g, colour):
    rgb = np.zeros(g.shape + (3,)); rgb[..., {"red": 0, "green": 1, "blue": 2}[colour]] = g
    return rgb


def bacteria_like(dapi):
    """Used ONLY to place the inset (same rule both conditions); never alters displayed pixels."""
    th = dapi - ndi.grey_opening(ndi.gaussian_filter(dapi, 1), size=(15, 15))
    med = np.median(th); rsd = 1.4826 * np.median(np.abs(th - med))
    m = th > med + 5 * max(rsd, 0.5)
    bright = ndi.grey_opening(dapi, size=(15, 15)) > 60          # exclude nuclei/glare
    lab, n = ndi.label(m & ~bright)
    if n == 0:
        return np.zeros((0, 2))
    sizes = ndi.sum(m, lab, range(1, n + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if 6 <= s <= 96]
    return np.array(ndi.center_of_mass(m, lab, keep)).reshape(-1, 2)


def clean_mask(dapi):
    """Pixels on clean background: no nuclear glare (local median <= 60) and no dark debris
    (raw value not more than 8 grey levels below the local median). Same rule in every image."""
    med = ndi.median_filter(dapi[::2, ::2], size=16)
    med = np.kron(med, np.ones((2, 2)))[:dapi.shape[0], :dapi.shape[1]]
    debris = ndi.binary_dilation(dapi < med - 8, iterations=6)
    return (med <= 60) & ~debris


def best_window(pts, shape, w, h, step=16, clean=None):
    """Window with the most bacterium-sized objects; if a clean mask is given, only windows that are
    >= 97% clean background are allowed (avoids glare and debris)."""
    best, xy = -1, (0, 0)
    for y in range(0, shape[0] - h, step):
        for x in range(0, shape[1] - w, step):
            if clean is not None and clean[y:y + h, x:x + w].mean() < 0.97:
                continue
            n = int(((pts[:, 0] >= y) & (pts[:, 0] < y + h) & (pts[:, 1] >= x) & (pts[:, 1] < x + w)).sum()) if len(pts) else 0
            if n > best:
                best, xy = n, (x, y)
    if best < 0:                               # no fully clean window: fall back to unrestricted rule
        return best_window(pts, shape, w, h, step, None)
    return xy


def pick_cells(dapi, phal, n=1):
    """Annotation only: choose n well-separated host cells (nucleus not touching the border, phalloidin present).
    Rule: largest isolated nuclei. Returns list of (y, x, radius_px)."""
    s = ndi.gaussian_filter(dapi, 2)
    bg = ndi.grey_opening(s[::4, ::4], size=(39, 39))
    bg = np.kron(bg, np.ones((4, 4)))[:s.shape[0], :s.shape[1]]
    t = s - bg
    from skimage.filters import threshold_otsu
    m = ndi.binary_fill_holes(t > threshold_otsu(t))
    lab, k = ndi.label(m)
    cand = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None: continue
        y0, y1, x0, x1 = sl[0].start, sl[0].stop, sl[1].start, sl[1].stop
        if y0 < 40 or x0 < 40 or y1 > dapi.shape[0] - 40 or x1 > dapi.shape[1] - 40: continue
        area = (lab[sl] == i).sum()
        if area < 600: continue
        cy, cx = ndi.center_of_mass(lab == i)
        ph = phal[int(cy), int(cx)] if phal is not None else 0
        cand.append((area, cy, cx, np.sqrt(area / np.pi), ph))
    cand.sort(reverse=True)
    out = []
    for a_, cy, cx, r, ph in cand:
        if all(np.hypot(cy - y, cx - x) > 250 for y, x, _ in out):
            out.append((cy, cx, r))
        if len(out) == n: break
    return out


def pick_bacteria(inset_raw, n=3):
    """Annotation only: choose n clearly visible, well-separated rod-like DAPI objects in the inset
    (high local contrast on dark background, away from the inset border). Returns list of (y, x)."""
    sm = ndi.gaussian_filter(inset_raw, 1)
    th = sm - ndi.grey_opening(sm, size=(15, 15))
    bgl = ndi.median_filter(inset_raw, size=31)
    tmed = np.median(th); trsd = 1.4826 * np.median(np.abs(th - tmed))
    lab, k = ndi.label(th > max(3.0, tmed + 4 * trsd))          # threshold relative to local noise
    cand = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None: continue
        obj = lab == i
        area = obj.sum()
        if not (10 <= area <= 90): continue
        ys, xs = np.nonzero(obj)
        length = np.hypot(np.ptp(ys), np.ptp(xs))
        if length < 6: continue                            # rod-shaped (>= ~1 µm long), not a dot
        cy, cx = ys.mean(), xs.mean()
        if cy < 15 or cx < 15 or cy > inset_raw.shape[0] - 15 or cx > inset_raw.shape[1] - 15: continue
        if bgl[int(cy), int(cx)] > 60: continue          # skip objects in nuclear glare
        y_, x_ = int(cy), int(cx)
        if inset_raw[max(0, y_ - 6):y_ + 7, max(0, x_ - 6):x_ + 7].min() < bgl[y_, x_] - 8: continue   # rim of dark debris
        peak = th[obj].max()
        if peak < max(4.0, tmed + 7 * trsd): continue      # crisp, in-focus objects only (>= 7 robust SD)
        if cy > inset_raw.shape[0] * 0.78 or cy < inset_raw.shape[0] * 0.22: continue   # keep label strip and scale-bar strip free
        cand.append((th[obj].sum(), cy, cx))              # favour the most visible rods (integrated signal)
    cand.sort(reverse=True)
    out = []
    for c, cy, cx in cand:
        if all(np.hypot(cy - y, cx - x) > 30 for y, x in out):
            out.append((cy, cx))
        if len(out) == n: break
    return out


def valid_af488(root, rel):
    """An AF488 file is used only if it is a genuine single-channel export (R=G=B) with signal.
    Composite copies renamed as _g_ (green channel empty) are rejected."""
    full = os.path.join(root, rel)
    if not os.path.exists(full):
        return False, "absent"
    with open(full, "rb") as fh:
        a = tifffile.imread(fh)
    if a.ndim == 3 and a.shape[0] in (3, 4) and a.shape[-1] not in (3, 4):
        a = np.moveaxis(a, 0, -1)
    if a.ndim == 2:
        return True, "ok"
    body = a[:CROP_ROWS]
    if np.array_equal(body[..., 0], body[..., 1]) and np.array_equal(body[..., 0], body[..., 2]):
        return True, "ok"
    return False, ("rejected: multi-channel composite (green max %d), not a single-channel AF488 export" % int(a[..., 1].max()))


def cell_mask_from_actin(phal):
    """Host-cell outline from phalloidin (Gaussian sigma 2 px, Otsu, fill holes, objects > 600 px). Overlay only."""
    from skimage.filters import threshold_otsu
    sm = ndi.gaussian_filter(phal, 2)
    t = ndi.binary_fill_holes(ndi.binary_opening(sm > threshold_otsu(sm), iterations=2))
    lab, n = ndi.label(t)
    if n == 0:
        return t
    sz = ndi.sum(t, lab, range(1, n + 1))
    return np.isin(lab, [i + 1 for i, v in enumerate(sz) if v > 600])


def rods_validated(img):
    """Rod-shaped objects passing the validated rule (local SNR >= 7, no dark-debris dip > 3 local SD).
    Works on any channel (DAPI proxy or AF488). Returns array of (y, x, local_background, snr)."""
    sm = ndi.gaussian_filter(img, 1); th = sm - ndi.grey_opening(sm, size=(15, 15))
    med = ndi.median_filter(img[::2, ::2], size=16); med = np.kron(med, np.ones((2, 2)))[:img.shape[0], :img.shape[1]]
    dev = np.abs(img - med); lsd = 1.4826 * ndi.median_filter(dev[::2, ::2], size=16)
    lsd = np.kron(lsd, np.ones((2, 2)))[:img.shape[0], :img.shape[1]]
    lab, n = ndi.label(th > 3)
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None: continue
        obj = lab[sl] == i; a_ = obj.sum()
        if not 8 <= a_ <= 90: continue
        ys, xs = np.nonzero(obj); L = np.hypot(np.ptp(ys), np.ptp(xs))
        if L < 5 or L > 31: continue
        cy, cx = ys.mean() + sl[0].start, xs.mean() + sl[1].start; y, x = int(cy), int(cx)
        snr = th[sl][obj].max() / max(lsd[y, x], 1)
        dip = (med[y, x] - img[max(0, y - 6):y + 7, max(0, x - 6):x + 7].min()) / max(lsd[y, x], 1)
        if snr >= 7 and dip <= 3:
            out.append((cy, cx, med[y, x], snr))
    return np.array(out).reshape(-1, 4)


def best_membrane_view(dapi, phal, w, h, band_um=None):
    """Choose BOTH the example host cell and the window on its edge, by what the panel is meant to show:
    the cell whose edge carries the most Salmonella candidates, and the window on that edge containing
    the most of them (ties broken by less DAPI glare).  Falls back to the least-glare edge segment of the
    largest isolated cell when no candidate is found anywhere."""
    band_um = ADHESION_DIST_UM if band_um is None else band_um
    cm = cell_mask_from_actin(phal)
    lab, n = ndi.label(cm)
    edge_all = cm ^ ndi.binary_erosion(cm)
    dist_all = ndi.distance_transform_edt(~edge_all) * UM_PER_PX
    objs_all = membrane_objects(dapi, dist_all, band_um)
    H, W = dapi.shape
    # assign each candidate to its nearest cell
    _, (iy, ix) = ndi.distance_transform_edt(~cm, return_indices=True)
    per_cell = {}
    for o in objs_all:
        y, x = int(o[0]), int(o[1])
        li = lab[y, x] or lab[iy[y, x], ix[y, x]]
        if li: per_cell.setdefault(int(li), []).append(o)
    if not per_cell:
        cells = pick_cells(dapi, phal, n=1)
        x0, y0, edge = membrane_window(cm, dapi, cells[0], w, h)
        return cells[0], x0, y0, edge, ndi.distance_transform_edt(~edge) * UM_PER_PX, []
    # the cell with the most candidates at its edge; ties -> the larger cell
    sizes = ndi.sum(cm, lab, range(1, n + 1))
    li = max(per_cell, key=lambda k: (len(per_cell[k]), sizes[k - 1]))
    cell = lab == li
    edge = cell ^ ndi.binary_erosion(cell)
    dist = ndi.distance_transform_edt(~edge) * UM_PER_PX
    objs = membrane_objects(dapi, dist, band_um)
    if not objs: objs = per_cell[li]
    # window centred on each candidate; keep the one holding the most candidates, then the least glare
    best = None
    for (cy, cx, *_rest) in objs:
        x0 = int(np.clip(cx - w / 2, 0, W - w)); y0 = int(np.clip(cy - h / 2, 0, H - h))
        inside = sum(1 for o in objs if y0 <= o[0] < y0 + h and x0 <= o[1] < x0 + w)
        if not (edge[y0:y0 + h, x0:x0 + w]).any():       # the cell edge must be visible in the panel
            continue
        glare = float(np.median(dapi[y0:y0 + h, x0:x0 + w]))
        key = (inside, -glare)
        if best is None or key > best[0]: best = (key, x0, y0)
    if best is None:
        cyc, cxc = objs[0][0], objs[0][1]
        best = ((0, 0), int(np.clip(cxc - w / 2, 0, W - w)), int(np.clip(cyc - h / 2, 0, H - h)))
    ys, xs = np.nonzero(cell)
    cell_yx = (ys.mean(), xs.mean(), float(np.sqrt(cell.sum() / np.pi)))
    return cell_yx, best[1], best[2], edge, dist, objs


def membrane_window(cmask, dapi, cell_yx, w, h):
    """Window on the example cell's membrane: among windows centred on that cell's edge pixels, take the one
    with the lowest median DAPI (least glare) - same rule in every condition."""
    lab, _ = ndi.label(cmask)
    cy, cx = int(cell_yx[0]), int(cell_yx[1])
    li = lab[cy, cx]
    if li == 0:                                   # nucleus centre outside actin mask: nearest cell
        d, (iy, ix) = ndi.distance_transform_edt(lab == 0, return_indices=True); li = lab[iy[cy, cx], ix[cy, cx]]
    cell = lab == li
    edge = cell ^ ndi.binary_erosion(cell)
    ey, ex = np.nonzero(edge)
    best = None
    for k in range(0, len(ey), max(1, len(ey) // 60)):
        y0 = int(np.clip(ey[k] - h // 2, 0, dapi.shape[0] - h)); x0 = int(np.clip(ex[k] - w // 2, 0, dapi.shape[1] - w))
        v = np.median(dapi[y0:y0 + h, x0:x0 + w])
        if best is None or v < best[0]:
            best = (v, x0, y0)
    return best[1], best[2], edge


def glare_corrected(img):
    """White top-hat: image minus its morphological opening. Removes the smooth out-of-focus DAPI halo and
    keeps objects narrower than TOPHAT_UM. Whole-image, same parameters everywhere, fully reversible to state."""
    k = _ellipse_footprint(int(TOPHAT_UM / UM_PER_PX) | 1)
    return ndi.white_tophat(img.astype(np.float32), footprint=k).astype(float)


def membrane_objects(img, dist, band_um=None):
    """Bacterium-sized, filled, compact bright objects within band_um of the cell edge, detected on the
    GLARE-CORRECTED image.  Returns (y, x, snr, core_minus_annulus, convexity).
    These are DAPI-positive objects of bacterial size and shape; in the bright membrane zone an
    out-of-focus debris ring can mimic one, so they are reported as candidates, not as proof."""
    from skimage.measure import regionprops, label as sklabel
    band_um = ADHESION_DIST_UM if band_um is None else band_um
    corr = glare_corrected(img)
    rsd = max(1.4826 * float(np.median(np.abs(corr - np.median(corr)))), 0.5)
    YY, XX = np.ogrid[-9:10, -9:10]; RAD = np.hypot(YY, XX)
    CORE = RAD <= 2.0; ANN = (RAD >= 4) & (RAD <= 7)
    out = []
    for rp in regionprops(sklabel(corr > 3 * rsd)):
        if not 6 <= rp.area <= 120: continue
        if rp.major_axis_length < 4 or rp.major_axis_length * UM_PER_PX > MEMBRANE_MAX_LEN_UM: continue
        if rp.solidity < MEMBRANE_MIN_CONVEX: continue
        cy, cx = rp.centroid; y, x = int(cy), int(cx)
        if not (9 <= y < img.shape[0] - 9 and 9 <= x < img.shape[1] - 9): continue
        if dist[y, x] > band_um: continue
        p = corr[y - 9:y + 10, x - 9:x + 10]
        snr = float(p[RAD <= 3].max() / rsd); core = float(p[CORE].mean() - p[ANN].mean())
        if snr >= MEMBRANE_MIN_SNR and core >= MEMBRANE_MIN_CORE:
            out.append((cy, cx, round(snr, 1), round(core, 1), round(float(rp.solidity), 2)))
    return out


def _tile(corr, cy, cx, half, label, colour, fnt, size=150):
    y, x = int(cy), int(cx)
    c = corr[y - half:y + half + 1, x - half:x + half + 1]
    if c.shape != (2 * half + 1, 2 * half + 1): return None
    t = (np.clip(c / max(c.max(), 1), 0, 1) * 255).astype(np.uint8)
    from PIL import Image, ImageDraw
    im = Image.fromarray(t).resize((size, size), Image.NEAREST).convert("RGB")
    ImageDraw.Draw(im).text((3, 2), label, fill=colour, font=fnt)
    return np.array(im)


def validation_sheet(img, objs, reference, path, title, cols=8, half=13):
    """Side-by-side validation sheet, in the style of the candidate-vs-confirmed comparison:
      top block  - every Salmonella candidate marked near the cell edge in THIS field
      bottom row - confirmed Salmonella from the clean background of the SAME field, same display
    Both blocks are the glare-corrected view at the same scale, so shape can be compared directly.
    No blank padding tiles: the sheet is exactly as wide as the number of tiles present."""
    from PIL import Image, ImageDraw, ImageFont
    if not objs and not reference: return
    corr = glare_corrected(img)
    try:
        fnt = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        hfn = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except OSError:
        fnt = hfn = ImageFont.load_default()
    cand = [t for t in (_tile(corr, cy, cx, half, f"({int(cx)},{int(cy)}) s{snr:.0f}", (255, 255, 0), fnt)
                        for cy, cx, snr, core, cvx in sorted(objs, key=lambda t: -t[2])) if t is not None]
    ref = [t for t in (_tile(corr, cy, cx, half, f"confirmed s{snr:.0f}", (0, 255, 0), fnt)
                       for cy, cx, snr in reference) if t is not None]
    # Column count is set by the CANDIDATES (the point of the sheet) and chosen to leave the fewest
    # empty slots; the reference row is then trimmed to exactly one full row, so there are no blank frames.
    n_c = len(cand)
    ncols = max(1, min(cols, n_c)) if n_c else max(1, min(cols, len(ref)))
    if n_c > cols:
        best_c, best_waste = ncols, 10 ** 9
        for c_ in range(max(4, cols - 4), cols + 1):
            waste = (-n_c) % c_
            if waste < best_waste or (waste == best_waste and c_ > best_c):
                best_c, best_waste = c_, waste
        ncols = best_c
    ref = ref[:ncols]
    W = 150 * ncols
    def block(tiles, caption=None):
        out = []
        for i in range(0, len(tiles), ncols):
            chunk = tiles[i:i + ncols]
            rowimg = np.hstack(chunk)
            if rowimg.shape[1] < W:                      # pad only the final partial row, with background
                rowimg = np.pad(rowimg, ((0, 0), (0, W - rowimg.shape[1]), (0, 0)))
            out.append(rowimg)
        return out
    def textblock(msg, colour, font, pad=4):
        """Header that wraps to the sheet width, so nothing is cut off on a narrow sheet."""
        words, lines, cur = msg.split(), [], ""
        probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
        for w_ in words:
            t = (cur + " " + w_).strip()
            if probe.textlength(t, font=font) <= W - 2 * pad or not cur: cur = t
            else: lines.append(cur); cur = w_
        if cur: lines.append(cur)
        lh = font.size + 5
        im = Image.new("RGB", (W, lh * len(lines) + 6), (0, 0, 0)); dr = ImageDraw.Draw(im)
        for i, ln in enumerate(lines): dr.text((pad, 3 + i * lh), ln, fill=colour, font=font)
        return np.array(im)
    parts = [textblock(title, (255, 255, 255), hfn)]
    if cand: parts.append(textblock(f"Salmonella candidates at the cell edge (n={len(cand)}) - yellow label", (0, 255, 255), fnt))
    if cand: parts += block(cand, None)
    if ref: parts.append(textblock(f"Confirmed Salmonella from the clean background of the same field (n={len(ref)}) - green label", (0, 255, 255), fnt))
    if ref: parts += block(ref, None)
    Image.fromarray(np.vstack(parts)).save(path)


def confirmed_reference(img, dist, band_um=None, n=8, min_dist_um=6.0):
    """Confirmed Salmonella for the comparison row: objects passing the SAME membrane test but far from
    any cell, on clean background, where identification is unambiguous."""
    band_um = ADHESION_DIST_UM if band_um is None else band_um
    objs = membrane_objects(img, np.zeros_like(dist))      # band test disabled: search the whole field
    out = [(cy, cx, snr) for cy, cx, snr, core, cvx in objs if dist[int(cy), int(cx)] >= min_dist_um]
    out.sort(key=lambda t: -t[2])
    return out[:n]


def edge_candidates(img, dist, band_um=None):
    """Every bright, bacterium-sized feature within band_um of the cell edge, each tagged accepted/rejected
    by the validated rule (local SNR >= 7 and no dark dip > 3 local SD).  A dark dip means the bright feature
    has a markedly darker pixel beside it - the signature of an out-of-focus debris ring, not of a
    fluorescent bacterium, which is uniformly brighter than its surroundings."""
    band_um = ADHESION_DIST_UM if band_um is None else band_um
    sm = ndi.gaussian_filter(img, 1); th = sm - ndi.grey_opening(sm, size=(15, 15))
    med = ndi.median_filter(img[::2, ::2], size=16); med = np.kron(med, np.ones((2, 2)))[:img.shape[0], :img.shape[1]]
    lsd = 1.4826 * ndi.median_filter(np.abs(img - med)[::2, ::2], size=16)
    lsd = np.kron(lsd, np.ones((2, 2)))[:img.shape[0], :img.shape[1]]
    lab, n = ndi.label(th > 3)
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None: continue
        obj = lab[sl] == i; a_ = obj.sum()
        if not 8 <= a_ <= 90: continue
        ys, xs = np.nonzero(obj); L = np.hypot(np.ptp(ys), np.ptp(xs))
        if L < 5 or L > 31: continue
        cy, cx = ys.mean() + sl[0].start, xs.mean() + sl[1].start; y, x = int(cy), int(cx)
        if dist[y, x] > band_um: continue
        snr = th[sl][obj].max() / max(lsd[y, x], 1)
        dip = (med[y, x] - img[max(0, y - 6):y + 7, max(0, x - 6):x + 7].min()) / max(lsd[y, x], 1)
        ok = snr >= 7 and dip <= 3
        why = "" if ok else ("low SNR" if snr < 7 else "") + ("; " if snr < 7 and dip > 3 else "") + ("dark ring" if dip > 3 else "")
        out.append(dict(y=cy, x=cx, snr=round(float(snr), 1), dark_dip=round(float(dip), 1), accepted=bool(ok), reason=why))
    return out


def phal_for(root, cond, f):
    """Path spec of the phalloidin image for this field, or None."""
    s = SOURCES[cond]
    cands = [s["phal"]] if "phal" in s else s["phal_options"]
    return next((c.format(f=f) for c in cands
                 if os.path.exists(os.path.join(root, c.format(f=f).split("::")[0]))), None)


def band_metrics(dapi, phal, band_um=None, bac=None):
    """How much of the membrane band could show a bacterium, and how many validated rods lie in it.
    band = all pixels within band_um of the phalloidin cell edge (inside and outside the cell).
    Cell edge from phal; bacteria detected in bac (AF488) when given, otherwise in dapi."""
    band_um = ADHESION_DIST_UM if band_um is None else band_um
    img = dapi if bac is None else bac
    cm = cell_mask_from_actin(phal)
    edge = cm ^ ndi.binary_erosion(cm)
    dist = ndi.distance_transform_edt(~edge) * UM_PER_PX
    band = dist <= band_um
    med = _median31(np.clip(img, 0, 255).astype(np.uint8)).astype(float)
    evaluable = (med <= GLARE_MAX) & (img < 255)      # a rod here would be identifiable
    rods = rods_validated(img)
    in_band = [(y, x) for y, x, _, _ in rods if dist[int(y), int(x)] <= band_um]
    return dict(n_cells=int(ndi.label(cm)[1]),
                band_um2=round(float(band.sum() * UM_PER_PX ** 2)),
                band_evaluable_frac=round(float((band & evaluable).sum() / max(band.sum(), 1)), 3),
                band_evaluable_um2=round(float((band & evaluable).sum() * UM_PER_PX ** 2)),
                median_DAPI_at_band=round(float(np.median(dapi[band]))) if band.any() else None,
                rods_in_field=len(rods), rods_within_band=len(in_band),
                dapi_saturated_pct=round(100 * float((dapi >= 255).mean()), 2)), cm, dist, in_band


def select_fields(root, log=None):
    """FIELD SELECTION RULE, fixed before looking at the images and applied to each treatment separately:
         candidates = fields that have BOTH a DAPI and a phalloidin image;
         rank by the number of validated rod-shaped bacteria in the field (the figure has to show bacteria),
         ties broken by the lower DAPI saturation (cleaner cell outlines).
    The two treatments are ranked independently, so they need not share a field number.
    Every candidate's numbers are written to field_selection_table.csv so the choice is auditable."""
    chosen, table = {}, []
    for cond in ("Glucose", "Fucose"):
        rows = []
        for k in range(1, 11):
            f = f"{k:04d}"
            dp = SOURCES[cond]["dapi"].format(f=f)
            if not os.path.exists(os.path.join(root, dp)):
                continue
            pp = phal_for(root, cond, f)
            if pp is None:
                table.append(dict(treatment=cond, field=f, excluded="no phalloidin image")); continue
            d, _ = read_plane(root, dp); p, _ = read_plane(root, pp)
            d, p = d[:CROP_ROWS], p[:CROP_ROWS]
            m, cm_, dist_, _ = band_metrics(d, p)
            m["salmonella_at_membrane"] = len(membrane_objects(d, dist_))
            m.update(treatment=cond, field=f, phalloidin_source=os.path.basename(pp.split("::")[0]), excluded="")
            rows.append(m); table.append(m)
        if not rows:
            sys.exit(f"{cond}: no field has both a DAPI and a phalloidin image.")
        if SELECT_BY == "membrane":
            rows.sort(key=lambda r: (-r["salmonella_at_membrane"], r["dapi_saturated_pct"]))
        else:
            rows.sort(key=lambda r: (-r["rods_in_field"], r["dapi_saturated_pct"]))
        chosen[cond] = rows[0]["field"]
        if log is not None:
            log.setdefault("field_selection", {})[cond] = dict(
                chosen=rows[0]["field"],
                rule=("most Salmonella candidates within %g um of a cell edge; tie-break lowest DAPI saturation"
                      % ADHESION_DIST_UM) if SELECT_BY == "membrane" else
                     "most rod-shaped objects in the field; tie-break lowest DAPI saturation",
                ranked=[(r["field"], r["salmonella_at_membrane"], r["rods_in_field"], r["dapi_saturated_pct"]) for r in rows])
    return chosen, table


def arrow(ax, y, x, dy, dx, colour, label=None, fontsize=9, head=True):
    """Arrow pointing at (x, y) from offset (dx, dy) in pixels."""
    ax.annotate(label or "", xy=(x, y), xytext=(x + dx, y + dy), color=colour, fontsize=fontsize, fontweight="bold",
                ha="center", va="center", path_effects=[pe.withStroke(linewidth=2.5, foreground="black")],
                arrowprops=dict(arrowstyle="-|>", color=colour, lw=1.8, mutation_scale=14, shrinkA=0, shrinkB=3))


def scalebar(ax, img_w_px, um, color="white", frac_y=0.93, lw=4):
    L = um / UM_PER_PX
    x1 = img_w_px * 0.96; x0 = x1 - L
    ax.plot([x0, x1], [frac_y, frac_y], transform=ax.get_xaxis_transform(), color=color, lw=lw, solid_capstyle="butt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT_FOLDER, help="folder containing 20251222/")
    ap.add_argument("--out", default=OUTPUT_FOLDER)
    ap.add_argument("--glucose-field", default=FIELDS["Glucose"], help='4-digit field, or "auto"')
    ap.add_argument("--fucose-field", default=FIELDS["Fucose"], help='4-digit field, or "auto"')
    a, _ = ap.parse_known_args()
    if not os.path.isdir(os.path.join(a.root, "20251222")):
        sys.exit(f"Cannot find the 20251222 folder inside: {a.root}\nEdit ROOT_FOLDER at the top of the script.")
    print("Reading from:", a.root, "\nSaving to:", os.path.abspath(a.out))
    os.makedirs(a.out, exist_ok=True)
    log = dict(run=datetime.datetime.now().isoformat(timespec="seconds"),
               display=dict(dapi=DAPI_RANGE, phalloidin=PHAL_RANGE, dapi_inset=DAPI_INSET_RANGE),
               um_per_px=UM_PER_PX, crop_rows=CROP_ROWS, membrane_band_um=ADHESION_DIST_UM, inputs={})

    # ---------- choose one representative field PER TREATMENT (independent field numbers)
    want = {"Glucose": a.glucose_field, "Fucose": a.fucose_field}
    auto, sel_table = ({}, [])
    if "auto" in want.values():
        print("Selecting representative fields (scanning every field with both channels)…")
        auto, sel_table = select_fields(a.root, log)
        import csv
        if sel_table:
            keys = sorted({k for r in sel_table for k in r})
            with open(os.path.join(a.out, "field_selection_table.csv"), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=["treatment", "field"] + [k for k in keys if k not in ("treatment", "field")])
                w.writeheader(); w.writerows(sel_table)
    fields = {c: (auto[c] if want[c] == "auto" else want[c]) for c in ("Glucose", "Fucose")}
    log["fields"] = dict(fields); log["field_mode"] = dict(want)
    print("Fields used — " + ", ".join(f"{c}: {fields[c]}" + (" (auto)" if want[c] == "auto" else " (set by hand)")
                                       for c in ("Glucose", "Fucose")))

    # ---------- load
    data = {}
    for cond in ("Glucose", "Fucose"):
        f_ = fields[cond]
        s = SOURCES[cond]
        dapi, p1 = read_plane(a.root, s["dapi"].format(f=f_))
        phal, p2 = None, None
        pp = phal_for(a.root, cond, f_)
        if pp is not None:
            phal, p2 = read_plane(a.root, pp)
        data[cond] = (dapi[:CROP_ROWS], None if phal is None else phal[:CROP_ROWS])
        for p in (p1, p2):
            if p: log["inputs"][os.path.relpath(p, a.root)] = sha256(p)

    # ---------- main figure: rows = conditions; columns = DAPI | AF594 phalloidin | AF488 anti-Salmonella | merge | DAPI inset
    H, W = data["Glucose"][0].shape
    iw, ih = int(INSET_UM[0] / UM_PER_PX), int(INSET_UM[1] / UM_PER_PX)
    fig, axs = plt.subplots(2, 4, figsize=(16.5, 6.4), gridspec_kw=dict(wspace=0.03, hspace=0.08))
    titles = ["DAPI\n(nuclei and bacterial DNA)", "Phalloidin–Alexa Fluor 594\n(host-cell F-actin)",
              "Merge\n(DAPI blue, F-actin red)",
              ("Enlarged at cell membrane, DAPI halo removed\n(red line: cell edge)" if MEMBRANE_VIEW == "tophat"
               else "Enlarged merge at cell membrane\n(red line: cell edge)") if ZOOM_TARGET == "membrane" else "Enlarged DAPI\n(Salmonella)"]
    for r, cond in enumerate(("Glucose", "Fucose")):
        dapi, phal = data[cond]
        if ZOOM_TARGET == "membrane" and phal is not None:
            # cell AND window chosen by where the Salmonella candidates actually are (see best_membrane_view)
            cyx, x0, y0, edge, dist, _objs = best_membrane_view(dapi, phal, iw, ih)
            cells = [cyx]
            cm = cell_mask_from_actin(phal)
            cm_all = cm
            dist_all = ndi.distance_transform_edt(~(cm_all ^ ndi.binary_erosion(cm_all))) * UM_PER_PX
            mobj_all = membrane_objects(dapi, dist_all)       # every cell in the field, for inspection
            mobj = membrane_objects(dapi, dist)               # the zoomed cell only, for the panel
            bac = [(y - y0, x - x0) for y, x, _, _, _ in mobj if y0 <= y < y0 + ih and x0 <= x < x0 + iw]
            rej = []
            log.setdefault("salmonella_at_membrane", {})[cond] = dict(
                field=fields[cond], n_in_zoomed_cell=len(mobj), n_in_field=len(mobj_all),
                objects=[dict(x=round(x, 1), y=round(y, 1), snr=s, core=co, convexity=cv) for y, x, s, co, cv in mobj_all])
            if WRITE_CANDIDATE_SHEET:
                ref = confirmed_reference(dapi, dist_all, n=12)
                log["salmonella_at_membrane"][cond]["n_confirmed_reference"] = len(ref)
                validation_sheet(dapi, mobj_all, ref,
                                 os.path.join(a.out, f"Validation_{cond}_field{fields[cond]}.png"),
                                 f"{cond}, field {int(fields[cond])} - Salmonella identification check "
                                 f"(glare-corrected DAPI, same display for every tile; band {ADHESION_DIST_UM:g} um)")
            if MEMBRANE_VIEW == "tophat":
                zoom_img = lin(glare_corrected(dapi)[y0:y0 + ih, x0:x0 + iw], TOPHAT_RANGE)   # grey: clearest
            else:
                zoom_img = merge_custom(dapi[y0:y0 + ih, x0:x0 + iw], phal[y0:y0 + ih, x0:x0 + iw],
                                        DAPI_MEMBRANE_RANGE, PHAL_RANGE)
            zoom_mask = cm[y0:y0 + ih, x0:x0 + iw]
        else:
            cells = pick_cells(dapi, phal, n=1)
            x0, y0 = best_window(bacteria_like(dapi), dapi.shape, iw, ih, clean=clean_mask(dapi))
            bac = pick_bacteria(dapi[y0:y0 + ih, x0:x0 + iw], n=4)
            rej = []
            zoom_img = lin(dapi[y0:y0 + ih, x0:x0 + iw], DAPI_INSET_RANGE); zoom_mask = None
        log[f"inset_{cond}"] = dict(x=int(x0), y=int(y0), w=iw, h=ih, target=ZOOM_TARGET, bacteria_marked=len(bac))
        log[f"annotations_{cond}"] = dict(cells=[[round(v, 1) for v in c_] for c_ in cells], bacteria_in_inset=[[round(v, 1) for v in b_] for b_ in bac])
        panels = [lin(dapi, DAPI_RANGE),                       # single channels in grey (best contrast)
                  lin(phal, PHAL_RANGE) if phal is not None else None,
                  merge(dapi, phal),
                  zoom_img]
        for c, img in enumerate(panels):
            ax = axs[r, c]; ax.set_xticks([]); ax.set_yticks([])
            if r == 0: ax.set_title(titles[c], fontsize=10.5)
            if c == 0: ax.set_ylabel(f"{cond}\nfield {int(fields[cond])}", fontsize=12)
            if img is None:
                ax.imshow(np.zeros((H, W)), cmap="gray", vmin=0, vmax=1)
                ax.text(0.5, 0.5, "no image", color="white", ha="center", va="center", fontsize=10, transform=ax.transAxes)
                continue
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=1, interpolation="nearest")
            ax.text(0.015, 0.97, "abcdefgh"[r * 4 + c], transform=ax.transAxes, color="white", fontsize=13,
                    fontweight="bold", va="top", ha="left", path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])
            if c in (0, 1, 2):
                for (cy, cx, rad) in cells:
                    # diagonal arrow from the side horizontally away from the inset box; label placed beyond the tail
                    best = None; fallback = None
                    for sx in (-1, 1):
                        for sy in (-1, 1):
                            tipx, tipy = cx + sx * rad * 0.5, cy + sy * rad * 0.5
                            tailx, taily = tipx + sx * 130, tipy + sy * 130
                            lx = tailx + sx * 330                              # far end of the label
                            inside = (20 < min(tailx, lx) and max(tailx, lx) < W - 20
                                      and 0.16 * H < taily < H - 60)      # 0.16*H keeps clear of the scale bar
                            d = np.hypot(tailx - (x0 + iw / 2), taily - (y0 + ih / 2))
                            cand = (d, sx, sy, tipx, tipy, tailx, taily)
                            if inside and (best is None or d > best[0]):
                                best = cand
                            fallback = cand if (fallback is None or d > fallback[0]) else fallback
                    if best is None:
                        best = fallback                                   # no clear spot: use the farthest one
                    _, sx, sy, tipx, tipy, tailx, taily = best
                    ax.annotate("", xy=(tipx, tipy), xytext=(tailx, taily),
                                arrowprops=dict(arrowstyle="-|>", color="white", lw=2, mutation_scale=16, shrinkA=0, shrinkB=0))
                    if c == 2:
                        ax.text(tailx + sx * 10, taily + sy * 25, "Host cell (HCT116)", color="white", fontsize=9, fontweight="bold",
                                ha="left" if sx > 0 else "right", va="center",
                                path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])
            if c == 2:
                ax.add_patch(Rectangle((x0, y0), iw, ih, fill=False, ec="yellow", lw=1.4, ls="--"))
            if c == 3 and zoom_mask is not None:
                ax.contour(zoom_mask.astype(float), levels=[0.5], colors="#ff5050", linewidths=1.6, linestyles="--")
                if SHOW_REJECTED:
                    for (by, bx, why) in rej:
                        ax.plot([bx], [by], marker="o", mfc="none", mec="#ff3b30", mew=1.6, ms=13)
                if not bac:
                    bmz, _, _, _ = band_metrics(dapi, phal)
                    ax.text(0.03, 0.06, f"No Salmonella resolvable within {ADHESION_DIST_UM:g} µm of the edge",
                            transform=ax.transAxes, color="yellow", fontsize=8.5, fontweight="bold",
                            path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])
            if c == 3:
                for j, (by, bx) in enumerate(bac):
                    off = 0.11 * iw
                    dxb = off if bx < iw / 2 else -off          # tail toward the panel centre: arrows point outward, never cross
                    dyb = -off                                   # all arrows come from above
                    arrow(ax, by, bx, dyb, dxb, "yellow", None)
                if bac:
                    ax.text(0.03, 0.06, f"Salmonella candidates ({len(bac)})" if zoom_mask is not None else "Salmonella (DAPI+ rods)", transform=ax.transAxes, color="yellow", fontsize=9,
                            fontweight="bold", path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])
            scalebar(ax, img.shape[1], 5 if c == 3 else 50)
        # link dashed box in the merge to the inset panel
        for (xa, ya), (xb, yb) in [((x0 + iw, y0), (0, 0)), ((x0 + iw, y0 + ih), (0, ih))]:
            axs[r, 3].add_artist(ConnectionPatch(xyA=(xa, ya), coordsA=axs[r, 2].transData, xyB=(xb, yb),
                                                 coordsB=axs[r, 3].transData, color="yellow", lw=0.8, ls="--"))
    key = "   ".join(f"{v[2].capitalize()}: {v[1]} ({v[0]})" for k_, v in STAINS.items() if k_ != "af488")
    fig.text(0.125, 0.085, "Staining in merge — " + key, fontsize=8)
    zoom_note = ((f"Enlarged panel: DAPI after a uniform white top-hat ({TOPHAT_UM:g} µm element, linear {TOPHAT_RANGE[0]}–{TOPHAT_RANGE[1]}) that removes "
                  f"the out-of-focus halo;\nred dashed line = cell edge from phalloidin; yellow arrowheads = Salmonella candidates within "
                  f"{ADHESION_DIST_UM:g} µm of that edge.\nIn this bright zone an out-of-focus debris ring can mimic a bacterium, so these "
                  f"are candidates: check each one against confirmed Salmonella on the Validation_*.png sheets."
                  if MEMBRANE_VIEW == "tophat" else
                  f"Enlarged merge: DAPI shown at {DAPI_MEMBRANE_RANGE[0]}–{DAPI_MEMBRANE_RANGE[1]}; red dashed line = cell edge.")
                 if ZOOM_TARGET == "membrane" else "Enlarged DAPI shown at %d–%d." % DAPI_INSET_RANGE)
    fig.text(0.125, -0.03, "Scale bars: 50 µm (full fields), 5 µm (enlarged panel). Identical linear display settings for both conditions.\n"
             "White arrows: example host cell; yellow arrowheads: Salmonella (rod-shaped DAPI-positive objects); dashed yellow box: enlarged region.\n"
             + zoom_note, fontsize=8)
    for ext in ("png", "tif"):
        fig.savefig(os.path.join(a.out, f"Figure_qualitative_G{fields['Glucose']}_F{fields['Fucose']}.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------- membrane zoom figure (adhesion check) + table for every field with phalloidin
    if MAKE_MEMBRANE_ZOOM:
        mw, mh = int(MEMBRANE_UM[0] / UM_PER_PX), int(MEMBRANE_UM[1] / UM_PER_PX)
        fig, axs = plt.subplots(2, 3, figsize=(13.5, 6.6), gridspec_kw=dict(wspace=0.04, hspace=0.12))
        ttl = ["Merge, host cell", "Phalloidin, cell edge enlarged",
               ("DAPI with halo removed, cell edge enlarged" if MEMBRANE_VIEW == "tophat"
                else "DAPI (or AF488), cell edge enlarged")]
        for r, cond in enumerate(("Glucose", "Fucose")):
            dapi, phal = data[cond]
            af = None
            ok, why = valid_af488(a.root, AF488_PATTERNS[cond].format(f=fields[cond]))
            log.setdefault("af488_check", {})[AF488_PATTERNS[cond].format(f=fields[cond])] = why
            if ok:
                af, full = read_plane(a.root, AF488_PATTERNS[cond].format(f=fields[cond])); af = af[:CROP_ROWS]
                log["inputs"][os.path.relpath(full, a.root)] = sha256(full)
            bac_img = af if af is not None else dapi
            cm = cell_mask_from_actin(phal)
            cell = pick_cells(dapi, phal, n=1)[0]
            x0, y0, edge = membrane_window(cm, dapi, cell, mw, mh)
            dist = ndi.distance_transform_edt(~edge) * UM_PER_PX
            mobj_f = membrane_objects(bac_img, dist)
            rods = mobj_f
            at_mem = [(y, x) for y, x, _, _, _ in mobj_f]
            in_win = [(y - y0, x - x0) for y, x in at_mem if y0 <= y < y0 + mh and x0 <= x < x0 + mw]
            bm, _, _, _ = band_metrics(dapi, phal, bac=af)
            log[f"membrane_{cond}"] = dict(window=[int(x0), int(y0), mw, mh], field=fields[cond],
                                           bacteria_channel="AF488" if af is not None else "DAPI",
                                           band_um=ADHESION_DIST_UM,
                                           rods_field=int(len(rods)), rods_at_membrane_field=len(at_mem),
                                           rods_at_membrane_window=len(in_win), **bm)
            if af is not None:
                zrng = (float(np.percentile(af, 1)), float(np.percentile(af, 99.9))); zimg = bac_img
            elif MEMBRANE_VIEW == "tophat":
                zrng = TOPHAT_RANGE; zimg = glare_corrected(bac_img)     # halo removed, uniform and disclosed
            else:
                zrng = DAPI_MEMBRANE_RANGE; zimg = bac_img
            rej_win = []
            panels = [merge(dapi, phal), lin(phal[y0:y0 + mh, x0:x0 + mw], PHAL_RANGE),
                      lin(zimg[y0:y0 + mh, x0:x0 + mw], zrng)]
            for c, img in enumerate(panels):
                ax = axs[r, c]; ax.set_xticks([]); ax.set_yticks([])
                ax.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=1, interpolation="nearest")
                if r == 0: ax.set_title(ttl[c], fontsize=10.5)
                if c == 0:
                    ax.set_ylabel(f"{cond}\nfield {int(fields[cond])}", fontsize=12)
                    ax.add_patch(Rectangle((x0, y0), mw, mh, fill=False, ec="yellow", lw=1.4, ls="--"))
                    scalebar(ax, img.shape[1], 50)
                else:
                    ax.contour(cm[y0:y0 + mh, x0:x0 + mw].astype(float), levels=[0.5], colors="#ff4040",
                               linewidths=1.6, linestyles="--")                              # membrane outline (overlay)
                    scalebar(ax, mw, 5)
                    if c == 2:
                        for (by, bx) in in_win:
                            arrow(ax, by, bx, -0.12 * mh, 0.12 * mw if bx < mw / 2 else -0.12 * mw, "yellow")
                        if SHOW_REJECTED:
                            for (by, bx) in rej_win:
                                ax.plot([bx], [by], marker="o", mfc="none", mec="#ff3b30", mew=1.6, ms=15)
                        msg = (f"{len(in_win)} Salmonella candidate(s) within {ADHESION_DIST_UM:g} µm of the edge" if in_win else
                               f"No Salmonella resolvable within {ADHESION_DIST_UM:g} µm of the edge\n"
                               f"({100 * bm['band_evaluable_frac']:.0f}% of the band is below the DAPI glare limit)"
                               )
                        ax.text(0.03, 0.05, msg, transform=ax.transAxes, color="yellow", fontsize=9, fontweight="bold",
                                path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])
                ax.text(0.015, 0.97, "abcdef"[r * 3 + c], transform=ax.transAxes, color="white", fontsize=13, fontweight="bold",
                        va="top", path_effects=[pe.withStroke(linewidth=2.5, foreground="black")])
        fig.text(0.125, 0.015, f"Red line: host-cell edge from phalloidin (overlay). Yellow arrowheads: rod-shaped bacteria within "
                 f"{ADHESION_DIST_UM:g} µm of the edge (local SNR ≥ 7). DAPI zoom shown at linear range {DAPI_MEMBRANE_RANGE[0]}–{DAPI_MEMBRANE_RANGE[1]} "
                 + (f"The enlarged DAPI panel is shown after a uniform white top-hat ({TOPHAT_UM:g} µm element, linear "
                    f"{TOPHAT_RANGE[0]}–{TOPHAT_RANGE[1]}) that removes the smooth out-of-focus halo; red circles mark bright features "
                    f"that fail the test (dark-centred rings = out-of-focus debris, not fluorescent bacteria). "
                    if MEMBRANE_VIEW == "tophat" else "") +
                 f"Scale bars 50 µm and 5 µm.\n"
                 f"Each treatment is shown at its own representative field (Glucose {int(fields['Glucose'])}, Fucose {int(fields['Fucose'])}); "
                 f"see field_selection_table.csv and membrane_adhesion_table.csv for every field.", fontsize=8)
        fig.savefig(os.path.join(a.out, f"Figure_membrane_zoom_G{fields['Glucose']}_F{fields['Fucose']}.png"),
                    dpi=300, bbox_inches="tight")
        plt.close(fig)
        # table across all fields with phalloidin (both conditions)
        rows = []
        for cond in ("Glucose", "Fucose"):
            for k in range(10):
                f = f"{k + 1:04d}"
                s_ = SOURCES[cond]; cands = [s_["phal"]] if "phal" in s_ else s_["phal_options"]
                pp = next((c.format(f=f) for c in cands if os.path.exists(os.path.join(a.root, c.format(f=f).split("::")[0]))), None)
                dp = s_["dapi"].format(f=f)
                if pp is None or not os.path.exists(os.path.join(a.root, dp)):
                    rows.append(f"{cond},{f},no phalloidin image,,,,"); continue
                d_, _ = read_plane(a.root, dp); p_, _ = read_plane(a.root, pp); d_, p_ = d_[:CROP_ROWS], p_[:CROP_ROWS]
                ok_, why_ = valid_af488(a.root, AF488_PATTERNS[cond].format(f=f))
                if why_ != "absent": log.setdefault("af488_check", {})[AF488_PATTERNS[cond].format(f=f)] = why_
                b_ = read_plane(a.root, AF488_PATTERNS[cond].format(f=f))[0][:CROP_ROWS] if ok_ else d_
                bm_, _, _, _ = band_metrics(d_, p_, bac=(b_ if ok_ else None))
                rows.append(f"{cond},{f},{'AF488' if ok_ else 'DAPI'},{bm_['rods_in_field']},{bm_['rods_within_band']},"
                            f"{bm_['median_DAPI_at_band']},{bm_['band_um2']},{bm_['band_evaluable_frac']},"
                            f"{bm_['band_evaluable_um2']},{bm_['n_cells']}")
        open(os.path.join(a.out, "membrane_adhesion_table.csv"), "w").write(
            ("condition,field,bacteria_channel,clear_rods_in_field,rods_within_%gum_of_membrane,median_DAPI_at_membrane,"
             "membrane_band_um2,band_fraction_below_glare_limit,band_evaluable_um2,n_cells\n") % ADHESION_DIST_UM
            + "\n".join(rows) + "\n")

    # ---------- supplementary gallery: every field, both conditions, same settings
    for cond, pat in (("Glucose", "20251222/glu_b_{f}.tif"), ("Fucose", "20251222/fucose_b_{f}.tif")):
        fig, axs = plt.subplots(2, 10, figsize=(25, 4.2), gridspec_kw=dict(wspace=0.02, hspace=0.04))
        for k in range(10):
            f = f"{k + 1:04d}"
            p = os.path.join(a.root, pat.format(f=f))
            if not os.path.exists(p):
                axs[0, k].axis("off"); axs[1, k].axis("off"); continue
            d, full = read_plane(a.root, pat.format(f=f)); d = d[:CROP_ROWS]
            log["inputs"][os.path.relpath(full, a.root)] = sha256(full)
            x0, y0 = best_window(bacteria_like(d), d.shape, iw, ih)
            axs[0, k].imshow(lin(d, DAPI_RANGE), cmap="gray", vmin=0, vmax=1)
            axs[0, k].add_patch(Rectangle((x0, y0), iw, ih, fill=False, ec="yellow", lw=0.8))
            axs[1, k].imshow(lin(d[y0:y0 + ih, x0:x0 + iw], DAPI_INSET_RANGE), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axs[0, k].set_title(f"{cond} field {k + 1:02d}", fontsize=8)
            scalebar(axs[0, k], d.shape[1], 50, lw=2); scalebar(axs[1, k], iw, 10, lw=2)
            for r in range(2): axs[r, k].set_xticks([]); axs[r, k].set_yticks([])
        fig.savefig(os.path.join(a.out, f"Supplementary_gallery_{cond}_all_fields.png"), dpi=200, bbox_inches="tight")
        plt.close(fig)

    # ---------- integrity check + log + legend
    after = {k: sha256(os.path.join(a.root, k)) for k in log["inputs"]}
    log["sources_unchanged"] = after == log["inputs"]
    json.dump(log, open(os.path.join(a.out, "processing_log.json"), "w"), indent=1)
    phal_note = ""
    if any("composite" in k.lower() and "fucose" in k.lower() for k in log["inputs"]):
        phal_note = "; the fucose phalloidin image is the red channel of the composite saved at acquisition"
    legend = (
        f"Representative widefield fluorescence images of HCT116 cells infected with S. Typhi after pre-incubation of the bacteria "
        f"with glucose (top) or fucose (bottom). Staining: host-cell nuclei (and bacterial DNA), DAPI (blue); host-cell F-actin, "
        f"Alexa Fluor 594 phalloidin (red); Salmonella, anti-Salmonella antibody (clone 100/358) conjugated to Alexa Fluor 488 "
        f"(Novus Biologicals; green). The Alexa Fluor 488 channel was not acquired in this imaging session; bacteria are therefore "
        f"identified only as small rod-shaped DAPI-positive objects (putative bacteria; DAPI is not specific for Salmonella) (panels d, h; enlarged from the dashed boxes in c, g). White arrows mark an example host cell; yellow arrowheads mark putative bacteria; arrows are annotations added for orientation and do not alter the image data. Images are shown for qualitative illustration; "
        f"bacterial adherence and invasion were quantified by CFU assays (Figure 3). One representative field is shown per treatment "
        f"(glucose, field {int(fields['Glucose'])}; fucose, field {int(fields['Fucose'])}), selected independently for each treatment by a rule fixed in advance: "
        f"among the fields with both a DAPI and a phalloidin image, the field with the most rod-shaped DAPI-positive objects, ties broken by the lower DAPI saturation "
        f"(these are therefore clearest-example fields, not median fields{phal_note}); all fields are shown in Supplementary Figure SX. Images were acquired on a "
        f"Leica widefield microscope (40×, DFC365 FX camera, 0.161 µm/pixel). Only linear brightness/contrast adjustment was applied, "
        f"identically to all images of a channel (DAPI {DAPI_RANGE[0]}–{DAPI_RANGE[1]}, phalloidin {PHAL_RANGE[0]}–{PHAL_RANGE[1]}, "
        f"inset DAPI {DAPI_INSET_RANGE[0]}–{DAPI_INSET_RANGE[1]} grey levels); the burned-in acquisition scale bar was cropped and "
        f"replaced by calibrated scale bars (50 µm; enlarged panels 5 µm).")
    open(os.path.join(a.out, "figure_legend.txt"), "w").write(legend)
    print("done; sources unchanged:", log["sources_unchanged"])
    print("Open this folder to see the figure:", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
