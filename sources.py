"""Data sources for the mask editor: where images and mask layers come from.

Both sources expose the same small interface used by server.py:
    pids()                      -> list of ids
    stem(id)                    -> on-disk filename stem for that id
    chest(id)                   -> 2D float image (the background)
    parenchyma(id, shape)       -> uint8 mask, layer 0
    nodule(id, shape)           -> uint8 mask, layer 1
(The chest/parenchyma/nodule names are historical; think image / layer0 / layer1.)
"""
import ast
import csv
import io
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_series(spec):
    """Return series stems 'P<patient>-S<last5>' from a --series-style spec.

    spec is a file or an inline string. A file may be a CSV with 'patient' and 'last5'
    columns, or one entry per line. An entry is either a stem (P100012-S22196) or a
    'patient,last5' pair (also accepts space/dash). Inline, separate pairs with ';'
    (e.g. "100012,22196; 100570,45005") or pass stems comma-separated.
    """
    p = Path(spec)
    if p.exists():
        text = p.read_text()
        head = text.splitlines()[0].lower() if text.strip() else ""
        if "patient" in head and "last5" in head:  # CSV with named columns
            return [f"P{r['patient'].strip()}-S{r['last5'].strip()}"
                    for r in csv.DictReader(io.StringIO(text)) if r.get("patient") and r.get("last5")]
        chunks = text.splitlines()
    else:
        chunks = re.split(r"[;\n]+", spec)  # inline: pairs separated by ';' or newline

    entries = []
    for ch in chunks:
        ch = ch.strip()
        if not ch or ch.startswith("#"):
            continue
        if "," in ch and ch.upper().startswith("P"):  # a comma list of stems
            entries += [c.strip() for c in ch.split(",") if c.strip()]
        else:
            entries.append(ch)

    out = []
    for e in entries:
        if re.match(r"P\d+-S\w+$", e):
            out.append(e)
            continue
        parts = re.split(r"[,\s\-]+", e)
        if len(parts) >= 2 and parts[0].isdigit():
            out.append(f"P{parts[0]}-S{parts[1]}")
    return out


_ID_ALIASES = ["image", "id", "filename", "stem", "series", "pid"]
_X_ALIASES = ["x", "coordx", "centroid_x", "cx", "col"]
_Y_ALIASES = ["y", "coordy", "centroid_y", "cy", "row"]


def _find_col(fieldnames, aliases, explicit=None):
    if explicit:
        for f in fieldnames:
            if f.strip().lower() == explicit.strip().lower():
                return f
        raise ValueError(f"column '{explicit}' not found (have: {fieldnames})")
    low = {f.strip().lower(): f for f in fieldnames}
    for a in aliases:
        if a in low:
            return low[a]
    return None


def parse_nodule_centroids(csv_path, id_col=None, x_col=None, y_col=None):
    """Read a nodule-centroid CSV: one row per nodule, an id column (matching a whole-slice
    <id>.npy stem) plus two pixel-coordinate columns. Column names are sniffed from common
    aliases (image/id/filename/stem/series/pid for the id; x/coordX/centroid_x/cx/col and
    y/coordY/centroid_y/cy/row for the coordinates) — same sniff-then-fallback approach as
    parse_series. Any of id_col/x_col/y_col can be given explicitly to override the sniff.
    If no column matches by name, the first three columns are used positionally. Multiple
    rows per id are all kept (one nodule each). Returns {id: [(x, y), ...]}.
    """
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        idc = _find_col(fieldnames, _ID_ALIASES, id_col)
        xc = _find_col(fieldnames, _X_ALIASES, x_col)
        yc = _find_col(fieldnames, _Y_ALIASES, y_col)
        if idc is None or xc is None or yc is None:
            if len(fieldnames) < 3:
                raise ValueError(f"can't find id/x/y columns in header: {fieldnames}")
            idc, xc, yc = idc or fieldnames[0], xc or fieldnames[1], yc or fieldnames[2]
        out = defaultdict(list)
        for r in reader:
            if not r.get(idc) or r.get(xc) in (None, "") or r.get(yc) in (None, ""):
                continue
            out[r[idc].strip()].append((float(r[xc]), float(r[yc])))
    return dict(out)


class StagingSource:
    """LUNA25 protocol_7 staging, addressed directly by series stem P<patient>-S<last5>.

    The caller supplies the series (patient + last5 of the SeriesInstanceUID) — there is
    no pid/visit resolution. chest and lung_parenchyma live at P<patient>-S<last5>.npy;
    the nodule layer is the union of the in-plane nodule crops (a nodule is in-plane iff
    chest[box] == crop exactly).
    """

    def __init__(self, dataset_dir, series=None):
        s = Path(dataset_dir)
        # accept either the protocol_7 nested layout (images/chest, masks/lung_parenchyma,
        # ...) or a flat folder of P*-S*.npy images (masks/crops then simply absent).
        nested = s / "images" / "chest"
        self.chest_dir = nested if nested.is_dir() else s
        self.crop_dir = s / "images" / "lung_nodule_crop"
        self.mask_dir = s / "masks" / "lung_parenchyma"
        # (patient, last5) -> [(nodule_id, y, x)] from the dataset's own metadata
        self.nodule_index = defaultdict(list)
        meta = s / "metadata" / "lung_nodule_metadata.csv"
        if meta.exists():
            with open(meta) as f:
                for r in csv.DictReader(f):
                    _z, y, x = ast.literal_eval(r["crop_center_voxel_indices"])
                    self.nodule_index[(r["patient_id"], r["series_instance_uid"][-5:])].append((r["nodule_id"], y, x))
        # no series list -> edit every chest slice in the folder
        if not series:
            series = sorted(p.stem for p in self.chest_dir.glob("P*-S*.npy"))
        # keep the requested order, drop series whose chest slice is absent
        self._ids = [st for st in series if (self.chest_dir / f"{st}.npy").exists()]
        self.missing = [st for st in series if not (self.chest_dir / f"{st}.npy").exists()]

    @staticmethod
    def _pl(stem):
        m = re.match(r"P(\d+)-S(\w+)$", stem)
        return (m.group(1), m.group(2)) if m else (None, None)

    def pids(self):
        return self._ids

    def stem(self, sid):
        return sid  # already the series-keyed filename stem

    def chest(self, sid):
        return np.load(self.chest_dir / f"{sid}.npy").astype(np.float32)

    def parenchyma(self, sid, shape):
        p = self.mask_dir / f"{sid}.npy"
        if p.exists():
            return (np.load(p) > 0).astype(np.uint8)
        return np.zeros(shape, np.uint8)

    def nodule(self, sid, shape):
        pat, last5 = self._pl(sid)
        chest = self.chest(sid)
        nm = np.zeros(chest.shape, bool)
        for nid, y, x in self.nodule_index.get((pat, last5), []):
            cp = self.crop_dir / f"P{pat}-S{last5}-N{int(nid):02d}.npy"
            if not cp.exists():
                continue
            crop = np.load(cp)
            ch, cw = crop.shape
            y0, x0 = y - ch // 2, x - cw // 2
            if y0 < 0 or x0 < 0 or y0 + ch > chest.shape[0] or x0 + cw > chest.shape[1]:
                continue
            if np.abs(chest[y0:y0 + ch, x0:x0 + cw] - crop).max() < 1e-4:
                nm[y0:y0 + ch, x0:x0 + cw] = True
        return nm.astype(np.uint8)


class GenericSource:
    """Backend-agnostic source: a folder of <id>.npy images and up to two folders of
    <id>.npy masks (layer 0, layer 1). Not tied to LUNA. chest() = image,
    parenchyma()/nodule() = mask layer 0 / 1 (empty if that layer has no folder or no
    file for this id).

    The nodule layer (layer 1) can also be seeded from nodule_centroids ({id: [(x,y),...]},
    from parse_nodule_centroids): a real mask file in mask_dirs[1] wins if one exists for
    that id, otherwise a CENTROID_BOX-px square centered on each centroid stands in as a
    rough starting mask to refine by hand.
    """

    CENTROID_BOX = 32

    def __init__(self, images_dir, mask_dirs, ids=None, nodule_centroids=None):
        self.images_dir = Path(images_dir)
        self.mask_dirs = [Path(d) if d else None for d in mask_dirs]  # 1 or 2 entries
        self.nodule_centroids = nodule_centroids or {}
        avail = sorted(p.stem for p in self.images_dir.glob("*.npy"))
        avail_set = set(avail)
        self._ids = [i for i in (ids or avail) if i in avail_set]
        self.missing = [i for i in (ids or []) if i not in avail_set]

    def pids(self):
        return self._ids

    def stem(self, i):
        return i

    def chest(self, i):
        return np.load(self.images_dir / f"{i}.npy").astype(np.float32)

    def _layer(self, i, li, shape):
        d = self.mask_dirs[li] if li < len(self.mask_dirs) else None
        if d is not None:
            p = d / f"{i}.npy"
            if p.exists():
                m = (np.load(p) > 0).astype(np.uint8)
                if m.shape == shape:
                    return m
        return None

    def parenchyma(self, i, shape):
        m = self._layer(i, 0, shape)
        return m if m is not None else np.zeros(shape, np.uint8)

    def nodule(self, i, shape):
        m = self._layer(i, 1, shape)
        if m is not None:
            return m
        if self.nodule_centroids:
            r = self.CENTROID_BOX // 2
            m = np.zeros(shape, np.uint8)
            for x, y in self.nodule_centroids.get(i, []):
                y0, y1 = max(0, round(y) - r), min(shape[0], round(y) + r)
                x0, x1 = max(0, round(x) - r), min(shape[1], round(x) + r)
                m[y0:y1, x0:x1] = 1
            return m
        return np.zeros(shape, np.uint8)


RED, GREEN = [255, 85, 85], [0, 230, 90]


def _slug(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_") or "layer"


def _shape_mismatches(images_dir, mask_dir, ids):
    """Compare each id's image shape against its mask's (where a mask file exists).
    mmap_mode="r" reads just the header, so this stays cheap even over thousands of ids.
    Returns a list of (id, image_shape, mask_shape) for every mismatch found.
    """
    images_dir, mask_dir = Path(images_dir), Path(mask_dir)
    bad = []
    for i in ids:
        mp = mask_dir / f"{i}.npy"
        if not mp.exists():
            continue
        img_shape = np.load(images_dir / f"{i}.npy", mmap_mode="r").shape
        mask_shape = np.load(mp, mmap_mode="r").shape
        if img_shape != mask_shape:
            bad.append((i, img_shape, mask_shape))
    return bad


def _check_mask_shapes(images_dir, mask_dir, ids, layer_name):
    bad = _shape_mismatches(images_dir, mask_dir, ids)
    if not bad:
        return
    examples = "; ".join(f"{i}: image {s[1]}x{s[0]} vs mask {m[1]}x{m[0]}" for i, s, m in bad[:3])
    more = f" (+{len(bad) - 3} more)" if len(bad) > 3 else ""
    raise ValueError(f"{len(bad)} {layer_name} mask(s) don't match their image size — {examples}{more}")


def build_generic_cfg(images, masks, masks2, names="lung,nodule", out_dir=None, ids=None, no_masked=False,
                       nodule_csv=None, nodule_cols=None):
    """Build a CFG dict (source, layers, out_*_dir, ...) for generic mode: an images folder
    plus up to two mask folders. nodule_csv is an alternative (or complement) to masks2: a
    nodule-centroid CSV (see parse_nodule_centroids) seeding layer 1 with a small square
    per centroid wherever masks2 doesn't already have a real mask file for that id.
    nodule_cols is an optional (id_col, x_col, y_col) override for its column names.
    Raises ValueError if the images folder can't be resolved, the CSV's columns can't be
    found, or a mask folder has files whose shape doesn't match their image — callers
    decide how to surface that (SystemExit for the CLI, a JSON error for the API).
    """
    out = Path(out_dir or "mask_edits")
    names = [n.strip() for n in (names or "").split(",") if n.strip()] or ["mask"]
    two = bool(masks2) or bool(nodule_csv) or len(names) >= 2
    names = (names + ["label2"])[:2] if two else names[:1]
    layers = [{"name": names[0], "color": RED}] + ([{"name": names[1], "color": GREEN}] if two else [])
    try:
        centroids = parse_nodule_centroids(nodule_csv, *(nodule_cols or (None, None, None))) if nodule_csv else None
    except FileNotFoundError:
        raise ValueError(f"nodule CSV not found: {nodule_csv}")
    source = GenericSource(images, [masks, masks2][:len(layers)], ids, nodule_centroids=centroids)
    if not source.images_dir.is_dir():
        raise ValueError(f"images dir not found: {source.images_dir}")
    if masks:
        _check_mask_shapes(images, masks, source.pids(), layers[0]["name"])
    if masks2:
        _check_mask_shapes(images, masks2, source.pids(), layers[1]["name"])
    return dict(
        source=source, layers=layers, nodules=two, save_masked=not no_masked, title="Mask editor",
        out_mask_dir=out / "masks" / _slug(layers[0]["name"]),
        out_nodule_dir=out / "masks" / (_slug(layers[1]["name"]) if two else "layer2"),
        out_masked_dir=out / "images" / "masked",
    )


def build_empty_cfg(out_dir=None):
    """CFG for the unconfigured splash-screen state: generic mode, no images folder set
    yet, zero pids. The splash's Images/Lung mask/Nodule mask fields populate this for
    real via /api/configure once the user picks something — this just lets the server
    start (and the splash render) before that happens.
    """
    out = Path(out_dir or "mask_edits")
    placeholder = out / "_unconfigured"  # never exists; GenericSource.glob() on it is a no-op
    source = GenericSource(placeholder, [None], [])
    return dict(
        source=source, layers=[{"name": "mask", "color": RED}], nodules=False,
        save_masked=True, title="Mask editor",
        out_mask_dir=out / "masks" / "mask",
        out_nodule_dir=out / "masks" / "layer2",
        out_masked_dir=out / "images" / "masked",
    )


def build_luna_cfg(dataset, series_spec=None, out_dir=None, no_masked=False):
    """Build a CFG dict (source, layers, out_*_dir, ...) for the LUNA25 protocol_7 preset:
    series stems 'P<patient>-S<last5>', fixed lung + nodule layers. series_spec is optional
    (a --series-style spec: a CSV/text file path or inline text) — without it every chest
    slice found under dataset is edited. Raises ValueError if the dataset's chest folder
    can't be found — callers decide how to surface that (SystemExit for the CLI, a JSON
    error for the API).
    """
    out = Path(out_dir or "mask_edits")
    series = parse_series(series_spec) if series_spec else None
    if series_spec and not series:
        raise ValueError("no series parsed from the series spec")
    source = StagingSource(dataset, series)
    if not source.chest_dir.is_dir():
        raise ValueError(f"chest folder not found: {source.chest_dir} (dataset must be a "
                          "protocol_7 root with images/chest, or a flat folder of P*-S*.npy)")
    return dict(
        source=source, nodules=True, save_masked=not no_masked, title="LUNA lung-mask editor",
        layers=[{"name": "lung", "color": RED}, {"name": "nodule", "color": GREEN}],
        out_mask_dir=out / "masks" / "lung_parenchyma",
        out_nodule_dir=out / "masks" / "lung_nodule",
        out_masked_dir=out / "images" / "lung_masked",
    )


def make_demo(dirpath, n=8):
    """Write n synthetic image+mask pairs so the app runs with zero real data."""
    d = Path(dirpath)
    (d / "images").mkdir(parents=True, exist_ok=True)
    (d / "masks").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    H = W = 256
    yy, xx = np.mgrid[0:H, 0:W]
    for k in range(n):
        img = 0.2 + 0.08 * rng.standard_normal((H, W))
        m = np.zeros((H, W), np.uint8)
        for _ in range(rng.integers(1, 4)):
            cy, cx = rng.integers(50, 206, 2)
            r = rng.integers(18, 46)
            blob = ((yy - cy) ** 2 + (xx - cx) ** 2) < r * r
            img[blob] += 0.55
            m[blob] = 1
        np.save(d / "images" / f"case{k:02d}.npy", np.clip(img, 0, 1).astype(np.float32))
        np.save(d / "masks" / f"case{k:02d}.npy", m)
    return d
