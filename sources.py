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
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


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
    """

    def __init__(self, images_dir, mask_dirs, ids=None):
        self.images_dir = Path(images_dir)
        self.mask_dirs = [Path(d) if d else None for d in mask_dirs]  # 1 or 2 entries
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
        return np.zeros(shape, np.uint8)

    def parenchyma(self, i, shape):
        return self._layer(i, 0, shape)

    def nodule(self, i, shape):
        return self._layer(i, 1, shape)


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
