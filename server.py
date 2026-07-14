"""Flask server for the mask editor: HTTP routes + image<->PNG helpers.

The active data source and output layout live in CFG, populated by mask_editor.py
before app.run(). The frontend is static (mask_editor.html + static/).
"""
import base64
import io
from pathlib import Path

import numpy as np
from flask import Flask, Response, jsonify, request
from PIL import Image

try:
    from scipy import ndimage
except Exception:  # scipy optional; fill-holes/clean disabled without it
    ndimage = None

app = Flask(__name__)
CFG = {}  # set by mask_editor.main(): source, layers, out_*_dir, nodules, save_masked, title
HTML_PATH = Path(__file__).with_name("mask_editor.html")


# ----------------------------- data helpers --------------------------------
def load_chest(pid):
    arr = CFG["source"].chest(pid).astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return arr


def load_mask(pid, shape, orig=False):
    # previously-saved edit wins (unless Reset asked for the original)
    if not orig:
        edit = CFG["out_mask_dir"] / f"{CFG['source'].stem(pid)}.npy"
        if edit.exists():
            return (np.load(edit) > 0).astype(np.uint8)
    return CFG["source"].parenchyma(pid, shape)


def load_nodule_mask(pid, shape, orig=False):
    """Layer-1 mask. A previously-saved edit wins unless the original is requested."""
    if not orig:
        e = CFG["out_nodule_dir"] / f"{CFG['source'].stem(pid)}.npy"
        if e.exists():
            m = (np.load(e) > 0).astype(np.uint8)
            if m.shape == shape:
                return m
    return CFG["source"].nodule(pid, shape)


def gray_png_b64(arr01):
    img = Image.fromarray((np.clip(arr01, 0, 1) * 255).astype(np.uint8))  # 2-D uint8 -> mode "L"
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def mask_png_b64(mask):
    img = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def decode_mask_png(data_url, shape):
    raw = base64.b64decode(data_url.split(",", 1)[1])
    img = Image.open(io.BytesIO(raw)).convert("L")
    m = (np.array(img) > 127).astype(np.uint8)
    if m.shape != shape:  # safety: resize nearest to native
        img = img.resize((shape[1], shape[0]), Image.NEAREST)
        m = (np.array(img) > 127).astype(np.uint8)
    return m


# ---------------------------- request validation ---------------------------
def _known(pid):
    return pid in set(CFG["source"].pids())


def _mask_request():
    """Validate a POST body carrying a mask PNG + dims. Returns (shape, json) or aborts."""
    data = request.get_json(silent=True) or {}
    for k in ("mask", "w", "h"):
        if k not in data:
            return None, (jsonify(error=f"missing '{k}'"), 400)
    try:
        shape = (int(data["h"]), int(data["w"]))
    except (TypeError, ValueError):
        return None, (jsonify(error="w/h must be integers"), 400)
    return (shape, data), None


# ------------------------------- routes ------------------------------------
@app.route("/")
def index():
    return Response(HTML_PATH.read_text(encoding="utf-8"), mimetype="text/html")


@app.route("/api/config")
def api_config():
    return jsonify(layers=CFG["layers"], nlayers=len(CFG["layers"]), title=CFG.get("title", "Mask editor"))


@app.route("/api/pids")
def api_pids():
    pids = CFG["source"].pids()
    edited = [(CFG["out_mask_dir"] / f"{CFG['source'].stem(p)}.npy").exists() for p in pids]
    return jsonify(pids=pids, edited=edited, count=len(pids), done=sum(edited))


@app.route("/api/slice/<pid>")
def api_slice(pid):
    if not _known(pid):
        return jsonify(error=f"unknown id '{pid}'"), 404
    chest = load_chest(pid)
    orig = request.args.get("orig") == "1"
    mask = load_mask(pid, chest.shape, orig=orig)
    nodule = load_nodule_mask(pid, chest.shape, orig=orig) if CFG["nodules"] else np.zeros(chest.shape, np.uint8)
    edited = (CFG["out_mask_dir"] / f"{CFG['source'].stem(pid)}.npy").exists()
    return jsonify(
        w=chest.shape[1], h=chest.shape[0],
        chest=gray_png_b64(chest), mask=mask_png_b64(mask),
        nodule=mask_png_b64(nodule), edited=edited,
    )


@app.route("/api/clean/<op>", methods=["POST"])
def api_clean(op):
    if ndimage is None:
        return jsonify(error="scipy not installed"), 400
    parsed, err = _mask_request()
    if err:
        return err
    shape, data = parsed
    m = decode_mask_png(data["mask"], shape)
    if op == "fill":
        m = ndimage.binary_fill_holes(m).astype(np.uint8)
    elif op == "islands":
        lbl, n = ndimage.label(m)
        if n:
            sizes = ndimage.sum(np.ones_like(m), lbl, range(1, n + 1))
            keep = np.argsort(sizes)[::-1][:2] + 1  # keep two largest (both lungs)
            m = np.isin(lbl, keep).astype(np.uint8)
    else:
        return jsonify(error=f"unknown op '{op}'"), 400
    return jsonify(mask=mask_png_b64(m))


@app.route("/api/save/<pid>", methods=["POST"])
def api_save(pid):
    if not _known(pid):
        return jsonify(error=f"unknown id '{pid}'"), 404
    parsed, err = _mask_request()
    if err:
        return err
    shape, data = parsed
    two = len(CFG["layers"]) == 2
    m = decode_mask_png(data["mask"], shape)                                     # layer 0
    nodule = decode_mask_png(data["nodule"], shape) if two and data.get("nodule") else np.zeros(shape, np.uint8)
    stem = CFG["source"].stem(pid)  # series-keyed (LUNA) or the id itself (generic)

    CFG["out_mask_dir"].mkdir(parents=True, exist_ok=True)
    np.save(CFG["out_mask_dir"] / f"{stem}.npy", m.astype(np.uint8))
    if two:
        CFG["out_nodule_dir"].mkdir(parents=True, exist_ok=True)
        np.save(CFG["out_nodule_dir"] / f"{stem}.npy", nodule.astype(np.uint8))
    if CFG.get("save_masked", True):
        CFG["out_masked_dir"].mkdir(parents=True, exist_ok=True)
        chest = CFG["source"].chest(pid).astype(np.float32)
        effective = ((m > 0) | (nodule > 0)).astype(np.uint8)  # image * (union of layers)
        np.save(CFG["out_masked_dir"] / f"{stem}.npy", (chest * effective).astype(np.float32))
    return jsonify(ok=True, mask_path=str(CFG["out_mask_dir"] / f"{stem}.npy"))
