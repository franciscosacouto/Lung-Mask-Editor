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

import segment as sam  # optional MobileSAM click-to-segment (lazy; safe to import)
from sources import GenericSource, build_generic_cfg, build_luna_cfg

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


PICK_PATH_SCRIPT = Path(__file__).with_name("pick_path.py")


def _pick_path(initial=None, kind="folder"):
    """Pop a native OS folder- or file-picker (kind: 'folder' or 'file', the latter for a
    series CSV) by running pick_path.py in its own subprocess (same interpreter, via
    sys.executable). Server and browser are the same machine, so this is safe to call
    from a request handler; the subprocess isolation means a picker-side misbehavior
    can't take the Flask server down with it — worst case this call errors or times out.

    pick_path.py uses crossfiledialog (pip install crossfiledialog), which calls the
    same modern IFileDialog COM API File Explorer itself uses on Windows — unlike
    System.Windows.Forms.FolderBrowserDialog, which is stuck on the legacy XP-era tree
    dialog under .NET Framework.
    """
    import subprocess
    import sys
    title = "Select CSV file" if kind == "file" else "Select folder"
    try:
        r = subprocess.run(
            [sys.executable, str(PICK_PATH_SCRIPT), "--kind", kind, "--initial", initial or "", "--title", title],
            capture_output=True, text=True, timeout=1800,
        )
    except FileNotFoundError:
        raise RuntimeError("python interpreter not found for the picker subprocess")
    except subprocess.TimeoutExpired:
        raise RuntimeError("dialog timed out")
    if r.returncode != 0 and not r.stdout.strip():
        raise RuntimeError(r.stderr.strip() or f"dialog exited with code {r.returncode}")
    return r.stdout.strip() or None


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
    src = CFG["source"]
    generic = isinstance(src, GenericSource)
    dirs = src.mask_dirs if generic else []
    sam_info = {layer: {"available": sam.available(layer), "backend": sam.backend(layer),
                        "prompt": sam.prompt_mode(layer)} for layer in ("lung", "nodule")}
    return jsonify(layers=CFG["layers"], nlayers=len(CFG["layers"]),
                   title=CFG.get("title", "Mask editor"), sam=sam_info,
                   mode="generic" if generic else "luna",
                   images=str(src.images_dir) if generic and src.images_dir.is_dir() else "",
                   masks=str(dirs[0]) if len(dirs) > 0 and dirs[0] else "",
                   masks2=str(dirs[1]) if len(dirs) > 1 and dirs[1] else "",
                   nodule_csv=CFG.get("_nodule_csv_spec", "") if generic else "",
                   nodule_cols=CFG.get("_nodule_cols_spec", "") if generic else "",
                   dataset="" if generic else CFG.get("_dataset_spec", ""),
                   series="" if generic else CFG.get("_series_spec", ""))


@app.route("/api/browse", methods=["POST"])
def api_browse():
    data = request.get_json(silent=True) or {}
    kind = data.get("kind", "folder")
    if kind not in ("folder", "file"):
        return jsonify(error="kind must be 'folder' or 'file'"), 400
    try:
        path = _pick_path(data.get("initial"), kind)
    except Exception as e:
        return jsonify(error=f"dialog unavailable: {e}"), 500
    return jsonify(path=path)


@app.route("/api/configure", methods=["POST"])
def api_configure():
    """Switch the running app to a new data source: either generic (images/masks/masks2)
    or a protocol_7/LUNA dataset (dataset/series)."""
    data = request.get_json(silent=True) or {}
    dataset = (data.get("dataset") or "").strip()
    images = (data.get("images") or "").strip()
    out_base = CFG.get("_out_base")
    spec = {}
    try:
        if dataset:
            series = (data.get("series") or "").strip() or None
            cfg = build_luna_cfg(dataset, series, out_base)
            spec = {"_dataset_spec": dataset, "_series_spec": series or ""}
        elif images:
            masks = (data.get("masks") or "").strip() or None
            masks2 = (data.get("masks2") or "").strip() or None
            nodule_csv = (data.get("nodule_csv") or "").strip() or None
            nodule_cols_raw = (data.get("nodule_cols") or "").strip()
            nodule_cols = None
            if nodule_cols_raw:
                nodule_cols = tuple(c.strip() for c in nodule_cols_raw.split(","))
                if len(nodule_cols) != 3:
                    return jsonify(error="nodule columns must be 'id,x,y'"), 400
            cfg = build_generic_cfg(images, masks, masks2, "lung,nodule", out_base,
                                     nodule_csv=nodule_csv, nodule_cols=nodule_cols)
            spec = {"_nodule_csv_spec": nodule_csv or "", "_nodule_cols_spec": nodule_cols_raw}
        else:
            return jsonify(error="provide an images folder or a dataset folder"), 400
    except ValueError as e:
        return jsonify(error=str(e)), 400
    CFG.clear()
    CFG.update(cfg)
    CFG["_out_base"] = out_base
    CFG.update(spec)
    return jsonify(ok=True, title=cfg["title"], count=len(cfg["source"].pids()))


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


def _run_sam(pid, layer, point=None, box=None):
    """Shared error handling for the SAM endpoints -> (json, status)."""
    try:
        m = sam.segment(load_chest(pid), layer, point=point, box=box, key=pid)
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 400
    except ImportError as e:
        # usually a missing dependency (torch / torchvision / timm), not the package itself
        return jsonify(error=f"SAM import failed: {e}"), 400
    except Exception as e:  # model/runtime problems shouldn't kill the session
        return jsonify(error=f"segment failed: {e}"), 500
    return jsonify(mask=mask_png_b64(m)), 200


@app.route("/api/segment/<pid>", methods=["POST"])
def api_segment(pid):
    """Prompt SAM with a click {x,y} or a box {box:[x0,y0,x1,y1]} -> mask. body.layer
    ('lung' or 'nodule') picks which model handles the prompt — see segment.py."""
    if not _known(pid):
        return jsonify(error=f"unknown id '{pid}'"), 404
    data = request.get_json(silent=True) or {}
    layer = data.get("layer") if data.get("layer") in ("lung", "nodule") else "lung"
    point = box = None
    if data.get("box") is not None:
        try:
            box = [float(v) for v in data["box"]]
            assert len(box) == 4
        except Exception:
            return jsonify(error="box must be [x0,y0,x1,y1]"), 400
    elif "x" in data and "y" in data:
        try:
            point = (float(data["x"]), float(data["y"]))
        except (TypeError, ValueError):
            return jsonify(error="need numeric x,y"), 400
    else:
        return jsonify(error="need a point {x,y} or a box [x0,y0,x1,y1]"), 400
    body, status = _run_sam(pid, layer, point=point, box=box)
    return body, status


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
