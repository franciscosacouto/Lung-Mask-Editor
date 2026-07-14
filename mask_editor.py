"""
Local browser app to paint / hand-edit 2D masks on .npy images with the mouse.
Flask serves the page (frontend in mask_editor.html + static/); offline, no SSH/X11.

    pip install flask numpy pillow scipy

Generic mode — any folder of <id>.npy images, with 0-2 mask layers:
    python mask_editor.py --images imgs/ --masks masks/ [--masks2 masks2/] [--names cell,nucleus]
    python mask_editor.py --demo          # synthetic data, zero setup

LUNA25 preset — supply the series (patient+last5); layers are lung + nodule:
    python mask_editor.py --series series.csv [--dataset .../protocol_7]

Edits save non-destructively under --out-dir (default ./mask_edits), one <stem>.npy
per layer plus an image×mask composite:
    masks/<layer0>/<stem>.npy, masks/<layer1>/<stem>.npy, images/masked/<stem>.npy
(LUNA preset mirrors protocol_7: masks/lung_parenchyma, masks/lung_nodule, images/lung_masked).

Code layout: sources.py (data), server.py (Flask routes), this file (CLI/wiring).
"""
import argparse
import csv
import io
import re
from pathlib import Path

from server import CFG, app
from sources import GenericSource, StagingSource, make_demo

# LUNA preset default: look next to this script (put your images folder beside tools/).
DEFAULT_DATASET = str(Path(__file__).resolve().parent)


def parse_series(spec):
    """Return series stems 'P<patient>-S<last5>' from --series.

    --series is a file or an inline string. A file may be a CSV with 'patient' and
    'last5' columns, or one entry per line. An entry is either a stem (P100012-S22196)
    or a 'patient,last5' pair (also accepts space/dash). Inline, separate pairs with ';'
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


def parse_ids(spec):
    """--ids: comma list or a file with one id per line."""
    p = Path(spec)
    if p.exists():
        return [ln.strip() for ln in p.read_text().splitlines()
                if ln.strip() and not ln.lstrip().startswith("#")]
    return [x.strip() for x in spec.replace("\n", ",").split(",") if x.strip()]


def _slug(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s).strip("_") or "layer"


def build_cli():
    ap = argparse.ArgumentParser(
        description="Local browser mask editor — paint masks on 2D .npy images.")
    # ---- generic mode ----
    ap.add_argument("--images", help="folder of <id>.npy images to annotate (generic mode)")
    ap.add_argument("--masks", help="folder of <id>.npy masks for layer 1 (optional)")
    ap.add_argument("--masks2", help="folder of <id>.npy masks for layer 2 (optional)")
    ap.add_argument("--names", default="mask,label2", help="layer display names, comma-separated")
    ap.add_argument("--ids", help="restrict to these ids: comma list or a file (one id/line)")
    ap.add_argument("--demo", action="store_true", help="generate synthetic data and edit that")
    # ---- LUNA25 preset ----
    ap.add_argument("--dataset", default=DEFAULT_DATASET,
                    help="LUNA preset: path to the images folder — either a protocol_7 root "
                    "(with images/chest) or a flat folder of P*-S*.npy. Default: next to this "
                    "script. Without --series, every image found is edited.")
    ap.add_argument("--series", help="LUNA preset (optional): restrict to these series "
                    "(patient,last5 / stems / CSV); default is all images in the folder")
    # ---- common ----
    ap.add_argument("--out-dir", help="where edits are written (default: ./mask_edits)")
    ap.add_argument("--skip-done", action="store_true", help="skip ids already saved in --out-dir (resume)")
    ap.add_argument("--no-masked", action="store_true", help="do not write the image×mask composite")
    ap.add_argument("--port", type=int, default=8000)
    return ap


RED, GREEN = [255, 85, 85], [0, 230, 90]


def configure(args):
    """Build the data source + output layout into CFG. Returns (source, label)."""
    out = Path(args.out_dir or "mask_edits")

    if args.demo:
        demo = make_demo(Path(args.out_dir or ".") / "demo_data")
        args.images, args.masks, args.masks2 = str(demo / "images"), str(demo / "masks"), None
        print(f"demo data written to {demo}")

    if args.images:  # -------- generic mode --------
        names = [n.strip() for n in args.names.split(",") if n.strip()] or ["mask"]
        two = bool(args.masks2) or len(names) >= 2
        names = (names + ["label2"])[:2] if two else names[:1]
        layers = [{"name": names[0], "color": RED}] + ([{"name": names[1], "color": GREEN}] if two else [])
        source = GenericSource(args.images, [args.masks, args.masks2][:len(layers)],
                               parse_ids(args.ids) if args.ids else None)
        if not source.images_dir.is_dir():
            raise SystemExit(f"images dir not found: {source.images_dir}")
        CFG.update(
            source=source, layers=layers, nodules=two, save_masked=not args.no_masked, title="Mask editor",
            out_mask_dir=out / "masks" / _slug(layers[0]["name"]),
            out_nodule_dir=out / "masks" / (_slug(layers[1]["name"]) if two else "layer2"),
            out_masked_dir=out / "images" / "masked",
        )
        return source, "generic"

    # -------- LUNA25 preset --------
    # --series is optional: without it, edit every P*-S*.npy in <dataset>/images/chest.
    series = parse_series(args.series) if args.series else None
    if args.series and not series:
        raise SystemExit("no series parsed from --series")
    source = StagingSource(args.dataset, series)
    if not source.chest_dir.is_dir():
        raise SystemExit(f"chest folder not found: {source.chest_dir}\n"
                         f"point --dataset at a protocol_7 root (it must contain images/chest)")
    CFG.update(
        source=source, nodules=True, save_masked=not args.no_masked, title="LUNA lung-mask editor",
        layers=[{"name": "lung", "color": RED}, {"name": "nodule", "color": GREEN}],
        out_mask_dir=out / "masks" / "lung_parenchyma",
        out_nodule_dir=out / "masks" / "lung_nodule",
        out_masked_dir=out / "images" / "lung_masked",
    )
    return source, "LUNA (protocol_7)"


def main():
    args = build_cli().parse_args()
    source, label = configure(args)

    if args.skip_done:
        before = len(source._ids)
        source._ids = [s for s in source._ids if not (CFG["out_mask_dir"] / f"{source.stem(s)}.npy").exists()]
        print(f"--skip-done: {before - len(source._ids)} already done, {len(source._ids)} remaining")

    if not source.pids():
        raise SystemExit("nothing to edit — check inputs" + (" (all done?)" if args.skip_done else ""))
    print(f"mode: {label}   layers: {[l['name'] for l in CFG['layers']]}")
    print(f"editable: {len(source.pids())}"
          + (f"  ({len(source.missing)} not found: {source.missing[:5]}…)" if source.missing else ""))
    print(f"edits -> {Path(args.out_dir or 'mask_edits')}\nopen http://localhost:{args.port}")
    app.run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
