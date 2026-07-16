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
from pathlib import Path

import segment
from server import CFG, app
from sources import build_empty_cfg, build_generic_cfg, build_luna_cfg, make_demo

# LUNA preset default: look next to this script (put your images folder beside tools/).
DEFAULT_DATASET = str(Path(__file__).resolve().parent)


def parse_ids(spec):
    """--ids: comma list or a file with one id per line."""
    p = Path(spec)
    if p.exists():
        return [ln.strip() for ln in p.read_text().splitlines()
                if ln.strip() and not ln.lstrip().startswith("#")]
    return [x.strip() for x in spec.replace("\n", ",").split(",") if x.strip()]


def build_cli():
    ap = argparse.ArgumentParser(
        description="Local browser mask editor — paint masks on 2D .npy images.")
    # ---- generic mode ----
    ap.add_argument("--images", help="folder of <id>.npy images to annotate (generic mode)")
    ap.add_argument("--masks", help="folder of <id>.npy masks for layer 1 (optional)")
    ap.add_argument("--masks2", help="folder of <id>.npy masks for layer 2 (optional)")
    ap.add_argument("--names", default="mask,label2", help="layer display names, comma-separated")
    ap.add_argument("--ids", help="restrict to these ids: comma list or a file (one id/line)")
    ap.add_argument("--nodule-csv", help="alternative to --masks2: a CSV of nodule centroids "
                    "(id + x + y columns, sniffed by name or overridden with --nodule-cols) "
                    "seeding a 32x32px square per centroid, refine by hand afterward")
    ap.add_argument("--nodule-cols", help="override the sniffed column names as 'id,x,y'")
    ap.add_argument("--demo", action="store_true", help="generate synthetic data and edit that")
    # ---- LUNA25 preset ----
    ap.add_argument("--dataset", default=None,
                    help="LUNA preset: path to the images folder — either a protocol_7 root "
                    "(with images/chest) or a flat folder of P*-S*.npy. Passing this (or "
                    "--series) switches to LUNA mode; default (next to this script) is used "
                    "if only --series is given. Without --series, every image found is edited.")
    ap.add_argument("--series", help="LUNA preset (optional): restrict to these series "
                    "(patient,last5 / stems / CSV); default is all images in the folder")
    # ---- common ----
    ap.add_argument("--out-dir", help="where edits are written (default: ./mask_edits)")
    ap.add_argument("--skip-done", action="store_true", help="skip ids already saved in --out-dir (resume)")
    ap.add_argument("--no-masked", action="store_true", help="do not write the image×mask composite")
    ap.add_argument("--port", type=int, default=8000)
    # ---- optional LiteMedSAM box/click segmentation ----
    ap.add_argument("--sam-checkpoint", help="lite_medsam.pth (default: models/lite_medsam.pth)")
    ap.add_argument("--litemedsam-repo", help="clone of the MedSAM LiteMedSAM branch "
                    "(default: a MedSAM/ folder next to the script)")
    # ---- optional MobileSAM point/box segmentation (lung layer) ----
    ap.add_argument("--mobilesam-checkpoint", help="mobile_sam.pt (default: models/mobile_sam.pt)")
    ap.add_argument("--mobilesam-repo", help="clone of MobileSAM "
                    "(default: a MobileSAM/ folder next to the script)")
    return ap


def configure(args):
    """Build the data source + output layout into CFG. Returns (source, label)."""
    out = Path(args.out_dir or "mask_edits")
    CFG["_out_base"] = str(out)  # remembered so /api/configure can reuse it later

    if args.demo:
        demo = make_demo(Path(args.out_dir or ".") / "demo_data")
        args.images, args.masks, args.masks2 = str(demo / "images"), str(demo / "masks"), None
        print(f"demo data written to {demo}")

    if not args.images and args.dataset is None and not args.series:
        # -------- nothing requested: start empty in generic mode --------
        # (LUNA mode only kicks in if the user explicitly asked for it via --dataset/--series;
        # bare `mask_editor.py` should land on the splash screen's generic fields, not LUNA.)
        CFG.update(build_empty_cfg(out))
        return CFG["source"], "generic (unconfigured)"

    if args.images:  # -------- generic mode --------
        CFG["_nodule_csv_spec"], CFG["_nodule_cols_spec"] = args.nodule_csv or "", args.nodule_cols or ""
        cols = tuple(c.strip() for c in args.nodule_cols.split(",")) if args.nodule_cols else None
        try:
            cfg = build_generic_cfg(args.images, args.masks, args.masks2, args.names, out,
                                     parse_ids(args.ids) if args.ids else None, args.no_masked,
                                     args.nodule_csv, cols)
        except ValueError as e:
            raise SystemExit(str(e))
        CFG.update(cfg)
        return cfg["source"], "generic"

    # -------- LUNA25 preset --------
    # --series is optional: without it, edit every P*-S*.npy in <dataset>/images/chest.
    dataset = args.dataset or DEFAULT_DATASET
    CFG["_dataset_spec"], CFG["_series_spec"] = dataset, args.series or ""
    try:
        cfg = build_luna_cfg(dataset, args.series, out, args.no_masked)
    except ValueError as e:
        raise SystemExit(str(e))
    CFG.update(cfg)
    return cfg["source"], "LUNA (protocol_7)"


def main():
    args = build_cli().parse_args()
    segment.configure(checkpoint=args.sam_checkpoint, repo=args.litemedsam_repo,
                       mobilesam_checkpoint=args.mobilesam_checkpoint, mobilesam_repo=args.mobilesam_repo)
    source, label = configure(args)

    if args.skip_done:
        before = len(source._ids)
        source._ids = [s for s in source._ids if not (CFG["out_mask_dir"] / f"{source.stem(s)}.npy").exists()]
        print(f"--skip-done: {before - len(source._ids)} already done, {len(source._ids)} remaining")

    print(f"mode: {label}   layers: {[l['name'] for l in CFG['layers']]}")
    if not source.pids():
        print("nothing to edit for the given inputs" + (" (all done?)" if args.skip_done else "")
              + " — pick a folder from the splash screen once the page loads")
    else:
        print(f"editable: {len(source.pids())}"
              + (f"  ({len(source.missing)} not found: {source.missing[:5]}…)" if source.missing else ""))
    print(f"edits -> {Path(args.out_dir or 'mask_edits')}\nopen http://localhost:{args.port}")
    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
