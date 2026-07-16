# Lung Mask Editor

![alt text](image.png)

A small local web app to hand-edit segmentation masks with the mouse. It runs entirely on
your own machine and offline: a Flask server serves a canvas you paint on in the browser.

Give it a folder of 2D images and it lets you paint up to two mask layers over them,
saving your edits as `.npy` files you can feed straight back into a pipeline.

## Requirements

- **Python 3.9+**
- `pip install flask numpy pillow scipy`
  - `scipy` is optional — without it, *Fill holes* and *Keep 2 largest* are disabled.
- `pip install crossfiledialog` — powers the splash screen's native folder/file browse
  buttons. Without it, Browse just errors; you can still type paths in by hand.
- *(Optional)* two assisted-segmentation models — see [Assisted segmentation](#assisted-segmentation-optional).

## Data layout

Everything is plain NumPy:

- **Images**: a folder of `<id>.npy`, each a **2D** array (grayscale). Any value range —
  it's min–max normalized for display.
- **Masks** *(optional)*: folders of `<id>.npy` with matching ids, non-zero = mask.
  Missing files just start as an empty layer you paint from scratch.
- **Nodule centroids** *(optional, alternative to a nodule mask folder)*: a CSV with an
  id column plus two coordinate columns (`image,x,y` by default, or any names — sniffed
  automatically, overridable). Each row seeds a 32×32px square around that centroid as a
  rough starting mask, for when you have nodule locations but not full masks yet.

The `<id>` is simply the filename, and it's what your saved edits are named after.

There's also a **LUNA25 dataset preset**: point it at a dataset root laid out as
`images/chest`, `masks/lung_parenchyma`, `images/lung_nodule_crop` +
`metadata/lung_nodule_metadata.csv`, and it reconstructs the lung + nodule layers from
that dataset's own metadata — see `--dataset`/`--series` below.

## Run

```bash
# no data configured yet — pick a folder from the splash screen once it loads:
python mask_editor.py

# try it with synthetic data, no setup:
python mask_editor.py --demo

# your own images, one editable layer:
python mask_editor.py --images imgs/

# images + an existing mask to correct:
python mask_editor.py --images imgs/ --masks masks/

# two layers, with your own names:
python mask_editor.py --images imgs/ --masks lungs/ --masks2 nodules/ --names lung,nodule

# nodule layer seeded from centroids instead of a mask folder:
python mask_editor.py --images imgs/ --masks lungs/ --nodule-csv nodules.csv

# LUNA25 dataset preset (every series in the dataset):
python mask_editor.py --dataset /path/to/dataset

# ...restricted to a subset of series:
python mask_editor.py --dataset /path/to/dataset --series series.csv
```

Then open the printed URL — **http://localhost:8000**.

## Picking data from the splash screen

You don't need any flags at all — `python mask_editor.py` starts the server with nothing
configured, and the splash screen lets you point it at data interactively (with a native
folder/file browse dialog), no restart needed:

- **Images folder** — required for generic mode.
- **Lung masks** toggle — reveals a folder picker for layer 1.
- **Lung nodule masks** toggle — reveals a **Folder** / **Centroid CSV** dropdown, then
  the matching picker (plus an optional column-override field for the CSV).
- **LUNA25 dataset** toggle — mutually exclusive with the above; reveals a
  Dataset folder picker and an optional Series CSV picker (blank = every series found).

Whatever you pass on the command line pre-fills these fields, so you can launch with
`--images`/`--dataset` as a default and still override it in the browser before hitting
**Start**. Picking a mask folder validates every mask's shape against its image and
fails clearly (in the splash, not silently) if any don't match.

## How it works

**Two layers.** Each is an independent binary mask drawn in its own colour (layer 1 red,
layer 2 green). Pick which one the brush edits with the **Edit layer** buttons or keys
`1` / `2` — the brush ring and the image border are tinted to the active layer so you
never paint the wrong one. With one layer configured, the second is hidden entirely.

**Tools.**
- **Paint / Erase** — left-drag paints, right-drag always erases.
- **✨ SAM** — segments a structure into the active layer using whichever model is
  assigned to it (optional — see below).
- **Brush size**, **mask opacity**, and **zoom** sliders. The cursor ring always shows the
  exact area the brush will cover.
- **Undo / Redo**, **Fill holes**, **Keep 2 largest** (these act on the active layer).
- **Show** toggles per layer, and **hold `H`** to hide the overlays and peek at the raw image.

**Moving through images.** Use the dropdown, **Prev/Next**, or **Next unedited**. A progress
bar tracks how many are done. Switching images **auto-saves** the current one if you changed
anything, so you can stop and resume any time — reopen with `--skip-done` to drop everything
already saved and continue where you left off.

## Saving

Edits are **non-destructive** — your source folders are never modified. Everything is
written under `--out-dir` (default `./mask_edits/`), named by image id:

- `masks/<layer1>/<id>.npy` — layer 1 mask (uint8, 0/1)
- `masks/<layer2>/<id>.npy` — layer 2 mask (uint8, 0/1)
- `images/masked/<id>.npy` — `image × (layer1 | layer2)`, the image with everything
  outside your masks zeroed (skip it with `--no-masked`)

## Assisted segmentation (optional)

Two separate models, one per layer — install either or both; whichever's missing just
hides its tools and everything else still works.

- **Lung layer → MobileSAM**: general-purpose SAM distilled for CPU, natively
  point-prompted — **one click** segments the lung.
- **Nodule layer → LiteMedSAM**: MedSAM distilled for CPU (CVPR24 *MedSAM on Laptop*),
  trained on **box** prompts only — small, medical-tuned, but needs a box, not a click.

```bash
pip install torch torchvision timm

# MobileSAM (lung, point prompt)
git clone --depth 1 https://github.com/ChaoningZhang/MobileSAM.git
# checkpoint -> models/mobile_sam.pt, from
# https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt

# LiteMedSAM (nodule, box prompt)
git clone --depth 1 -b LiteMedSAM https://github.com/bowang-lab/MedSAM.git
# checkpoint -> models/lite_medsam.pth, from the LiteMedSAM branch's README
```

`MobileSAM/` and `MedSAM/` folders beside the script are auto-detected; use
`--mobilesam-repo`/`--mobilesam-checkpoint` and `--litemedsam-repo`/`--sam-checkpoint`
(or the `MOBILESAM_REPO`/`MOBILESAM_CHECKPOINT` and `LITEMEDSAM_REPO`/`SAM_CHECKPOINT`
env vars) if you keep them elsewhere.

> LiteMedSAM is **not** a drop-in SAM checkpoint — its TinyViT and 256px pipeline differ
> from stock SAM/MobileSAM — which is why its own model code is required, separately
> from MobileSAM's.

**Using it** — pick the layer first, then:
- **Lung**: click **✨ MobileSAM**, then click the structure. A drag also works as a box
  prompt if you want one.
- **Nodule**: click **✨ MedSAM** to arm it — the button relabels itself
  **⚡ Automatic segmentation** — then either **drag a box** around the structure, or
  click the button *again* to re-segment from whatever's already on the layer (e.g. a
  centroid-seeded 32×32 square), no drag needed.
- **✨ Refine from its box**: the same "re-segment from what's already there" action,
  available as its own button on either layer.

Either way the result **replaces whatever was in that box** as a single undo step
(`Ctrl+Z`), so you can refine with the brush afterwards. The first prompt on an image
runs the encoder (a few seconds on CPU); further prompts on the same image are fast.

### Citing the models

If you use this tool's assisted segmentation, please cite the underlying models:

**MobileSAM** ([repo](https://github.com/ChaoningZhang/MobileSAM)):
```bibtex
@article{mobile_sam,
  title={Faster Segment Anything: Towards Lightweight SAM for Mobile Applications},
  author={Zhang, Chaoning and Han, Dongshen and Qiao, Yu and Kim, Jung Uk and Bae, Sung-Ho and Lee, Seungkyu and Hong, Choong Seon},
  journal={arXiv preprint arXiv:2306.14289},
  year={2023}
}
```

**MedSAM** ([repo](https://github.com/bowang-lab/MedSAM)), whose LiteMedSAM branch this
tool's nodule model comes from — Ma, J., He, Y., Li, F., Han, L., You, C., and Wang, B.,
*"Segment Anything in Medical Images"*, Nature Communications, 2024. See the repo for
the exact citation / BibTeX.

## Options

| Option | Meaning |
|---|---|
| `--images DIR` | folder of `<id>.npy` images to annotate (generic mode) |
| `--masks DIR` | starting masks for layer 1 (optional) |
| `--masks2 DIR` | starting masks for layer 2 (optional) |
| `--names a,b` | layer display names (default `mask,label2`) |
| `--ids ...` | restrict to these ids — comma list or a file, one per line |
| `--nodule-csv CSV` | alternative to `--masks2`: nodule centroids (id+x+y), seeds a 32×32px square per row |
| `--nodule-cols id,x,y` | override the sniffed CSV column names |
| `--demo` | generate synthetic images and edit those |
| `--dataset DIR` | LUNA25 preset: dataset root (or a flat `P*-S*.npy` folder). Passing this (or `--series`) switches to LUNA mode |
| `--series ...` | LUNA preset (optional): restrict to these series — comma list, stems, or a CSV |
| `--out-dir DIR` | where edits are written (default `./mask_edits`) |
| `--skip-done` | skip ids already saved in `--out-dir` (resume) |
| `--no-masked` | don't write the `image × mask` composite |
| `--port N` | server port (default 8000) |
| `--sam-checkpoint` | path to `lite_medsam.pth` (default `models/lite_medsam.pth`) |
| `--litemedsam-repo` | clone of the MedSAM LiteMedSAM branch (default: `MedSAM/` beside the script) |
| `--mobilesam-checkpoint` | path to `mobile_sam.pt` (default `models/mobile_sam.pt`) |
| `--mobilesam-repo` | clone of MobileSAM (default: `MobileSAM/` beside the script) |

## Shortcuts

`1` / `2` layer · `P` / `E` paint / erase · `[` `]` brush size · `←` `→` image ·
`U` next unedited · `H` hold to peek · `Ctrl+Z` undo · `Ctrl+S` save · right-drag erases

## Code layout

| File | Role |
|---|---|
| `mask_editor.py` | CLI and wiring — run this |
| `server.py` | Flask routes, image ⇄ PNG helpers, native folder/file picker plumbing |
| `sources.py` | where images and masks are read from (generic folders, centroid CSVs, LUNA25 preset) |
| `segment.py` | optional assisted segmentation — MobileSAM (lung, point) + LiteMedSAM (nodule, box) |
| `pick_path.py` | native OS folder/file picker, run in its own subprocess (uses `crossfiledialog`) |
| `mask_editor.html`, `static/` | the frontend (markup, `style.css`, `app.js`) |

![alt text](image-1.png)
