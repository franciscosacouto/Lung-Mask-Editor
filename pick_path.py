"""Native OS folder/file picker, run in its own subprocess by server.py's _pick_path().
Isolated from the Flask server process: if the picker misbehaves, it can't take the
whole app down with it — worst case this subprocess errors or times out.

Uses crossfiledialog (pip install crossfiledialog), which calls the same modern
IFileDialog COM API File Explorer itself uses on Windows (and the native pickers on
macOS/Linux) — unlike System.Windows.Forms.FolderBrowserDialog, which is stuck on the
legacy XP-era tree dialog under .NET Framework.
"""
import argparse

import crossfiledialog

CSV_FILTER = {"CSV files": "*.csv", "All files": "*.*"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["folder", "file"], default="folder")
    ap.add_argument("--initial", default="")
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    if args.kind == "file":
        path = crossfiledialog.open_file(title=args.title or "Select CSV file",
                                          start_dir=args.initial or None, filter=CSV_FILTER)
    else:
        path = crossfiledialog.choose_folder(title=args.title or "Select folder",
                                              start_dir=args.initial or None)
    if path:
        print(path)


if __name__ == "__main__":
    main()
