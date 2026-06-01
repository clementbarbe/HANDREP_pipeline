"""
Interactive hand annotation tool.

Changes vs original:
- file browser opens directly in ``data/raw``
- re-annotating the same image **overwrites** previous data for that image

Launch:
    python -m pipeline.tools.annotate
"""

import os
import csv
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ── Config ────────────────────────────────────────────────
LABELS = [
    "thumb_Z1",  "thumb_Z2",
    "index_Z1",  "index_Z2",
    "middle_Z1", "middle_Z2",
    "ring_Z1",   "ring_Z2",
    "little_Z1", "little_Z2",
]
N_POINTS = len(LABELS)
OUTPUT_DIR = Path("data/ref")

# Where to open the file browser
_RAW_DIR_CANDIDATES = [
    Path("data/raw"),
    Path(__file__).resolve().parent.parent.parent / "data" / "raw",
]


def _initial_directory():
    """Find the best starting directory for the file dialog."""
    for d in _RAW_DIR_CANDIDATES:
        if d.exists():
            return str(d.resolve())
    return str(Path.cwd())


# ── Metadata ──────────────────────────────────────────────

def _parse_metadata(image_path):
    """Extract (subject, session) from a BIDS-like path."""
    p = Path(image_path)
    subject, session = None, None
    for part in p.parts:
        if part.startswith("sub-"):
            subject = part.replace("sub-", "")
        elif part.startswith("ses-"):
            session = part.replace("ses-", "")
    return subject or "UNKNOWN", session or "1"


# ── Save with overwrite ──────────────────────────────────

def _save_annotations(image_path, points):
    """
    Save the 10 landmark annotations to a reference CSV.

    If the CSV already contains rows for the **same image**,
    those rows are **replaced** (overwritten).  Rows belonging
    to other images in the same subject/session are preserved.
    """
    subject, session = _parse_metadata(image_path)
    image_name = os.path.basename(image_path)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"{subject}_ses-{session}_ref.csv"

    header = ["subject", "session", "finger", "zone",
              "landmark", "x_real", "y_real", "image"]

    # ── Load existing rows (excluding current image) ──────
    existing = []
    if csv_path.is_file():
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("image") != image_name:
                    existing.append(row)

    # ── Write header + kept rows + new rows ───────────────
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        # Re-write other images
        for row in existing:
            writer.writerow([row[h] for h in header])

        # Current image
        for label, (x, y) in zip(LABELS, points):
            finger, zone = label.split("_")
            writer.writerow([
                subject, session, finger, zone,
                label, x, y, image_name,
            ])

    n_kept = len(existing)
    action = "overwritten" if csv_path.stat().st_size > 0 else "created"
    print(f"✓ Annotations {action} → {csv_path}")
    if n_kept:
        print(f"  ({n_kept} rows from other images preserved)")


# ── Interactive annotation ────────────────────────────────

def annotate_image(image_path):
    img = mpimg.imread(image_path)
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.canvas.manager.set_window_title(
        f"Annotate: {os.path.basename(image_path)}"
    )
    ax.imshow(img)
    ax.set_title(
        f"Click {N_POINTS} points  |  Scroll = zoom  |  "
        f"SPACE = save  |  R = reset",
        fontsize=11,
    )

    points = []
    plots  = []

    # ── Zoom ──────────────────────────────────────────────
    def on_scroll(event):
        if event.inaxes != ax:
            return
        s = 1 / 1.2 if event.button == "up" else 1.2
        xm, xx = ax.get_xlim(); ym, yx = ax.get_ylim()
        ax.set_xlim([event.xdata - (event.xdata - xm) * s,
                     event.xdata + (xx - event.xdata) * s])
        ax.set_ylim([event.ydata - (event.ydata - ym) * s,
                     event.ydata + (yx - event.ydata) * s])
        fig.canvas.draw_idle()

    # ── Click ─────────────────────────────────────────────
    def on_click(event):
        if event.button != 1 or event.inaxes != ax:
            return
        if len(points) >= N_POINTS:
            print(f"⚠ {N_POINTS} points already placed")
            return
        x, y = int(event.xdata), int(event.ydata)
        points.append((x, y))
        label = LABELS[len(points) - 1]
        p = ax.scatter(x, y, c="red", s=40)
        ax.text(x, y, label, color="red", fontsize=8)
        plots.append(p)
        print(f"  [{len(points):2d}/{N_POINTS}] {label} → ({x}, {y})")
        fig.canvas.draw()

    # ── Keyboard ──────────────────────────────────────────
    def on_key(event):
        if event.key == " ":
            if len(points) != N_POINTS:
                print(f"⚠ {len(points)}/{N_POINTS} points — need exactly {N_POINTS}")
                return
            _save_annotations(image_path, points)
            plt.close(fig)

        elif event.key == "r":
            points.clear()
            for p in plots:
                p.remove()
            plots.clear()
            ax.images.clear()
            ax.imshow(img)
            fig.canvas.draw()
            print("🔄 Reset — click again")

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


# ── Main ──────────────────────────────────────────────────

def main():
    root = tk.Tk()
    root.withdraw()

    path = filedialog.askopenfilename(
        title="Select a hand reference image",
        initialdir=_initial_directory(),
        filetypes=[("Images", "*.jpg *.jpeg *.png")],
    )

    if not path:
        print("❌ No image selected")
        return

    print(f"Image: {path}")
    annotate_image(path)


if __name__ == "__main__":
    main()