"""
Interactive hand annotation tool.

Launch with:
    python -m pipeline.tools.annotate
"""

import os
import csv
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

LABELS = [
    "thumb_Z1", "thumb_Z2", "index_Z1", "index_Z2",
    "middle_Z1", "middle_Z2", "ring_Z1", "ring_Z2",
    "little_Z1", "little_Z2",
]
N_POINTS = len(LABELS)
OUTPUT_DIR = "data/ref"


def _parse_metadata(image_path):
    p = Path(image_path)
    subject, session = None, None
    for part in p.parts:
        if part.startswith("sub-"):
            subject = part.replace("sub-", "")
        elif part.startswith("ses-"):
            session = part.replace("ses-", "")
    return subject or "UNKNOWN", session or "1"


def annotate_image(image_path):
    """Open a matplotlib window for annotating 10 hand landmarks."""
    img = mpimg.imread(image_path)
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.canvas.manager.set_window_title(f"Annotate: {os.path.basename(image_path)}")
    ax.imshow(img)
    ax.set_title(f"Click {N_POINTS} points | Scroll=zoom | SPACE=save | R=reset")

    points, plots = [], []

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

    def on_click(event):
        if event.button != 1 or event.inaxes != ax or len(points) >= N_POINTS:
            return
        x, y = int(event.xdata), int(event.ydata)
        points.append((x, y))
        label = LABELS[len(points) - 1]
        p = ax.scatter(x, y, c="red", s=40)
        ax.text(x, y, label, color="red", fontsize=8)
        plots.append(p); fig.canvas.draw()

    def on_key(event):
        if event.key == " ":
            if len(points) != N_POINTS:
                print(f"Need {N_POINTS} points, got {len(points)}")
                return
            subject, session = _parse_metadata(image_path)
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            fpath = os.path.join(OUTPUT_DIR, f"{subject}_ses-{session}_ref.csv")
            exists = os.path.isfile(fpath)
            with open(fpath, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if not exists:
                    w.writerow(["subject", "session", "finger", "zone",
                                "landmark", "x_real", "y_real", "image"])
                for lab, (x, y) in zip(LABELS, points):
                    finger, zone = lab.split("_")
                    w.writerow([subject, session, finger, zone, lab,
                                x, y, os.path.basename(image_path)])
            print(f"✓ Saved → {fpath}")
            plt.close(fig)
        elif event.key == "r":
            points.clear()
            for p in plots:
                p.remove()
            plots.clear(); ax.images.clear(); ax.imshow(img)
            fig.canvas.draw(); print("Reset")

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


def main():
    root = tk.Tk(); root.withdraw()
    path = filedialog.askopenfilename(
        title="Select image", filetypes=[("Images", "*.jpg *.jpeg *.png")],
    )
    if not path:
        print("No image selected"); return
    annotate_image(path)


if __name__ == "__main__":
    main()