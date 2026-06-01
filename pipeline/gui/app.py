"""
PyQt6 GUI for running NN predictions on a folder of images.

Launch with:
    python -m pipeline.gui.app
"""

import sys
import csv
import traceback
from pathlib import Path

import cv2
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QPushButton, QCheckBox,
    QSpinBox, QProgressBar, QTableWidget,
    QTableWidgetItem, QHeaderView,
    QFileDialog, QMessageBox,
)

from pipeline.preprocessing.images import collect_images
from pipeline.io.discovery import parse_image_filename, parse_subject_session_from_path
from pipeline.segmentation.predictor import Predictor

WEIGHTS_DIR = Path(__file__).resolve().parent.parent.parent / "weights"


class PredictionWorker(QThread):
    progress     = pyqtSignal(int, int)
    result_ready = pyqtSignal(str, int, int)
    error        = pyqtSignal(str)
    done         = pyqtSignal()

    def __init__(self, predictor, image_paths, mark, marker_radius, out_dir):
        super().__init__()
        self.predictor     = predictor
        self.image_paths   = image_paths
        self.mark          = mark
        self.marker_radius = marker_radius
        self.out_dir       = out_dir

    def run(self):
        total = len(self.image_paths)
        for i, p in enumerate(self.image_paths):
            try:
                img = cv2.imread(str(p))
                if img is None:
                    self.error.emit(f"Cannot read: {p.name}")
                    self.progress.emit(i + 1, total)
                    continue

                x, y = self.predictor.predict(img)
                self.result_ready.emit(p.name, x, y)

                if self.mark and self.out_dir is not None:
                    h, w = img.shape[:2]
                    px, py = min(max(x, 0), w - 1), min(max(y, 0), h - 1)
                    r = self.marker_radius
                    if r == 0:
                        img[py, px] = [0, 0, 255]
                    else:
                        cv2.drawMarker(img, (px, py), (0, 0, 255),
                                       cv2.MARKER_CROSS, 2 * r + 1, 1,
                                       cv2.LINE_AA)
                    cv2.imwrite(str(self.out_dir / p.name), img)

            except Exception as e:
                self.error.emit(f"{p.name}: {e}")
            self.progress.emit(i + 1, total)
        self.done.emit()


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Propriohand — NN Prediction GUI")
        self.setMinimumSize(780, 560)
        self.predictor = None
        self.folder = None
        self.results = []
        self._build_ui()
        self._load_models()

    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central)

        grp = QGroupBox("Image folder")
        h = QHBoxLayout()
        self.lbl_folder = QLabel("No folder selected")
        btn = QPushButton("Browse…"); btn.clicked.connect(self._browse)
        h.addWidget(self.lbl_folder, 1); h.addWidget(btn)
        grp.setLayout(h); root.addWidget(grp)

        grp2 = QGroupBox("Options"); g = QGridLayout()
        self.chk_mark = QCheckBox("Save annotated images")
        g.addWidget(self.chk_mark, 0, 0, 1, 2)
        g.addWidget(QLabel("Marker radius (0 = 1px):"), 1, 0)
        self.spn_radius = QSpinBox(); self.spn_radius.setRange(0, 30)
        self.spn_radius.setSuffix(" px"); g.addWidget(self.spn_radius, 1, 1)
        grp2.setLayout(g); root.addWidget(grp2)

        self.btn_run = QPushButton("▶  Run predictions")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self._run); root.addWidget(self.btn_run)

        self.progress = QProgressBar(); root.addWidget(self.progress)
        self.lbl_status = QLabel(""); root.addWidget(self.lbl_status)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Image", "X", "Y"])
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        root.addWidget(self.table, 1)

        hx = QHBoxLayout()
        self.btn_csv = QPushButton("Export CSV")
        self.btn_csv.clicked.connect(self._export_csv); self.btn_csv.setEnabled(False)
        hx.addWidget(self.btn_csv); root.addLayout(hx)

    def _load_models(self):
        coarse = WEIGHTS_DIR / "hrnet_weights.pth"
        refine = WEIGHTS_DIR / "refine_weights.pth"
        if not coarse.exists() or not refine.exists():
            QMessageBox.critical(self, "Weights not found",
                                 f"Expected in: {WEIGHTS_DIR}")
            return
        try:
            self.predictor = Predictor(str(coarse), str(refine))
            self.lbl_status.setText(f"✓ Models loaded on {self.predictor.device}")
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Select image folder")
        if not d:
            return
        self.folder = Path(d)
        imgs = collect_images(self.folder)
        self.lbl_folder.setText(f"{d}  ({len(imgs)} images)")
        self.btn_run.setEnabled(len(imgs) > 0 and self.predictor is not None)

    def _run(self):
        if not self.folder:
            return
        paths = collect_images(self.folder)
        if not paths:
            return
        self.results.clear(); self.table.setRowCount(0)
        self.btn_run.setEnabled(False); self.btn_csv.setEnabled(False)
        self.progress.setMaximum(len(paths)); self.progress.setValue(0)
        out = (self.folder / "annotated") if self.chk_mark.isChecked() else None
        if out:
            out.mkdir(exist_ok=True)
        self.worker = PredictionWorker(self.predictor, paths,
                                       self.chk_mark.isChecked(),
                                       self.spn_radius.value(), out)
        self.worker.progress.connect(lambda c, t: (
            self.progress.setValue(c),
            self.lbl_status.setText(f"{c}/{t}"),
        ))
        self.worker.result_ready.connect(self._on_result)
        self.worker.done.connect(self._on_done)
        self.worker.start()

    def _on_result(self, name, x, y):
        self.results.append((name, x, y))
        r = self.table.rowCount(); self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(name))
        ix = QTableWidgetItem(str(x)); ix.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        iy = QTableWidgetItem(str(y)); iy.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(r, 1, ix); self.table.setItem(r, 2, iy)

    def _on_done(self):
        self.btn_run.setEnabled(True)
        self.btn_csv.setEnabled(len(self.results) > 0)
        self.lbl_status.setText(f"✓ {len(self.results)} image(s) processed")
        self._auto_export()

    def _auto_export(self):
        if not self.results or not self.folder:
            return
        subject, session = parse_subject_session_from_path(self.folder)
        out_dir = self.folder
        for parent in self.folder.parents:
            if parent.name == "data":
                out_dir = parent / "nn_outputs"; break
        out_dir.mkdir(exist_ok=True)
        path = out_dir / f"{subject}_{session}_predictions.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["subject", "session", "trial", "finger", "zone",
                        "x_est", "y_est", "filename"])
            for name, x, y in self.results:
                _, trial, finger, zone = parse_image_filename(name)
                w.writerow([subject, session, trial, finger, zone, x, y, name])

    def _export_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save CSV",
                                              "predictions.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["subject", "session", "trial", "finger", "zone",
                        "x_est", "y_est", "filename"])
            for name, x, y in self.results:
                s, t, fg, z = parse_image_filename(name)
                w.writerow([s, "1", t, fg, z, x, y, name])
        QMessageBox.information(self, "CSV", f"Saved: {path}")


def main():
    app = QApplication(sys.argv); app.setStyle("Fusion")
    win = MainWindow(); win.show(); sys.exit(app.exec())


if __name__ == "__main__":
    main()