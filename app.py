"""
Image Inspect

Pick a folder, type a description, and see the images CLIP thinks match it
best. Click a thumbnail to enlarge it.
"""

import os
import sys

from PIL import Image
from PyQt6.QtCore import QSettings, QSize, Qt, QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (QApplication, QCheckBox, QDialog, QFileDialog,
                             QHBoxLayout, QLabel, QLineEdit, QListWidget,
                             QListWidgetItem, QMainWindow, QMessageBox,
                             QProgressBar, QPushButton, QSpinBox, QVBoxLayout,
                             QWidget)

import clip_search
from image_io import open_image


class SearchWorker(QThread):
    """Downloads models if needed, indexes the folder, then ranks it."""

    progress = pyqtSignal(str, int, int)
    finished_ok = pyqtSignal(list)      # [(path, score, thumbnail bytes)]
    failed = pyqtSignal(str)

    def __init__(self, state, folder, recursive, prompt, limit):
        super().__init__()
        self.state = state              # dict holding the Clip and Index
        self.folder, self.recursive = folder, recursive
        self.prompt, self.limit = prompt, limit

    def run(self):
        try:
            if "clip" not in self.state:
                clip_search.ensure_models(self.progress.emit)
                self.progress.emit("Loading CLIP model", 0, 0)
                self.state["clip"] = clip_search.Clip()
                self.state["index"] = clip_search.Index()
            clip, index = self.state["clip"], self.state["index"]

            self.progress.emit("Scanning folder", 0, 0)
            paths = clip_search.find_images(self.folder, self.recursive)
            index.build(clip, paths, self.progress.emit,
                        self.isInterruptionRequested)

            results = index.search(clip.embed_text(self.prompt), self.limit)
            self.finished_ok.emit([(path, score, index.thumbnail(path))
                                   for path, score in results])
        except Exception as e:                               # noqa: BLE001
            self.failed.emit(str(e))


class ImageDialog(QDialog):
    """One image, scaled to fit the screen."""

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.setWindowTitle(path)
        self.path = path

        screen = self.screen().availableGeometry()
        max_size = (int(screen.width() * 0.9), int(screen.height() * 0.85))
        with open_image(path, draft_size=max_size) as img:
            img = img.convert("RGB")
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            data = img.tobytes()
            qimage = QImage(data, img.width, img.height, 3 * img.width,
                            QImage.Format.Format_RGB888).copy()

        label = QLabel()
        label.setPixmap(QPixmap.fromImage(qimage))
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        open_button = QPushButton("Open in default viewer")
        open_button.clicked.connect(lambda: QDesktopServices.openUrl(
            QUrl.fromLocalFile(self.path)))
        folder_button = QPushButton("Show in folder")
        folder_button.clicked.connect(self.show_in_folder)

        buttons = QHBoxLayout()
        buttons.addWidget(QLabel(path), 1)
        buttons.addWidget(folder_button)
        buttons.addWidget(open_button)

        layout = QVBoxLayout(self)
        layout.addWidget(label, 1)
        layout.addLayout(buttons)

    def show_in_folder(self):
        if sys.platform == "win32":
            import subprocess
            subprocess.Popen(["explorer", "/select,", os.path.normpath(self.path)])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(self.path)))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Inspect")
        self.resize(1200, 800)
        self.settings = QSettings("ImageInspect", "ImageInspect")
        self.state = {}
        self.worker = None

        self.folder_edit = QLineEdit(self.settings.value("folder", ""))
        self.folder_edit.setPlaceholderText("Folder of images")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self.choose_folder)
        self.recursive = QCheckBox("Include subfolders")
        self.recursive.setChecked(self.settings.value("recursive", True, bool))

        self.prompt_edit = QLineEdit()
        self.prompt_edit.setPlaceholderText("Describe what you are looking for, "
                                            "e.g. \"a dog on the beach\"")
        self.prompt_edit.returnPressed.connect(self.search)
        self.limit = QSpinBox()
        self.limit.setRange(1, 1000)
        self.limit.setValue(100)
        self.limit.setPrefix("Top ")
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.search)

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setMaximumWidth(300)
        self.status = QLabel("Choose a folder and enter a description.")

        self.results = QListWidget()
        self.results.setViewMode(QListWidget.ViewMode.IconMode)
        self.results.setIconSize(QSize(clip_search.THUMB_SIZE * 3 // 4,
                                       clip_search.THUMB_SIZE * 3 // 4))
        self.results.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.results.setMovement(QListWidget.Movement.Static)
        self.results.setUniformItemSizes(True)
        self.results.setSpacing(6)
        self.results.itemClicked.connect(self.enlarge)

        row1 = QHBoxLayout()
        row1.addWidget(self.folder_edit, 1)
        row1.addWidget(browse)
        row1.addWidget(self.recursive)
        row2 = QHBoxLayout()
        row2.addWidget(self.prompt_edit, 1)
        row2.addWidget(self.limit)
        row2.addWidget(self.search_button)
        row3 = QHBoxLayout()
        row3.addWidget(self.status, 1)
        row3.addWidget(self.progress)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addWidget(self.results, 1)
        layout.addLayout(row3)
        self.setCentralWidget(central)
        self.prompt_edit.setFocus()

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Choose image folder", self.folder_edit.text())
        if folder:
            self.folder_edit.setText(os.path.normpath(folder))

    def search(self):
        if self.worker is not None:
            return
        folder, prompt = self.folder_edit.text().strip(), self.prompt_edit.text().strip()
        if not os.path.isdir(folder):
            QMessageBox.warning(self, "Image Inspect", "Choose a folder of images first.")
            return
        if not prompt:
            return
        self.settings.setValue("folder", folder)
        self.settings.setValue("recursive", self.recursive.isChecked())

        self.search_button.setEnabled(False)
        self.worker = SearchWorker(self.state, folder, self.recursive.isChecked(),
                                   prompt, self.limit.value())
        self.worker.progress.connect(self.on_progress)
        self.worker.finished_ok.connect(self.show_results)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.on_worker_done)
        self.worker.start()

    def on_progress(self, message, done, total):
        if total:
            self.progress.setRange(0, total)
            self.progress.setValue(done)
            if message.startswith("Downloading"):
                self.status.setText(f"{message}: {done >> 20} / {total >> 20} MB")
            else:
                self.status.setText(f"{message}: {done:,} / {total:,}")
        else:
            self.progress.setRange(0, 0)    # busy indicator
            self.status.setText(message + "…")

    def show_results(self, results):
        self.results.clear()
        for path, score, thumb in results:
            pixmap = QPixmap()
            if thumb:
                pixmap.loadFromData(thumb)
            item = QListWidgetItem(QIcon(pixmap), f"{score:.3f}")
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setToolTip(path)
            self.results.addItem(item)
        count = len(self.state["index"].paths)
        self.status.setText(f"Showing {len(results)} best of {count:,} images "
                            f"for \"{self.worker.prompt}\".")

    def on_failed(self, message):
        self.status.setText("Search failed.")
        QMessageBox.critical(self, "Image Inspect", message)

    def on_worker_done(self):
        self.worker = None
        self.search_button.setEnabled(True)
        self.progress.setRange(0, 1)
        self.progress.setValue(0)

    def enlarge(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        try:
            ImageDialog(path, self).exec()
        except Exception as e:                               # noqa: BLE001
            QMessageBox.warning(self, "Image Inspect", f"Could not open {path}:\n{e}")

    def closeEvent(self, event):
        if self.worker is not None:
            self.worker.requestInterruption()
            self.worker.wait()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
