"""
Image Inspect

Pick a folder, then describe what you are looking for, add guide images
that look like it, or both, and see the images CLIP thinks match best.
Click a thumbnail to enlarge it; right-click it to search for similar ones.
"""

import os
import sys

import numpy as np
from PIL import Image
from PyQt6.QtCore import (QElapsedTimer, QSettings, QSize, Qt, QThread,
                          QTimer, QUrl, pyqtSignal)
from PyQt6.QtGui import (QColor, QDesktopServices, QIcon, QImage, QPainter,
                         QPixmap)
from PyQt6.QtWidgets import (QApplication, QCheckBox, QDialog, QFileDialog,
                             QHBoxLayout, QLabel, QLineEdit, QListWidget,
                             QListWidgetItem, QMainWindow, QMenu, QMessageBox,
                             QProgressBar, QPushButton, QSlider, QSpinBox,
                             QVBoxLayout, QWidget)

import clip_search
from image_io import open_image

ICON_SIZE = clip_search.THUMB_SIZE * 3 // 4
GUIDE_ICON_SIZE = 64
GUIDE_COLOR = QColor(255, 190, 40, 90)


def same_path(a, b):
    return os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))


def square_pixmap(pixmap, size):
    """pixmap scaled to fit, centred on a transparent size x size square.

    Every item then has the same decoration size, which the results list's
    uniform item sizes rely on; otherwise a portrait thumbnail can overflow
    its cell and push its score out of view.
    """
    canvas = QPixmap(size, size)
    canvas.fill(Qt.GlobalColor.transparent)
    if not pixmap.isNull():
        pixmap = pixmap.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
        painter = QPainter(canvas)
        painter.drawPixmap((size - pixmap.width()) // 2,
                           (size - pixmap.height()) // 2, pixmap)
        painter.end()
    return canvas


def load_pixmap(path, max_size):
    with open_image(path, draft_size=max_size) as img:
        img = img.convert("RGB")
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        data = img.tobytes()
        qimage = QImage(data, img.width, img.height, 3 * img.width,
                        QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimage)


class SearchWorker(QThread):
    """Downloads models if needed, indexes the folder, then ranks it."""

    progress = pyqtSignal(str, int, int)
    finished_ok = pyqtSignal(list)      # [(path, scores, thumbnail bytes)]
    failed = pyqtSignal(str)

    def __init__(self, state, folder, recursive, prompt, guides, image_weight,
                 limit):
        super().__init__()
        self.state = state              # dict holding the Clip and Index
        self.folder, self.recursive = folder, recursive
        self.prompt, self.guides = prompt, guides
        self.image_weight, self.limit = image_weight, limit

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

            self.progress.emit("Ranking images", 0, 0)
            text = clip.embed_text(self.prompt) if self.prompt else None
            guides = None
            if self.guides:
                guides = np.stack([
                    v if (v := index.embedding(p)) is not None
                    else clip.embed_image_file(p) for p in self.guides])
            results = index.search(text, guides, self.image_weight, self.limit)
            self.finished_ok.emit([(path, scores, index.thumbnail(path))
                                   for path, scores in results])
        except Exception as e:                               # noqa: BLE001
            self.failed.emit(str(e))


class ImageDialog(QDialog):
    """One image, scaled to fit the screen."""

    def __init__(self, path, parent=None, on_find_similar=None):
        super().__init__(parent)
        self.setWindowTitle(path)
        self.path = path

        screen = self.screen().availableGeometry()
        max_size = (int(screen.width() * 0.9), int(screen.height() * 0.85))
        label = QLabel()
        label.setPixmap(load_pixmap(path, max_size))
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        open_button = QPushButton("Open in default viewer")
        open_button.clicked.connect(lambda: QDesktopServices.openUrl(
            QUrl.fromLocalFile(self.path)))
        folder_button = QPushButton("Show in folder")
        folder_button.clicked.connect(self.show_in_folder)

        buttons = QHBoxLayout()
        buttons.addWidget(QLabel(path), 1)
        if on_find_similar is not None:
            similar_button = QPushButton("Find similar images")
            similar_button.clicked.connect(lambda: (self.accept(),
                                                    on_find_similar(self.path)))
            buttons.addWidget(similar_button)
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
        self.setAcceptDrops(True)
        self.settings = QSettings("ImageInspect", "ImageInspect")
        self.state = {}
        self.worker = None
        self.clock = QElapsedTimer()
        self.ticker = QTimer(self)
        self.ticker.setInterval(250)
        self.ticker.timeout.connect(self.update_elapsed)
        self.progress_text = self.query_text = self.result_text = ""

        self.folder_edit = QLineEdit(self.settings.value("folder", ""))
        self.folder_edit.setPlaceholderText("Folder of images")
        self.browse_button = QPushButton("Browse…")
        self.browse_button.clicked.connect(self.choose_folder)
        self.recursive = QCheckBox("Include subfolders")
        self.recursive.setChecked(self.settings.value("recursive", True, bool))

        self.prompt_edit = QLineEdit()
        self.prompt_edit.setPlaceholderText(
            "Describe what you are looking for, e.g. \"a dog on the beach\" "
            "(optional when using guide images)")
        self.prompt_edit.setClearButtonEnabled(True)
        self.prompt_edit.returnPressed.connect(self.search)
        self.prompt_edit.textChanged.connect(self.update_controls)
        self.add_guide_button = QPushButton("Add guide images…")
        self.add_guide_button.clicked.connect(self.choose_guides)
        self.limit = QSpinBox()
        self.limit.setRange(1, 1000)
        self.limit.setValue(100)
        self.limit.setPrefix("Top ")
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.search_or_cancel)

        self.guide_list = QListWidget()
        self.guide_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.guide_list.setFlow(QListWidget.Flow.LeftToRight)
        self.guide_list.setWrapping(False)
        self.guide_list.setMovement(QListWidget.Movement.Static)
        self.guide_list.setIconSize(QSize(GUIDE_ICON_SIZE, GUIDE_ICON_SIZE))
        self.guide_list.setFixedHeight(GUIDE_ICON_SIZE + 24)
        self.guide_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.guide_list.customContextMenuRequested.connect(self.guide_menu)
        self.guide_list.itemDoubleClicked.connect(
            lambda item: self.remove_guide(item.data(Qt.ItemDataRole.UserRole)))
        self.clear_guides_button = QPushButton("Clear guides")
        self.clear_guides_button.clicked.connect(self.clear_guides)
        self.weight = QSlider(Qt.Orientation.Horizontal)
        self.weight.setRange(0, 100)
        self.weight.setValue(self.settings.value("image_weight", 50, int))
        self.weight.setMaximumWidth(160)
        self.weight.setToolTip("How much the guide images count against the "
                               "description when using both")
        self.weight_label = QLabel()
        self.weight.valueChanged.connect(self.update_controls)

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setMaximumWidth(300)
        self.status = QLabel("Choose a folder, then enter a description or "
                             "add guide images.")
        self.elapsed = QLabel()

        self.results = QListWidget()
        self.results.setViewMode(QListWidget.ViewMode.IconMode)
        self.results.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        # A fixed cell with room for two lines of label under each icon.
        line = self.results.fontMetrics().lineSpacing()
        self.results.setGridSize(QSize(ICON_SIZE + 24, ICON_SIZE + 2 * line + 16))
        self.results.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.results.setMovement(QListWidget.Movement.Static)
        self.results.setUniformItemSizes(True)
        self.results.setWordWrap(True)
        self.results.itemClicked.connect(self.enlarge)
        self.results.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.results.customContextMenuRequested.connect(self.result_menu)

        row1 = QHBoxLayout()
        row1.addWidget(self.folder_edit, 1)
        row1.addWidget(self.browse_button)
        row1.addWidget(self.recursive)
        row2 = QHBoxLayout()
        row2.addWidget(self.prompt_edit, 1)
        row2.addWidget(self.add_guide_button)
        row2.addWidget(self.limit)
        row2.addWidget(self.search_button)
        self.guide_row = QWidget()
        row_guides = QHBoxLayout(self.guide_row)
        row_guides.setContentsMargins(0, 0, 0, 0)
        row_guides.addWidget(QLabel("Guide images:"))
        row_guides.addWidget(self.guide_list, 1)
        weight_box = QVBoxLayout()
        weight_box.addWidget(self.weight_label)
        weight_box.addWidget(self.weight)
        row_guides.addLayout(weight_box)
        row_guides.addWidget(self.clear_guides_button)
        row3 = QHBoxLayout()
        row3.addWidget(self.status, 1)
        row3.addWidget(self.elapsed)
        row3.addWidget(self.progress)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addLayout(row1)
        layout.addLayout(row2)
        layout.addWidget(self.guide_row)
        layout.addWidget(self.results, 1)
        layout.addLayout(row3)
        self.setCentralWidget(central)
        self.prompt_edit.setFocus()
        self.update_controls()

    # ------------------------------------------------------------------
    # Guide images
    # ------------------------------------------------------------------

    def guides(self):
        return [self.guide_list.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(self.guide_list.count())]

    def add_guides(self, paths):
        for path in paths:
            path = os.path.normpath(os.path.abspath(path))
            if any(same_path(path, g) for g in self.guides()):
                continue
            try:
                pixmap = load_pixmap(path, (2 * GUIDE_ICON_SIZE, 2 * GUIDE_ICON_SIZE))
            except Exception as e:                           # noqa: BLE001
                QMessageBox.warning(self, "Image Inspect",
                                    f"Could not open {path}:\n{e}")
                continue
            item = QListWidgetItem(QIcon(square_pixmap(pixmap, GUIDE_ICON_SIZE)), "")
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setToolTip(f"{path}\n(double-click or right-click to remove)")
            self.guide_list.addItem(item)
        self.update_controls()

    def remove_guide(self, path):
        for i in range(self.guide_list.count()):
            if same_path(self.guide_list.item(i).data(Qt.ItemDataRole.UserRole), path):
                self.guide_list.takeItem(i)
                break
        self.update_controls()

    def clear_guides(self):
        self.guide_list.clear()
        self.update_controls()

    def choose_guides(self):
        patterns = " ".join(f"*{e}" for e in clip_search.IMAGE_EXTENSIONS)
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Choose guide images", self.folder_edit.text(),
            f"Images ({patterns});;All files (*)")
        self.add_guides(paths)

    def find_similar(self, path):
        """Search with path as the only guide and no description."""
        if self.worker is not None:
            return
        self.guide_list.clear()
        self.prompt_edit.clear()
        self.add_guides([path])
        self.search()

    def guide_menu(self, pos):
        item = self.guide_list.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        menu.addAction("Remove", lambda: self.remove_guide(
            item.data(Qt.ItemDataRole.UserRole)))
        menu.exec(self.guide_list.viewport().mapToGlobal(pos))

    def result_menu(self, pos):
        item = self.results.itemAt(pos)
        if item is None or self.worker is not None:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        menu.addAction("Find similar images", lambda: self.find_similar(path))
        if any(same_path(path, g) for g in self.guides()):
            menu.addAction("Remove from guide images", lambda: self.remove_guide(path))
        else:
            menu.addAction("Add to guide images", lambda: self.add_guides([path]))
        menu.exec(self.results.viewport().mapToGlobal(pos))

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and self.worker is None:
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        self.add_guides([p for p in paths if os.path.splitext(p)[1].lower()
                         in clip_search.IMAGE_EXTENSIONS])

    def update_controls(self):
        """Reflect the query mode and whether a search is running."""
        busy = self.worker is not None
        has_guides = self.guide_list.count() > 0
        has_text = bool(self.prompt_edit.text().strip())
        for widget in (self.folder_edit, self.browse_button, self.recursive,
                       self.prompt_edit, self.add_guide_button, self.limit,
                       self.guide_row):
            widget.setEnabled(not busy)
        self.guide_row.setVisible(has_guides)
        self.weight.setEnabled(not busy and has_text)
        self.weight_label.setText(f"Text {100 - self.weight.value()}% · "
                                  f"Images {self.weight.value()}%"
                                  if has_text else "Images only")
        self.search_button.setText("Cancel" if busy else "Search")
        self.search_button.setEnabled(busy or has_text or has_guides)

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Choose image folder", self.folder_edit.text())
        if folder:
            self.folder_edit.setText(os.path.normpath(folder))

    def search_or_cancel(self):
        if self.worker is not None:
            self.worker.requestInterruption()
            self.search_button.setEnabled(False)
            self.progress_text = "Cancelling…"
            self.update_elapsed()
        else:
            self.search()

    def describe_query(self, prompt, guides):
        parts = []
        if prompt:
            parts.append(f"\"{prompt}\"")
        if guides:
            parts.append(f"{len(guides)} guide image" + ("s" if len(guides) > 1 else ""))
        return " + ".join(parts)

    def search(self):
        if self.worker is not None:
            return
        folder, prompt = self.folder_edit.text().strip(), self.prompt_edit.text().strip()
        guides = self.guides()
        if not os.path.isdir(folder):
            QMessageBox.warning(self, "Image Inspect", "Choose a folder of images first.")
            return
        if not prompt and not guides:
            return
        self.settings.setValue("folder", folder)
        self.settings.setValue("recursive", self.recursive.isChecked())
        self.settings.setValue("image_weight", self.weight.value())

        # Clear the old results straight away, so it is plain that a new
        # search has started even if indexing new files takes a while.
        self.results.clear()
        self.result_text = ""
        self.query_text = self.describe_query(prompt, guides)
        self.worker = SearchWorker(self.state, folder, self.recursive.isChecked(),
                                   prompt, guides, self.weight.value() / 100,
                                   self.limit.value())
        self.worker.progress.connect(self.on_progress)
        self.worker.finished_ok.connect(self.show_results)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.on_worker_done)
        self.update_controls()
        self.progress_text = "Starting…"
        self.clock.start()
        self.ticker.start()
        self.update_elapsed()
        self.worker.start()

    def on_progress(self, message, done, total):
        if total:
            self.progress.setRange(0, total)
            self.progress.setValue(done)
            if message.startswith("Downloading"):
                text = f"{message}: {done >> 20} / {total >> 20} MB"
            else:
                text = f"{message}: {done:,} / {total:,}"
        else:
            self.progress.setRange(0, 0)    # busy indicator
            text = message + "…"
        if not self.progress_text.startswith("Cancelling"):
            self.progress_text = text
        self.update_elapsed()

    def update_elapsed(self):
        self.status.setText(f"Searching for {self.query_text} — {self.progress_text}")
        self.elapsed.setText(f"{self.clock.elapsed() / 1000:.1f} s")

    def show_results(self, results):
        self.results.clear()
        guides = self.worker.guides
        for path, scores, thumb in results:
            pixmap = QPixmap()
            if thumb:
                pixmap.loadFromData(thumb)
            if len(scores) == 1:
                label = f"{next(iter(scores.values())):.3f}"
            else:
                label = f"text {scores['text']:.3f} · image {scores['image']:.3f}"
            is_guide = any(same_path(path, g) for g in guides)
            if is_guide:
                label = "★ guide\n" + label
            item = QListWidgetItem(QIcon(square_pixmap(pixmap, ICON_SIZE)), label)
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setToolTip(path + ("\n(guide image)" if is_guide else ""))
            if is_guide:
                item.setBackground(GUIDE_COLOR)
            self.results.addItem(item)
        count = len(self.state["index"].paths)
        cancelled = self.worker.isInterruptionRequested()
        self.result_text = (
            f"Showing {len(results)} best of {count:,} "
            f"{'indexed ' if cancelled else ''}images for {self.query_text}"
            f"{' (indexing was cancelled)' if cancelled else ''}. "
            "Edit the description or guide images and press Search to search again.")

    def on_failed(self, message):
        self.result_text = "Search failed."
        QMessageBox.critical(self, "Image Inspect", message)

    def on_worker_done(self):
        self.ticker.stop()
        self.elapsed.setText(f"{self.clock.elapsed() / 1000:.1f} s")
        self.status.setText(self.result_text)
        self.worker = None
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.update_controls()
        self.results.scrollToTop()
        self.prompt_edit.setFocus()
        self.prompt_edit.selectAll()

    def enlarge(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        try:
            ImageDialog(path, self, on_find_similar=self.find_similar).exec()
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
