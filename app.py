"""
Image Inspect

Pick a folder, then describe what you are looking for, add guide images
that look like it, or both, and see the images CLIP thinks match best.
Guide images can instead pick out a person, matched by face recognition.
Click a thumbnail to enlarge it; right-click it to search for similar ones
or have a local language model describe it.
"""

import os
import sys

import numpy as np
from PIL import Image
from PyQt6.QtCore import (QElapsedTimer, QEvent, QObject, QPoint, QSettings,
                          QSize, Qt, QThread, QTimer, QUrl, pyqtSignal)
from PyQt6.QtGui import (QAction, QColor, QContextMenuEvent, QDesktopServices,
                         QIcon, QImage, QKeySequence, QPainter, QPalette,
                         QPixmap)
from PyQt6.QtWidgets import (QApplication, QButtonGroup, QCheckBox, QComboBox,
                             QDialog, QFileDialog, QFrame, QGridLayout,
                             QHBoxLayout, QLabel, QLineEdit, QListWidget,
                             QListWidgetItem, QMainWindow, QMenu, QMessageBox,
                             QPlainTextEdit, QProgressBar, QPushButton, QRadioButton,
                             QScrollArea, QSlider, QSpacerItem, QSpinBox,
                             QStyle, QToolButton, QVBoxLayout, QWidget)

import clip_search
import describe
import face_search
import models
from image_io import open_image
from version import __version__
from ui_support import (THEME_CHANGE, HintLabel, StatusLabel, WidestTextLabel,
                        WrapLabel, announce, apply_windows_appearance,
                        initial_window_rect, make_secondary,
                        refresh_scaled_fonts, scale_font,
                        secondary_text_color, set_tone)

# Where bundled files are: next to this file, or in the installed app's
# _internal folder.
BUNDLE_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
APP_ICON = os.path.join(BUNDLE_DIR, "packaging", "icon.ico")

ICON_SIZE = clip_search.THUMB_SIZE * 3 // 4
GUIDE_ICON_SIZE = 64
# Drawn over photos, so the same in light and dark themes.
FACE_BOX_COLOR = QColor(255, 190, 40)

# What the guide images are matched on.
MATCH_LOOK, MATCH_PERSON = "look", "person"


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


def path_key(path):
    return os.path.normcase(os.path.abspath(path))


class SearchWorker(QThread):
    """Downloads models if needed, indexes the folder, then ranks it.

    For a person search, the guide images' faces are found first. Guides
    whose face is not yet known are reported through guide_faces; if any
    shows several people the search stops there, with needs_choice set, so
    the window can ask which one is meant and search again.
    """

    progress = pyqtSignal(str, int, int)
    finished_ok = pyqtSignal(list)      # [(path, scores, thumbnail, face box)]
    failed = pyqtSignal(str)
    guide_faces = pyqtSignal(list)      # [(path, [face_search.Face])]

    def __init__(self, state, folder, recursive, prompt, guides, image_weight,
                 limit, match=MATCH_LOOK, known_faces=None, threshold=0.0):
        super().__init__()
        self.state = state              # dict holding the models and indexes
        self.folder, self.recursive = folder, recursive
        self.prompt, self.guides = prompt, guides
        self.image_weight, self.limit = image_weight, limit
        self.match, self.threshold = match, threshold
        # path_key -> face embedding, or None for a guide with no face
        self.known_faces = dict(known_faces or {})
        self.needs_choice = False

    def run(self):
        try:
            if "clip" not in self.state:
                clip_search.ensure_models(self.progress.emit)
                self.progress.emit("Loading CLIP model", 0, 0)
                self.state["clip"] = clip_search.Clip()
                self.state["index"] = clip_search.Index()
            clip, index = self.state["clip"], self.state["index"]

            template = None
            if self.match == MATCH_PERSON:
                template = self.person_template()
                if template is None:
                    return

            self.progress.emit("Scanning folder", 0, 0)
            paths = clip_search.find_images(self.folder, self.recursive)
            index.build(clip, paths, self.progress.emit,
                        self.isInterruptionRequested)

            text = clip.embed_text(self.prompt) if self.prompt else None
            if template is not None:
                face_index = self.state["face_index"]
                face_index.build(self.state["faces"], paths, self.progress.emit,
                                 self.isInterruptionRequested)
                self.progress.emit("Ranking images", 0, 0)
                results = index.search(
                    text, None, self.image_weight, self.limit,
                    person=face_index.person_scores(template, index.paths),
                    person_threshold=self.threshold)
                self.finished_ok.emit([
                    (path, scores, index.thumbnail(path),
                     face_index.best_box(template, path))
                    for path, scores in results])
                return

            self.progress.emit("Ranking images", 0, 0)
            guides = None
            if self.guides:
                guides = np.stack([
                    v if (v := index.embedding(p)) is not None
                    else clip.embed_image_file(p) for p in self.guides])
            results = index.search(text, guides, self.image_weight, self.limit)
            self.finished_ok.emit([(path, scores, index.thumbnail(path), None)
                                   for path, scores in results])
        except Exception as e:                               # noqa: BLE001
            self.failed.emit(str(e))

    def person_template(self):
        """The face embedding to search for, or None to stop here."""
        if "faces" not in self.state:
            face_search.ensure_models(self.progress.emit)
            self.progress.emit("Loading face recognition model", 0, 0)
            self.state["faces"] = face_search.FaceModel()
            self.state["face_index"] = face_search.FaceIndex()
        model = self.state["faces"]

        found = []
        for path in self.guides:
            if path_key(path) not in self.known_faces:
                self.progress.emit(f"Finding faces in {os.path.basename(path)}", 0, 0)
                faces = model.analyze_file(path, crops=True)
                found.append((path, faces))
                if len(faces) == 1:
                    self.known_faces[path_key(path)] = faces[0].embedding
                elif not faces:
                    self.known_faces[path_key(path)] = None
        if found:
            self.guide_faces.emit(found)
        if any(len(faces) > 1 for _, faces in found):
            self.needs_choice = True
            return None

        embeddings = [e for p in self.guides
                      if (e := self.known_faces.get(path_key(p))) is not None]
        if not embeddings:
            self.failed.emit("No face was found in the guide images. "
                             "Use photos where the person's face is clearly "
                             "visible, or search for Things & scenes instead.")
            return None
        return face_search.person_template(np.stack(embeddings))


def set_fraction(bar, done, total):
    """Show done of total on a progress bar; in thousandths, since a
    progress bar holds 32-bit numbers and downloads run past 2 GB."""
    bar.setRange(0, 1000)
    bar.setValue(1000 * done // total)


def pixmap_from_bytes(data):
    pixmap = QPixmap()
    if data:
        pixmap.loadFromData(data)
    return pixmap


def describe_place(box):
    """Where box (fractions of the image) sits, in words: "top left",
    "centre", "bottom"."""
    x = (box[0] + box[2]) / 2
    y = (box[1] + box[3]) / 2
    row = "top" if y < 1 / 3 else "bottom" if y > 2 / 3 else ""
    column = "left" if x < 1 / 3 else "right" if x > 2 / 3 else ""
    return " ".join(p for p in (row, column) if p) or "centre"


def label_for(widget, text, name=None):
    """A visible label for widget, which screen readers also hear as its
    name, and whose Alt access key moves to it."""
    label = QLabel(text)
    label.setBuddy(widget)
    widget.setAccessibleName(name or text.replace("&", ""))
    return label


def heading(text, buddy=None, name=None):
    """A bold section label, sized from the window font so it follows
    later font changes; with buddy, also that input's spoken name."""
    label = label_for(buddy, text, name) if buddy is not None else QLabel(text)
    scale_font(label, 1.0, bold=True)
    return label


class ThumbnailList(QListWidget):
    """An icon grid whose context menu opens on the current item when
    asked for from the keyboard (Menu key or Shift+F10), and that shows a
    hint while empty. The hint is also its accessible description, since
    screen readers cannot see painted text."""

    menu_requested = pyqtSignal(object, QPoint)   # item or None, global pos

    def __init__(self, parent=None):
        super().__init__(parent)
        self.hint = ""
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setMovement(QListWidget.Movement.Static)

    def set_hint(self, hint):
        self.hint = hint
        self.setAccessibleDescription(hint)
        self.viewport().update()

    def contextMenuEvent(self, event):
        if event.reason() == QContextMenuEvent.Reason.Keyboard:
            item = self.currentItem()
            rect = self.visualItemRect(item) if item else self.viewport().rect()
            self.menu_requested.emit(item, self.viewport().mapToGlobal(rect.center()))
        else:
            self.menu_requested.emit(self.itemAt(event.pos()), event.globalPos())

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.count() == 0 and self.hint:
            painter = QPainter(self.viewport())
            painter.setPen(secondary_text_color(self.palette()))
            margin = self.fontMetrics().averageCharWidth() * 2
            painter.drawText(self.viewport().rect().adjusted(margin, 0, -margin, 0),
                             Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                             self.hint)
            painter.end()


def with_badge(pixmap, icon):
    """pixmap with icon drawn small in its bottom-right corner."""
    pixmap = QPixmap(pixmap)
    side = pixmap.width() // 3
    painter = QPainter(pixmap)
    painter.drawPixmap(pixmap.width() - side, pixmap.height() - side,
                       icon.pixmap(side, side))
    painter.end()
    return pixmap


class DescribeWorker(QThread):
    """Writes one description, downloading and loading the model first if
    need be. If the GPU build will not load and the CPU build is not
    downloaded, it stops with needs_cpu_model set so the user can be asked
    about that download."""

    progress = pyqtSignal(str, int, int)
    text = pyqtSignal(str)              # the description so far
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, state, path, variant):
        super().__init__()
        self.state, self.path, self.variant = state, path, variant
        self.view = None                    # the DescriptionPanel shown to
        self.needs_cpu_model = False

    def run(self):
        try:
            text = describe.describe_file(
                self.state, self.path, self.variant, self.progress.emit,
                self.text.emit, self.isInterruptionRequested)
            if text is not None:
                self.finished_ok.emit(text)
        except models.Cancelled:
            pass
        except describe.GpuUnavailable:
            self.needs_cpu_model = True
        except Exception as e:                               # noqa: BLE001
            self.failed.emit(str(e))


def download_question(variant, size, gpu_failed=False):
    """The question asked before downloading the description model, as
    (text, informative text)."""
    if gpu_failed:
        hardware = ("The graphics card could not run the version already "
                    "downloaded, so this is the version for the processor. "
                    "It takes about half a minute per photo.")
    elif variant == "cuda":
        hardware = (f"It will run on the graphics card ({models.nvidia_gpu()}), "
                    "taking a few seconds per photo.")
    else:
        hardware = ("It will run on the processor, taking about half a "
                    "minute per photo.")
    return (f"Describing photos needs {describe.MODEL_LABEL}, a "
            f"{describe.format_size(size)} download.",
            "It runs on this computer, so your photos are not sent anywhere. "
            f"{hardware}\n\nIt is downloaded once, from Hugging Face, to "
            f"{models.MODEL_DIR}.")


def agree_to_download(parent, variant, gpu_failed=False):
    """Ask before downloading the description model; True to go ahead,
    including when there is nothing to download."""
    size = describe.download_size(variant)
    if not size:
        return True
    free = describe.free_space()
    if free < size + (1 << 30):
        QMessageBox.warning(
            parent, "Image Inspect",
            f"Describing photos needs a {describe.format_size(size)} "
            f"download, but there is only {describe.format_size(free)} free "
            f"on the drive holding {models.MODEL_DIR}.")
        return False
    text, details = download_question(variant, size, gpu_failed)
    box = QMessageBox(QMessageBox.Icon.Question, "Download description model?",
                      text, parent=parent)
    box.setInformativeText(details)
    download = box.addButton("&Download", QMessageBox.ButtonRole.AcceptRole)
    box.addButton("Not now", QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(download)
    box.exec()
    return box.clickedButton() is download


class DescriptionService(QObject):
    """Writes descriptions for the enlarged views, one at a time, keeping
    the model loaded between them.

    A view that closes stops its description; one asked for while another
    is still stopping waits its turn.
    """

    device_changed = pyqtSignal()

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.worker = None
        self.pending = None                 # (path, variant, view) to start next
        self.cpu_only = False               # The GPU could not run the model

    def cache(self):
        if "descriptions" not in self.state:
            self.state["descriptions"] = describe.DescriptionCache()
        return self.state["descriptions"]

    def start(self, path, view):
        """Describe path into view, asking first if the model needs
        downloading. False if the user would rather not."""
        cached = self.cache().get(path)
        if cached is not None:
            view.show_done(cached, cached=True)
            return True
        describer = self.state.get("describer")
        if describer is not None:
            variant = describer.variant
        else:
            variant = "cpu" if self.cpu_only else describe.preferred_variant()
            if not agree_to_download(view.window(), variant):
                view.show_stopped("The description model was not downloaded")
                return False
        view.show_started()
        if self.worker is not None:
            self.worker.requestInterruption()
            self.worker.view = None
            self.pending = (path, variant, view)
        else:
            self.run(path, variant, view)
        return True

    def run(self, path, variant, view):
        worker = self.worker = DescribeWorker(self.state, path, variant)
        worker.view = view
        worker.progress.connect(
            lambda m, d, t: worker.view and worker.view.show_progress(m, d, t))
        worker.text.connect(lambda t: worker.view and worker.view.show_text(t))
        worker.finished_ok.connect(
            lambda t: worker.view and worker.view.show_done(t))
        worker.failed.connect(lambda m: worker.view and worker.view.show_failed(m))
        worker.finished.connect(self.on_worker_done)
        worker.start()

    def stop(self, view):
        """Stop describing for view, which is closing or was told to stop."""
        if self.pending and self.pending[2] is view:
            self.pending = None
        if self.worker is not None and self.worker.view is view:
            self.worker.requestInterruption()
            self.worker.view = None

    def on_worker_done(self):
        worker, self.worker = self.worker, None
        self.device_changed.emit()
        if self.pending:
            path, variant, view = self.pending
            self.pending = None
            self.run(path, variant, view)
        elif worker.needs_cpu_model and worker.view is not None:
            view = worker.view
            self.cpu_only = True
            if agree_to_download(view.window(), "cpu", gpu_failed=True):
                self.run(worker.path, "cpu", view)
            else:
                view.show_stopped("The description model was not downloaded")

    def shutdown(self):
        self.pending = None
        if self.worker is not None:
            self.worker.requestInterruption()
            self.worker.wait()


class DescriptionPanel(QWidget):
    """Where an enlarged photo's description appears as it is written, with
    its progress and a Copy button. What happens is announced to screen
    readers, ending with the description itself."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard)
        self.heading = heading("Desc&ription", self.text)
        self.status = StatusLabel(wrap=True)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setAccessibleName("Description progress")
        self.note = WrapLabel("Written by an AI model, which can get details "
                              "wrong, such as how many people there are.")
        make_secondary(self.note)
        self.copy_button = QPushButton("&Copy")
        self.copy_button.setAccessibleName("Copy description")
        self.copy_button.clicked.connect(self.copy)
        self.stop_button = QPushButton("S&top")
        self.stop_button.setAccessibleName("Stop describing")
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.copy_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.heading)
        layout.addWidget(self.text, 1)
        layout.addWidget(self.status)
        layout.addWidget(self.progress)
        layout.addWidget(self.note)
        layout.addLayout(buttons)
        self.clock = QElapsedTimer()
        self.last_spoken = 0
        self.set_busy(False)

    def set_busy(self, busy):
        self.progress.setVisible(busy)
        self.stop_button.setVisible(busy)
        self.copy_button.setEnabled(not busy and bool(self.text.toPlainText()))

    def show_started(self):
        self.text.clear()
        self.status.set_status("Starting…")
        self.progress.setRange(0, 0)
        self.set_busy(True)
        self.clock.start()
        self.last_spoken = 0
        announce(self, "Describing the photo")

    def show_progress(self, message, done, total):
        if total:
            set_fraction(self.progress, done, total)
            self.status.set_status(f"{message}: {done >> 20:,} / {total >> 20:,} MB")
            if self.clock.elapsed() - self.last_spoken >= ANNOUNCE_PROGRESS_EVERY:
                self.last_spoken = self.clock.elapsed()
                announce(self, f"{message.split(' (')[0]}, "
                         f"{round(100 * done / total)} percent")
        else:
            self.progress.setRange(0, 0)
            self.status.set_status(message + "…")

    def show_text(self, text):
        self.text.setPlainText(text)

    def show_done(self, text, cached=False):
        self.text.setPlainText(text)
        self.set_busy(False)
        self.status.set_status("Described earlier" if cached else
                               f"Written by {describe.MODEL_LABEL} on the "
                               f"{models.device()}", "done")
        announce(self, f"Description: {text}")

    def show_failed(self, message):
        self.set_busy(False)
        self.status.set_status(f"Could not describe the photo: {message}",
                               "problem")
        announce(self, self.status.accessibleName(), important=True)

    def show_stopped(self, message):
        self.set_busy(False)
        self.status.set_status(message)
        announce(self, message)

    def copy(self):
        QApplication.clipboard().setText(self.text.toPlainText())
        announce(self, "Description copied")


class ImageDialog(QDialog):
    """One image, scaled to fit the screen, with the matched face boxed.

    With descriptions (a DescriptionService), it can also describe the
    photo, in a panel beside it.
    """

    PANEL_WIDTH_EM = 22

    def __init__(self, path, parent=None, on_find_similar=None,
                 on_find_person=None, face_box=None, descriptions=None,
                 describe_now=False):
        super().__init__(parent)
        self.setWindowTitle(path)
        self.path = path
        self.descriptions = descriptions

        screen = self.screen().availableGeometry()
        max_size = (int(screen.width() * 0.9), int(screen.height() * 0.85))
        self.max_size = max_size
        pixmap = load_pixmap(path, max_size)
        if face_box is not None:
            x1, y1, x2, y2 = face_box
            w, h = pixmap.width(), pixmap.height()
            painter = QPainter(pixmap)
            pen = painter.pen()
            pen.setColor(FACE_BOX_COLOR)
            pen.setWidth(3)
            painter.setPen(pen)
            painter.drawRect(round(x1 * w), round(y1 * h),
                             round((x2 - x1) * w), round((y2 - y1) * h))
            painter.end()
        self.pixmap = pixmap
        self.image = QLabel()
        self.image.setPixmap(pixmap)
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setAccessibleName(f"Photo {os.path.basename(path)}")
        if face_box is not None:
            self.image.setAccessibleDescription(
                f"The matching face is boxed, at the {describe_place(face_box)}")

        open_button = QPushButton("&Open in default viewer")
        open_button.clicked.connect(lambda: QDesktopServices.openUrl(
            QUrl.fromLocalFile(self.path)))
        folder_button = QPushButton("Show in &folder")
        folder_button.clicked.connect(self.show_in_folder)

        path_label = QLabel(path)
        path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        buttons = QHBoxLayout()
        buttons.addWidget(path_label, 1)
        if on_find_similar is not None:
            similar_button = QPushButton("Find &similar images")
            similar_button.clicked.connect(lambda: (self.accept(),
                                                    on_find_similar(self.path)))
            buttons.addWidget(similar_button)
        if on_find_person is not None:
            person_button = QPushButton("Find this &person")
            person_button.clicked.connect(lambda: (self.accept(),
                                                   on_find_person(self.path)))
            buttons.addWidget(person_button)
        self.panel = None
        if descriptions is not None:
            self.describe_button = QPushButton("&Describe")
            self.describe_button.setToolTip("Describe the photo in words (Ctrl+D)")
            self.describe_button.clicked.connect(self.describe)
            buttons.addWidget(self.describe_button)
            describe_action = QAction(self)
            describe_action.setShortcut(QKeySequence(DESCRIBE_SHORTCUT))
            describe_action.triggered.connect(self.describe)
            self.addAction(describe_action)
            self.panel = DescriptionPanel()
            self.panel.stop_button.clicked.connect(self.stop_describing)
            self.panel.hide()
        buttons.addWidget(folder_button)
        buttons.addWidget(open_button)

        top = QHBoxLayout()
        top.addWidget(self.image, 1)
        if self.panel is not None:
            top.addWidget(self.panel)
        layout = QVBoxLayout(self)
        layout.addLayout(top, 1)
        layout.addLayout(buttons)
        if describe_now and descriptions is not None:
            QTimer.singleShot(0, self.describe)

    def describe(self):
        """Show the description panel and write the photo's description,
        unless one is already being written."""
        if self.panel.isHidden():
            width = self.PANEL_WIDTH_EM * self.fontMetrics().height()
            self.panel.setFixedWidth(width)
            room = QSize(self.max_size[0] - width, self.max_size[1])
            if self.pixmap.width() > room.width():
                self.image.setPixmap(self.pixmap.scaled(
                    room, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))
            self.panel.show()
            self.adjustSize()
        elif not self.panel.stop_button.isHidden():
            return
        self.descriptions.start(self.path, self.panel)

    def stop_describing(self):
        self.descriptions.stop(self.panel)
        self.panel.show_stopped("Stopped")

    def done(self, result):
        if self.panel is not None:
            self.descriptions.stop(self.panel)
        super().done(result)

    def show_in_folder(self):
        if sys.platform == "win32":
            import subprocess
            subprocess.Popen(["explorer", "/select,", os.path.normpath(self.path)])
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(self.path)))


class FacePickerDialog(QDialog):
    """Asks which of the people in a guide image is meant.

    The faces are shown left to right as they stand in the photo, and each
    is named for screen readers by its place in it.
    """

    COLUMNS = 6

    def __init__(self, path, faces, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Which person?")
        self.choice = None
        order = sorted(range(len(faces)),
                       key=lambda i: (faces[i].box[0] + faces[i].box[2]) / 2)
        biggest = max(range(len(faces)), key=lambda i: (
            (faces[i].box[2] - faces[i].box[0]) * (faces[i].box[3] - faces[i].box[1])))
        grid = QGridLayout()
        self.buttons = []
        for n, i in enumerate(order):
            face = faces[i]
            button = QToolButton()
            button.setIcon(QIcon(pixmap_from_bytes(face.crop)))
            button.setIconSize(QSize(face_search.CROP_SIZE, face_search.CROP_SIZE))
            button.setAutoRaise(True)
            button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            name = (f"Person {n + 1} of {len(faces)}, "
                    f"{describe_place(face.box)}"
                    + (", largest face" if i == biggest else ""))
            button.setAccessibleName(name)
            button.setToolTip(name)
            button.clicked.connect(lambda _, i=i: self.choose(i))
            grid.addWidget(button, n // self.COLUMNS, n % self.COLUMNS)
            self.buttons.append(button)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(cancel)
        layout = QVBoxLayout(self)
        layout.addWidget(WrapLabel(
            f"{os.path.basename(path)} shows {len(faces)} people. Choose the "
            "one to search for; they are in order from left to right."))
        layout.addLayout(grid)
        layout.addLayout(buttons)
        self.buttons[order.index(biggest)].setFocus()

    def choose(self, i):
        self.choice = i
        self.accept()


# Window-wide keys for the main actions. Each presses its button, so a key
# does nothing a click could not.
SHORTCUTS = {
    "choose_folder": ["Ctrl+O"],
    "add_guides": ["Ctrl+Shift+O"],
    "search": ["Ctrl+Return", "Ctrl+Enter"],
    "cancel": ["Esc"],
}

# Describes the current result, in the main window and enlarged view.
DESCRIBE_SHORTCUT = "Ctrl+D"

# How often indexing progress is spoken, in milliseconds.
ANNOUNCE_PROGRESS_EVERY = 15000


class MainWindow(QMainWindow):
    """A fixed sidebar holding the query, the results grid beside it, and
    progress in the status bar.

    Colours come from the palette and sizes from the font, so the window
    follows the Windows light/dark theme and text size. Every control has
    a spoken name and is reachable from the keyboard, and what happens
    away from the focus (results arriving, progress, cancelling) is
    announced to screen readers.
    """

    RECENT_FOLDERS = 10

    def __init__(self, settings=None):
        super().__init__()
        self.setWindowTitle("Image Inspect")
        self.setAcceptDrops(True)
        self.settings = settings or QSettings("ImageInspect", "ImageInspect")
        self.state = {}
        self.worker = None
        self.descriptions = DescriptionService(self.state, self)
        self.descriptions.device_changed.connect(self.update_device)
        self.clock = QElapsedTimer()
        self.ticker = QTimer(self)
        self.ticker.setInterval(250)
        self.ticker.timeout.connect(self.update_elapsed)
        self.progress_text = ""
        self.last_spoken = 0                # clock time progress was announced
        self.enlarging = False
        # path_key -> (face embedding, face crop pixmap), or (None, None) for
        # a guide image with no face; filled in by person searches.
        self.guide_faces = {}
        self.pending_choices = []           # [(path, faces)] to ask about

        # Folder
        self.folder = QComboBox()
        self.folder.setEditable(True)
        self.folder.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.folder.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.folder.setMinimumContentsLength(10)
        self.folder.lineEdit().setPlaceholderText("Folder of images")
        self.folder.addItems(self.recent_folders())
        self.folder.setCurrentText(self.settings.value("folder", ""))
        self.folder.lineEdit().returnPressed.connect(self.search)
        self.folder.setToolTip("The folder to search; recent folders are in the list")
        folder_heading = heading("&Folder", self.folder, "Folder to search")
        self.browse_button = QToolButton()
        self.browse_button.setText("…")
        self.browse_button.setAccessibleName("Choose folder")
        self.browse_button.setToolTip("Choose a folder")
        self.browse_button.clicked.connect(self.choose_folder)
        self.recursive = QCheckBox("Include s&ubfolders")
        self.recursive.setChecked(self.settings.value("recursive", True, bool))

        # Mode. Radio buttons rather than a segmented switch of push
        # buttons: the windows11 style draws a checked push button's text
        # unreadably in dark mode.
        self.look_button = QRadioButton("&Things && scenes")
        self.person_button = QRadioButton("&Person")
        self.look_button.setToolTip("Find images that look like the description "
                                    "and guide images (CLIP)")
        self.person_button.setToolTip("Find photos of the person in the guide "
                                      "photos (face recognition)")
        self.mode = QButtonGroup(self)
        for button in (self.look_button, self.person_button):
            self.mode.addButton(button)
        (self.person_button if self.settings.value("match") == MATCH_PERSON
         else self.look_button).setChecked(True)
        self.mode.buttonToggled.connect(self.update_controls)

        # Description
        self.prompt_edit = QLineEdit()
        self.prompt_edit.setClearButtonEnabled(True)
        self.prompt_edit.returnPressed.connect(self.search)
        self.prompt_edit.textChanged.connect(self.update_controls)
        prompt_heading = heading("&Description", self.prompt_edit)

        # Guide images
        self.add_guide_button = QToolButton()
        self.add_guide_button.setText("+")
        self.add_guide_button.setAccessibleName("Add guide images")
        self.add_guide_button.setToolTip("Add guide images")
        self.add_guide_button.clicked.connect(self.choose_guides)
        self.guide_list = ThumbnailList()
        self.guide_list.setFlow(QListWidget.Flow.LeftToRight)
        self.guide_list.setWrapping(True)
        self.guide_list.setIconSize(QSize(GUIDE_ICON_SIZE, GUIDE_ICON_SIZE))
        self.guide_list.setSpacing(4)
        self.guide_list.menu_requested.connect(self.guide_menu)
        self.guide_list.itemDoubleClicked.connect(
            lambda item: self.remove_guide(item.data(Qt.ItemDataRole.UserRole)))
        remove = QAction(self.guide_list)
        remove.setShortcuts([QKeySequence(QKeySequence.StandardKey.Delete),
                             QKeySequence(Qt.Key.Key_Backspace)])
        remove.setShortcutContext(Qt.ShortcutContext.WidgetShortcut)
        remove.triggered.connect(self.remove_current_guide)
        self.guide_list.addAction(remove)
        self.guide_heading = heading("", self.guide_list)
        self.clear_guides_button = QPushButton("&Clear")
        self.clear_guides_button.setAccessibleName("Clear guide images")
        self.clear_guides_button.clicked.connect(self.clear_guides)
        self.guide_hint = HintLabel("Double-click or press Delete to remove")

        # Balance and strictness
        self.weight = QSlider(Qt.Orientation.Horizontal)
        self.weight.setRange(0, 100)
        self.weight.setValue(self.settings.value("image_weight", 50, int))
        self.weight.valueChanged.connect(self.update_controls)
        self.weight_text = label_for(self.weight, "Te&xt")
        self.weight_visual = QLabel()
        self.strictness = QSlider(Qt.Orientation.Horizontal)
        self.strictness.setRange(5, 95)
        self.strictness.setValue(round(100 * self.settings.value(
            "face_threshold", face_search.DEFAULT_MATCH_THRESHOLD, float)))
        self.strictness.setToolTip(
            "How alike a face must be to count as the same person. Lower it "
            "to find more photos (and more strangers), raise it to be stricter.")
        self.strictness.valueChanged.connect(self.update_controls)
        self.strictness_label = label_for(self.strictness, "St&rictness",
                                          "Face match strictness")
        self.strictness_value = WidestTextLabel("0.00")

        self.search_button = QPushButton("&Search")
        self.search_button.setDefault(True)
        self.search_button.clicked.connect(self.search)

        # Results
        self.summary = StatusLabel(wrap=True)
        scale_font(self.summary, 1.0, bold=True)
        self.limit = QSpinBox()
        self.limit.setRange(1, 1000)
        self.limit.setValue(self.settings.value("limit", 100, int))
        self.limit.setToolTip("How many results to show at most")
        limit_label = label_for(self.limit, "Sh&ow", "Most results to show")
        self.results = ThumbnailList()
        self.results.setAccessibleName("Results")
        self.results.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        self.results.setUniformItemSizes(True)
        self.results.setWordWrap(True)
        self.results.itemClicked.connect(self.enlarge)
        self.results.itemActivated.connect(self.enlarge)     # Enter
        self.results.menu_requested.connect(self.result_menu)

        # Status bar
        self.status = QLabel("Choose a folder, then enter a description or "
                             "add guide images.")
        self.status.setIndent(self.fontMetrics().averageCharWidth())
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setAccessibleName("Search progress")
        self.elapsed = QLabel()
        make_secondary(self.elapsed)
        self.cancel_button = QPushButton("C&ancel")
        self.cancel_button.clicked.connect(self.cancel)
        # Which hardware the models run on, once one has loaded.
        self.device = QLabel()
        make_secondary(self.device)
        bar = self.statusBar()
        bar.addWidget(self.status, 1)
        bar.addPermanentWidget(self.progress)
        bar.addPermanentWidget(self.elapsed)
        bar.addPermanentWidget(self.cancel_button)
        bar.addPermanentWidget(self.device)

        # Sidebar layout
        folder_row = QHBoxLayout()
        folder_row.addWidget(self.folder, 1)
        folder_row.addWidget(self.browse_button)
        mode_row = QHBoxLayout()
        mode_row.addWidget(self.look_button)
        mode_row.addWidget(self.person_button)
        mode_row.addStretch(1)
        guide_head = QHBoxLayout()
        guide_head.addWidget(self.guide_heading, 1)
        guide_head.addWidget(self.add_guide_button)
        guide_foot = QHBoxLayout()
        guide_foot.addWidget(self.guide_hint, 1)
        guide_foot.addWidget(self.clear_guides_button)
        tuning = QGridLayout()
        tuning.addWidget(self.weight_text, 0, 0)
        tuning.addWidget(self.weight, 0, 1)
        tuning.addWidget(self.weight_visual, 0, 2)
        tuning.addWidget(self.strictness_label, 1, 0)
        tuning.addWidget(self.strictness, 1, 1)
        tuning.addWidget(self.strictness_value, 1, 2)
        tuning.setColumnStretch(1, 1)

        self.sidebar = QWidget()
        side = QVBoxLayout(self.sidebar)
        self.gaps = []
        side.addWidget(folder_heading)
        side.addLayout(folder_row)
        side.addWidget(self.recursive)
        self.add_gap(side)
        side.addWidget(heading("Search for"))
        side.addLayout(mode_row)
        self.add_gap(side)
        side.addWidget(prompt_heading)
        side.addWidget(self.prompt_edit)
        self.add_gap(side)
        side.addLayout(guide_head)
        side.addWidget(self.guide_list)
        side.addLayout(guide_foot)
        self.add_gap(side)
        side.addLayout(tuning)
        self.add_gap(side)
        side.addWidget(self.search_button)
        side.addStretch(1)
        # Scrolls rather than clipping when large text makes it too tall;
        # not a tab stop itself, only its contents are.
        self.side_scroll = QScrollArea()
        self.side_scroll.setWidget(self.sidebar)
        self.side_scroll.setWidgetResizable(True)
        self.side_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.side_scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.side_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        header = QHBoxLayout()
        header.addWidget(self.summary, 1)
        header.addWidget(limit_label)
        header.addWidget(self.limit)
        main = QVBoxLayout()
        main.addLayout(header)
        main.addWidget(self.results, 1)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.addWidget(self.side_scroll)
        layout.addLayout(main, 1)
        self.setCentralWidget(central)

        self.build_shortcuts()
        self.apply_metrics()
        em = self.fontMetrics().height()
        self.setGeometry(initial_window_rect(
            self.screen().availableGeometry(), QSize(75 * em, 50 * em)))
        self.prompt_edit.setFocus()
        self.update_controls()

    def add_gap(self, layout):
        """Space between sidebar sections, a line of text high."""
        gap = QSpacerItem(0, 0)
        layout.addItem(gap)
        self.gaps.append(gap)

    def build_shortcuts(self):
        buttons = {"choose_folder": self.browse_button,
                   "add_guides": self.add_guide_button,
                   "search": self.search_button,
                   "cancel": self.cancel_button}
        self.shortcut_actions = {}
        for name, keys in SHORTCUTS.items():
            button = buttons[name]
            action = QAction(self)
            action.setShortcuts([QKeySequence(k) for k in keys])
            action.triggered.connect(button.click)
            self.addAction(action)
            self.shortcut_actions[name] = action
            shown = QKeySequence(keys[0]).toString(
                QKeySequence.SequenceFormat.NativeText)
            tip = button.toolTip()
            button.setToolTip(f"{tip} ({shown})" if tip else shown)
        # Describes the current result, which has no button of its own.
        self.describe_action = QAction(self)
        self.describe_action.setShortcut(QKeySequence(DESCRIBE_SHORTCUT))
        self.describe_action.triggered.connect(self.describe_current)
        self.addAction(self.describe_action)

    def apply_metrics(self):
        """Size things from the font, so larger text gets room to fit."""
        em = self.fontMetrics().height()
        for gap in self.gaps:
            gap.changeSize(0, em)
        # Allow for the scroll bar, so showing it does not squeeze the
        # sidebar's contents.
        scroll_bar = self.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent)
        self.side_scroll.setFixedWidth(max(19 * em, 4 * GUIDE_ICON_SIZE)
                                       + scroll_bar)
        self.sidebar.layout().setContentsMargins(0, 0, scroll_bar, 0)
        self.sidebar.layout().invalidate()
        # Room for two rows of guides before the list scrolls.
        self.guide_list.setFixedHeight(2 * (GUIDE_ICON_SIZE + 16) + 8)
        # A fixed cell with room for two lines of label under each icon.
        line = self.results.fontMetrics().lineSpacing()
        self.results.setGridSize(QSize(ICON_SIZE + 24, ICON_SIZE + 2 * line + 16))
        self.progress.setFixedWidth(12 * em)
        self.limit.setMinimumWidth(
            self.limit.fontMetrics().horizontalAdvance("0000") + 4 * em)

    def changeEvent(self, event):
        if event.type() == QEvent.Type.FontChange:
            refresh_scaled_fonts(self)
            self.apply_metrics()
        elif event.type() in (QEvent.Type.PaletteChange, THEME_CHANGE):
            self.recolor_results()
        super().changeEvent(event)

    def recent_folders(self):
        folders = self.settings.value("recent_folders", [], list)
        return [f for f in folders if isinstance(f, str) and f]

    def remember_folder(self, folder):
        folders = [folder] + [f for f in self.recent_folders()
                              if not same_path(f, folder)]
        folders = folders[:self.RECENT_FOLDERS]
        self.settings.setValue("recent_folders", folders)
        self.settings.setValue("folder", folder)
        self.folder.blockSignals(True)
        self.folder.clear()
        self.folder.addItems(folders)
        self.folder.setCurrentText(folder)
        self.folder.blockSignals(False)

    # ------------------------------------------------------------------
    # Guide images
    # ------------------------------------------------------------------

    def guides(self):
        return [self.guide_list.item(i).data(Qt.ItemDataRole.UserRole)
                for i in range(self.guide_list.count())]

    def add_guides(self, paths):
        added = 0
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
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setData(Qt.ItemDataRole.UserRole + 1,
                         square_pixmap(pixmap, GUIDE_ICON_SIZE))
            self.guide_list.addItem(item)
            added += 1
        if added:
            announce(self, f"{added} guide image{'s' if added > 1 else ''} added, "
                     f"{self.guide_list.count()} in all")
        self.update_controls()

    def person_mode(self):
        return self.person_button.isChecked()

    def set_person_mode(self, on):
        (self.person_button if on else self.look_button).setChecked(True)

    def update_guide_items(self):
        """Show each guide's chosen face in person mode, its photo otherwise,
        flagging guides in which no face was found."""
        warning = self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxWarning)
        for i in range(self.guide_list.count()):
            item = self.guide_list.item(i)
            path = item.data(Qt.ItemDataRole.UserRole)
            pixmap = item.data(Qt.ItemDataRole.UserRole + 1)
            spoken = f"Guide image {os.path.basename(path)}"
            tip = path
            if self.person_mode() and path_key(path) in self.guide_faces:
                _, crop = self.guide_faces[path_key(path)]
                if crop is not None:
                    pixmap = square_pixmap(crop, GUIDE_ICON_SIZE)
                    spoken += ", showing the face searched for"
                else:
                    pixmap = with_badge(pixmap, warning)
                    tip += "\nNo face found, so this photo is not used"
                    spoken += ", warning: no face found, not used"
            item.setIcon(QIcon(pixmap))
            item.setToolTip(tip + "\n(double-click or press Delete to remove)")
            item.setData(Qt.ItemDataRole.AccessibleTextRole, spoken)

    def remove_guide(self, path):
        for i in range(self.guide_list.count()):
            if same_path(self.guide_list.item(i).data(Qt.ItemDataRole.UserRole), path):
                self.guide_list.takeItem(i)
                announce(self, f"Removed guide image {os.path.basename(path)}")
                break
        self.guide_faces.pop(path_key(path), None)
        self.update_controls()

    def remove_current_guide(self):
        item = self.guide_list.currentItem()
        if item is not None and self.worker is None:
            self.remove_guide(item.data(Qt.ItemDataRole.UserRole))

    def clear_guides(self):
        had = self.guide_list.count()
        self.guide_list.clear()
        self.guide_faces.clear()
        if had:
            announce(self, "Guide images cleared")
        self.update_controls()

    def choose_guides(self):
        patterns = " ".join(f"*{e}" for e in clip_search.IMAGE_EXTENSIONS)
        # Start where guides were last picked from, which is often not the
        # folder being searched.
        start = self.settings.value("guide_folder", "")
        if not os.path.isdir(start):
            start = self.folder.currentText()
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Choose guide images", start,
            f"Images ({patterns});;All files (*)")
        if paths:
            self.settings.setValue("guide_folder", os.path.dirname(paths[0]))
        self.add_guides(paths)

    def find_similar(self, path):
        """Search with path as the only guide and no description."""
        if self.worker is not None:
            return
        self.guide_list.clear()
        self.guide_faces.clear()
        self.prompt_edit.clear()
        self.set_person_mode(False)
        self.add_guides([path])
        self.search()

    def find_person(self, path):
        """Search for the person in path, asking which if there are several."""
        if self.worker is not None:
            return
        self.guide_list.clear()
        self.guide_faces.clear()
        self.prompt_edit.clear()
        self.set_person_mode(True)
        self.add_guides([path])
        self.search()

    def guide_menu(self, item, pos):
        if item is None or self.worker is not None:
            return
        menu = QMenu(self)
        menu.addAction("&Remove", lambda: self.remove_guide(
            item.data(Qt.ItemDataRole.UserRole)))
        menu.exec(pos)

    def result_menu(self, item, pos):
        if item is None or self.worker is not None:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        menu = QMenu(self)
        menu.addAction("&Open", lambda: self.enlarge(item))
        menu.addAction(f"&Describe image	{DESCRIBE_SHORTCUT}",
                       lambda: self.enlarge(item, describe_now=True))
        menu.addAction("Find &similar images", lambda: self.find_similar(path))
        menu.addAction("Find this &person", lambda: self.find_person(path))
        if any(same_path(path, g) for g in self.guides()):
            menu.addAction("&Remove from guide images", lambda: self.remove_guide(path))
        else:
            menu.addAction("&Add to guide images", lambda: self.add_guides([path]))
        menu.exec(pos)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and self.worker is None:
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        self.add_guides([p for p in paths if os.path.splitext(p)[1].lower()
                         in clip_search.IMAGE_EXTENSIONS])

    def update_controls(self):
        """Reflect the mode and whether a search is running. Controls that
        do not apply are disabled rather than hidden, so nothing moves; their
        labels are only quietened, as a disabled label is read out as
        unavailable."""
        busy = self.worker is not None
        person = self.person_mode()
        has_guides = self.guide_list.count() > 0
        has_text = bool(self.prompt_edit.text().strip())
        self.sidebar.setEnabled(not busy)
        self.limit.setEnabled(not busy)

        self.prompt_edit.setPlaceholderText(
            "Optional, e.g. at the beach" if person else "e.g. a dog on the beach")
        self.guide_heading.setText("&Who" if person else "&Looks like")
        self.guide_list.setAccessibleName(
            "Guide photos of the person" if person else "Guide images")
        self.guide_list.set_hint(
            "Drop photos of the person here, or press + to choose them"
            if person else "Drop images here, or press + to choose them (optional)")
        self.clear_guides_button.setEnabled(has_guides)
        self.guide_hint.setVisible(has_guides)
        self.update_guide_items()

        visual = "Person" if person else "Images"
        self.weight_visual.setText(visual)
        self.weight.setAccessibleName(f"Balance of text against {visual.lower()}")
        balance = has_text and has_guides
        self.weight.setEnabled(balance)
        for label in (self.weight_text, self.weight_visual):
            set_tone(label, None if balance else "secondary")
        self.weight.setToolTip(
            f"Text {100 - self.weight.value()}% · {visual} {self.weight.value()}%"
            if balance else "Balances the description against the guide "
            "images when you use both")
        self.weight.setAccessibleDescription(
            f"Text {100 - self.weight.value()} percent, {visual.lower()} "
            f"{self.weight.value()} percent")
        self.strictness.setEnabled(person)
        for label in (self.strictness_label, self.strictness_value):
            set_tone(label, None if person else "secondary")
        self.strictness_value.setText(f"{self.strictness.value() / 100:.2f}")
        self.strictness.setAccessibleDescription(
            f"Similarity of at least {self.strictness.value() / 100:.2f}")

        self.search_button.setEnabled(
            not busy and (has_guides if person else has_text or has_guides))
        self.progress.setVisible(busy)
        self.cancel_button.setVisible(busy)
        self.cancel_button.setEnabled(busy and not (
            self.worker.isInterruptionRequested()))

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Choose image folder", self.folder.currentText())
        if folder:
            self.folder.setCurrentText(os.path.normpath(folder))

    def cancel(self):
        if self.worker is not None and not self.worker.isInterruptionRequested():
            self.worker.requestInterruption()
            self.cancel_button.setEnabled(False)
            self.progress_text = "Cancelling…"
            announce(self, "Cancelling")
            self.update_elapsed()

    def search(self):
        if self.worker is not None:
            return
        folder = self.folder.currentText().strip()
        prompt = self.prompt_edit.text().strip()
        guides = self.guides()
        if not os.path.isdir(folder):
            QMessageBox.warning(self, "Image Inspect", "Choose a folder of images first.")
            return
        if not guides and (self.person_mode() or not prompt):
            return
        self.remember_folder(os.path.normpath(folder))
        self.settings.setValue("recursive", self.recursive.isChecked())
        self.settings.setValue("image_weight", self.weight.value())
        self.settings.setValue("match", MATCH_PERSON if self.person_mode()
                               else MATCH_LOOK)
        self.settings.setValue("face_threshold", self.strictness.value() / 100)
        self.settings.setValue("limit", self.limit.value())

        # Clear the old results straight away, so it is plain that a new
        # search has started even if indexing new files takes a while.
        self.results.clear()
        self.summary.set_status("Searching…")
        self.worker = SearchWorker(self.state, folder, self.recursive.isChecked(),
                                   prompt, guides, self.weight.value() / 100,
                                   self.limit.value(),
                                   MATCH_PERSON if self.person_mode() else MATCH_LOOK,
                                   {k: v[0] for k, v in self.guide_faces.items()},
                                   self.strictness.value() / 100)
        self.pending_choices = []
        self.worker.progress.connect(self.on_progress)
        self.worker.guide_faces.connect(self.on_guide_faces)
        self.worker.finished_ok.connect(self.show_results)
        self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.on_worker_done)
        self.update_controls()
        self.progress_text = "Starting…"
        self.clock.start()
        self.last_spoken = 0
        self.ticker.start()
        self.update_elapsed()
        announce(self, "Searching")
        self.worker.start()

    def on_progress(self, message, done, total):
        if total:
            set_fraction(self.progress, done, total)
            if message.startswith("Downloading"):
                text = f"{message}: {done >> 20} / {total >> 20} MB"
                spoken = f"{message}, {round(100 * done / total)} percent"
            else:
                text = f"{message}: {done:,} / {total:,}"
                spoken = f"{message.split(' (')[0]}, {done:,} of {total:,}"
            # A long job is spoken now and then, not file by file.
            if self.clock.elapsed() - self.last_spoken >= ANNOUNCE_PROGRESS_EVERY:
                self.last_spoken = self.clock.elapsed()
                announce(self, spoken)
        else:
            self.progress.setRange(0, 0)    # busy indicator
            text = message + "…"
        if not self.progress_text.startswith("Cancelling"):
            self.progress_text = text
        self.update_elapsed()

    def update_elapsed(self):
        self.status.setText(self.progress_text)
        self.elapsed.setText(f"{self.clock.elapsed() / 1000:.1f} s")

    def guide_highlight(self):
        """The background marking guide images among the results."""
        color = QColor(self.palette().color(QPalette.ColorRole.Highlight))
        color.setAlpha(70)
        return color

    def recolor_results(self):
        if not hasattr(self, "results"):
            return                          # Still being built.
        for i in range(self.results.count()):
            item = self.results.item(i)
            if item.data(Qt.ItemDataRole.UserRole + 2):
                item.setBackground(self.guide_highlight())

    def show_results(self, results):
        self.results.clear()
        worker = self.worker
        guides = worker.guides
        person = worker.match == MATCH_PERSON
        for n, (path, scores, thumb, face_box) in enumerate(results):
            # One score under the thumbnail: the guide part's when there is
            # one; the full breakdown goes in the tooltip.
            shown = scores.get("person", scores.get("image", scores.get("text")))
            label = f"{shown:.3f}"
            name = os.path.basename(path)
            tip = path + "".join(f"\n{k}: {v:.3f}" for k, v in scores.items())
            spoken = (f"{n + 1}. {name}, "
                      + ", ".join(f"{k} similarity {v:.2f}" for k, v in scores.items()))
            is_guide = any(same_path(path, g) for g in guides)
            if is_guide:
                label = "★ " + label
                tip += "\n(guide image)"
                spoken += ", guide image"
            if face_box is not None:
                spoken += f", matching face at the {describe_place(face_box)}"
            item = QListWidgetItem(
                QIcon(square_pixmap(pixmap_from_bytes(thumb), ICON_SIZE)), label)
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setData(Qt.ItemDataRole.UserRole + 1, face_box)
            item.setData(Qt.ItemDataRole.UserRole + 2, is_guide)
            item.setData(Qt.ItemDataRole.AccessibleTextRole, spoken)
            item.setToolTip(tip)
            if is_guide:
                item.setBackground(self.guide_highlight())
            self.results.addItem(item)
        if self.results.count():
            self.results.setCurrentRow(0)

        count = len(self.state["index"].paths)
        noun = "photo" if len(results) == 1 else "photos"
        if person:
            summary = (f"{len(results)} {noun} of the person"
                       if len(results) < worker.limit
                       else f"The best {len(results)} {noun} of the person")
        else:
            summary = f"The best {len(results)} of {count:,} images"
        if worker.prompt:
            summary += f" matching “{worker.prompt}”"
        notes = []
        if person:
            no_face = sum(1 for p in guides if path_key(p) in worker.known_faces
                          and worker.known_faces[path_key(p)] is None)
            if no_face:
                notes.append(f"{no_face} guide photo{'s' if no_face > 1 else ''} "
                             "with no face ignored")
        if worker.isInterruptionRequested():
            notes.append("indexing was cancelled, so only part of the folder "
                         "was searched")
        if notes:
            self.summary.set_status(f"{summary} ({'; '.join(notes)})", "problem")
        else:
            self.summary.set_status(summary)
        announce(self, self.summary.accessibleName(), important=True)

    def on_guide_faces(self, found):
        for path, faces in found:
            if len(faces) == 1:
                self.guide_faces[path_key(path)] = (
                    faces[0].embedding, pixmap_from_bytes(faces[0].crop))
            elif not faces:
                self.guide_faces[path_key(path)] = (None, None)
            else:
                self.pending_choices.append((path, faces))
        self.update_guide_items()

    def ask_which_faces(self):
        """Ask which person is meant in each guide showing several; False
        if the user cancels."""
        for path, faces in self.pending_choices:
            picker = FacePickerDialog(path, faces, self)
            if not picker.exec() or picker.choice is None:
                return False
            face = faces[picker.choice]
            self.guide_faces[path_key(path)] = (face.embedding,
                                                pixmap_from_bytes(face.crop))
        self.update_guide_items()
        return True

    def on_failed(self, message):
        self.summary.set_status("Search failed", "problem")
        QMessageBox.critical(self, "Image Inspect", message)

    def on_worker_done(self):
        self.ticker.stop()
        seconds = self.clock.elapsed() / 1000
        self.elapsed.setText("")
        cancelled = self.worker.isInterruptionRequested()
        self.status.setText(f"{'Cancelled' if cancelled else 'Done'} "
                            f"in {seconds:.1f} s")
        needs_choice = self.worker.needs_choice
        self.worker = None
        self.update_controls()
        self.update_device()
        self.results.scrollToTop()
        self.prompt_edit.setFocus()
        self.prompt_edit.selectAll()
        if needs_choice:
            # The search stopped to ask which person is meant; carry on
            # once that is known.
            self.summary.set_status("Choose which person to search for")
            if self.ask_which_faces():
                QTimer.singleShot(0, self.search)
            else:
                self.summary.set_status("Search cancelled")
                announce(self, "Search cancelled")

    def describe_current(self):
        item = self.results.currentItem()
        if item is None:
            announce(self, "No photo to describe; search first")
            return
        self.enlarge(item, describe_now=True)

    def update_device(self):
        """Show which hardware the models run on, once one has loaded."""
        if self.state.keys() & {"clip", "faces", "describer"}:
            self.device.setText(f"Running on {models.device()}")

    def enlarge(self, item, describe_now=False):
        if self.enlarging or item is None:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        self.enlarging = True
        try:
            ImageDialog(path, self, on_find_similar=self.find_similar,
                        on_find_person=self.find_person,
                        face_box=item.data(Qt.ItemDataRole.UserRole + 1),
                        descriptions=self.descriptions,
                        describe_now=describe_now).exec()
        except Exception as e:                               # noqa: BLE001
            QMessageBox.warning(self, "Image Inspect", f"Could not open {path}:\n{e}")
        finally:
            self.enlarging = False

    def closeEvent(self, event):
        if self.worker is not None:
            self.worker.requestInterruption()
            self.worker.wait()
        self.descriptions.shutdown()
        super().closeEvent(event)


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--diagnose":
        import diagnose
        diagnose.run(sys.argv[2], sys.argv[3] if len(sys.argv) > 3
                     else "diagnose.txt")
        return
    app = QApplication(sys.argv)
    app.setApplicationName("ImageInspect")
    app.setApplicationVersion(__version__)
    app.setWindowIcon(QIcon(APP_ICON))
    apply_windows_appearance(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
