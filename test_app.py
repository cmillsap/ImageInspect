"""
Tests for the main window's accessibility: spoken names, announcements,
keyboard access and sizing that follows the system font and theme.

The search itself is stood in for; these never load a model.
"""

import os

import numpy as np
import pytest
from PIL import Image

# Qt needs an offscreen platform plugin under CI / headless runs.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QPoint, QSettings, Qt                # noqa: E402
from PyQt6.QtGui import QContextMenuEvent, QKeySequence, QPalette  # noqa: E402
from PyQt6.QtWidgets import (QAbstractButton, QAbstractSpinBox,  # noqa: E402
                             QApplication, QComboBox, QFileDialog, QLabel,
                             QLineEdit, QListWidget, QProgressBar,
                             QScrollArea, QSlider)

import app                                                    # noqa: E402
import describe                                               # noqa: E402
import face_search                                            # noqa: E402
import models                                                 # noqa: E402
import ui_support                                             # noqa: E402
from app import SHORTCUTS, FacePickerDialog, ImageDialog, MainWindow  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    """A single QApplication for the whole test session."""
    application = QApplication.instance() or QApplication([])
    yield application


@pytest.fixture
def settings(tmp_path):
    """Settings in a throwaway file, never the user's real preferences."""
    return QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)


@pytest.fixture
def window(qapp, settings):
    w = MainWindow(settings)
    yield w
    w.close()


@pytest.fixture
def announcements():
    """What the window has asked screen readers to say, as (text, important)."""
    heard = []
    previous = ui_support.set_announcer(
        lambda widget, text, important: heard.append((text, important)))
    yield heard
    ui_support.set_announcer(previous)


@pytest.fixture
def photos(tmp_path):
    """A few small image files."""
    paths = []
    for i, name in enumerate(("beach.jpg", "party.png", "garden.jpg")):
        path = tmp_path / name
        Image.new("RGB", (120, 80), (40 * i, 100, 160)).save(path)
        paths.append(str(path))
    return paths


def settle(qapp):
    for _ in range(3):
        qapp.processEvents()


def face(x1, y1, x2, y2):
    return face_search.Face(np.array([x1, y1, x2, y2], np.float32), 0.9,
                            np.zeros(512, np.float32), None)


class FakeWorker:
    """Enough of a finished SearchWorker for show_results."""

    def __init__(self, guides=(), match=app.MATCH_LOOK, prompt="",
                 known_faces=None, limit=100, cancelled=False):
        self.guides, self.match, self.prompt = list(guides), match, prompt
        self.known_faces = known_faces or {}
        self.limit, self.cancelled = limit, cancelled
        self.needs_choice = False

    def isInterruptionRequested(self):
        return self.cancelled

    def requestInterruption(self):
        self.cancelled = True


class FakeIndex:
    def __init__(self, count):
        self.paths = [f"img{i}.jpg" for i in range(count)]


def show(window, results, **worker):
    """Show results as if a search had just produced them."""
    window.state["index"] = FakeIndex(50)
    window.worker = FakeWorker(**worker)
    window.show_results(results)
    window.worker = None


# ============================================================================
# Screen readers
# ============================================================================

def inputs(window):
    """Every control a screen reader user can land on and operate."""
    found = []
    for widget in window.findChildren((QAbstractSpinBox, QSlider, QComboBox,
                                       QLineEdit, QProgressBar, QListWidget,
                                       QAbstractButton)):
        if isinstance(widget.parentWidget(), (QAbstractSpinBox, QComboBox,
                                              QLineEdit)):
            continue  # Parts of a spin box, combo box or line edit (its
            # clear button), named by their owner or by Qt.
        if isinstance(widget, QAbstractButton) and any(
                c.isalpha() for c in widget.text()):
            continue  # A button's own words name it.
        found.append(widget)
    return found


class TestScreenReaders:

    def test_every_input_has_a_name(self, window):
        unnamed = [type(w).__name__ for w in inputs(window)
                   if not w.accessibleName()]
        assert unnamed == []

    def test_names_tell_inputs_apart(self, window):
        names = [w.accessibleName() for w in inputs(window)]
        assert len(names) == len(set(names))

    def test_symbol_buttons_are_named_in_words(self, window):
        assert window.browse_button.accessibleName() == "Choose folder"
        assert window.add_guide_button.accessibleName() == "Add guide images"

    def test_names_follow_the_mode(self, window):
        window.set_person_mode(True)
        assert window.guide_list.accessibleName() == "Guide photos of the person"
        assert "person" in window.weight.accessibleName()
        window.set_person_mode(False)
        assert window.guide_list.accessibleName() == "Guide images"
        assert "images" in window.weight.accessibleName()

    def test_empty_guide_list_describes_how_to_add(self, window):
        assert "press +" in window.guide_list.accessibleDescription()

    def test_sliders_describe_their_value(self, window):
        window.strictness.setValue(40)
        assert "0.40" in window.strictness.accessibleDescription()
        window.weight.setValue(70)
        assert "Text 30 percent" in window.weight.accessibleDescription()

    def test_labels_of_unused_controls_stay_enabled(self, window):
        # Disabled labels are read as "unavailable"; they are only quietened.
        window.set_person_mode(False)
        assert not window.strictness.isEnabled()
        for label in (window.strictness_label, window.strictness_value,
                      window.weight_text, window.weight_visual):
            assert label.isEnabled()
            assert label.property("tone") == "secondary"

    def test_result_items_are_spoken_in_full(self, window):
        show(window, [("C:/p/beach.jpg", {"person": 0.81}, b"",
                       np.array([0.0, 0.0, 0.2, 0.2])),
                      ("C:/p/party.png", {"person": 0.52}, b"", None)],
             guides=["C:/p/beach.jpg"], match=app.MATCH_PERSON)
        first = window.results.item(0).data(Qt.ItemDataRole.AccessibleTextRole)
        assert first.startswith("1. beach.jpg, person similarity 0.81")
        assert "guide image" in first
        assert "matching face at the top left" in first
        second = window.results.item(1).data(Qt.ItemDataRole.AccessibleTextRole)
        assert second == "2. party.png, person similarity 0.52"

    def test_both_scores_are_spoken(self, window):
        show(window, [("C:/p/beach.jpg", {"text": 0.25, "image": 0.71}, b"", None)],
             guides=["C:/p/x.jpg"], prompt="sea")
        spoken = window.results.item(0).data(Qt.ItemDataRole.AccessibleTextRole)
        assert "text similarity 0.25" in spoken
        assert "image similarity 0.71" in spoken

    def test_guide_items_are_spoken(self, window, photos):
        window.add_guides(photos[:2])
        window.set_person_mode(True)
        window.guide_faces[app.path_key(photos[1])] = (None, None)
        window.update_guide_items()
        first = window.guide_list.item(0).data(Qt.ItemDataRole.AccessibleTextRole)
        second = window.guide_list.item(1).data(Qt.ItemDataRole.AccessibleTextRole)
        assert first == "Guide image beach.jpg"
        assert second == "Guide image party.png, warning: no face found, not used"

    def test_warnings_in_the_summary_are_spoken_as_warnings(self, window):
        show(window, [], guides=["C:/p/a.jpg"], match=app.MATCH_PERSON,
             known_faces={app.path_key("C:/p/a.jpg"): None})
        assert window.summary.accessibleName().startswith("Warning: ")
        assert "with no face ignored" in window.summary.accessibleName()


class TestDescribePlace:

    @pytest.mark.parametrize("box, words", [
        ((0.0, 0.0, 0.2, 0.2), "top left"),
        ((0.4, 0.4, 0.6, 0.6), "centre"),
        ((0.8, 0.1, 1.0, 0.2), "top right"),
        ((0.4, 0.8, 0.6, 1.0), "bottom"),
        ((0.0, 0.4, 0.2, 0.6), "left"),
    ])
    def test_places(self, box, words):
        assert app.describe_place(box) == words


class TestFacePicker:

    @pytest.fixture
    def picker(self, qapp):
        # Listed by detector confidence, not position.
        faces = [face(0.7, 0.3, 0.9, 0.6), face(0.05, 0.1, 0.15, 0.25),
                 face(0.4, 0.3, 0.6, 0.7)]
        dialog = FacePickerDialog("C:/p/group.jpg", faces)
        yield dialog
        dialog.close()

    def test_faces_are_named_left_to_right(self, picker):
        names = [b.accessibleName() for b in picker.buttons]
        assert names == ["Person 1 of 3, top left",
                         "Person 2 of 3, centre, largest face",
                         "Person 3 of 3, right"]

    def test_choosing_returns_the_original_face(self, picker):
        picker.buttons[0].click()
        assert picker.choice == 1           # The second face given.

    def test_starts_on_the_largest_face(self, qapp, picker):
        picker.show()
        settle(qapp)
        assert picker.focusWidget() is picker.buttons[1]


class TestImageDialog:

    def test_photo_and_matched_face_are_described(self, qapp, photos):
        dialog = ImageDialog(photos[0], face_box=(0.7, 0.0, 0.9, 0.3))
        assert dialog.image.accessibleName() == "Photo beach.jpg"
        assert dialog.image.accessibleDescription() == \
            "The matching face is boxed, at the top right"
        dialog.close()

    def test_no_face_no_description(self, qapp, photos):
        dialog = ImageDialog(photos[0])
        assert dialog.image.accessibleDescription() == ""
        dialog.close()


# ============================================================================
# Announcements
# ============================================================================

class TestAnnouncements:

    def test_nothing_is_said_on_startup(self, qapp, settings, announcements):
        MainWindow(settings).close()
        assert announcements == []

    def test_adding_and_removing_guides(self, window, photos, announcements):
        window.add_guides(photos[:2])
        assert announcements[-1] == ("2 guide images added, 2 in all", False)
        window.remove_guide(photos[0])
        assert announcements[-1] == ("Removed guide image beach.jpg", False)
        window.clear_guides()
        assert announcements[-1] == ("Guide images cleared", False)

    def test_adding_a_duplicate_says_nothing(self, window, photos, announcements):
        window.add_guides(photos[:1])
        announcements.clear()
        window.add_guides(photos[:1])
        assert announcements == []

    def test_results_are_announced_and_can_interrupt(self, window, announcements):
        show(window, [("C:/p/a.jpg", {"text": 0.3}, b"", None)], prompt="dog")
        assert announcements[-1] == (
            "The best 1 of 50 images matching \u201cdog\u201d", True)

    def test_person_results(self, window, announcements):
        show(window, [("C:/p/a.jpg", {"person": 0.6}, b"", None)] * 3,
             guides=["C:/p/g.jpg"], match=app.MATCH_PERSON)
        assert announcements[-1] == ("3 photos of the person", True)

    def test_cancelled_search_is_a_warning(self, window, announcements):
        show(window, [], prompt="dog", cancelled=True)
        text, important = announcements[-1]
        assert text.startswith("Warning: ") and "cancelled" in text
        assert important

    def test_progress_is_spoken_now_and_then(self, window, announcements):
        window.clock.start()
        window.last_spoken = -app.ANNOUNCE_PROGRESS_EVERY
        window.on_progress("Indexing new images (a.jpg)", 10, 300)
        window.on_progress("Indexing new images (b.jpg)", 11, 300)
        assert announcements == [("Indexing new images, 10 of 300", False)]

    def test_busy_steps_are_not_spoken(self, window, announcements):
        window.clock.start()
        window.last_spoken = -app.ANNOUNCE_PROGRESS_EVERY
        window.on_progress("Loading CLIP model", 0, 0)
        assert announcements == []

    def test_cancel(self, window, announcements):
        window.worker = FakeWorker()
        window.update_controls()
        window.cancel()
        window.cancel()                     # A second press says nothing more.
        assert announcements == [("Cancelling", False)]
        assert window.worker.cancelled
        window.worker = None


# ============================================================================
# Keyboard
# ============================================================================

def tab_stops(window):
    """Visible widgets that Tab lands on, in the order it visits them."""
    start = window.folder
    stops, widget = [], start
    while True:
        if widget.isVisible() and (widget.focusPolicy()
                                   & Qt.FocusPolicy.TabFocus):
            stops.append(widget)
        widget = widget.nextInFocusChain()
        if widget is start:
            return stops


def mnemonic(text):
    """The Alt access key marked in text with a single &, else None."""
    i = text.replace("&&", "").find("&")
    return text.replace("&&", "")[i + 1].lower() if i >= 0 else None


class TestKeyboard:

    def test_tab_follows_the_screen_top_to_bottom(self, qapp, window, photos):
        window.add_guides(photos[:1])
        window.show()
        settle(qapp)
        expected = [window.folder, window.browse_button, window.recursive,
                    window.look_button, window.person_button,
                    window.prompt_edit, window.add_guide_button,
                    window.guide_list, window.clear_guides_button,
                    window.weight, window.strictness, window.search_button,
                    window.limit, window.results]
        stops = tab_stops(window)
        assert [w for w in stops if w in expected] == expected

    def test_scroll_area_is_not_a_tab_stop(self, qapp, window):
        window.show()
        settle(qapp)
        assert window.findChild(QScrollArea) not in tab_stops(window)

    def test_shortcut_keys(self, window):
        for name, keys in SHORTCUTS.items():
            bound = window.shortcut_actions[name].shortcuts()
            assert bound == [QKeySequence(k) for k in keys]

    def test_tooltips_mention_the_shortcut(self, window):
        assert "Ctrl+O" in window.browse_button.toolTip()
        assert "Ctrl+Shift+O" in window.add_guide_button.toolTip()
        assert "Ctrl+Return" in window.search_button.toolTip()

    def test_choose_folder_shortcut(self, window, monkeypatch):
        opened = []
        monkeypatch.setattr(QFileDialog, "getExistingDirectory",
                            lambda *a, **k: opened.append(a) or "")
        window.shortcut_actions["choose_folder"].trigger()
        assert opened

    def test_add_guides_shortcut(self, window, monkeypatch):
        opened = []
        monkeypatch.setattr(QFileDialog, "getOpenFileNames",
                            lambda *a, **k: opened.append(a) or ([], ""))
        window.shortcut_actions["add_guides"].trigger()
        assert opened

    def test_search_shortcut_only_when_ready(self, window, monkeypatch, tmp_path):
        started = []
        monkeypatch.setattr(app, "SearchWorker", lambda *a: started.append(a)
                            or pytest.fail("search should not have started"))
        window.shortcut_actions["search"].trigger()
        assert started == []                # Nothing to search for yet.

    def test_escape_cancels(self, window):
        window.worker = FakeWorker()
        window.update_controls()
        window.shortcut_actions["cancel"].trigger()
        assert window.worker.cancelled
        window.worker = None

    def test_enter_opens_a_result(self, window, monkeypatch):
        opened = []
        monkeypatch.setattr(app, "ImageDialog", lambda path, *a, **k:
                            opened.append(path) or type("D", (), {"exec": lambda s: 0})())
        show(window, [("C:/p/a.jpg", {"text": 0.3}, b"", None)], prompt="dog")
        window.results.itemActivated.emit(window.results.currentItem())
        assert opened == ["C:/p/a.jpg"]

    def test_first_result_is_current(self, window):
        show(window, [("C:/p/a.jpg", {"text": 0.3}, b"", None),
                      ("C:/p/b.jpg", {"text": 0.2}, b"", None)], prompt="dog")
        assert window.results.currentRow() == 0

    def test_menu_key_opens_the_menu_on_the_current_result(self, qapp, window):
        show(window, [("C:/p/a.jpg", {"text": 0.3}, b"", None),
                      ("C:/p/b.jpg", {"text": 0.2}, b"", None)], prompt="dog")
        window.results.setCurrentRow(1)
        asked = []
        window.results.menu_requested.disconnect()
        window.results.menu_requested.connect(lambda item, pos: asked.append(item))
        event = QContextMenuEvent(QContextMenuEvent.Reason.Keyboard, QPoint(0, 0))
        window.results.contextMenuEvent(event)
        assert asked == [window.results.item(1)]

    def test_delete_removes_the_current_guide(self, window, photos):
        window.add_guides(photos[:2])
        window.guide_list.setCurrentRow(0)
        window.guide_list.actions()[0].trigger()
        assert window.guides() == [os.path.normpath(photos[1])]

    @pytest.mark.parametrize("person", [False, True])
    def test_access_keys_are_unique(self, window, person):
        window.set_person_mode(person)
        texts = [b.text() for b in window.findChildren(QAbstractButton)]
        texts += [label.text() for label in window.findChildren(QLabel)
                  if label.buddy() is not None]
        keys = [k for k in map(mnemonic, texts) if k]
        assert len(keys) == len(set(keys)), sorted(keys)
        assert len(keys) >= 12

    def test_access_keys_on_labels_reach_an_input(self, window):
        for label in window.findChildren(QLabel):
            if mnemonic(label.text()):
                assert label.buddy() is not None, label.text()


# ============================================================================
# Sizing and theme
# ============================================================================

def set_point_size(qapp, window, point_size):
    """Stand in for the user's system text size."""
    font = window.font()
    font.setPointSize(point_size)
    window.setFont(font)
    window.show()
    settle(qapp)


class TestSizing:

    def test_window_fits_the_screen(self, window):
        available = window.screen().availableGeometry()
        assert available.contains(window.geometry())

    def test_headings_follow_the_font(self, qapp, window):
        heading = window.guide_heading
        set_point_size(qapp, window, 9)
        small = heading.font().pointSizeF()
        set_point_size(qapp, window, 18)
        assert heading.font().pointSizeF() == pytest.approx(2 * small)
        assert heading.font().bold()

    @pytest.mark.parametrize("point_size", [9, 14, 20])
    def test_sidebar_controls_are_not_clipped(self, qapp, window, point_size):
        set_point_size(qapp, window, point_size)
        clipped = [w.text() if hasattr(w, "text") else type(w).__name__
                   for w in window.sidebar.findChildren((QLabel, QAbstractButton,
                                                         QComboBox, QSlider))
                   if w.isVisible() and not isinstance(w, ui_support.HintLabel)
                   and w.width() < w.minimumSizeHint().width()]
        assert clipped == []

    def test_sidebar_widens_with_the_font(self, qapp, window):
        set_point_size(qapp, window, 9)
        narrow = window.side_scroll.width()
        set_point_size(qapp, window, 18)
        assert window.side_scroll.width() > narrow


class TestTheme:

    def test_no_hard_coded_styles(self, window):
        for widget in [window, *window.findChildren(QLabel),
                       *window.findChildren(QAbstractButton)]:
            assert widget.styleSheet() == ""

    def test_guide_results_use_the_theme_highlight(self, window):
        show(window, [("C:/p/a.jpg", {"image": 0.9}, b"", None)],
             guides=["C:/p/a.jpg"])
        highlight = window.palette().color(QPalette.ColorRole.Highlight)
        background = window.results.item(0).background().color()
        assert background.rgb() == highlight.rgb()

    def test_guide_results_follow_a_new_palette(self, window):
        show(window, [("C:/p/a.jpg", {"image": 0.9}, b"", None)],
             guides=["C:/p/a.jpg"])
        palette = QPalette(window.palette())
        palette.setColor(QPalette.ColorRole.Highlight, Qt.GlobalColor.darkGreen)
        window.setPalette(palette)
        background = window.results.item(0).background().color()
        assert background.rgb() == palette.color(QPalette.ColorRole.Highlight).rgb()


class TestWindowsAppearance:

    @pytest.fixture
    def font_size(self, qapp):
        before = qapp.font()
        font = qapp.font()
        font.setPointSizeF(9)
        qapp.setFont(font)
        yield
        qapp.setFont(before)

    @pytest.fixture
    def on_windows(self, monkeypatch):
        monkeypatch.setattr(ui_support.sys, "platform", "win32")

    def test_text_size_scales_the_font(self, qapp, font_size, on_windows,
                                       monkeypatch):
        monkeypatch.setattr(ui_support, "text_scale_factor", lambda: 1.5)
        ui_support.apply_windows_appearance(qapp)
        assert qapp.font().pointSizeF() == pytest.approx(13.5)

    def test_already_scaled_font_is_left_alone(self, qapp, font_size,
                                               on_windows, monkeypatch):
        font = qapp.font()
        font.setPointSizeF(13.5)
        qapp.setFont(font)
        monkeypatch.setattr(ui_support, "text_scale_factor", lambda: 1.5)
        ui_support.apply_windows_appearance(qapp)
        assert qapp.font().pointSizeF() == pytest.approx(13.5)

    def test_normal_text_size_changes_nothing(self, qapp, font_size,
                                              on_windows, monkeypatch):
        monkeypatch.setattr(ui_support, "text_scale_factor", lambda: 1.0)
        ui_support.apply_windows_appearance(qapp)
        assert qapp.font().pointSizeF() == pytest.approx(9)


# ============================================================================
# Describing photos
# ============================================================================

class FakeService:
    """A DescriptionService that only records what it is asked."""

    def __init__(self):
        self.started, self.stopped = [], []

    def start(self, path, view):
        self.started.append(path)
        view.show_started()
        return True

    def stop(self, view):
        self.stopped.append(view)


@pytest.fixture
def described(qapp, photos):
    """An enlarged view that can describe its photo."""
    service = FakeService()
    dialog = ImageDialog(photos[0], descriptions=service)
    dialog.service = service
    yield dialog
    dialog.close()


@pytest.fixture
def service(window, tmp_path, monkeypatch):
    """The window's DescriptionService, with its cache in a scratch file and
    the model never downloaded or run."""
    window.state["descriptions"] = describe.DescriptionCache(str(tmp_path / "db"))
    monkeypatch.setattr(describe, "preferred_variant", lambda: "cuda")
    runs = []
    monkeypatch.setattr(window.descriptions, "run",
                        lambda path, variant, view: runs.append((path, variant)))
    window.descriptions.runs = runs
    return window.descriptions


class TestDescribing:

    def test_describe_button_opens_the_panel(self, described, photos):
        assert described.panel.isHidden()
        described.describe_button.click()
        assert not described.panel.isHidden()
        assert described.service.started == [photos[0]]

    def test_describing_twice_at_once_starts_once(self, described):
        described.describe()
        described.describe()
        assert len(described.service.started) == 1

    def test_ctrl_d_describes(self, described):
        action = next(a for a in described.actions()
                      if a.shortcut() == QKeySequence(app.DESCRIBE_SHORTCUT))
        action.trigger()
        assert len(described.service.started) == 1

    def test_closing_stops_the_description(self, described):
        described.describe()
        described.reject()
        assert described.service.stopped == [described.panel]

    def test_stop_button(self, described, announcements):
        described.describe()
        described.panel.stop_button.click()
        assert described.service.stopped == [described.panel]
        assert announcements[-1] == ("Stopped", False)
        assert described.panel.stop_button.isHidden()

    def test_access_keys_are_unique(self, described):
        described.describe()
        texts = [b.text() for b in described.findChildren(QAbstractButton)]
        texts += [label.text() for label in described.findChildren(QLabel)
                  if label.buddy() is not None]
        keys = [k for k in map(mnemonic, texts) if k]
        assert len(keys) == len(set(keys)), sorted(keys)

    def test_without_a_service_there_is_no_describe_button(self, qapp, photos):
        dialog = ImageDialog(photos[0])
        assert not hasattr(dialog, "describe_button")
        dialog.close()


class TestDescriptionPanel:

    @pytest.fixture
    def panel(self, qapp):
        panel = app.DescriptionPanel()
        yield panel
        panel.close()

    def test_text_box_is_named(self, panel):
        assert panel.text.accessibleName() == "Description"
        assert panel.heading.buddy() is panel.text

    def test_starting_is_announced(self, panel, announcements):
        panel.show_started()
        assert announcements == [("Describing the photo", False)]
        assert not panel.copy_button.isEnabled()

    def test_the_description_is_read_out_when_done(self, panel, announcements,
                                                   monkeypatch):
        monkeypatch.setattr(models, "device", lambda: "GPU (Test GPU)")
        panel.show_started()
        panel.show_text("A dog")
        assert len(announcements) == 1      # Not word by word.
        panel.show_done("A dog on a beach.")
        assert announcements[-1] == ("Description: A dog on a beach.", False)
        assert panel.text.toPlainText() == "A dog on a beach."
        assert panel.status.accessibleName() == \
            "Done: Written by Phi-3.5-vision on the GPU (Test GPU)"
        assert panel.copy_button.isEnabled()

    def test_failure_is_an_important_warning(self, panel, announcements):
        panel.show_failed("out of memory")
        text, important = announcements[-1]
        assert text.startswith("Warning: ") and "out of memory" in text
        assert important

    def test_download_progress_is_spoken_now_and_then(self, panel, announcements):
        panel.show_started()
        panel.last_spoken = -app.ANNOUNCE_PROGRESS_EVERY
        panel.show_progress("Downloading Phi-3.5-vision (2.6 GB)", 1 << 30, 2 << 30)
        panel.show_progress("Downloading Phi-3.5-vision (2.6 GB)", 1 << 30, 2 << 30)
        assert announcements[1:] == [("Downloading Phi-3.5-vision, 50 percent", False)]
        assert panel.status.text() == \
            "Downloading Phi-3.5-vision (2.6 GB): 1,024 / 2,048 MB"

    def test_copy(self, qapp, panel, announcements):
        panel.show_done("A cat.")
        panel.copy_button.click()
        assert qapp.clipboard().text() == "A cat."
        assert announcements[-1] == ("Description copied", False)


class TestDescriptionService:

    def test_earlier_descriptions_need_no_model(self, service, photos, monkeypatch):
        monkeypatch.setattr(app, "agree_to_download", lambda *a, **k: 1 / 0)
        service.cache().put(photos[0], "Earlier.")
        panel = app.DescriptionPanel()
        assert service.start(photos[0], panel)
        assert panel.text.toPlainText() == "Earlier."
        assert service.runs == []

    def test_declining_the_download(self, service, photos, monkeypatch):
        monkeypatch.setattr(app, "agree_to_download", lambda *a, **k: False)
        panel = app.DescriptionPanel()
        assert not service.start(photos[0], panel)
        assert service.runs == []
        assert "not downloaded" in panel.status.text()

    def test_agreeing_starts_the_preferred_build(self, service, photos, monkeypatch):
        asked = []
        monkeypatch.setattr(app, "agree_to_download",
                            lambda parent, variant, **k: asked.append(variant) or True)
        service.start(photos[0], app.DescriptionPanel())
        assert asked == ["cuda"]
        assert service.runs == [(photos[0], "cuda")]

    def test_a_loaded_model_is_not_asked_about(self, service, photos, monkeypatch):
        monkeypatch.setattr(app, "agree_to_download", lambda *a, **k: 1 / 0)
        service.state["describer"] = type("D", (), {"variant": "cpu"})()
        service.start(photos[0], app.DescriptionPanel())
        assert service.runs == [(photos[0], "cpu")]

    def test_gpu_failure_offers_the_cpu_build(self, service, photos, monkeypatch):
        asked = []
        monkeypatch.setattr(app, "agree_to_download", lambda parent, variant, **k:
                            asked.append((variant, k)) or True)
        panel = app.DescriptionPanel()
        worker = app.DescribeWorker(service.state, photos[0], "cuda")
        worker.view, worker.needs_cpu_model = panel, True
        service.worker = worker
        service.on_worker_done()
        assert asked == [("cpu", {"gpu_failed": True})]
        assert service.runs == [(photos[0], "cpu")]
        assert service.cpu_only

    def test_a_new_request_waits_for_the_last_to_stop(self, service, photos,
                                                      monkeypatch):
        monkeypatch.setattr(app, "agree_to_download", lambda *a, **k: True)
        old, new = app.DescriptionPanel(), app.DescriptionPanel()
        worker = app.DescribeWorker(service.state, photos[1], "cuda")
        worker.view = old
        service.worker = worker
        service.start(photos[0], new)
        assert worker.view is None          # Hears no more from the old one.
        assert service.runs == []
        service.on_worker_done()
        assert service.runs == [(photos[0], "cuda")]


class TestDownloadQuestion:

    def test_gpu(self, monkeypatch):
        monkeypatch.setattr(models, "nvidia_gpu", lambda: "Test GPU")
        text, details = app.download_question("cuda", 2_600_000_000)
        assert text == "Describing photos needs Phi-3.5-vision, a 2.6 GB download."
        assert "graphics card (Test GPU)" in details
        assert "not sent anywhere" in details

    def test_cpu(self):
        _, details = app.download_question("cpu", 3_200_000_000)
        assert "half a minute" in details

    def test_after_the_gpu_failed(self):
        _, details = app.download_question("cpu", 3_200_000_000, gpu_failed=True)
        assert details.startswith("It runs on this computer")
        assert "could not run" in details

    def test_not_enough_space(self, qapp, monkeypatch):
        warned = []
        monkeypatch.setattr(describe, "download_size", lambda v: 3_000_000_000)
        monkeypatch.setattr(describe, "free_space", lambda: 2_000_000_000)
        monkeypatch.setattr(app.QMessageBox, "warning",
                            lambda parent, title, text: warned.append(text))
        assert not app.agree_to_download(None, "cpu")
        assert "only 2.0 GB free" in warned[0]

    def test_nothing_to_download(self, monkeypatch):
        monkeypatch.setattr(describe, "download_size", lambda v: 0)
        assert app.agree_to_download(None, "cpu")


class TestDescribeFromTheWindow:

    def test_ctrl_d_describes_the_current_result(self, window, monkeypatch):
        opened = []
        monkeypatch.setattr(window, "enlarge",
                            lambda item, describe_now=False: opened.append(
                                (item.data(Qt.ItemDataRole.UserRole), describe_now)))
        show(window, [("C:/p/a.jpg", {"text": 0.3}, b"", None)], prompt="dog")
        assert window.describe_action.shortcut() == \
            QKeySequence(app.DESCRIBE_SHORTCUT)
        window.describe_action.trigger()
        assert opened == [("C:/p/a.jpg", True)]

    def test_progress_past_2_gb(self, window):
        window.clock.start()
        window.on_progress("Downloading model", 3 << 30, 4 << 30)
        assert window.progress.value() == 750

    def test_ctrl_d_without_results_says_why(self, window, announcements):
        window.describe_action.trigger()
        assert announcements[-1] == ("No photo to describe; search first", False)

    def test_device_is_shown_once_a_model_has_loaded(self, window, monkeypatch):
        monkeypatch.setattr(models, "device", lambda: "GPU (Test GPU)")
        window.update_device()
        assert window.device.text() == ""
        window.state["clip"] = object()
        window.update_device()
        assert window.device.text() == "Running on GPU (Test GPU)"
