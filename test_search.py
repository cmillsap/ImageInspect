"""
Tests for ranking and face bookkeeping that need no model: combining scores,
the person cut-off, per-photo best faces, and the alignment geometry.
"""

import numpy as np
import pytest
from PIL import Image

import clip_search
import face_search


def unit(*values):
    v = np.array(values, np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def index(tmp_path):
    """A CLIP index of three images with hand-made 2-d embeddings."""
    idx = clip_search.Index(str(tmp_path / "index.sqlite"))
    idx.paths = ["a.jpg", "b.jpg", "c.jpg"]
    idx.embeddings = np.stack([unit(1, 0), unit(1, 1), unit(0, 1)])
    return idx


class TestIndexSearch:

    def test_text_alone_ranks_by_similarity(self, index):
        results = index.search(unit(0, 1), None, 0.5, 10)
        assert [p for p, _ in results] == ["c.jpg", "b.jpg", "a.jpg"]
        assert set(results[0][1]) == {"text"}

    def test_person_scores_replace_the_guide_images(self, index):
        person = np.array([0.2, 0.7, 0.5], np.float32)
        results = index.search(None, np.stack([unit(1, 0)]), 0.5, 10,
                               person=person, person_threshold=0.3)
        assert [p for p, _ in results] == ["b.jpg", "c.jpg"]
        assert results[0][1] == {"person": pytest.approx(0.7)}

    def test_person_cut_off_applies_with_text_too(self, index):
        person = np.array([0.9, 0.1, 0.4], np.float32)
        results = index.search(unit(0, 1), None, 0.5, 10,
                               person=person, person_threshold=0.3)
        assert {p for p, _ in results} == {"a.jpg", "c.jpg"}
        assert set(results[0][1]) == {"text", "person"}

    def test_limit(self, index):
        assert len(index.search(unit(1, 0), None, 0.5, 2)) == 2


@pytest.fixture
def faces(tmp_path):
    """Two photos with faces: one with two people, one with one."""
    idx = face_search.FaceIndex(str(tmp_path / "index.sqlite"))
    idx.paths = ["group.jpg", "solo.jpg"]
    idx.embeddings = np.stack([unit(1, 0), unit(0, 1), unit(1, 1)])
    idx.boxes = np.array([[0, 0, .1, .1], [.5, .5, .6, .6], [.2, .2, .4, .4]],
                         np.float32)
    idx.starts = np.array([0, 2, 3])
    idx._row = {p: i for i, p in enumerate(idx.paths)}
    return idx


class TestFaceIndex:

    def test_a_photo_scores_its_best_face(self, faces):
        scores = faces.person_scores(unit(0, 1), ["solo.jpg", "group.jpg"])
        assert scores == pytest.approx([unit(1, 1)[1], 1.0])

    def test_photos_without_faces_score_minus_one(self, faces):
        assert faces.person_scores(unit(0, 1), ["empty.jpg"]) == [-1]

    def test_best_box_is_the_matching_face(self, faces):
        assert list(faces.best_box(unit(0, 1), "group.jpg")) == \
            pytest.approx([.5, .5, .6, .6])
        assert faces.best_box(unit(0, 1), "empty.jpg") is None

    def test_template_is_the_normalised_mean(self):
        template = face_search.person_template(np.stack([unit(1, 0), unit(0, 1)]))
        assert template == pytest.approx(unit(1, 1))


class TestGeometry:

    def test_nms_drops_overlapping_boxes(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]],
                         np.float32)
        keep = face_search._nms(boxes, np.array([0.9, 0.8, 0.7]))
        assert keep == [0, 2]

    def test_align_maps_landmarks_onto_the_template(self):
        # A face whose landmarks already sit on ArcFace's template, shifted
        # and doubled in size, comes back as the template crop.
        image = Image.new("RGB", (400, 400))
        pixels = np.asarray(image).copy()
        landmarks = face_search.ARCFACE_LANDMARKS * 2 + 50
        for x, y in landmarks.round().astype(int):
            pixels[y - 2:y + 3, x - 2:x + 3] = 255
        aligned = face_search.align(Image.fromarray(pixels), landmarks)
        assert aligned.shape == (112, 112, 3)
        for x, y in face_search.ARCFACE_LANDMARKS.round().astype(int):
            assert aligned[y, x].max() > 128

    def test_detector_input_keeps_the_aspect_ratio(self):
        pixels, scale = face_search.detector_input(Image.new("RGB", (1280, 640)))
        assert pixels.shape == (640, 640, 3)
        assert scale == pytest.approx(0.5)
