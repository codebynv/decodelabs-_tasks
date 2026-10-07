"""Unit + integration tests for the Project 4 OCR pipeline.

Run with:  python tests/test_ocr_pipeline.py
       or: python -m pytest tests/

The integration tests at the bottom need the external Tesseract engine; they
are skipped automatically (with a clear reason) when it is not installed.
No test ever fabricates an OCR result.
"""

import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from src import ocr_pipeline  # noqa: E402

SAMPLE_IMAGE = (
    Path(__file__).resolve().parent.parent / "data" / "sample" / "sample_text.png"
)


def tesseract_available() -> bool:
    try:
        ocr_pipeline.ensure_tesseract()
    except RuntimeError:
        return False
    return True


HAS_TESSERACT = tesseract_available()
SKIP_REASON = (
    "Tesseract engine is not installed/configured - integration test skipped "
    "(pytesseract is only a wrapper, see README 'Tesseract Installation')"
)


def make_text_image(width: int = 600, height: int = 300) -> np.ndarray:
    """Small synthetic image with a black bar, used for deskew tests."""
    image = np.full((height, width), 255, np.uint8)
    cv2.rectangle(image, (60, 120), (width - 60, 180), 0, -1)
    return image


def rotate(image: np.ndarray, degrees: float) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
    return cv2.warpAffine(
        image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=255
    )


class TestLoadImage(unittest.TestCase):
    def test_loads_the_shipped_sample(self):
        # 1. Image loading: valid image is decoded as a colour array.
        image = ocr_pipeline.load_image(SAMPLE_IMAGE)
        self.assertEqual(image.shape, (420, 900, 3))
        self.assertEqual(image.dtype, np.uint8)

    def test_missing_path_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            ocr_pipeline.load_image("does/not/exist.png")

    def test_non_image_file_raises_value_error(self):
        with tempfile.TemporaryDirectory() as folder:
            fake = Path(folder) / "fake.png"
            fake.write_text("this is not an image", encoding="utf-8")
            with self.assertRaises(ValueError):
                ocr_pipeline.load_image(fake)

    def test_directory_path_raises_value_error(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                ocr_pipeline.load_image(folder)


class TestPreprocessing(unittest.TestCase):
    # 2. Pre-processing integrity: grayscale conversion.
    def test_convert_to_grayscale_collapses_channels(self):
        image = ocr_pipeline.load_image(SAMPLE_IMAGE)
        gray = ocr_pipeline.convert_to_grayscale(image)
        self.assertEqual(gray.ndim, 2)
        self.assertEqual(gray.shape, (420, 900))
        self.assertEqual(gray.dtype, np.uint8)

    def test_convert_to_grayscale_accepts_gray_input(self):
        gray = np.full((10, 10), 7, np.uint8)
        np.testing.assert_array_equal(ocr_pipeline.convert_to_grayscale(gray), gray)

    def test_convert_to_grayscale_rejects_empty_image(self):
        with self.assertRaises(ValueError):
            ocr_pipeline.convert_to_grayscale(np.array([]))

    def test_gaussian_blur_keeps_shape_and_smooths_noise(self):
        rng = np.random.default_rng(0)
        noisy = rng.integers(0, 256, size=(120, 160), dtype=np.uint8)
        blurred = ocr_pipeline.apply_gaussian_blur(noisy)
        self.assertEqual(blurred.shape, noisy.shape)
        self.assertEqual(blurred.dtype, np.uint8)
        self.assertFalse(np.array_equal(blurred, noisy))
        # Noise must decrease, not increase.
        self.assertLess(blurred.std(), noisy.std())

    def test_gaussian_blur_rejects_colour_input(self):
        with self.assertRaises(ValueError):
            ocr_pipeline.apply_gaussian_blur(np.zeros((5, 5, 3), np.uint8))

    # 2. Adaptive thresholding produces a true binary image.
    def test_adaptive_threshold_is_binary(self):
        image = ocr_pipeline.load_image(SAMPLE_IMAGE)
        binary = ocr_pipeline.apply_adaptive_threshold(
            ocr_pipeline.apply_gaussian_blur(ocr_pipeline.convert_to_grayscale(image))
        )
        self.assertTrue(set(np.unique(binary)).issubset({0, 255}))

    def test_adaptive_threshold_rejects_even_block_size(self):
        gray = np.full((50, 50), 128, np.uint8)
        with self.assertRaises(ValueError):
            ocr_pipeline.apply_adaptive_threshold(gray, block_size=30)
        with self.assertRaises(ValueError):
            ocr_pipeline.apply_adaptive_threshold(gray, block_size=1)

    def test_adaptive_threshold_rejects_colour_input(self):
        with self.assertRaises(ValueError):
            ocr_pipeline.apply_adaptive_threshold(np.zeros((5, 5, 3), np.uint8))

    def test_preprocess_image_returns_binary_same_shape(self):
        image = ocr_pipeline.load_image(SAMPLE_IMAGE)
        processed = ocr_pipeline.preprocess_image(image)
        self.assertEqual(processed.shape[:2], image.shape[:2])
        self.assertTrue(set(np.unique(processed)).issubset({0, 255}))

    def test_preprocess_image_is_deterministic(self):
        image = ocr_pipeline.load_image(SAMPLE_IMAGE)
        first = ocr_pipeline.preprocess_image(image)
        second = ocr_pipeline.preprocess_image(image)
        np.testing.assert_array_equal(first, second)

    def test_preprocess_sample_needs_no_rotation(self):
        # The shipped sample is horizontal: deskew must not touch it.
        image = ocr_pipeline.load_image(SAMPLE_IMAGE)
        gray = ocr_pipeline.convert_to_grayscale(image)
        corrected, angle = ocr_pipeline.deskew_image(gray)
        self.assertAlmostEqual(angle, 0.0, places=6)
        np.testing.assert_array_equal(corrected, gray)


class TestDeskew(unittest.TestCase):
    def test_straight_image_is_untouched(self):
        image = make_text_image()
        corrected, angle = ocr_pipeline.deskew_image(image)
        self.assertAlmostEqual(angle, 0.0, places=6)
        np.testing.assert_array_equal(corrected, image)

    def test_rotated_image_is_straightened(self):
        tilted = rotate(make_text_image(), 7.0)
        corrected, angle = ocr_pipeline.deskew_image(tilted)
        # Content rotated +7 deg must be corrected by about -7 deg.
        self.assertAlmostEqual(angle, -7.0, delta=1.0)
        # ... and the corrected image must now measure as horizontal.
        _, residual = ocr_pipeline.deskew_image(corrected)
        self.assertAlmostEqual(residual, 0.0, delta=1.0)

    def test_blank_image_is_untouched(self):
        blank = np.full((100, 100), 255, np.uint8)
        corrected, angle = ocr_pipeline.deskew_image(blank)
        self.assertEqual(angle, 0.0)
        np.testing.assert_array_equal(corrected, blank)

    def test_rejects_colour_input(self):
        with self.assertRaises(ValueError):
            ocr_pipeline.deskew_image(np.zeros((5, 5, 3), np.uint8))


class TestConfidenceFiltering(unittest.TestCase):
    # 5. Confidence handling: MIN_CONFIDENCE is the specified 80%.
    def test_min_confidence_constant(self):
        self.assertEqual(ocr_pipeline.MIN_CONFIDENCE, 80.0)

    def _data(self, texts, confs, lines=None, blocks=None):
        count = len(texts)
        return {
            "text": texts,
            "conf": confs,
            "left": [10] * count,
            "top": [20] * count,
            "width": [50] * count,
            "height": [30] * count,
            "line_num": lines or [0] * count,
            "block_num": blocks or [0] * count,
            "par_num": [0] * count,
        }

    def test_filters_detections_below_threshold(self):
        data = self._data(
            ["keep", "drop", "boundary"],
            ["95", "79.9", "80"],
        )
        detections = ocr_pipeline.parse_image_data(data)
        by_text = {d.text: d for d in detections}
        self.assertTrue(by_text["keep"].accepted)
        self.assertFalse(by_text["drop"].accepted)
        # The threshold is inclusive: exactly 80.0% is accepted.
        self.assertTrue(by_text["boundary"].accepted)
        self.assertEqual(len(detections), 3)

    def test_raw_confidence_values_are_preserved(self):
        data = self._data(["word"], ["91.0"])
        detection = ocr_pipeline.parse_image_data(data)[0]
        self.assertIsInstance(detection.confidence, float)
        self.assertEqual(detection.confidence, 91.0)

    def test_empty_text_entries_are_ignored(self):
        data = self._data(["", "  ", "word"], ["-1", "-1", "90"])
        detections = ocr_pipeline.parse_image_data(data)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].text, "word")

    def test_invalid_confidence_values_are_ignored(self):
        data = self._data(["a", "b", "c"], ["abc", "-1", "70"])
        detections = ocr_pipeline.parse_image_data(data)
        # "abc" and the -1 placeholder are dropped, "c" is kept but rejected.
        self.assertEqual([d.text for d in detections], ["c"])
        self.assertFalse(detections[0].accepted)

    def test_missing_geometry_keys_use_defaults(self):
        detections = ocr_pipeline.parse_image_data({"text": ["solo"], "conf": ["88"]})
        self.assertEqual(detections[0].box, (0, 0, 0, 0))
        self.assertTrue(detections[0].accepted)

    def test_line_numbers_group_words(self):
        data = self._data(
            ["one", "two", "three"],
            ["90", "90", "90"],
            lines=[1, 1, 2],
            blocks=[1, 1, 2],
        )
        detections = ocr_pipeline.parse_image_data(data)
        self.assertEqual([d.line_number for d in detections], [0, 0, 1])


class TestResultFormatting(unittest.TestCase):
    def _result(self):
        detections = [
            ocr_pipeline.OcrDetection("DecodeLabs", 95.0, 10, 20, 100, 40, True, 0),
            ocr_pipeline.OcrDetection("AI", 96.0, 120, 20, 30, 40, True, 0),
            ocr_pipeline.OcrDetection("noise", 42.0, 10, 80, 60, 30, False, 1),
        ]
        return ocr_pipeline.OcrResult(detections=detections, psm=6, min_confidence=80.0)

    def test_accepted_text_contains_only_accepted_words(self):
        self.assertEqual(self._result().accepted_text, "DecodeLabs AI")

    def test_format_result_reports_settings_and_status(self):
        text = ocr_pipeline.format_result(self._result())
        self.assertIn("Confidence threshold   : 80.0%", text)
        self.assertIn("Page Segmentation Mode : 6", text)
        self.assertIn("2 accepted, 1 rejected", text)
        self.assertIn("DecodeLabs AI", text)
        self.assertIn("'noise' 42.0% [REJECTED]", text)
        self.assertIn("Confidence (rejected) : min 42.0%", text)

    def test_format_result_with_no_accepted_text(self):
        result = ocr_pipeline.OcrResult(
            detections=[
                ocr_pipeline.OcrDetection("weak", 12.0, 0, 0, 10, 10, False, 0)
            ],
            psm=3,
            min_confidence=80.0,
        )
        text = ocr_pipeline.format_result(result)
        self.assertIn("<none met the threshold>", text)

    def test_confidence_summary(self):
        empty = [ocr_pipeline.OcrDetection("x", 50.0, 0, 0, 1, 1, False, 0)]
        self.assertEqual(ocr_pipeline.confidence_summary([]), "n/a (no detections)")
        self.assertEqual(
            ocr_pipeline.confidence_summary(empty),
            "min 50.0% | mean 50.0% | max 50.0%",
        )


class TestVisualConfirmation(unittest.TestCase):
    def test_draw_annotations_keeps_size_and_source_pixels(self):
        image = ocr_pipeline.load_image(SAMPLE_IMAGE)
        result = ocr_pipeline.OcrResult(
            detections=[
                ocr_pipeline.OcrDetection("DecodeLabs", 95.0, 60, 60, 280, 40, True, 0),
                ocr_pipeline.OcrDetection("junk", 10.0, 60, 200, 100, 40, False, 1),
            ],
            psm=6,
            min_confidence=80.0,
        )
        annotated = ocr_pipeline.draw_annotations(image, result)
        self.assertEqual(annotated.shape, image.shape)
        self.assertEqual(annotated.dtype, image.dtype)
        # The source image must not be modified in place.
        self.assertFalse(np.array_equal(annotated, image))

    def test_save_image_creates_parent_folders(self):
        image = np.full((20, 20, 3), 255, np.uint8)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "nested" / "out.png"
            written = ocr_pipeline.save_image(path, image)
            self.assertTrue(written.is_file())
            self.assertIsNotNone(cv2.imread(str(written)))


class TestCli(unittest.TestCase):
    def test_requires_image_argument(self):
        parser = ocr_pipeline.build_arg_parser()
        with self.assertRaises(SystemExit) as context:
            parser.parse_args([])
        self.assertEqual(context.exception.code, 2)

    def test_rejects_unsupported_psm(self):
        parser = ocr_pipeline.build_arg_parser()
        with self.assertRaises(SystemExit) as context:
            parser.parse_args(["--image", "x.png", "--psm", "99"])
        self.assertEqual(context.exception.code, 2)

    def test_accepts_supported_psm_values(self):
        parser = ocr_pipeline.build_arg_parser()
        for psm in ocr_pipeline.SUPPORTED_PSM:
            args = parser.parse_args(["--image", "x.png", "--psm", str(psm)])
            self.assertEqual(args.psm, psm)

    def test_default_psm_is_six(self):
        args = ocr_pipeline.build_arg_parser().parse_args(["--image", "x.png"])
        self.assertEqual(args.psm, 6)
        self.assertEqual(args.output, os.path.join("outputs", "ocr_result.png"))

    def test_main_reports_missing_image_and_returns_1(self):
        stderr = io.StringIO()
        with mock.patch.object(
            ocr_pipeline, "ensure_tesseract", return_value="fake-tesseract (v0.0)"
        ):
            with contextlib.redirect_stderr(stderr):
                code = ocr_pipeline.main(["--image", "missing_file.png"])
        self.assertEqual(code, 1)
        self.assertIn("Image not found", stderr.getvalue())

    def test_main_reports_non_image_file_and_returns_1(self):
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as folder:
            fake = Path(folder) / "fake.png"
            fake.write_text("plain text, not an image", encoding="utf-8")
            with mock.patch.object(
                ocr_pipeline, "ensure_tesseract", return_value="fake-tesseract (v0.0)"
            ):
                with contextlib.redirect_stderr(stderr):
                    code = ocr_pipeline.main(["--image", str(fake)])
        self.assertEqual(code, 1)
        self.assertIn("could not decode", stderr.getvalue())

    def test_main_reports_missing_tesseract_and_returns_1(self):
        stderr = io.StringIO()
        with mock.patch.object(
            ocr_pipeline,
            "ensure_tesseract",
            side_effect=RuntimeError(ocr_pipeline.TESSERACT_SETUP_MESSAGE),
        ):
            with contextlib.redirect_stderr(stderr):
                code = ocr_pipeline.main(["--image", str(SAMPLE_IMAGE)])
        self.assertEqual(code, 1)
        self.assertIn("pytesseract is only a Python", stderr.getvalue())
        self.assertIn("does NOT install Tesseract", stderr.getvalue())


@unittest.skipUnless(HAS_TESSERACT, SKIP_REASON)
class TestTesseractIntegration(unittest.TestCase):
    """Real OCR execution - only runs when the Tesseract engine exists."""

    def test_end_to_end_pipeline_on_sample(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "ocr_result.png"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = ocr_pipeline.main(
                    [
                        "--image",
                        str(SAMPLE_IMAGE),
                        "--psm",
                        "6",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(code, 0, stderr.getvalue())
            self.assertTrue(output.is_file())
            preprocessed = output.parent / "preprocessed.png"
            self.assertTrue(preprocessed.is_file())

            binary = cv2.imread(str(preprocessed), cv2.IMREAD_GRAYSCALE)
            self.assertIsNotNone(binary)
            self.assertTrue(set(np.unique(binary)).issubset({0, 255}))

    def test_ocr_text_and_confidence_on_sample(self):
        image = ocr_pipeline.load_image(SAMPLE_IMAGE)
        result = ocr_pipeline.run_ocr(
            ocr_pipeline.preprocess_image(image), psm=6
        )
        self.assertTrue(result.detections, "Tesseract returned no detections")
        for detection in result.detections:
            self.assertGreaterEqual(detection.confidence, 0.0)
            self.assertLessEqual(detection.confidence, 100.0)

        accepted_text = result.accepted_text
        self.assertIn("DecodeLabs", accepted_text)
        self.assertIn("Project", accepted_text)
        self.assertIn("Recognition", accepted_text)

        for detection in result.accepted:
            self.assertGreaterEqual(detection.confidence, ocr_pipeline.MIN_CONFIDENCE)
        for detection in result.rejected:
            self.assertLess(detection.confidence, ocr_pipeline.MIN_CONFIDENCE)

    def test_all_supported_psm_modes_execute(self):
        image = ocr_pipeline.preprocess_image(
            ocr_pipeline.load_image(SAMPLE_IMAGE)
        )
        for psm in ocr_pipeline.SUPPORTED_PSM:
            with self.subTest(psm=psm):
                result = ocr_pipeline.run_ocr(image, psm=psm)
                self.assertIsInstance(result, ocr_pipeline.OcrResult)
                self.assertEqual(result.psm, psm)


class TestRunOcrValidation(unittest.TestCase):
    def test_run_ocr_rejects_unsupported_psm(self):
        gray = np.full((50, 50), 255, np.uint8)
        with self.assertRaises(ValueError):
            ocr_pipeline.run_ocr(gray, psm=99)


if __name__ == "__main__":
    unittest.main(verbosity=2)
