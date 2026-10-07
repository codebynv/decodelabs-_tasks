"""OCR pipeline for Project 4 - Image/Text Recognition (Basic).

Pipeline (Decode Labs Project 4, Path 1):

    Input Image
        -> load_image()             validate path + OpenCV decode
        -> convert_to_grayscale()   collapse RGB to 1-channel intensity
        -> apply_gaussian_blur()    smooth micro-noise
        -> deskew_image()           snap tilted text back to horizontal
        -> apply_adaptive_threshold() force a black/white decision per pixel
        -> run_ocr()                pytesseract.image_to_data (text + confidence)
        -> confidence filter        drop detections below MIN_CONFIDENCE
        -> report + visual confirmation (bounding boxes)

The core functions are pure (no printing, no CLI parsing) so they can be unit
tested without a Tesseract installation. Terminal I/O lives in main().
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image

# --- Project specification constants -------------------------------------
MIN_CONFIDENCE = 80.0
DEFAULT_PSM = 6
SUPPORTED_PSM = (3, 6, 7, 11)

# --- Preprocessing constants ---------------------------------------------
GAUSSIAN_KERNEL = (5, 5)
ADAPTIVE_BLOCK_SIZE = 31
ADAPTIVE_C = 10
MIN_SKEW_DEGREES = 0.5
MAX_SKEW_DEGREES = 45.0

# Extra locations searched when the engine is not on PATH.
TESSERACT_SEARCH_PATHS = (
    r"%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    "/usr/bin/tesseract",
    "/usr/local/bin/tesseract",
    "/opt/homebrew/bin/tesseract",
)

TESSERACT_SETUP_MESSAGE = """\
Tesseract OCR engine not found.

pytesseract is only a Python *wrapper* around the Tesseract engine; installing
the pip package does NOT install Tesseract itself, so no OCR can be executed.

Install the engine, then run this script again:

  Windows : winget install UB-Mannheim.TesseractOCR
            (installer: https://github.com/UB-Mannheim/tesseract/releases)
            default path: C:\\Program Files\\Tesseract-OCR\\tesseract.exe
  macOS   : brew install tesseract
  Linux   : sudo apt install tesseract-ocr

If Tesseract is installed in a non-standard location, set the TESSERACT_CMD
environment variable to the full path of the executable."""


# --- Result representation ------------------------------------------------
@dataclass(frozen=True)
class OcrDetection:
    """One word/line candidate returned by Tesseract."""

    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int
    accepted: bool
    line_number: int = 0

    @property
    def box(self) -> tuple[int, int, int, int]:
        """Bounding box as (left, top, width, height) in pixels."""
        return (self.left, self.top, self.width, self.height)


@dataclass(frozen=True)
class OcrResult:
    """All detections of one OCR run plus the settings used to produce them."""

    detections: list[OcrDetection] = field(default_factory=list)
    psm: int = DEFAULT_PSM
    min_confidence: float = MIN_CONFIDENCE

    @property
    def accepted(self) -> list[OcrDetection]:
        return [d for d in self.detections if d.accepted]

    @property
    def rejected(self) -> list[OcrDetection]:
        return [d for d in self.detections if not d.accepted]

    @property
    def accepted_text(self) -> str:
        """Accepted words grouped back into lines, newline separated."""
        if not self.accepted:
            return ""
        lines: dict[int, list[str]] = {}
        for detection in self.accepted:
            lines.setdefault(detection.line_number, []).append(detection.text)
        return "\n".join(" ".join(words) for _, words in sorted(lines.items()))


# --- Tesseract discovery --------------------------------------------------
def find_tesseract() -> str | None:
    """Locate the Tesseract executable (PATH, TESSERACT_CMD, common paths)."""
    candidates: list[str] = []
    override = os.environ.get("TESSERACT_CMD")
    if override:
        candidates.append(override)
    on_path = shutil.which("tesseract")
    if on_path:
        candidates.append(on_path)
    candidates.extend(os.path.expandvars(path) for path in TESSERACT_SEARCH_PATHS)

    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return candidate
    return None


def ensure_tesseract() -> str:
    """Return a usable tesseract path or raise RuntimeError with setup help."""
    executable = find_tesseract()
    if executable is None:
        raise RuntimeError(TESSERACT_SETUP_MESSAGE)

    pytesseract.pytesseract.tesseract_cmd = executable
    try:
        version = pytesseract.get_tesseract_version()
    except pytesseract.TesseractNotFoundError as error:
        raise RuntimeError(TESSERACT_SETUP_MESSAGE) from error
    except OSError as error:
        raise RuntimeError(
            f"Tesseract found at {executable} but it could not be executed:\n{error}"
        ) from error
    return f"{executable} (v{version})"


# --- Preprocessing --------------------------------------------------------
def load_image(image_path: str | Path) -> np.ndarray:
    """Read an image from disk and validate it.

    Raises FileNotFoundError when the path does not exist and ValueError when
    OpenCV cannot decode the file contents.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode this file as an image: {path}")
    return image


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Collapse a colour image into a single 8-bit intensity channel."""
    if image is None or image.size == 0:
        raise ValueError("convert_to_grayscale received an empty image.")
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(
    grayscale: np.ndarray, kernel: tuple[int, int] = GAUSSIAN_KERNEL
) -> np.ndarray:
    """Smooth micro-imperfections and artefact noise (PDF preprocessing step 2)."""
    if grayscale.ndim != 2:
        raise ValueError("apply_gaussian_blur expects a grayscale image.")
    return cv2.GaussianBlur(grayscale, kernel, 0)


def deskew_image(
    grayscale: np.ndarray,
    min_angle: float = MIN_SKEW_DEGREES,
    max_angle: float = MAX_SKEW_DEGREES,
) -> tuple[np.ndarray, float]:
    """Straighten tilted text (PDF preprocessing step 3).

    The rotation angle is taken from the minimum area rectangle around the
    dark text pixels. Images whose skew is below `min_angle` are returned
    untouched with an applied angle of 0.0.

    Returns (image, applied_angle) where a negative angle means the image was
    rotated clockwise. Assumes the text block is wider than it is tall.
    """
    if grayscale.ndim != 2:
        raise ValueError("deskew_image expects a grayscale image.")

    _, binary = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coordinates = cv2.findNonZero(binary)
    if coordinates is None or len(coordinates) < 5:
        return grayscale, 0.0

    rectangle_angle = cv2.minAreaRect(coordinates)[-1]
    # OpenCV reports angles in [-90, 0): -90 means "already horizontal".
    correction = rectangle_angle + 90.0 if rectangle_angle < -45.0 else rectangle_angle

    if abs(correction) < min_angle or abs(correction) > max_angle:
        return grayscale, 0.0

    height, width = grayscale.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), correction, 1.0)
    rotated = cv2.warpAffine(
        grayscale,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderValue=255,
    )
    return rotated, correction


def apply_adaptive_threshold(
    grayscale: np.ndarray,
    block_size: int = ADAPTIVE_BLOCK_SIZE,
    c: float = ADAPTIVE_C,
) -> np.ndarray:
    """Force every pixel to black or white using a locally computed threshold."""
    if grayscale.ndim != 2:
        raise ValueError("apply_adaptive_threshold expects a grayscale image.")
    if block_size < 3 or block_size % 2 == 0:
        raise ValueError("block_size must be an odd integer >= 3.")
    return cv2.adaptiveThreshold(
        grayscale,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c,
    )


def preprocess_image(image: np.ndarray, deskew: bool = True) -> np.ndarray:
    """Full preprocessing chain: grayscale -> blur -> deskew -> threshold."""
    grayscale = convert_to_grayscale(image)
    blurred = apply_gaussian_blur(grayscale)
    if deskew:
        blurred, _ = deskew_image(blurred)
    return apply_adaptive_threshold(blurred)


# --- OCR ------------------------------------------------------------------
def _to_confidence(raw_value: object) -> float | None:
    """Convert a raw Tesseract `conf` entry to float, or None if unusable."""
    try:
        confidence = float(str(raw_value).strip())
    except (TypeError, ValueError):
        return None
    if confidence < 0:  # Tesseract marks non-text rows with -1
        return None
    return confidence


def parse_image_data(data: dict, min_confidence: float = MIN_CONFIDENCE) -> list[OcrDetection]:
    """Turn pytesseract `image_to_data` output into typed detections.

    Empty text entries are ignored, confidence values are kept exactly as
    Tesseract reported them, and `accepted` is True only when
    confidence >= min_confidence.
    """
    texts = data.get("text", [])
    confidences = data.get("conf", [])

    def row(key: str, default: int = 0) -> list:
        """Return one parallel array from Tesseract output (defaults if absent)."""
        values = data.get(key)
        return values if values else [default] * len(texts)

    lefts = row("left")
    tops = row("top")
    widths = row("width")
    heights = row("height")
    blocks = row("block_num")
    paragraphs = row("par_num")
    lines = row("line_num")

    detections: list[OcrDetection] = []
    current_line = -1
    previous_key: tuple[int, int, int] | None = None

    for index, raw_text in enumerate(texts):
        text = str(raw_text).strip()
        if not text:
            continue
        confidence = _to_confidence(confidences[index])
        if confidence is None:
            continue

        line_key = (int(blocks[index]), int(paragraphs[index]), int(lines[index]))
        if line_key != previous_key:
            current_line += 1
            previous_key = line_key

        detections.append(
            OcrDetection(
                text=text,
                confidence=confidence,
                left=int(lefts[index]),
                top=int(tops[index]),
                width=int(widths[index]),
                height=int(heights[index]),
                accepted=confidence >= min_confidence,
                line_number=current_line,
            )
        )
    return detections


def run_ocr(
    preprocessed_image: np.ndarray,
    psm: int = DEFAULT_PSM,
    min_confidence: float = MIN_CONFIDENCE,
) -> OcrResult:
    """Execute Tesseract on a preprocessed image and return typed detections."""
    if psm not in SUPPORTED_PSM:
        raise ValueError(
            f"Unsupported PSM {psm}. Choose one of {SUPPORTED_PSM}."
        )
    if preprocessed_image is None or preprocessed_image.size == 0:
        raise ValueError("run_ocr received an empty image.")

    pil_image = Image.fromarray(preprocessed_image)
    data = pytesseract.image_to_data(
        pil_image,
        config=f"--psm {psm}",
        output_type=pytesseract.Output.DICT,
    )
    return OcrResult(
        detections=parse_image_data(data, min_confidence),
        psm=psm,
        min_confidence=min_confidence,
    )


# --- Reporting / visual confirmation -------------------------------------
def confidence_summary(detections: list[OcrDetection]) -> str:
    """min/mean/max of raw confidences, or a placeholder for an empty list."""
    if not detections:
        return "n/a (no detections)"
    values = [d.confidence for d in detections]
    return (
        f"min {min(values):.1f}% | mean {sum(values) / len(values):.1f}% "
        f"| max {max(values):.1f}%"
    )


def format_result(result: OcrResult) -> str:
    """Human readable OCR report: accepted text, raw confidences, boxes."""
    accepted = result.accepted
    rejected = result.rejected

    output: list[str] = []
    output.append("=== OCR Result ===")
    output.append(f"Page Segmentation Mode : {result.psm}")
    output.append(f"Confidence threshold   : {result.min_confidence:.1f}%")
    output.append(
        f"Detections             : {len(accepted)} accepted, "
        f"{len(rejected)} rejected (of {len(result.detections)})"
    )
    output.append("")
    output.append(f"Accepted text (confidence >= {result.min_confidence:.1f}%):")
    output.append(result.accepted_text if accepted else "  <none met the threshold>")
    output.append("")
    output.append("Per-detection detail (raw Tesseract confidence, not accuracy):")
    for detection in result.detections:
        status = "accepted" if detection.accepted else "REJECTED"
        output.append(
            f"  line {detection.line_number}: {detection.text!r} "
            f"{detection.confidence:.1f}% [{status}] box={detection.box}"
        )
    output.append("")
    output.append(f"Confidence (accepted) : {confidence_summary(accepted)}")
    if rejected:
        output.append(f"Confidence (rejected) : {confidence_summary(rejected)}")
    return "\n".join(output)


def draw_annotations(image: np.ndarray, result: OcrResult) -> np.ndarray:
    """Draw bounding boxes and confidence labels on a copy of the image.

    Green = accepted (>= threshold), red = rejected. Labels show the raw
    confidence percentage.
    """
    canvas = image.copy()
    for detection in result.detections:
        colour = (0, 200, 0) if detection.accepted else (0, 0, 220)
        x, y, width, height = detection.box
        cv2.rectangle(canvas, (x, y), (x + width, y + height), colour, 2)
        label = f"{detection.confidence:.0f}%"
        label_y = y - 6 if y >= 22 else y + height + 16
        cv2.putText(
            canvas,
            label,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colour,
            1,
            cv2.LINE_AA,
        )

    legend = (
        f"green: accepted >= {result.min_confidence:.0f}%  "
        f"red: rejected  psm {result.psm}"
    )
    cv2.putText(
        canvas,
        legend,
        (10, canvas.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (60, 60, 60),
        1,
        cv2.LINE_AA,
    )
    return canvas


def save_image(image_path: str | Path, image: np.ndarray) -> Path:
    """Write an image to disk, creating parent directories as needed."""
    path = Path(image_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"OpenCV could not write the image to {path}")
    return path


# --- CLI ------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.ocr_pipeline",
        description=(
            "Content-free OCR pipeline: preprocess an image, run Tesseract, "
            "filter detections by confidence and draw the result."
        ),
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the input image (e.g. data/sample/sample_text.png).",
    )
    parser.add_argument(
        "--psm",
        type=int,
        choices=SUPPORTED_PSM,
        default=DEFAULT_PSM,
        help=(
            "Tesseract Page Segmentation Mode: 3 (automatic), 6 (single block, "
            "default), 7 (single line), 11 (sparse text)."
        ),
    )
    parser.add_argument(
        "--output",
        default=str(Path("outputs") / "ocr_result.png"),
        help="Where to write the annotated result image "
        "(preprocessed.png is written to the same folder).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns 0 on success, 1 on a handled error."""
    args = build_arg_parser().parse_args(argv)

    try:
        engine = ensure_tesseract()
    except RuntimeError as error:
        print(str(error), file=sys.stderr)
        return 1

    try:
        image = load_image(args.image)
    except (FileNotFoundError, ValueError) as error:
        print(f"Input error: {error}", file=sys.stderr)
        return 1

    preprocessed = preprocess_image(image)
    output_path = Path(args.output)
    preprocessed_path = output_path.parent / "preprocessed.png"

    try:
        result = run_ocr(preprocessed, psm=args.psm, min_confidence=MIN_CONFIDENCE)
    except (ValueError, pytesseract.TesseractError) as error:
        print(f"OCR error: {error}", file=sys.stderr)
        return 1

    save_image(preprocessed_path, preprocessed)
    save_image(output_path, draw_annotations(image, result))

    print(f"Tesseract         : {engine}")
    print(f"Input image       : {args.image} ({image.shape[1]}x{image.shape[0]} px)")
    print(f"Preprocessed file : {preprocessed_path}")
    print(f"Result image      : {output_path}")
    print()
    print(format_result(result))

    if not result.accepted:
        raw = confidence_summary(result.detections)
        print()
        print(
            f"WARNING: no detection reached the {MIN_CONFIDENCE:.1f}% acceptance "
            f"threshold. Highest raw confidence observed: {raw}. "
            "Nothing was accepted, so no OCR text is claimed."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
