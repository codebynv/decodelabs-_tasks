# Project 4 — Image/Text Recognition (OCR)

## Overview
This is the perception phase of the Decode Labs Industrial Training Kit (Batch 2026). Projects 1–3 handled text, data and logic; Project 4 gives the machine eyes. This repository implements **Path 1: Optical Character Recognition** — a complete Input → Process → Output pipeline that ingests a raw image, cleans it with OpenCV, reads it with Google's Tesseract engine through `pytesseract`, filters every detection through an **80% confidence gate**, and prints both a machine-readable result and a visual confirmation with bounding boxes.

Nothing in this project is simulated: every number you see in the terminal comes from a real `tesseract` process reading real pixels.

## Objective
Taken from the Project 4 specification ("AI — Image/Text Recognition"):

- **Goal:** "Engineer a Python script capable of ingesting raw visual data and extracting accurate, machine-readable intelligence."
- **Toolkit:** `pytesseract` (Google's OCR engine wrapper) + OpenCV (preprocessing). The alternative `cv2.dnn`/MobileNet-SSD object-detection path (Path 2) was not required for the OCR deliverable.
- **The deliverable:** "A fully functioning recognition pipeline that proves the machine can see text … with validated confidence."
- **Key skills:** image preprocessing, OCR integration, confidence thresholding, visual result validation.

## Decode Labs Requirements
How each requirement from the Project 4 PDF maps to this codebase:

| PDF requirement | How this project satisfies it |
| --- | --- |
| Path 1 — OCR via `pytesseract` | `run_ocr()` calls `pytesseract.image_to_data(..., config="--psm N")` on the preprocessed image |
| Page Segmentation Mode tuning (`--psm 3/6/7/11`) | `SUPPORTED_PSM = (3, 6, 7, 11)`; CLI `--psm` rejects anything else (exit code 2); default is 6 |
| Pre-processing: grayscale conversion | `convert_to_grayscale()` (BGR/BGRA → single 8-bit channel) |
| Pre-processing: blur to remove noise | `apply_gaussian_blur()` — 5×5 Gaussian kernel |
| Pre-processing: adaptive thresholding | `apply_adaptive_threshold()` — Gaussian adaptive, blockSize 31, C 10 |
| "The Gate": `if confidence >= 0.80` | `MIN_CONFIDENCE = 80.0` applied inclusively in `parse_image_data()` (`accepted=confidence >= min_confidence`) |
| "80% is the absolute minimum standard" | The CLI always filters at `MIN_CONFIDENCE`; the constant is asserted by a unit test |
| Visual confirmation (boxes + labels) | `draw_annotations()` → `outputs/ocr_result.png` (green = accepted, red = rejected, confidence label on every box) |
| Honest, non-fabricated output | `main()` prints an explicit WARNING when nothing met the threshold instead of inventing text |

### The four milestone validations
| # | Validation from the PDF | Evidence in this project |
| --- | --- | --- |
| 1 | **Library Integration** — seamless `pytesseract`/`cv2.dnn` usage | Engine discovery + `ensure_tesseract()`; `pytesseract`/`cv2` imported and used error-free (see "Technologies Used") |
| 2 | **Pre-Processing Integrity** — grayscale + adaptive thresholding demonstrably executed | `outputs/preprocessed.png` is written on every run; unit tests assert binary `{0,255}` output |
| 3 | **Accuracy Benchmarking** — minimum validated confidence of 80% | Sample run: 8/8 detections accepted, min **91.0%**, mean **95.4%** (raw confidences printed per word) |
| 4 | **Visual Confirmation** — pristine output with bounding boxes/labels | `outputs/ocr_result.png` — every box labelled with its confidence percentage |

## OCR Pipeline
```
INPUT IMAGE (RGB, 900x420)
        │  load_image()                 validate path + OpenCV decode
        ▼
   colour image                          FileNotFoundError / ValueError on bad input
        │  convert_to_grayscale()        collapse to one intensity channel
        ▼
   grayscale
        │  apply_gaussian_blur()         5x5 kernel, smooth micro-noise
        ▼
   blurred
        │  deskew_image()                Otsu → minAreaRect → rotate (skip if < 0.5°)
        ▼
   straightened
        │  apply_adaptive_threshold()    per-pixel black/white decision (31, C=10)
        ▼
   BINARY IMAGE {0, 255}  ──────────────► outputs/preprocessed.png   (Milestone 2)
        │  run_ocr()                     pytesseract.image_to_data + --psm
        ▼
   raw detections (text, confidence, box, line)
        │  parse_image_data()            drop empty/(-1) rows, apply >= 80% gate
        ▼
   accepted + rejected detections
        ├──► format_result()             terminal report (Milestone 3)
        └──► draw_annotations()          outputs/ocr_result.png      (Milestone 4)
```

Each function is pure (no printing, no CLI parsing), so the whole pipeline is unit-testable without Tesseract installed. Terminal I/O lives only in `main()`.

## Sample Image
`data/sample/sample_text.png` — 900×420 px, black text on white:

```
DecodeLabs AI Internship
Project 4
Optical Character Recognition
```

It is generated deterministically (no external dataset download). The generator used Pillow with Consolas at 52 px, 60 px padding, 34 px line spacing:

```python
# How data/sample/sample_text.png was generated (run once)
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

font = ImageFont.truetype(r"C:\Windows\Fonts\consola.ttf", 52)
image = Image.new("RGB", (900, 420), "white")
draw = ImageDraw.Draw(image)
y = 60
for line in ["DecodeLabs AI Internship", "Project 4", "Optical Character Recognition"]:
    draw.text((60, y), line, fill="black", font=font)
    y = draw.textbbox((60, y), line, font=font)[3] + 34
image.save(Path("data/sample/sample_text.png"))
```

Consolas was chosen deliberately: with Arial, Tesseract misread `AI` as `Al`. A monospace font keeps every glyph cleanly separated, so the sample tests the pipeline rather than the font renderer.

## Page Segmentation Modes
The PDF stresses that "layout configuration is critical for accuracy". Measured on the shipped sample (real Tesseract 5.4.0):

| `--psm` | PDF description | Measured result on `sample_text.png` |
| --- | --- | --- |
| `3` | Fully automatic (default for varied layouts) | 8/8 accepted, min 91.0% / mean 95.4% |
| `6` | Single uniform block of text (project default) | 8/8 accepted, min 91.0% / mean 95.4% |
| `7` | Single text line (number plates/headers) | **1 detection, `mieten` at 7.0% → 0 accepted** |
| `11` | Sparse, scattered text (invoices) | 8/8 accepted, min 91.0% / mean 95.4% |

PSM 7 is the instructive failure: the sample is *three* lines, but PSM 7 promises Tesseract *one* line, so it smears everything into a single low-confidence guess. The gate rejects it and the report says so — that is exactly what the confidence filter is for.

## Confidence Filtering Logic
Tesseract attaches a confidence score to every word it extracts (0–100). The PDF's gatekeeper rule is:

```python
if confidence >= 0.80:   # keep
else:                    # discard
```

Implemented as `MIN_CONFIDENCE = 80.0` and `accepted=confidence >= min_confidence` in `parse_image_data()`:

- **The threshold is inclusive** — exactly 80.0% passes (unit-tested at the boundary).
- **Raw values are preserved** — 91.0 stays 91.0; nothing is rounded or invented.
- **Garbage rows are dropped before filtering** — empty text and Tesseract's `-1` placeholder rows never become detections.
- **Words are regrouped into lines** — accepted words keep their `line_num` so the output reads as natural text lines, not a word soup.
- **Nothing is fabricated** — if zero detections pass, the CLI prints a WARNING with the raw statistics and claims no OCR text.

> **OCR confidence is not the same thing as overall model accuracy.** The number is Tesseract's own statistical self-assessment of a single guess — "how sure am I of this word?" — not a measurement of how many characters of the document were transcribed correctly. A run can show 96% mean confidence while still having swapped `AI` for `Al`, and a genuinely correct word can be scored conservatively. Confidence is a *filter*; accuracy would require comparing the output against known ground truth (see "Limitations").

Note the PDF's trade-off: high thresholds minimise false positives but risk false negatives. 80% is the spec's absolute minimum, not a claim of perfection.

## Pre-Processing Details
1. **Grayscale** — `cv2.cvtColor` (BGR2GRAY / BGRA2GRAY). OCR only cares about intensity; colour is noise.
2. **Gaussian blur** — 5×5 kernel. Removes scan artefacts and anti-aliasing flecks before thresholding.
3. **Deskew** — Otsu-binarise the blurred image, fit `cv2.minAreaRect` around the dark pixels, rotate back with `cv2.warpAffine` (white border). Angles below 0.5° or above 45° are ignored, so an already-straight page is bit-for-bit untouched (unit-tested). Empirically, OpenCV 5 reports `minAreaRect` angles in `[−90, 0)`, where −90 means "already horizontal", so the correction is `angle + 90 if angle < −45 else angle` — a content-rotated +7° test image is corrected by ≈ −7° and then measures ≈ 0°.
4. **Adaptive threshold** — `cv2.adaptiveThreshold` with `ADAPTIVE_THRESH_GAUSSIAN_C`, blockSize 31, C 10. A *local* threshold handles uneven lighting that a single global value would wreck; blockSize must be odd ≥ 3 (enforced with a clear error).

## Project Structure
```
project-04-image-text-recognition/
├── README.md                     ← you are here
├── requirements.txt              ← Python dependencies
├── .gitignore                    ← ignores __pycache__, generated PNGs
├── src/
│   ├── __init__.py
│   └── ocr_pipeline.py           ← the entire pipeline + CLI (532 lines)
├── data/
│   └── sample/
│       └── sample_text.png       ← deterministic test input
├── outputs/
│   ├── .gitkeep                  ← folder exists in git
│   ├── preprocessed.png          ← generated: binary image (Milestone 2)
│   └── ocr_result.png            ← generated: annotated result (Milestone 4)
└── tests/
    └── test_ocr_pipeline.py      ← 43 unit + integration tests
```

## Technologies Used
| Component | Version here | Role |
| --- | --- | --- |
| Python | 3.13.14 | language |
| OpenCV (`opencv-python`) | 5.0.0 | loading, grayscale, blur, deskew, adaptive threshold, drawing |
| `pytesseract` | 0.3.13 | Python wrapper that shells out to the Tesseract binary |
| Tesseract OCR engine | 5.4.0.20240606 (eng + osd) | the actual OCR model (separate system install) |
| NumPy | 2.5.3 | image arrays |
| Pillow | 12.3.0 | array ⇄ PIL conversion for Tesseract; sample generation |
| `unittest` (stdlib) | — | test suite |

## Installation
```bash
# 1. Python dependencies
pip install -r requirements.txt
```

```bash
# 2. The Tesseract ENGINE (separate from pip!)
# Windows:
winget install UB-Mannheim.TesseractOCR
# macOS:
brew install tesseract
# Linux:
sudo apt install tesseract-ocr
```

**`pip install pytesseract` does NOT install Tesseract.** `pytesseract` is only a wrapper that launches the `tesseract` executable; without the engine there is no OCR. If this script cannot find the engine it prints setup instructions and exits with code 1 instead of pretending to work.

On this machine the engine is not on `PATH`, so `find_tesseract()` also searches `TESSERACT_CMD` (environment override) and well-known locations including `%LOCALAPPDATA%\Tesseract-OCR\tesseract.exe`, `C:\Program Files\Tesseract-OCR\tesseract.exe`, `/usr/bin/tesseract` and `/usr/local/bin/tesseract`. To use a custom install:

```powershell
$env:TESSERACT_CMD = "D:\tools\Tesseract-OCR\tesseract.exe"
```

## How to Run
```bash
python -m src.ocr_pipeline --image data/sample/sample_text.png --psm 6 --output outputs/ocr_result.png
```

| Option | Required | Default | Notes |
| --- | --- | --- | --- |
| `--image PATH` | yes | — | input image (png/jpg/bmp/tiff) |
| `--psm N` | no | `6` | one of `3, 6, 7, 11` — anything else exits 2 |
| `--output PATH` | no | `outputs/ocr_result.png` | `preprocessed.png` is written beside it |

**Exit codes:** `0` success (even if every detection was rejected — that is a valid, reported result), `1` handled error (missing engine, missing image, undecodable file, OCR failure), `2` bad command-line arguments.

Run the tests:
```bash
python tests/test_ocr_pipeline.py      # or: python -m pytest tests/
```

## Example Interaction
Real, unedited output on this machine (`--psm 6`):

```
Tesseract         : C:\Users\Nirav\AppData\Local\Tesseract-OCR\tesseract.exe (v5.4.0.20240606)
Input image       : data/sample/sample_text.png (900x420 px)
Preprocessed file : outputs\preprocessed.png
Result image      : outputs\ocr_result.png

=== OCR Result ===
Page Segmentation Mode : 6
Confidence threshold   : 80.0%
Detections             : 8 accepted, 0 rejected (of 8)

Accepted text (confidence >= 80.0%):
DecodeLabs AI Internship
Project 4
Optical Character Recognition

Per-detection detail (raw Tesseract confidence, not accuracy):
  line 0: 'DecodeLabs' 91.0% [accepted] box=(62, 62, 284, 39)
  line 0: 'AI' 96.0% [accepted] box=(379, 65, 54, 35)
  line 0: 'Internship' 96.0% [accepted] box=(469, 61, 284, 49)
  line 1: 'Project' 96.0% [accepted] box=(63, 144, 197, 50)
  line 1: '4' 96.0% [accepted] box=(292, 148, 28, 35)
  line 2: 'Optical' 96.0% [accepted] box=(61, 228, 199, 49)
  line 2: 'Character' 96.0% [accepted] box=(294, 229, 257, 39)
  line 2: 'Recognition' 96.0% [accepted] box=(585, 228, 312, 50)

Confidence (accepted) : min 91.0% | mean 95.4% | max 96.0%
```

The recovered text matches the sample **exactly** — including the tricky `AI` token — and every word clears the 80% gate. Open `outputs/ocr_result.png` to see the green boxes with `91%`/`96%` labels (Milestone 4), and `outputs/preprocessed.png` for the binary image Tesseract actually read (Milestone 2).

The gate failing honestly (`--psm 7`, single-line mode on a three-line image):

```
Detections             : 0 accepted, 1 rejected (of 1)

Accepted text (confidence >= 80.0%):
  <none met the threshold>

Per-detection detail (raw Tesseract confidence, not accuracy):
  line 0: 'mieten' 7.0% [REJECTED] box=(61, 61, 808, 217)

WARNING: no detection reached the 80.0% acceptance threshold. Highest raw
confidence observed: min 7.0% | mean 7.0% | max 7.0%. Nothing was accepted,
so no OCR text is claimed.
```

Error handling:

```
$ python -m src.ocr_pipeline --image data/does_not_exist.png
Input error: Image not found: data\does_not_exist.png        (exit 1)

$ python -m src.ocr_pipeline --image sample.png --psm 99
usage: python -m src.ocr_pipeline [-h] --image IMAGE [--psm {3,6,7,11}] ...
python -m src.ocr_pipeline: error: argument --psm: invalid choice: '99'  (exit 2)
```

## Testing
43 tests in `tests/test_ocr_pipeline.py` — run with `python tests/test_ocr_pipeline.py` or `python -m pytest tests/`.

```
Ran 43 tests in 1.063s
OK
```

| Group | What it proves |
| --- | --- |
| `TestLoadImage` (4) | valid image decodes; missing path / non-image file / directory raise clearly |
| `TestPreprocessing` (9) | grayscale collapses channels; blur reduces noise; threshold output is strictly `{0,255}`; even block sizes rejected; preprocessing is deterministic; the straight sample is not rotated |
| `TestDeskew` (4) | straight/blank images untouched; +7° tilt corrected by ≈ −7° then ≈ 0° residual; colour input rejected |
| `TestConfidenceFiltering` (7) | `MIN_CONFIDENCE == 80.0`; **boundary test: 79.9 rejected, 80.0 accepted**; raw confidences preserved; empty/`-1`/garbage rows dropped; missing geometry defaults; line grouping |
| `TestResultFormatting` (4) | report shows threshold, mode, accept/reject counts, `<none met the threshold>` placeholder, min/mean/max |
| `TestVisualConfirmation` (2) | annotated image keeps size/dtype and does not mutate the source; saved images create parent folders |
| `TestCli` (7) | required `--image`, PSM choices, defaults; `main()` returns 1 for missing image / non-image file / missing engine (mocked) |
| `TestRunOcrValidation` (1) | unsupported PSM raises before any OCR runs |
| `TestTesseractIntegration` (4) | **real OCR**: end-to-end CLI run writes both output images; recovered text contains `DecodeLabs`/`Project`/`Recognition`; all accepted ≥ 80 and all rejected < 80; all four PSM modes execute |

The integration tests are decorated with `@unittest.skipUnless(HAS_TESSERACT, ...)` — on a machine without the engine they skip with the message *"Tesseract engine is not installed/configured — integration test skipped (pytesseract is only a wrapper…)"* instead of failing or, worse, inventing a result. On this machine all 43 run.

## What I Learned
- **`pytesseract` ≠ Tesseract.** The pip package is ~40 KB of wrapper; the engine is a 50 MB system binary. The error message when it is missing is part of the deliverable, not an afterthought.
- **PSM choice matters more than tuning.** Same image, same threshold: PSM 3/6/11 all score 95.4% mean, PSM 7 collapses to 7%. "Layout configuration is critical for accuracy" is empirically true.
- **OpenCV angle conventions are a trap.** `minAreaRect` returns angles in `[−90, 0)` in OpenCV 5; a naïve `if angle: rotate(angle)` spins straight text 90°. The `angle < −45 → angle + 90` rule plus a straight-image regression test keeps it honest.
- **Confidence filtering is easy; honest reporting is the real work.** The spec's `if confidence >= 0.80` is one line. Deciding what to print when *nothing* passes — and not printing invented text — is what makes the output trustworthy.
- **Font choice is an input to accuracy.** The same words read as `AI` in Consolas and `Al` in Arial. Benchmarks are meaningless if the input rendering is the actual variable.

## Limitations
- **Confidence ≠ accuracy** — the whole pipeline filters on Tesseract's self-reported certainty. True accuracy (CER/WER) would need ground-truth transcripts; none are claimed here.
- **English only** — only the `eng` language data is installed; other scripts need `tesseract-ocr-<lang>` plus `--lang` support (not implemented).
- **Clean, printed text only** — the sample is synthetic and high-contrast. Handwriting, low-contrast photos, heavy perspective distortion and complex multi-column layouts were not tested.
- **Deskew assumes text is wider than it is tall** and skew < 45°; rotated pages beyond that are returned untouched by design.
- **Path 2 not implemented** — the PDF's MobileNet-SSD/`cv2.dnn` object-detection alternative is out of scope for this OCR deliverable; no model weights are bundled.
- **Single image per run** — no batch or directory mode, no PDF page ingestion.

## Future Improvements
- **Measure real accuracy** against ground-truth transcripts (CER/WER) instead of relying on confidence alone.
- **Language selection** (`--lang`) and multi-language packs; `--oem` engine-mode exposure.
- **Batch mode** over a directory with a CSV/JSON report of per-image statistics.
- **Auto-PSM probe** — run 3/6/11 and keep the layout whose mean confidence is highest (turning the PSM lesson into automation).
- **Perspective correction** (four-point warp) before deskew for photographed documents.
- **Path 2:** `cv2.dnn` + MobileNet-SSD object detection alongside the OCR path.
- **Confidence histogram / annotated rejected words** in the terminal to make false negatives easier to diagnose.
