"""
pdf_ocr.py - Fast, high-quality OCR for image-based (scanned) PDFs.

Output is plain text with unmistakable page boundaries:

    ################################################################
    >>>>>>>>>>  PAGE 3 of 120  -  START  <<<<<<<<<<
    ################################################################
    ...page text...
    ################################################################
    >>>>>>>>>>  PAGE 3 of 120  -  END  <<<<<<<<<<
    ################################################################

Design for speed
  * pypdfium2 renders pages directly to grayscale (no poppler, no temp PNGs).
  * Pages are OCR'd in parallel (one Tesseract process per CPU core,
    each pinned to 1 thread so they don't fight each other).
  * Pages that already contain a real text layer are copied, not OCR'd
    (disable with --force-ocr).
  * Blank pages are detected and skipped.
  * Results stream to disk in page order as soon as they are ready.

Design for quality
  * Renders at 300 DPI by default (Tesseract's sweet spot).
  * Grayscale + contrast stretch; optional denoise and binarisation.
  * Tesseract LSTM engine (--oem 1), configurable page segmentation mode.
  * Optional automatic rotation fix (--auto-rotate) and deskew (--deskew).
  * Per-page confidence score; low-confidence pages are flagged.

Requirements
    sudo apt install tesseract-ocr            # + tesseract-ocr-<lang> as needed
    pip install pypdfium2 pytesseract pillow  # tqdm and numpy optional

Examples
    python pdf_ocr.py                                  # opens a file explorer to pick PDF(s)
    python pdf_ocr.py scan.pdf                         # -> scan.txt
    python pdf_ocr.py scan.pdf -o out.txt -l eng+deu   # multiple languages
    python pdf_ocr.py scan.pdf --pages 5-20 --dpi 400  # page range, higher DPI
    python pdf_ocr.py scan.pdf --auto-rotate --deskew  # messy scans
    python pdf_ocr.py scan.pdf --per-page-dir pages/   # also one .txt per page
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional

# Keep Tesseract single-threaded per process; we parallelise across pages.
# Must be set before any Tesseract process is spawned.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

import pypdfium2 as pdfium
import pytesseract
from PIL import Image, ImageFilter, ImageOps, ImageStat

try:
    import numpy as np
except ImportError:  # numpy is optional (only needed for --deskew)
    np = None

try:
    from tqdm import tqdm
except ImportError:  # progress bar is optional
    tqdm = None


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass
class PageResult:
    page: int                      # 1-based page number
    text: str
    method: str                    # "ocr" | "text-layer" | "blank" | "error"
    confidence: Optional[float]    # mean word confidence 0-100 (OCR only)
    seconds: float
    rotation: int = 0              # degrees corrected (with --auto-rotate)
    error: Optional[str] = None


# Worker globals (set once per process by _init_worker)
_CFG: dict = {}
_PDF: Optional[pdfium.PdfDocument] = None


def _init_worker(cfg: dict) -> None:
    """Open the PDF once per worker process instead of once per page."""
    global _CFG, _PDF
    _CFG = cfg
    _PDF = pdfium.PdfDocument(cfg["pdf_path"], password=cfg.get("password"))


# --------------------------------------------------------------------------- #
# Image preprocessing
# --------------------------------------------------------------------------- #

def _is_blank(img: Image.Image, threshold: float = 1.5) -> bool:
    """A page is blank if almost no pixel variation exists after downsizing."""
    small = img.resize((max(1, img.width // 8), max(1, img.height // 8)))
    small = ImageOps.autocontrast(small, cutoff=1)
    stat = ImageStat.Stat(small)
    return stat.stddev[0] < threshold


def _deskew(img: Image.Image, max_angle: float = 8.0) -> Image.Image:
    """Projection-profile deskew. Fast on a downsampled copy."""
    if np is None:
        return img
    small = img.copy()
    small.thumbnail((900, 900))
    arr = np.asarray(ImageOps.invert(small.point(lambda p: 255 if p > 160 else 0)))
    if arr.sum() == 0:
        return img
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-max_angle, max_angle + 0.01, 0.5):
        rotated = np.asarray(Image.fromarray(arr).rotate(angle, resample=Image.NEAREST))
        score = float(np.var(rotated.sum(axis=1)))
        if score > best_score:
            best_angle, best_score = float(angle), score
    if abs(best_angle) < 0.3:
        return img
    return img.rotate(best_angle, resample=Image.BICUBIC, expand=True, fillcolor=255)


def _preprocess(img: Image.Image, cfg: dict) -> Image.Image:
    # cutoff=0: pure min/max stretch. A non-zero cutoff clips the darkest pixels,
    # which on sparse pages (text < 1% of pixels) wipes out the text entirely.
    img = ImageOps.autocontrast(img, cutoff=0)
    if cfg["denoise"]:
        img = img.filter(ImageFilter.MedianFilter(3))
    if cfg["binarize"]:
        # Mean-based threshold: cheap and effective on uneven scans.
        mean = ImageStat.Stat(img).mean[0]
        img = img.point(lambda p, t=mean * 0.88: 255 if p > t else 0)
    if cfg["deskew"]:
        img = _deskew(img)
    return img


# --------------------------------------------------------------------------- #
# Per-page processing (runs inside worker processes)
# --------------------------------------------------------------------------- #

def _render(page: "pdfium.PdfPage", dpi: int) -> Image.Image:
    bitmap = page.render(scale=dpi / 72.0, grayscale=True)
    return bitmap.to_pil().convert("L")


def _clean_text(text: str) -> str:
    text = text.replace("\x0c", "")                      # form feeds
    text = re.sub(r"[ \t]+\n", "\n", text)               # trailing spaces
    text = re.sub(r"\n{3,}", "\n\n", text)               # collapse blank runs
    return text.strip()


def _ocr_image(img: Image.Image, cfg: dict) -> tuple[str, Optional[float]]:
    tess_cfg = f"--oem {cfg['oem']} --psm {cfg['psm']} -c preserve_interword_spaces=1"
    if cfg["want_conf"]:
        data = pytesseract.image_to_data(
            img, lang=cfg["lang"], config=tess_cfg, output_type=pytesseract.Output.DICT
        )
        # Rebuild text line by line from word data.
        lines: dict[tuple, list[str]] = {}
        confs = []
        for i, word in enumerate(data["text"]):
            if not word.strip():
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            lines.setdefault(key, []).append(word)
            c = float(data["conf"][i])
            if c >= 0:
                confs.append(c)
        out, last_par = [], None
        for key in sorted(lines):
            par = key[:2]
            if last_par is not None and par != last_par:
                out.append("")
            out.append(" ".join(lines[key]))
            last_par = par
        return "\n".join(out), (sum(confs) / len(confs) if confs else None)
    return pytesseract.image_to_string(img, lang=cfg["lang"], config=tess_cfg), None


def _process_page(page_no: int) -> PageResult:
    """page_no is 1-based."""
    t0 = time.perf_counter()
    cfg = _CFG
    try:
        page = _PDF[page_no - 1]

        # 1) Reuse an existing text layer when it's substantial.
        if not cfg["force_ocr"]:
            tp = page.get_textpage()
            existing = tp.get_text_range().strip()
            tp.close()
            if len(existing) >= cfg["min_text_chars"]:
                return PageResult(page_no, _clean_text(existing), "text-layer",
                                  None, time.perf_counter() - t0)

        # 2) Render + blank check.
        img = _render(page, cfg["dpi"])
        if _is_blank(img):
            return PageResult(page_no, "", "blank", None, time.perf_counter() - t0)

        # 3) Optional orientation fix (90/180/270 degree scans).
        rotation = 0
        if cfg["auto_rotate"]:
            try:
                osd = pytesseract.image_to_osd(
                    img, config="--psm 0 -c min_characters_to_try=50",
                    output_type=pytesseract.Output.DICT)
                rotation = int(osd.get("rotate", 0))
                if rotation:
                    img = img.rotate(-rotation, expand=True, fillcolor=255)
            except pytesseract.TesseractError:
                pass  # OSD needs enough text; just continue unrotated

        # 4) Clean up + OCR.
        img = _preprocess(img, cfg)
        text, conf = _ocr_image(img, cfg)
        return PageResult(page_no, _clean_text(text), "ocr", conf,
                          time.perf_counter() - t0, rotation)

    except Exception as exc:  # never let one bad page kill a 500-page job
        return PageResult(page_no, "", "error", None,
                          time.perf_counter() - t0, error=f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# Output formatting
# --------------------------------------------------------------------------- #

BAR = "#" * 64


def format_page(r: PageResult, total: int, low_conf: float) -> str:
    notes = []
    if r.method == "blank":
        notes.append("blank page")
    elif r.method == "text-layer":
        notes.append("embedded text")
    elif r.method == "error":
        notes.append(f"OCR FAILED: {r.error}")
    if r.confidence is not None:
        flag = "  LOW CONFIDENCE - please review" if r.confidence < low_conf else ""
        notes.append(f"confidence {r.confidence:.0f}%{flag}")
    if r.rotation:
        notes.append(f"rotated {r.rotation} deg")
    note_line = f"[{'; '.join(notes)}]\n" if notes else ""

    head = f"{BAR}\n>>>>>>>>>>  PAGE {r.page} of {total}  -  START  <<<<<<<<<<\n{BAR}\n"
    foot = f"\n{BAR}\n>>>>>>>>>>  PAGE {r.page} of {total}  -  END  <<<<<<<<<<\n{BAR}\n"
    body = r.text if r.text else "(no text)"
    return f"{head}{note_line}{body}{foot}\n"


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def parse_pages(spec: Optional[str], total: int) -> list[int]:
    """'1-3,7,10-' -> [1,2,3,7,10,...,total]"""
    if not spec:
        return list(range(1, total + 1))
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            start = int(a) if a else 1
            end = int(b) if b else total
            pages.update(range(start, end + 1))
        else:
            pages.add(int(part))
    return sorted(p for p in pages if 1 <= p <= total)


def check_tesseract(lang: str) -> None:
    try:
        available = set(pytesseract.get_languages(config=""))
    except pytesseract.TesseractNotFoundError:
        sys.exit("Tesseract is not installed. Try: sudo apt install tesseract-ocr "
                 "(macOS: brew install tesseract)")
    missing = [l for l in lang.split("+") if l not in available]
    if missing:
        sys.exit(f"Missing Tesseract language data: {', '.join(missing)}. "
                 f"Installed: {', '.join(sorted(available))}")


def ocr_file(args: argparse.Namespace, pdf_path: Path, multi: bool = False) -> int:
    if not pdf_path.is_file():
        sys.exit(f"File not found: {pdf_path}")
    check_tesseract(args.lang)

    with pdfium.PdfDocument(str(pdf_path), password=args.password) as doc:
        total = len(doc)
    pages = parse_pages(args.pages, total)
    if not pages:
        sys.exit("No pages selected.")

    out_path = Path(args.output) if args.output and not multi else pdf_path.with_suffix(".txt")
    per_page_dir = Path(args.per_page_dir) if args.per_page_dir else None
    if per_page_dir and multi:
        per_page_dir = per_page_dir / pdf_path.stem
    if per_page_dir:
        per_page_dir.mkdir(parents=True, exist_ok=True)

    workers = max(1, min(args.workers or os.cpu_count() or 1, len(pages)))
    cfg = dict(
        pdf_path=str(pdf_path), password=args.password, dpi=args.dpi, lang=args.lang,
        psm=args.psm, oem=args.oem, force_ocr=args.force_ocr,
        min_text_chars=args.min_text_chars, auto_rotate=args.auto_rotate,
        deskew=args.deskew, denoise=args.denoise, binarize=args.binarize,
        want_conf=not args.no_confidence,
    )

    print(f"{pdf_path.name}: {len(pages)}/{total} pages | {workers} worker(s) | "
          f"{args.dpi} DPI | lang={args.lang}", file=sys.stderr)

    t_start = time.perf_counter()
    results: list[PageResult] = []

    # chunksize=1 keeps load balanced; imap preserves page order for streaming.
    ctx = mp.get_context("spawn" if sys.platform in ("win32", "darwin") else "fork")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(cfg,)) as pool, \
            open(out_path, "w", encoding="utf-8") as fout:

        stream: Iterable[PageResult] = pool.imap(_process_page, pages, chunksize=1)
        if tqdm and not args.quiet:
            stream = tqdm(stream, total=len(pages), unit="page", file=sys.stderr)

        for r in stream:
            results.append(r)
            fout.write(format_page(r, total, args.low_conf))
            fout.flush()
            if per_page_dir:
                (per_page_dir / f"page_{r.page:04d}.txt").write_text(r.text, encoding="utf-8")

        fout.write(f"{BAR}\n>>>>>>>>>>  END OF DOCUMENT  ({len(results)} pages)  <<<<<<<<<<\n{BAR}\n")

    elapsed = time.perf_counter() - t_start

    if args.json:
        json_path = Path(args.json)
        if multi:
            json_path = json_path.parent / f"{pdf_path.stem}.json"
        json_path.write_text(
            json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2),
            encoding="utf-8")

    # Summary
    n_ocr = sum(r.method == "ocr" for r in results)
    n_txt = sum(r.method == "text-layer" for r in results)
    n_blank = sum(r.method == "blank" for r in results)
    errors = [r for r in results if r.method == "error"]
    low = [r.page for r in results if r.confidence is not None and r.confidence < args.low_conf]
    confs = [r.confidence for r in results if r.confidence is not None]

    print(f"\nDone in {elapsed:.1f}s ({elapsed / len(pages):.2f}s/page) -> {out_path}",
          file=sys.stderr)
    print(f"  OCR: {n_ocr} | text layer: {n_txt} | blank: {n_blank} | errors: {len(errors)}",
          file=sys.stderr)
    if confs:
        print(f"  Mean confidence: {sum(confs) / len(confs):.1f}%", file=sys.stderr)
    if low:
        print(f"  Low-confidence pages (<{args.low_conf:.0f}%): {low}", file=sys.stderr)
    for r in errors:
        print(f"  Page {r.page} failed: {r.error}", file=sys.stderr)
    return 1 if errors else 0


# --------------------------------------------------------------------------- #
# File explorer (GUI) support
# --------------------------------------------------------------------------- #

def pick_pdfs() -> list[str]:
    """Open the native file explorer dialog; Ctrl/Shift-click selects several PDFs."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        sys.exit("No file explorer available (tkinter missing). "
                 "Pass the PDF path on the command line, or install tkinter "
                 "(Linux: sudo apt install python3-tk).")
    try:
        root = tk.Tk()
    except tk.TclError:
        sys.exit("No display found for the file explorer. Pass the PDF path on the command line.")
    root.withdraw()
    root.attributes("-topmost", True)
    paths = filedialog.askopenfilenames(
        parent=root, title="Select scanned PDF(s) to convert to text",
        filetypes=[("PDF files", "*.pdf *.PDF"), ("All files", "*.*")])
    root.destroy()
    return list(paths)


def _gui_message(title: str, text: str, error: bool = False) -> None:
    try:
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        (messagebox.showerror if error else messagebox.showinfo)(title, text, parent=root)
        root.destroy()
    except Exception:
        pass


def run(args: argparse.Namespace) -> int:
    gui = not args.pdf
    pdfs = list(args.pdf) if args.pdf else pick_pdfs()
    if not pdfs:
        print("No file selected.", file=sys.stderr)
        return 0
    if len(pdfs) > 1 and args.output:
        sys.exit("-o/--output can only be used with a single PDF.")

    rc = 0
    outputs = []
    try:
        for i, p in enumerate(pdfs, 1):
            if len(pdfs) > 1:
                print(f"\n[{i}/{len(pdfs)}] {p}", file=sys.stderr)
            pdf_path = Path(p)
            rc |= ocr_file(args, pdf_path, multi=len(pdfs) > 1)
            outputs.append(str(Path(args.output) if args.output and len(pdfs) == 1
                               else pdf_path.with_suffix(".txt")))
    except SystemExit as exc:
        if gui and isinstance(exc.code, str):
            _gui_message("OCR failed", exc.code, error=True)
        raise
    if gui:
        _gui_message("OCR finished", "Text saved to:\n\n" + "\n".join(outputs)
                     + ("\n\nSome pages had errors - see the console." if rc else ""))
    return rc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fast, high-quality OCR for scanned PDFs with clear page markers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    languagesss = input("eng/deu?")
    dpiselect = input("DPI?")
    p.add_argument("pdf", nargs="*", help="input PDF(s); omit to pick files in a file explorer window")
    p.add_argument("-o", "--output", help="output .txt (default: <pdf>.txt)")
    p.add_argument("-l", "--lang", default=languagesss, help="Tesseract language(s), e.g. eng+fra")
    p.add_argument("--dpi", type=int, default=dpiselect, help="render resolution (200 fast, 300 balanced, 400 small text)")
    p.add_argument("--pages", help="page selection, e.g. '1-5,9,12-'")
    p.add_argument("-j", "--workers", type=int, default=0, help="parallel processes (0 = all CPU cores)")
    p.add_argument("--psm", type=int, default=3,help="Tesseract page segmentation (3=auto, 4=single column, 6=uniform block)")
    p.add_argument("--oem", type=int, default=1, help="OCR engine (1 = LSTM, best quality)")
    p.add_argument("--auto-rotate", action="store_true", help="detect & fix 90/180/270 rotated pages (slower)")
    p.add_argument("--deskew", action="store_true", help="straighten slightly tilted scans (needs numpy)")
    p.add_argument("--binarize", action="store_true", help="black/white thresholding (helps faint/grey scans)")
    p.add_argument("--denoise", action="store_true", help="median-filter speckle noise (for dirty scans; slower)")
    p.add_argument("--force-ocr", action="store_true", help="OCR even when a text layer exists")
    p.add_argument("--min-text-chars", type=int, default=25, help="min chars for a text layer to be trusted")
    p.add_argument("--no-confidence", action="store_true", help="skip confidence scoring (a bit faster)")
    p.add_argument("--low-conf", type=float, default=70.0, help="flag pages below this confidence")
    p.add_argument("--per-page-dir", help="also write page_0001.txt, page_0002.txt, ... here")
    p.add_argument("--json", help="also write a JSON report (text + method + confidence per page)")
    p.add_argument("--password", help="PDF password, if encrypted")
    p.add_argument("-q", "--quiet", action="store_true", help="no progress bar")
    return p


if __name__ == "__main__":
    sys.exit(run(build_parser().parse_args()))