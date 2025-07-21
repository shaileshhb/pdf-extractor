import os
import copy
import pdfplumber
from PyPDF2.generic import RectangleObject
from PyPDF2 import PdfReader, PdfWriter


def detect_and_fix_pdf(
    input_pdf: str,
    output_pdf: str,
    angle: int = 90,
    rotate_threshold: float = 0.0,
    split_ar_threshold: float = 1.3
) -> str:
    """
    1) If output_pdf exists, return it.
    2) Otherwise detect pages (or halves of pages) with rotated text and rotate them.
       - If a page’s aspect‐ratio (width/height) > split_ar_threshold, treat it as “two A5s
         side by side” and split into left+right halves. Check each half for rotation.
       - Otherwise, check the full page for rotation.
    3) Write all resulting pages (or split+rotated halves) into output_pdf in order.
    """
    # Early exit if already done
    if os.path.exists(output_pdf):
        print(f"✅ Fixed PDF already exists: {output_pdf}")
        return output_pdf

    reader = PdfReader(input_pdf)
    writer = PdfWriter()

    with pdfplumber.open(input_pdf) as plumber_pdf:
        total_pages = len(plumber_pdf.pages)

        for page_idx in range(total_pages):
            pm_page = plumber_pdf.pages[page_idx]    # pdfplumber page for analysis
            py_page = reader.pages[page_idx]         # PyPDF2 PageObject for writing/cropping

            w = float(pm_page.width)
            h = float(pm_page.height)

            # If the page is “wide” enough to be two A5‐style halves:
            if (w / h) > split_ar_threshold:
                # Define left & right bounding boxes for cropping:
                left_box  = RectangleObject((0,    0,   w/2, h))
                right_box = RectangleObject((w/2,  0,   w,   h))

                # For detection, grab the corresponding halves via pdfplumber:
                left_pm  = pm_page.within_bbox((0,    0,   w/2, h))
                right_pm = pm_page.within_bbox((w/2,  0,   w,   h))

                # Now clone the PyPDF2 page and assign its mediabox to “left_box”:
                left_py = copy.copy(py_page)
                left_py.mediabox = left_box
                # If more than rotate_threshold of chars are tilted in left_pm:
                if is_rotated_text(left_pm, threshold=rotate_threshold):
                    left_py.rotate(angle)
                writer.add_page(left_py)

                # Similarly for the right half:
                right_py = copy.copy(py_page)
                right_py.mediabox = right_box
                if is_rotated_text(right_pm, threshold=rotate_threshold):
                    right_py.rotate(angle)
                writer.add_page(right_py)

            else:
                # Not a “wide” page: check entire page for rotation:
                if is_rotated_text(pm_page, threshold=rotate_threshold):
                    py_page.rotate(angle)
                writer.add_page(py_page)

    # Write out the new PDF
    with open(output_pdf, "wb") as out_f:
        writer.write(out_f)

    print(f"🔁 Rotated (and split) pages written to: {output_pdf}")
    return output_pdf

def is_rotated_text(page: pdfplumber.page.Page, threshold: float = 0.2) -> bool:
    """
    Return True if more than `threshold` fraction of the page's characters
    are drawn non-upright.  (Default threshold=0.2 means “if >20% of chars
    are tilted, treat page as rotated.”)
    """
    chars = page.chars
    if not chars:
        return False

    # count how many characters are “non-upright”
    non_upright = sum(1 for ch in chars if not ch.get("upright", True))
    ratio = non_upright / max(len(chars), 1)

    print(f"[debug] page half tilt ratio = {ratio:.2f}")

    return (non_upright / len(chars)) > threshold
