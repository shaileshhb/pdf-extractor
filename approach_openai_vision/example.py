import os
import fitz  # PyMuPDF
import base64
import numpy as np
import cv2
import json
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

client = OpenAI()

# Function to create a file with the Files API


def create_file(file_path):
    with open(file_path, "rb") as file_content:
        result = client.files.create(
            file=file_content,
            purpose="vision",
        )
    return result.id


# Function to encode the image
def encode_image(image_file):
    try:
        _, buffer = cv2.imencode(".jpg", image_file)
        return base64.b64encode(buffer).decode("utf-8")
    except Exception as e:
        print(f"[ERROR] Failed to encode image: {e}")
        return ""


def extract_html_from_img(image_file):
    base64_image = encode_image(image_file)
    if not base64_image:
        return ""

    try:
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=[{
                "role": "user",
                "content": [{
                    "type": "input_text", 
                    "text": 
                    """
                    You are an AI assistant extracting content from an image of a financial document (e.g., an annual report). 
                
                    **Task:** 
                    Extract all visible text and tables from this page and return them as well-structured semantic HTML.

                    **Strict OCR-like behavior:** 
                    - Do NOT infer, estimate, or guess any content. If text or numbers are unclear or unreadable, leave them blank or indicate with [UNREADABLE].
                    - Do NOT hallucinate or fabricate numbers or words.
                    - Extract EVERYTHING visible — do not skip footnotes, captions, or small text blocks.
                    
                    **Requirements:** 
                    - Do NOT include <html>, <head>, or <body> tags. 
                    - Preserve content hierarchy using semantic tags: <h1>, <h2>, <h3>, <p>, <ul>, <ol>, <table>, <thead>, <tbody>, <th>, <td>, etc.
                    - Maintain the exact structure, order, and alignment as seen in the document.
                    - Retain formatting cues such as bold (<b>), italic (<i>), and underline (<u>).
                    - Preserve multi-level/nested tables if present, using nested <table> tags.
                    - Maintain line breaks (<br>) where necessary to reflect layout.
                    - Do NOT summarize, infer, or invent any content. Only extract what is visible.
                    - Do NOT assume or add groupings that are not explicitly present.
                    - Extract numbers and financial values exactly as shown (no rounding or reformatting).
                    - Include footnotes, captions, and smaller text blocks if present.
                    - Ignore page numbers, headers, footers, and watermarks unless they contain meaningful content.
                    - For any diagram or chart with visible text, extract and include the text in <p> or <figcaption>.

                    **Output:** 
                    Provide ONLY the structured HTML content (no extra explanation, no wrapping <html>, <head>, or <body> tags).
                    """
                    }, {
                        "type": "input_image", 
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    }],
            }],
        )
        return response.output_text
    except Exception as e:
        print(f"[GPT VISION API ERROR]: {e}")
        return ""


def process_pdf(pdf_path, start_page=0, end_page=None):
    """
    Extract HTML from each page in the given PDF and return a list of HTML strings.
    """
    text_lines = []

    # Open PDF safely
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[ERROR] Failed to open PDF: {e}")
        return text_lines

    total_pages = len(doc)
    if end_page is None:
        end_page = total_pages

    start_page = max(start_page, 0)
    end_page = min(end_page, total_pages)

    selected_pages = range(start_page, end_page)

    for page_num in selected_pages:
        try:
            print(f"[INFO] Processing page {page_num + 1}/{end_page}...")
            page = doc.load_page(page_num)
            pix = page.get_pixmap(dpi=300)
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            image = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

            # should we add a check to which checks if table exist in current page. if no table then use paddleocr to extract text as it is cheaper
            # the only problem with this approach is with the sequencing of extracted text

            output = extract_html_from_img(image)
            if output.strip():
                text_lines.append(output)
            else:
                print(f"[WARNING] No content extracted from page {page_num + 1}.")
        except Exception as e:
            print(f"[ERROR] Failed to process page {page_num + 1}: {e}")
            continue

    return text_lines


if __name__ == '__main__':
    PDF_INPUT_PATH = "docs/bkt_fixed.pdf"
    OUTPUT_JSON_PATH = "./outputs/bkt.json"
    OUTPUT_HTML_PATH = "./outputs/bkt.html"

    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)

    # Extract notes
    notes = process_pdf(
        pdf_path=PDF_INPUT_PATH,
        start_page=170,
        end_page=219,
    )

    # Save as JSON
    try:
        with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as file:
            json.dump(notes, file, indent=2, ensure_ascii=False)
        print(f"[INFO] HTML JSON saved to {OUTPUT_JSON_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to save JSON: {e}")

    # Save as HTML
    try:
        combined_html = "\n<hr>\n".join(notes)  # Separate pages with a horizontal rule
        with open(OUTPUT_HTML_PATH, "w", encoding="utf-8") as file:
            file.write(combined_html)
        print(f"[INFO] Combined HTML saved to {OUTPUT_HTML_PATH}")
    except Exception as e:
        print(f"[ERROR] Failed to save HTML: {e}")