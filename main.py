import os
import json
import fitz  # PyMuPDF
import cv2
import numpy as np
from ultralytics import YOLO
from helper.pdf import detect_and_fix_pdf
from llm.note_detection import (process_notes, sanitize_and_join)
from ocr.paddle import (
    extract_text,
    extract_table,
)
from dotenv import load_dotenv


# Load environment variables
load_dotenv()

# Class-to-color map for drawing boxes
CLASS_COLOR_MAP = {
    "Text": (0, 255, 0),
    "Image": (255, 0, 0),
    "Table": (0, 0, 255),
}


def draw_boxes_on_image(image, detections):
    """
    Draws bounding boxes and labels on the image.
    """
    for det in detections:
        x1, y1, x2, y2 = det['box']
        label = det['label']
        conf = det['confidence']
        color = CLASS_COLOR_MAP.get(label, (128, 128, 128))

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image


def crop_and_save(image, detections, page_num, base_output_dir):
    """
    Crop detected regions and save them. Also run OCR on 'Text' regions.
    """
    text_dir = os.path.join(base_output_dir, "text")
    # table_dir = os.path.join(base_output_dir, "tables")

    os.makedirs(text_dir, exist_ok=True)
    # os.makedirs(table_dir, exist_ok=True)

    text_output_path = os.path.join(text_dir, f"page_{page_num + 1}.txt")
    text_lines = []

    counter = {}

    for det in detections:
        label = det['label']
        box = det['box']
        x1, y1, x2, y2 = box

        counter[label] = counter.get(label, 0) + 1
        count = counter[label]

        cropped = image[y1:y2, x1:x2]
        filename = f"{label}_{count}.jpg"

        # OCR for 'Text' class
        if label == "Text":
            try:
                text_result = extract_text(cropped)

                if text_result:
                    text_lines.append(f"{text_result}\n")
            except Exception as e:
                print(f"[TEXT OCR ERROR] {filename}: {e}")

        if label == "Table":
            try:
                table_result = extract_table(cropped)

                if table_result:
                    text_lines.append(f"{table_result}\n")
            except Exception as e:
                print(f"[TABLE OCR ERROR] {filename}: {e}")

    # ✅ Save OCR results to text file
    if text_lines:
        with open(text_output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(text_lines))
        print(f"[OCR] Text extracted and saved: {text_output_path}")

    return sanitize_and_join(text_lines)


def generate_yolo_detections(
        pdf_path: str, model_path: str, output_dir: str = "output_images",
        dpi: int = 300, conf_threshold: float = 0.25,
        image_format: str = "jpg", save_json: bool = True,
        pages: list[int] = None, start_page: int = 0, end_page: int = None):
    """
    Processes a PDF with YOLO layout detection and saves annotated images.
    """
    os.makedirs(output_dir, exist_ok=True)
    results_dict = {}

    try:
        model = YOLO(model_path)
        class_names = model.names
        print("Loaded model classes:", class_names)
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO model: {e}")
        return {}

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[ERROR] Failed to open PDF: {e}")
        return {}

    # total_pages = len(doc)
    # selected_pages = pages if pages else range(total_pages)
    total_pages = len(doc)
    if end_page is None:
        end_page = total_pages

    start_page = max(start_page, 0)
    end_page = min(end_page, total_pages)
    text_lines = []

    if pages:
        selected_pages = [p for p in pages if start_page <= p < end_page]
    else:
        selected_pages = range(start_page, end_page)

    for page_num in selected_pages:
        if page_num >= end_page:
            print(f"[WARNING] Page {page_num + 1} out of range.")
            continue

        print(f"[INFO] Processing page {page_num + 1}/{end_page}...")
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=dpi)

        img_data = np.frombuffer(
            pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        image = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

        result = model(image)[0]
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        detections = []
        for i in range(len(boxes)):
            if confs[i] < conf_threshold:
                continue
            box = [int(c) for c in boxes[i]]
            label = class_names[int(class_ids[i])]
            detections.append({
                "box": box,
                "label": label,
                "confidence": float(confs[i])
            })

        section_text = crop_and_save(image, detections, page_num, output_dir)
        text_lines.append(section_text)
        results_dict[page_num] = detections

    doc.close()

    print("[INFO] PDF layout detection complete.")
    return text_lines


def process_pdf(pdf_path, model_path, output_dir="output_images",
                dpi=300, conf_threshold=0.25, image_format="png",
                save_json=True, pages=None, start_page=0, end_page=None):
    """
    Main function to process the PDF and generate YOLO detections.
    """
    text_lines = generate_yolo_detections(
        pdf_path, model_path, output_dir, dpi, conf_threshold,
        image_format, save_json, pages, start_page, end_page
    )

    extracted_notes = []
    output = {}
    section = "notes"

    # Iterate through each multi-line string in the original array
    for str in text_lines:
        # Split the string into individual lines using newline as the delimiter
        lines = str.split('\n')
        # Extend the result list with the individual lines
        extracted_notes.extend(lines)

    json_path = os.path.join("./output_images", "extracted_notes.json")
    with open(json_path, "w") as f:
        json.dump(extracted_notes, f, indent=2)
    print(f"[INFO] Detection JSON saved to {json_path}")

    output[section] = process_notes(extracted_notes)
    return output


if __name__ == '__main__':
    PDF_INPUT_PATH = "docs/fixed.pdf"
    YOLO_MODEL_PATH = "ai_models/layout_detection.pt"

    fixed_path = detect_and_fix_pdf(PDF_INPUT_PATH, "./docs/fixed.pdf", 90)

    notes = process_pdf(
        pdf_path=PDF_INPUT_PATH,
        model_path=YOLO_MODEL_PATH,
        dpi=200,
        conf_threshold=0.3,
        start_page=179, end_page=240,
    )

    # notes = generate_yolo_detections(
    #     pdf_path=PDF_INPUT_PATH,
    #     model_path=YOLO_MODEL_PATH,
    #     conf_threshold=0.3,
    #     dpi=200,
    #     output_dir="output_images",
    #     image_format="png",
    #     start_page=186, end_page=190
    #     # start_page=174, end_page=225
    # )

    # if detections:
    #     print("\n--- Detection Summary ---")
    #     for page, items in detections.items():
    #         print(f"\nPage {page + 1}:")
    #         if not items:
    #             print("  No layouts detected.")
    #         for det in items:
    #             print(
    #                 f"  - {det['label']} ({det['confidence']:.2f}): {det['box']}")
    # else:
    # print("[INFO] Notes - ", notes)
       
    json_path = os.path.join("./output_images", "notes.json")
    with open(json_path, "w") as f:
        json.dump(notes, f, indent=2)
    print(f"[INFO] Detection JSON saved to {json_path}")

