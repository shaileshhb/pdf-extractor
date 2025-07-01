import os
import json
import fitz  # PyMuPDF
import cv2
import numpy as np
from ultralytics import YOLO
import pdfplumber
import pytesseract

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


# def crop_and_save(image, detections, page_num, base_output_dir):
#     """
#     Crop detected regions and save them as individual images.
#     """
#     crop_dir = os.path.join(base_output_dir, "crops", f"page_{page_num + 1}")
#     os.makedirs(crop_dir, exist_ok=True)

#     counter = {}  # Track how many crops of each type per page

#     for det in detections:
#         label = det['label']
#         box = det['box']
#         x1, y1, x2, y2 = box

#         # Update count for this label
#         counter[label] = counter.get(label, 0) + 1
#         count = counter[label]

#         # Crop the image
#         cropped = image[y1:y2, x1:x2]

#         # Save the crop
#         filename = f"{label}_{count}.jpg"
#         save_path = os.path.join(crop_dir, filename)
#         cv2.imwrite(save_path, cropped)
#         print(f"[CROP] Saved: {save_path}")


def crop_and_save(image, detections, page_num, base_output_dir):
    """
    Crop detected regions and save them. Also run OCR on 'Text' regions.
    """
    crop_dir = os.path.join(base_output_dir, "crops", f"page_{page_num + 1}")
    text_dir = os.path.join(base_output_dir, "text")
    os.makedirs(crop_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)

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
        save_path = os.path.join(crop_dir, filename)
        cv2.imwrite(save_path, cropped)
        print(f"[CROP] Saved: {save_path}")

        # ✅ OCR for 'Text' class
        if label == "Text":
            try:
                ocr_result = pytesseract.image_to_string(cropped, lang='eng').strip()
                if ocr_result:
                    text_lines.append(f"[{filename}]:\n{ocr_result}\n")
            except Exception as e:
                print(f"[OCR ERROR] {filename}: {e}")

    # ✅ Save OCR results to text file
    if text_lines:
        with open(text_output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(text_lines))
        print(f"[OCR] Text extracted and saved: {text_output_path}")



def extract_text_with_pdfplumber(pdf_path, detections_by_page, output_dir="output_images/text_pdfplumber"):
    os.makedirs(output_dir, exist_ok=True)

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, detections in detections_by_page.items():
            if page_num >= len(pdf.pages):
                continue

            page = pdf.pages[page_num]
            text_lines = []

            for i, det in enumerate(detections):
                if det['label'] != 'Text':
                    continue

                x1, y1, x2, y2 = det['box']

                # pdfplumber uses bottom-left origin, so convert bbox accordingly
                page_height = page.height
                crop_box = (x1, page_height - y2, x2, page_height - y1)  # (x0, top, x1, bottom)

                try:
                    cropped = page.within_bbox(crop_box)
                    extracted_text = cropped.extract_text()
                    if extracted_text:
                        text_lines.append(f"[Text_{i + 1}]:\n{extracted_text.strip()}\n")
                except Exception as e:
                    print(f"[ERROR] Failed to extract from page {page_num+1}, box {i+1}: {e}")

            # Save per-page text
            if text_lines:
                text_file = os.path.join(output_dir, f"page_{page_num + 1}.txt")
                with open(text_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(text_lines))
                print(f"[PDFPlumber] Text saved to {text_file}")


def generate_yolo_detections(pdf_path: str, model_path: str, output_dir: str = "output_images",
                            dpi: int = 300, conf_threshold: float = 0.25,
                            image_format: str = "jpg", save_json: bool = True,
                            pages: list[int] = None):
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

    total_pages = len(doc)
    selected_pages = pages if pages else range(total_pages)

    for page_num in selected_pages:
        if page_num >= total_pages:
            print(f"[WARNING] Page {page_num + 1} out of range.")
            continue

        print(f"[INFO] Processing page {page_num + 1}/{total_pages}...")
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=dpi)

        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
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

        crop_and_save(image, detections, page_num, output_dir)
        annotated = draw_boxes_on_image(image, detections)
        img_file = os.path.join(output_dir, f"page_{page_num + 1}.{image_format}")
        cv2.imwrite(img_file, annotated)
        print(f"[INFO] Saved: {img_file}\n")

        results_dict[page_num] = detections

    doc.close()

    if save_json:
        json_path = os.path.join(output_dir, "detections.json")
        with open(json_path, "w") as f:
            json.dump(results_dict, f, indent=2)
        print(f"[INFO] Detection JSON saved to {json_path}")

    print("[INFO] PDF layout detection complete.")
    return results_dict


if __name__ == '__main__':
    PDF_INPUT_PATH = "docs/cams-edited-3.pdf"
    YOLO_MODEL_PATH = "my_model/my_model.pt"

    detections = generate_yolo_detections(
        pdf_path=PDF_INPUT_PATH,
        model_path=YOLO_MODEL_PATH,
        conf_threshold=0.3,
        dpi=200,
        output_dir="output_images",
        image_format="png"
    )
    
    extract_text_with_pdfplumber(PDF_INPUT_PATH, detections)

    if detections:
        print("\n--- Detection Summary ---")
        for page, items in detections.items():
            print(f"\nPage {page + 1}:")
            if not items:
                print("  No layouts detected.")
            for det in items:
                print(f"  - {det['label']} ({det['confidence']:.2f}): {det['box']}")
    else:
        print("[INFO] No detections found.")
