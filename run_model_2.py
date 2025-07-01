import os
import fitz  # PyMuPDF
import cv2
import numpy as np
from ultralytics import YOLO


def draw_boxes_on_image(image, detections):
    """
    Draws bounding boxes and labels on the image.

    Args:
        image (np.ndarray): The original image.
        detections (list): List of detections with "box" and "label".

    Returns:
        np.ndarray: Annotated image.
    """
    for detection in detections:
        x1, y1, x2, y2 = detection['box']
        label = detection['label']
        confidence = detection['confidence']
        
        # Color coding by class
        color = (0, 255, 0) if label == "Text" else (255, 0, 0) if label == "Image" else (0, 0, 255)

        # Draw rectangle and label
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image


def generate_yolo_detections(pdf_path: str, model_path: str, output_dir: str = "output_images"):
    """
    Processes a PDF with a trained YOLO model to detect layouts and save annotated images.

    Args:
        pdf_path (str): Path to the input PDF file.
        model_path (str): Path to the trained YOLO model weights (.pt file).
        output_dir (str): Directory to save annotated images.

    Returns:
        dict: A dictionary with page numbers as keys and detections as values.
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        model = YOLO(model_path)
        print("Model Class Names:", model.names)
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return {}

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening PDF file: {e}")
        return {}
    
    yolo_model_output = {}

    for page_num in range(len(doc)):
        print(f"Processing page {page_num + 1}/{len(doc)}...")
        page = doc.load_page(page_num)
        pix = page.get_pixmap(dpi=300)

        img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        image_np = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)

        results = model(image_np)
        result = results[0]

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        page_detections = []
        for i in range(len(boxes)):
            box = [int(coord) for coord in boxes[i]]
            class_id = int(class_ids[i])
            label = model.names[class_id]
            page_detections.append({
                "box": box,
                "label": label,
                "confidence": float(confs[i])
            })

        # Save annotated image
        annotated_image = draw_boxes_on_image(image_np.copy(), page_detections)
        output_image_path = os.path.join(output_dir, f"page_{page_num + 1}.jpg")
        cv2.imwrite(output_image_path, annotated_image)
        print(f"Saved: {output_image_path}")

        yolo_model_output[page_num] = page_detections

    print("PDF processing complete.")
    doc.close()
    return yolo_model_output


if __name__ == '__main__':
    PDF_INPUT_PATH = "docs/cams-edited-2.pdf"
    YOLO_MODEL_PATH = "my_model/my_model.pt"

    detections = generate_yolo_detections(pdf_path=PDF_INPUT_PATH, model_path=YOLO_MODEL_PATH)

    if detections:
        print("\n--- Generated Detections ---")
        for page_num, page_content in detections.items():
            print(f"\n--- Page {page_num + 1} ---")
            if not page_content:
                print("No layouts detected.")
                continue
            for item in page_content:
                print(f"  - Detected '{item['label']}' with confidence {item['confidence']:.2f} at bbox: {item['box']}")
    else:
        print("No detections found.")
