from paddleocr import PaddleOCR
import re
from pathlib import Path
from llm.wrapper import recreate_html_table


text_rec_path = Path(__file__).parent.parent / "model" / \
    "paddle_ocr" / "en_PP-OCRv4_mobile_rec_infer"
text_rec_relative_path = text_rec_path.relative_to(Path.cwd())

text_dect_path = Path(__file__).parent.parent / "model" / \
    "paddle_ocr" / "PP-OCRv5_mobile_det_infer"
text_dect_relative_path = text_dect_path.relative_to(Path.cwd())


# PP-OCRv5_server_rec <- text recogintion
# PP-LCNet_x1_0_table_cls <- table classification model
# SLANeXt_wired <- Table Structure Recognition
# SLANet_plus <- Table Structure Recognition
# RT-DETR-L_wired_table_cell_det <- Table Cell Detection
# RT-DETR-L_wireless_table_cell_det <-  Table Cell Detection
# PP-FormulaNet_plus-L <- Formula Recognition
# PP-Chart2Table <- not found
# picodet_lcnet_x1_0_fgd_layout_table_infer

# PP-DocBlockLayout
# PP-DocLayout_plus-L
# RT-DETR-L_wired_table_cell_det
# RT-DETR-L_wireless_table_cell_det
text_model = PaddleOCR(
    lang='en',
    use_textline_orientation=False,
    use_doc_unwarping=False,
    use_doc_orientation_classify=False,

    # text detection
    text_detection_model_dir=text_dect_relative_path,
    text_detection_model_name="PP-OCRv5_mobile_det",

    # text recognition
    text_recognition_model_dir=text_rec_relative_path,
    text_recognition_model_name="en_PP-OCRv4_mobile_rec",
)

def extract_text(image):
    try:
        result = text_model.predict(image)
        extracted = result[0]['rec_texts']
        return extracted
    except Exception as e:
        print(f"[OCR ERROR] {e}")


def extract_table(img):
    try:
        result = text_model.predict(img)
        text_result = result[0]['rec_texts']
        joined_text = "\n".join(text_result)
        html_table = recreate_html_table(joined_text)
        return html_table
    except Exception as e:
        print(f"[OCR ERROR] {e}")


def extract_html_body(pred_html):
    match = re.search(r'<body[^>]*>(.*?)</body>', pred_html, re.DOTALL | re.IGNORECASE)
    if match:
        body_content = match.group(1).strip()
        return body_content
    else:
        return ""
