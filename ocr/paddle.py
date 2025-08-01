from paddleocr import PaddleOCR, PPStructureV3
import re
from pathlib import Path
from llm.wrapper import recreate_html_table


text_rec_path = Path(__file__).parent.parent / "ai_models" / \
    "paddleocr_models" / "en_PP-OCRv4_mobile_rec_infer"
text_rec_relative_path = text_rec_path.relative_to(Path.cwd())

text_dect_path = Path(__file__).parent.parent / "ai_models" / \
    "paddleocr_models" / "PP-OCRv5_mobile_det_infer"
text_dect_relative_path = text_dect_path.relative_to(Path.cwd())

table_classification_path = Path(
    __file__).parent.parent / "ai_models" / "paddleocr_models" / "PP-LCNet_x1_0_table_cls"
table_classification_relative_path = table_classification_path.relative_to(
    Path.cwd())

slan_wired_path = Path(__file__).parent.parent / \
    "ai_models" / "paddleocr_models" / "SLANeXt_wired_infer"
slan_wired_relative_path = slan_wired_path.relative_to(Path.cwd())

slanet_wireless_path = Path(__file__).parent.parent / \
    "ai_models" / "paddleocr_models" / "SLANeXt_wireless_infer"
slanet_wireless_relative_path = slanet_wireless_path.relative_to(Path.cwd())

slanet_plus_path = Path(__file__).parent.parent / \
    "ai_models" / "paddleocr_models" / "SLANet_plus_infer"
slanet_plus_relative_path = slanet_plus_path.relative_to(Path.cwd())

doc_orientation_path = Path(__file__).parent.parent / "ai_models" / \
    "paddleocr_models" / "PP-LCNet_x1_0_doc_ori_infer"
doc_orientation_relative_path = doc_orientation_path.relative_to(Path.cwd())

layout_doc_path = Path(__file__).parent.parent / "ai_models" / \
    "paddleocr_models" / "PP-DocLayout_plus-L_infer"
pico_layout_relative_path = layout_doc_path.relative_to(Path.cwd())

pico_layout_path = Path(__file__).parent.parent / "ai_models" / \
    "paddleocr_models" / "PicoDet_layout_1x_table_infer"
pico_layout_relative_path = pico_layout_path.relative_to(Path.cwd())

text_clas_path = Path(__file__).parent.parent / "ai_models" / \
    "paddleocr_models" / "PP-LCNet_x1_0_table_cls_infer"
table_clas_relative_path = text_clas_path.relative_to(Path.cwd())

wired_table_cell_path = Path(__file__).parent.parent / "ai_models" / \
    "paddleocr_models" / "RT-DETR-L_wired_table_cell_det_infer"
wired_table_cell_relative_path = wired_table_cell_path.relative_to(Path.cwd())

wireless_table_cell_path = Path(__file__).parent.parent / "ai_models" / \
    "paddleocr_models" / "RT-DETR-L_wireless_table_cell_det_infer"
wireless_table_cell_relative_path = wireless_table_cell_path.relative_to(Path.cwd())



# PP-OCRv5_server_rec <- text recogintion
# PP-LCNet_x1_0_table_cls <- table classification model
# SLANeXt_wired <- Table Structure Recognition
# SLANet_plus <- Table Structure Recognition
# RT-DETR-L_wired_table_cell_det <- Table Cell Detection
# RT-DETR-L_wireless_table_cell_det <-  Table Cell Detection
# PP-FormulaNet_plus-L <- Formula Recognition
# PP-Chart2Table <- not found

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


table_model = PPStructureV3(
    use_formula_recognition=False,
    use_chart_recognition=False,
    use_seal_recognition=False,
    use_textline_orientation=False,

    use_doc_orientation_classify=True,
    use_doc_unwarping=True,
    use_table_recognition=True,
    use_region_detection=True,

    layout_detection_model_dir=pico_layout_path,
    layout_detection_model_name="PicoDet_layout_1x_table",

    # text recognition
    text_recognition_model_dir=text_rec_relative_path,
    text_recognition_model_name="en_PP-OCRv4_mobile_rec",

    # text detection
    text_detection_model_dir=text_dect_relative_path,
    text_detection_model_name="PP-OCRv5_mobile_det",

    wired_table_structure_recognition_model_dir=slan_wired_relative_path,
    wired_table_structure_recognition_model_name="SLANeXt_wired",

    wireless_table_structure_recognition_model_dir=slanet_plus_relative_path,
    wireless_table_structure_recognition_model_name="SLANet_plus",

    table_classification_model_dir=table_clas_relative_path,
    table_classification_model_name="PP-LCNet_x1_0_table_cls",

    wired_table_cells_detection_model_dir=wired_table_cell_relative_path,
    wired_table_cells_detection_model_name="RT-DETR-L_wired_table_cell_det",

    wireless_table_cells_detection_model_dir=wireless_table_cell_relative_path,
    wireless_table_cells_detection_model_name="RT-DETR-L_wireless_table_cell_det",
)

# picodet_lcnet_x1_0_fgd_layout_table_infer
def extract_text(image):
    try:
        result = text_model.predict(image)
        if not result or not isinstance(result, list) or len(result) == 0:
            print("[OCR ERROR] No result returned from text_model.predict.")
            return ""
        if 'rec_texts' not in result[0] or not result[0]['rec_texts']:
            print("[OCR ERROR] 'rec_texts' missing or empty in result.")
            return ""
        extracted = "\n".join(result[0]['rec_texts'])
        return extracted
    except Exception as e:
        print(f"[OCR ERROR] {e}")
        return ""


def extract_table(img):
    try:
        result = text_model.predict(img)
        if not result or not isinstance(result, list) or len(result) == 0:
            print("[OCR ERROR] No result returned from text_model.predict.")
            return ""
        if 'rec_texts' not in result[0] or not result[0]['rec_texts']:
            print("[OCR ERROR] 'rec_texts' missing or empty in result.")
            return ""
        
        # return "\n".join(result[0]['rec_texts'])
        extracted = result[0]['rec_texts']
        html_table = recreate_html_table(extracted)
        return html_table
    except Exception as e:
        print(f"[OCR ERROR] {e}")
    # result = table_model.predict(img)

    # table_res_list = result[0].get("table_res_list", [])

    # if not table_res_list:
    #     print(f"No table detected in {img}")
    #     return None

    # pred_html = table_res_list[0].get("pred_html", "")

    # for idx, table_info in enumerate(result[0].get("table_res_list", [])):
    #     print(f"Table #{idx+1} used model: {table_info.get('structure_type')}")
    #     print("HTML output:\n", table_info.get("pred_html"))

    # # pred_html = result[0]['table_res_list'][0]['pred_html']
    # return extract_html_body(pred_html)


def extract_html_body(pred_html):
    match = re.search(r'<body[^>]*>(.*?)</body>', pred_html, re.DOTALL | re.IGNORECASE)
    if match:
        body_content = match.group(1).strip()
        return body_content
    else:
        return ""
