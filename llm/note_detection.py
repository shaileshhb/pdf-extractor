from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
from pathlib import Path
from llm.wrapper import fill_observations
from models.financial_models import NotesWrapper

# 1. Load from the folder where you saved
mode_dir_path = Path(__file__).parent.parent / \
    "ai_models" / "note_header_model"
model_dir_relative_path = mode_dir_path.relative_to(Path.cwd())


tokenizer = AutoTokenizer.from_pretrained(model_dir_relative_path)
model = AutoModelForSequenceClassification.from_pretrained(model_dir_relative_path)
model.eval()  # set to inference mode


def predict_note_header(text: str) -> dict:
    """
    Returns a dict with:
      - label: int (predicted class)
      - score: float (softmax probability)
      - logits: List[float] (raw model logits)
    """
    # Tokenize + build batched tensors
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",  # pad to your chosen max_length
        max_length=64,
    )
    # If using GPU, uncomment:
    # inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0]  # shape: [num_labels]
    probs = torch.nn.functional.softmax(logits, dim=-1)

    predicted_class = torch.argmax(probs).item()
    confidence = probs[predicted_class].item()

    # return {"label": predicted_class, "score": confidence, "logits": logits.tolist()}
    if predicted_class == 1:
        return True

    return False


def process_notes(notes_text):
    print(f"[INFO] Processing {len(notes_text)} notes...")
    extracted_notes = {}
    note_text = []
    current_note_number = 0
    collecting = False

    for text in notes_text:
        result = predict_note_header(text)
        if result:
            if collecting:
                description = sanitize_and_join(note_text)

                extracted_notes[current_note_number] = {
                    "note_no": current_note_number,
                    "description": description,
                    "observation": "",
                }
                note_text = []
                note_num = extract_note_number(text)
                if note_num:
                    current_note_number = note_num
            else:
                collecting = True  # Start collecting after the first match
                note_num = extract_note_number(text)
                if note_num:
                    current_note_number = note_num

        if collecting:
            note_text.append(text)

    if note_text:
        description = sanitize_and_join(note_text)

        extracted_notes[current_note_number] = {
            "note_no": current_note_number,
            "description": description,
            "observation": "",
        }

    notes = {}
    notes["notes"] = extracted_notes

    # updated_notes = fill_observations(NotesWrapper(**notes))
    # notes["notes"] = updated_notes

    return notes["notes"]


def extract_note_number(text):
    # Match patterns like "33.1", "3A", or "Note 4:"
    match = re.search(
        r"\b(?:Note\s*)?(\d+(?:\.\d+)?[A-Z]?)\b", text, re.IGNORECASE)
    return match.group(1) if match else None

def append_line(existing_block: str, new_line: str) -> str:
    # Clean the new line
    new_line = new_line.strip()

    if not new_line:
        return existing_block  # Skip empty lines

    if not existing_block:
        return new_line  # First line — no newline needed

    return existing_block.rstrip() + "\n" + new_line

def sanitize_and_join(lines):
    cleaned_lines = [line.strip() for line in lines if line.strip() != ""]
    return "\n".join(cleaned_lines)