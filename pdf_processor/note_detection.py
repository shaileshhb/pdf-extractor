import re

from llm.note_detection import predict_note_header, sanitize_and_join
from llm.wrapper import fill_observations
from models.financial_models import NotesWrapper

def process_notes(notes_text):
    print(f"[INFO] Processing notes...")
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
                    "html_str": "",
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
            "html_str": ""
        }

    notes = {}
    notes["notes"] = extracted_notes

    updated_notes = fill_observations(NotesWrapper(**notes))
    notes["notes"] = updated_notes

    return notes["notes"]

def extract_note_number(text):
    # Match patterns like "33.1", "3A", or "Note 4:"
    match = re.search(
        r"\b(?:Note\s*)?(\d+(?:\.\d+)?[A-Z]?)\b", text, re.IGNORECASE)
    return match.group(1) if match else None
