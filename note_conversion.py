import json

# Read the JSON file
with open("./output_images/extracted_notes.json", "r") as f:
    lines = json.load(f)  # This gives a Python list

# Convert each string into desired format
result = [{"text": line.strip(), "label": 0} for line in lines if line.strip()]

# Save to a new JSON file (optional)
with open("notes_with_labels.json", "w") as f:
    json.dump(result, f, indent=2)

print(result)