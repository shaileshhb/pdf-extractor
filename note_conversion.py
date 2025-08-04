import json
import os

# lines = []
# # Read the JSON file
# with open("./outputs/cdsl.json", "r") as f:
#     data = json.load(f)  # each file returns a list
#     lines.extend(data)

# # Read the JSON file
# with open("./outputs/hdfc.json", "r") as f:
#     data = json.load(f)  # each file returns a list
#     lines.extend(data)

# # Convert each string into desired format
# result = [{"text": line.strip(), "label": 0} for line in lines if line.strip()]

# # # Save to a new JSON file (optional)
# with open("notes_with_labels.json", "w") as f:
#     json.dump(result, f, indent=2)



def html_to_json_lines(html_path, output_json_path):
    try:
        # Read HTML file
        with open(html_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        # Strip newlines and keep as array of strings
        lines = [line.rstrip("\n") for line in lines]

        # Save to JSON file
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as json_file:
            json.dump(lines, json_file, indent=2, ensure_ascii=False)
        
        print(f"[INFO] JSON file saved at: {output_json_path}")
    except Exception as e:
        print(f"[ERROR] Failed to convert HTML to JSON: {e}")

if __name__ == "__main__":
    HTML_INPUT_PATH = "./outputs/hdfc.html"
    JSON_OUTPUT_PATH = "./outputs/hdfc.json"
    
    html_to_json_lines(HTML_INPUT_PATH, JSON_OUTPUT_PATH)
