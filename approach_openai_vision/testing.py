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

# Getting the file ID
file_id = create_file("./openai_vision/page_233.png")

response = client.responses.create(
    model="gpt-4.1-mini",
    input=[{
        "role": "user",
        "content": [
            {"type": "input_text", "text": "Extract all the structured content from this document page and return it in HTML format. Do not include <html>, <head>, or <body> tags. Preserve tables, lists, headings. Maintain hierarchy. Output only the HTML content."},
            {
                "type": "input_image",
                "file_id": file_id,
            },
        ],
    }],
)

print(response.output_text)