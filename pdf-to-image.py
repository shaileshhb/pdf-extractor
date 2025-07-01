from pdf2image import convert_from_path
import os

# === Configuration ===
pdf_path = 'docs/cams.pdf'               # Path to your PDF file
output_folder = 'docs/cams'       # Folder to save images
dpi = 200                             # Image quality (recommended: 200+)

# === Ensure output folder exists ===
os.makedirs(output_folder, exist_ok=True)

# === Convert PDF to list of images ===
images = convert_from_path(pdf_path, dpi=dpi)

# === Save each page as image ===
for i, img in enumerate(images):
    image_path = os.path.join(output_folder, f'page_{i + 1}.png')
    img.save(image_path, 'PNG')
    print(f'Saved {image_path}')
