import fitz  # PyMuPDF
import os

# Define the folder containing your PDFs
PDF_FOLDER = "/Users/balachikkala/New_bot/articles"

def extract_text_from_pdf(pdf_path):
    """Extract text from a single PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text.strip()

# Dictionary to store extracted text
pdf_texts = {}

# Loop through all PDFs in the folder and extract text
for filename in os.listdir(PDF_FOLDER):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(PDF_FOLDER, filename)
        pdf_texts[filename] = extract_text_from_pdf(pdf_path)
        print(f"✅ Processed: {filename}")

# Preview first 500 characters of each document
for name, text in pdf_texts.items():
    print(f"\n📜 {name} Preview:\n{text[:500]}\n{'-'*50}")

# Save extracted text for later processing
import json
with open("pdf_texts.json", "w") as f:
    json.dump(pdf_texts, f)
    
print("✅ Extracted text saved to pdf_texts.json")
