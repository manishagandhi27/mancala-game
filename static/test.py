import os
import json
import re
from docx import Document
from PIL import Image
import pytesseract
from langchain.document_loaders import UnstructuredWordDocumentLoader

# ----------- Step 1: Process Text, Lists, and Headings -----------

def parse_word_doc(file_path):
    """Extracts headings, paragraphs, and lists from a Word document."""
    doc = Document(file_path)
    extracted_data = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        
        if para.style.name.startswith("Heading"):
            extracted_data.append({"type": "heading", "content": text})
        elif para.style.name.startswith("List") or re.match(r"^\d+\.", text) or text.startswith("- "):
            extracted_data.append({"type": "list_item", "content": text})
        else:
            extracted_data.append({"type": "paragraph", "content": text})

    return extracted_data

# ----------- Step 2: Extract Tables -----------

def extract_tables(file_path):
    """Extracts tables from a Word document and returns structured table data."""
    doc = Document(file_path)
    tables = []

    for table in doc.tables:
        table_data = []
        for row in table.rows:
            row_data = [cell.text.strip() for cell in row.cells]
            table_data.append(row_data)
        tables.append({"type": "table", "content": table_data})

    return tables

# ----------- Step 3: Extract Images & Perform OCR -----------

def extract_images_from_docx(doc_path, output_folder="extracted_images"):
    """Extracts images from a Word document and applies OCR for text extraction."""
    doc = Document(doc_path)
    os.makedirs(output_folder, exist_ok=True)
    
    image_data = []
    for i, shape in enumerate(doc.inline_shapes):
        try:
            img_path = os.path.join(output_folder, f"image_{i}.png")
            with open(img_path, "wb") as f:
                f.write(shape._inline.graphic.graphicData.pic.blipFill.blip.blob)
            
            # Apply OCR to extract text from the image
            img_text = pytesseract.image_to_string(Image.open(img_path)).strip()
            image_data.append({"type": "image", "image_path": img_path, "extracted_text": img_text})
        except Exception as e:
            print(f"Error extracting image {i}: {e}")

    return image_data

# ----------- Step 4: Alternative - Use LangChain Unstructured Loader -----------

def load_using_langchain(file_path):
    """Uses UnstructuredWordDocumentLoader to extract text content."""
    loader = UnstructuredWordDocumentLoader(file_path)
    docs = loader.load()

    structured_data = []
    
    for doc in docs:
        lines = doc.page_content.split("\n")
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if line.endswith(":"):  # Heading-like text
                structured_data.append({"type": "heading", "content": line})
            elif line.startswith("- ") or re.match(r"^\d+\.", line):  # List detection
                structured_data.append({"type": "list_item", "content": line})
            else:
                structured_data.append({"type": "paragraph", "content": line})

    return structured_data

# ----------- Step 5: Full Document Processing -----------

def process_document(file_path):
    """Combines all extraction functions for a fully processed document."""
    parsed_text = parse_word_doc(file_path)
    tables = extract_tables(file_path)
    images = extract_images_from_docx(file_path)
    
    # Merge results
    final_output = parsed_text + tables + images
    
    return final_output

# ----------- Execute the Code -----------

if __name__ == "__main__":
    file_path = "sample.docx"  # Change to your actual document path
    structured_content = process_document(file_path)
    
    # Pretty print structured output
    print(json.dumps(structured_content, indent=2))
