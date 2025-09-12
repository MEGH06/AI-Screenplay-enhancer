from docx import Document
import fitz

def extract_docx_text(docx_path):
    """Extract text from DOCX file (without image processing)"""
    doc = Document(docx_path)
    all_text = []
    # Extract regular text
    full_text = []
    for paragraph in doc.paragraphs:
        if paragraph.text.strip():
            full_text.append(paragraph.text)
    if full_text:
        all_text.append("Document Text:\n" + '\n'.join(full_text))
    return '\n\n'.join(all_text)

def extract_pdf_text(pdf_path):
    """Extract text from PDF (without image or table processing)"""
    doc = fitz.open(pdf_path)
    all_text = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text()
        if page_text.strip():
            all_text.append(f"Page {page_num + 1}:\n{page_text}")
    doc.close()
    return '\n\n'.join(all_text)

def extract_txt_text(file):
    """Extract text from TXT file"""
    try:
        with open(file, 'r', encoding='utf-8') as file:
            content = file.read()
        return content.strip()
    except UnicodeDecodeError:
        # Try with different encoding if utf-8 fails
        with open(file, 'r', encoding='latin-1') as file:
            content = file.read()
        return [content.strip()] if content.strip() else []
    except Exception as e:
        print(f"Error reading text file: {e}")
        return ""