import os
from pypdf import PdfReader

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    reader = PdfReader(pdf_path)
    text = ""
    
    for page in reader.pages:
        text += page.extract_text() + "\n"
    
    return text

if __name__ == "__main__":
    pdf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PathRAG Paper.pdf")
    extracted_text = extract_text_from_pdf(pdf_path)
    
    # Save the extracted text to a file
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pathrag_paper_text.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(extracted_text)
    
    print(f"Text extracted and saved to {output_path}")
    print(f"First 1000 characters:\n{extracted_text[:1000]}")
