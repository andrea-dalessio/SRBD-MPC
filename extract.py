import sys
import os

pdf_path = r"c:\Users\Francesco\.antigravity\SRBD-MPC\Papers\A_Real-Time_Approach_for_Humanoid_Robot_Walking_Including_Dynamic_Obstacles_Avoidance.pdf"
out_path = r"c:\Users\Francesco\.antigravity\SRBD-MPC\paper_extracted_text.txt"

def extract_fitz():
    import fitz
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print("Extracted using PyMuPDF (fitz) to paper_extracted_text.txt")

def extract_pypdf2():
    import PyPDF2
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Extracted using PyPDF2 to paper_extracted_text.txt. Extracted {len(text)} characters.")

try:
    extract_fitz()
    sys.exit(0)
except ImportError:
    pass

try:
    extract_pypdf2()
    sys.exit(0)
except ImportError:
    pass

print("FAILED: No known PDF extraction library (PyMuPDF or PyPDF2) is installed. You can install one with: pip install PyMuPDF")
