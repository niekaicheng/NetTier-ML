import PyPDF2
import sys

def extract_text(pdf_path, out_path):
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        with open(out_path, 'w', encoding='utf-8') as out_f:
            for i, page in enumerate(reader.pages):
                out_f.write(f"\n--- Page {i+1} ---\n\n")
                out_f.write(page.extract_text() or '')

if __name__ == "__main__":
    if len(sys.argv) > 2:
        extract_text(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python extract.py <input.pdf> <output.txt>")
