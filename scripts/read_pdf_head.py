import sys
from pypdf import PdfReader

def read_pdf_head(file_path, num_pages=5):
    try:
        reader = PdfReader(file_path)
        total_pages = len(reader.pages)
        print(f"File: {file_path}")
        print(f"Total Pages: {total_pages}")
        print(f"Reading first {min(num_pages, total_pages)} pages...\n")
        
        for i in range(min(num_pages, total_pages)):
            text = reader.pages[i].extract_text()
            print(f"--- Page {i+1} --- ")
            print(text[:1000]) # 페이지당 최대 1000자까지만 출력
            print("\n")
            
    except Exception as e:
        print(f"Error reading PDF: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python read_pdf_head.py <pdf_file_path>")
    else:
        read_pdf_head(sys.argv[1])

