import os
import json
import sys
from pathlib import Path
import fitz  # PyMuPDF


def clean_text(text):

    text = ' '.join(text.split())

    return text

def extract_text_from_pdf(pdf_path):

    pages_data = []
    
    try:
        # Open the PDF
        pdf_document = fitz.open(pdf_path)
        filename = os.path.basename(pdf_path)
        
        print(f"Processing: {filename}")
        
        # Extract text from each page
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text = page.get_text()
            
            # Clean the text
            cleaned_text = clean_text(text)
            char_count = len(cleaned_text)
            
            # Create entry for this page
            page_entry = {
                "source": filename,
                "page": page_num + 1,  # Pages are 1-indexed
                "char_count": char_count,
                "text": cleaned_text
            }
            
            pages_data.append(page_entry)
        
        pdf_document.close()
        return pages_data
    
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return []

def main():
    """Main function to extract all PDFs"""
    

    pdfs_folder = Path("data/pdfs")
    output_file = Path("data/corpus.json")
    sample_file = Path("data/sample.json")
    

    if not pdfs_folder.exists():
        print(f"Error: {pdfs_folder} folder not found!")
        print("Please place your PDFs in the data/pdfs/ folder")
        sys.exit(1)
    

    pdf_files = list(pdfs_folder.glob("*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {pdfs_folder}")
        sys.exit(1)
    
    print(f"Found {len(pdf_files)} PDF files\n")
    

    all_pages = []
    total_pdfs = 0
    empty_pages = []
    
    for pdf_path in sorted(pdf_files):
        pages = extract_text_from_pdf(pdf_path)
        if pages:
            all_pages.extend(pages)
            total_pdfs += 1
        

        for page_data in pages:
            if page_data["char_count"] < 50:
                empty_pages.append({
                    "source": page_data["source"],
                    "page": page_data["page"],
                    "char_count": page_data["char_count"]
                })
    
    if not all_pages:
        print("No pages extracted. Check your PDFs.")
        sys.exit(1)
    

    Path("data").mkdir(exist_ok=True)
    
    # Save full corpus
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_pages, f, indent=2, ensure_ascii=False)
    print(f"+ Saved full corpus to {output_file}")
    

    sample_pages = all_pages[:10]
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(sample_pages, f, indent=2, ensure_ascii=False)
    print(f"+ Saved sample to {sample_file}")
    

    total_pages = len(all_pages)
    total_chars = sum(page["char_count"] for page in all_pages)
    avg_chars_per_page = total_chars / total_pages if total_pages > 0 else 0
    
    print("\n" + "="*50)
    print("EXTRACTION STATISTICS")
    print("="*50)
    print(f"Number of documents (PDFs): {total_pdfs}")
    print(f"Total pages: {total_pages}")
    print(f"Total characters: {total_chars:,}")
    print(f"Average characters per page: {avg_chars_per_page:.0f}")
    print(f"Empty/near-empty pages (< 50 chars): {len(empty_pages)}")
    
    if empty_pages:
        print("\nEmpty pages found:")
        for ep in empty_pages:
            print(f"  - {ep['source']} page {ep['page']} ({ep['char_count']} chars)")
    
    print("="*50)

if __name__ == "__main__":
    main()
