
import fitz
import json
import os

PDF_PATH = "documents/nist_ai_rmf.pdf"
OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

doc = fitz.open(PDF_PATH)

pages = []

for page_num, page in enumerate(doc, start=1):
    text = page.get_text().strip()

    if text:
        pages.append({
            "page": page_num,
            "text": text
        })

print("PDF pages:", len(doc))
print("Pages with text:", len(pages))

# Save extracted text
with open("results/extracted_pages.json", "w", encoding="utf-8") as f:
    json.dump(pages, f, indent=2, ensure_ascii=False)

def create_chunks(pages, chunk_size, overlap=50):
    chunks = []

    for page in pages:
        words = page["text"].split()
        start = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))

            chunk_text = " ".join(words[start:end])

            chunks.append({
                "chunk_id": len(chunks),
                "page": page["page"],
                "text": chunk_text
            })

            if end == len(words):
                break

            start = end - overlap

    return chunks

# Two chunk sizes for retrieval comparison
for chunk_size in [200, 500]:
    chunks = create_chunks(pages, chunk_size, overlap=50)

    output_path = f"results/chunks_{chunk_size}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"Chunk size {chunk_size}: {len(chunks)} chunks saved.")

print("Document preparation complete.")
