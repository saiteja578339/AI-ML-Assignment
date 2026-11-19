import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# Use CPU because this computer does not have an NVIDIA GPU.
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Source pages verified against the NIST PDF.
# These are physical PDF page numbers.
gold_pages = {
    1: [7],
    2: [17],
    3: [8],
    4: [27],
    5: [30],
    6: [33],
    7: [36, 37],
    8: [19],
    9: [22],
    10: [28],
    11: [23],
    12: [30],
    13: [38],
    14: [22],
    15: [7]
}

os.makedirs("results", exist_ok=True)

with open("results/questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME, device="cpu")

def evaluate(chunk_file):
    print(f"\nEvaluating {chunk_file}...")

    with open(chunk_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    chunk_texts = [chunk["text"] for chunk in chunks]
    question_texts = [q["question"] for q in questions]

    print(f"Embedding {len(chunks)} document chunks...")
    chunk_embeddings = model.encode(
        chunk_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("Embedding questions...")
    question_embeddings = model.encode(
        question_texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Cosine similarity: normalized vectors + dot product.
    similarity_matrix = np.dot(question_embeddings, chunk_embeddings.T)

    results = []
    hits = 0

    for i, question in enumerate(questions):
        top_indices = np.argsort(similarity_matrix[i])[::-1][:5]
        relevant_pages = gold_pages.get(question["id"], [])

        retrieved = []
        hit = False

        for index in top_indices:
            chunk = chunks[int(index)]
            score = float(similarity_matrix[i][index])

            retrieved.append({
                "chunk_id": chunk["chunk_id"],
                "page": chunk["page"],
                "score": round(score, 4),
                "text": chunk["text"]
            })

            if chunk["page"] in relevant_pages:
                hit = True

        if hit:
            hits += 1

        results.append({
            "question_id": question["id"],
            "question": question["question"],
            "gold_pages": relevant_pages,
            "hit_at_5": hit,
            "top_5": retrieved
        })

    recall_at_5 = hits / len(questions) if questions else 0

    output = {
        "chunk_file": chunk_file,
        "total_questions": len(questions),
        "hits_at_5": hits,
        "recall_at_5": round(recall_at_5, 4),
        "recall_at_5_percent": round(recall_at_5 * 100, 2),
        "results": results
    }

    output_file = "results/" + os.path.basename(chunk_file).replace(
        ".json", "_retrieval.json"
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Questions retrieved successfully: {hits}/{len(questions)}")
    print(f"Recall@5: {recall_at_5 * 100:.2f}%")
    print(f"Saved results to: {output_file}")

    return output

if __name__ == "__main__":
    result_200 = evaluate("results/chunks_200.json")
    result_500 = evaluate("results/chunks_500.json")

    print("\n========== FINAL COMPARISON ==========")
    print("200-word chunks:", result_200["recall_at_5_percent"], "%")
    print("500-word chunks:", result_500["recall_at_5_percent"], "%")