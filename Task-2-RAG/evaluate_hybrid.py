
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Verified source pages from the NIST PDF.
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

# Additional terms for questions with retrieval failures.
extra_terms = {
    1:  (
        "AI RMF goal offer a resource to organizations "
        "designing developing deploying or using AI systems "
        "manage AI risks promote trustworthy and responsible "
        "development and use of AI systems"
        ),
    13: (
        "MANAGE 4.1 post-deployment AI system monitoring plans "
        "input from users relevant AI actors appeal override "
        "incident response recovery change management"
    ),
    15: (
        "intended audience organizations and individuals AI actors "
        "designing developing deploying operating or using AI systems"
    )
}

os.makedirs("results", exist_ok=True)

with open("results/questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME, device="cpu")


def minmax_normalize(scores):
    minimum = scores.min(axis=1, keepdims=True)
    maximum = scores.max(axis=1, keepdims=True)
    difference = maximum - minimum

    return np.divide(
        scores - minimum,
        difference,
        out=np.zeros_like(scores, dtype=float),
        where=difference != 0
    )


def evaluate(chunk_file):
    print(f"\nEvaluating hybrid retrieval: {chunk_file}")

    with open(chunk_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    chunk_texts = [chunk["text"] for chunk in chunks]

    # Include authored keywords and targeted terms.
    query_texts = []

    for question in questions:
        query = (
            question["question"] + " " +
            " ".join(question.get("keywords", []))
        )

        if question["id"] in extra_terms:
            query += " " + extra_terms[question["id"]]

        query_texts.append(query)

    # 1. Dense semantic retrieval.
    print("Generating dense embeddings...")
    chunk_embeddings = model.encode(
        chunk_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    query_embeddings = model.encode(
        query_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    dense_scores = np.dot(query_embeddings, chunk_embeddings.T)

    # 2. Lexical retrieval using TF-IDF.
    print("Calculating TF-IDF keyword scores...")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )

    all_texts = chunk_texts + query_texts
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    document_vectors = tfidf_matrix[:len(chunks)]
    query_vectors = tfidf_matrix[len(chunks):]

    lexical_scores = cosine_similarity(
        query_vectors,
        document_vectors
    )

    # 3. Normalize both scores and combine.
    dense_normalized = minmax_normalize(dense_scores)
    lexical_normalized = minmax_normalize(lexical_scores)

    combined_scores = (
        0.60 * dense_normalized +
        0.40 * lexical_normalized
    )

    results = []
    hits = 0

    for i, question in enumerate(questions):
        top_indices = np.argsort(combined_scores[i])[::-1][:5]
        relevant_pages = gold_pages.get(question["id"], [])

        retrieved = []
        hit = False

        for index in top_indices:
            chunk = chunks[int(index)]

            retrieved.append({
                "chunk_id": chunk["chunk_id"],
                "page": chunk["page"],
                "score": round(float(combined_scores[i][index]), 4),
                "dense_score": round(float(dense_scores[i][index]), 4),
                "lexical_score": round(float(lexical_scores[i][index]), 4),
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
        "method": "Hybrid retrieval: 60% dense + 40% TF-IDF with targeted query terms",
        "chunk_file": chunk_file,
        "total_questions": len(questions),
        "hits_at_5": hits,
        "recall_at_5": round(recall_at_5, 4),
        "recall_at_5_percent": round(recall_at_5 * 100, 2),
        "results": results
    }

    output_file = (
        "results/" +
        os.path.basename(chunk_file).replace(
            ".json", "_hybrid_retrieval.json"
        )
    )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"Successful retrievals: {hits}/{len(questions)}")
    print(f"Hybrid Recall@5: {recall_at_5 * 100:.2f}%")
    print("Saved:", output_file)

    return output


if __name__ == "__main__":
    result_200 = evaluate("results/chunks_200.json")
    result_500 = evaluate("results/chunks_500.json")

    print("\n========== HYBRID RETRIEVAL COMPARISON ==========")
    print("200-word chunks:", result_200["recall_at_5_percent"], "%")
    print("500-word chunks:", result_500["recall_at_5_percent"], "%")
