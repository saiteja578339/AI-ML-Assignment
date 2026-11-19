import json

for filename in [
    "results/chunks_500_hybrid_retrieval.json",
    "results/chunks_500_hybrid_retrieval.json"
]:
    print("\n" + "=" * 65)
    print("FILE:", filename)
    print("=" * 65)

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data["results"]:
        if not item["hit_at_5"]:
            print("\nQuestion ID:", item["question_id"])
            print("Question:", item["question"])
            print("Expected source pages:", item["gold_pages"])
            print("Retrieved top 5:")

            for chunk in item["top_5"]:
                print(
                    f"\nPage {chunk['page']} | "
                    f"Chunk {chunk['chunk_id']} | "
                    f"Score {chunk['score']}"
                )
                print(chunk["text"][:400])