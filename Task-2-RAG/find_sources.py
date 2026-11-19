
import json
import re

with open("results/extracted_pages.json", encoding="utf-8") as f:
    pages = json.load(f)

with open("results/questions.json", encoding="utf-8") as f:
    questions = json.load(f)

def words(text):
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))

for q in questions:
    query_words = words(q["question"] + " " + " ".join(q["keywords"]))

    scored = []

    for page in pages:
        page_words = words(page["text"])
        score = len(query_words & page_words)

        scored.append((score, page))

    scored.sort(key=lambda x: x[0], reverse=True)

    print("\n" + "=" * 70)
    print("QUESTION", q["id"], ":", q["question"])

    for score, page in scored[:2]:
        print(f"\nCandidate PDF page: {page['page']} | Keyword score: {score}")
        print(page["text"][:1200].replace("\n", " "))

