
import json

with open("results/extracted_pages.json", encoding="utf-8") as f:
    pages = json.load(f)

search_terms = {
    "Q7 - Manage function": ["The MANAGE function", "MANAGE function"],
    "Q10 - Accountability": ["accountability", "responsibility"],
    "Q11 - Fairness": ["fairness", "harmful bias"],
    "Q13 - Monitoring": ["monitoring", "regularly while in operation"]
}

for question, terms in search_terms.items():
    print("\n" + "=" * 75)
    print(question)

    found = False

    for page in pages:
        text = page["text"]
        lower_text = text.lower()

        if any(term.lower() in lower_text for term in terms):
            print(f"\nPDF page: {page['page']}")
            print(text[:1800].replace("\n", " "))
            found = True

    if not found:
        print("No matching passage found.")
