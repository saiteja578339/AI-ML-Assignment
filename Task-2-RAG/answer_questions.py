
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"

torch.set_num_threads(4)

print("Loading language model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32
)

model.eval()

# Load hybrid retrieval results.
with open(
    "results/chunks_500_hybrid_retrieval.json",
    "r",
    encoding="utf-8"
) as f:
    retrieval_data = json.load(f)

# Load expected answers for evaluation only.
with open(
    "results/questions.json",
    "r",
    encoding="utf-8"
) as f:
    questions_data = json.load(f)

expected_answers = {
    q["id"]: q["answer"]
    for q in questions_data
}

final_results = []

for item in retrieval_data["results"]:
    question_id = item["question_id"]
    question = item["question"]

    # Use the top 3 retrieved passages.
    top_chunks = item["top_5"][:3]

    # Limit each passage to 1200 characters.
    context_parts = []

    for chunk in top_chunks:
        context_parts.append(
            f"Source page {chunk['page']}:\n"
            f"{chunk['text'][:1200]}"
        )

    context = "\n\n".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a precise document question-answering assistant. "
                "Answer using only facts explicitly stated in the context. "
                "Do not use outside knowledge. "
                "Do not invent names, numbers, or details. "
                "Give a direct answer in 1 or 2 short sentences. "
                "Do not copy long passages. "
                "Do not repeat sentences or list unrelated information. "
                "If the context does not contain the answer, say: "
                "The retrieved passages do not provide enough information."
            )
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Document context:\n{context}\n\n"
                "Answer:"
            )
        }
    ]

    # Convert chat messages into a text prompt.
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize the prompt into a model input dictionary.
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1400
    )

    # Generate the answer.
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=80,
            do_sample=False,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Extract only newly generated tokens.
    generated_tokens = outputs[0][
        inputs["input_ids"].shape[1]:
    ]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    final_results.append({
        "question_id": question_id,
        "question": question,
        "generated_answer": answer,
        "expected_answer_for_evaluation": expected_answers.get(
            question_id, ""
        ),
        "retrieved_source_pages": [
            chunk["page"] for chunk in top_chunks
        ],
        "retrieval_hit_at_5": item["hit_at_5"]
    })

    print("\n" + "=" * 60)
    print("Question:", question)
    print("Generated answer:", answer)
    print("Expected answer:", expected_answers.get(question_id, ""))
    print("Source pages:", [c["page"] for c in top_chunks])

# Save all generated answers.
with open(
    "results/rag_answers.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(final_results, f, indent=2, ensure_ascii=False)

print("\nRAG answer generation completed.")
print("Total questions:", len(final_results))
print("Saved to results/rag_answers.json")
