
import json
import os
import re
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"
ADAPTER_PATH = "adapters/smollm2-lora"
TEST_FILE = "dataset/test.jsonl"

# Load the 100 unseen test examples
test_data = []

with open(TEST_FILE, "r", encoding="utf-8") as f:
    for line in f:
        test_data.append(json.loads(line))

print("Test examples:", len(test_data))

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def extract_json(text):
    """Extract the first valid JSON object from generated text."""
    decoder = json.JSONDecoder()

    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
            return obj
        except json.JSONDecodeError:
            continue

    return None


def evaluate(model, model_name):
    model.eval()
    correct = 0
    failures = []

    print("\nEvaluating:", model_name)

    for index, example in enumerate(test_data):
        prompt = (
            "Instruction: " + example["instruction"] +
            "\nInput: " + example["input"] +
            "\nOutput: "
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=192
        )

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        predicted = extract_json(generated_text)

        try:
            expected = json.loads(example["output"])
        except json.JSONDecodeError:
            expected = None

        is_correct = (
            predicted is not None
            and expected is not None
            and json.dumps(predicted, sort_keys=True)
            == json.dumps(expected, sort_keys=True)
        )

        if is_correct:
            correct += 1
        elif len(failures) < 3:
            failures.append({
                "input": example["input"],
                "expected": expected,
                "predicted_text": generated_text,
                "parsed_prediction": predicted
            })

        if (index + 1) % 10 == 0:
            print(f"Processed {index + 1}/{len(test_data)}")

    accuracy = correct / len(test_data) * 100

    print(f"\n{model_name} accuracy: {accuracy:.2f}%")
    print(f"Correct: {correct}/{len(test_data)}")

    return {
        "model": model_name,
        "correct": correct,
        "total": len(test_data),
        "accuracy_percent": accuracy,
        "failures": failures
    }


# 1. Evaluate original base model
print("\nLoading base model...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

base_results = evaluate(base_model, "Base SmolLM2")

del base_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()


# 2. Evaluate fine-tuned model
print("\nLoading fine-tuned model...")

base_for_adapter = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

fine_tuned_model = PeftModel.from_pretrained(
    base_for_adapter,
    ADAPTER_PATH
)

fine_tuned_results = evaluate(
    fine_tuned_model,
    "Fine-tuned SmolLM2 + LoRA"
)

# Save results
os.makedirs("results", exist_ok=True)

all_results = {
    "base_model": base_results,
    "fine_tuned_model": fine_tuned_results
}

with open("results/task1_evaluation.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

print("\nEvaluation completed!")
print("Results saved to results/task1_evaluation.json")
