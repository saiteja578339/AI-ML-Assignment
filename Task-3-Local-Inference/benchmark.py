
import csv
import json
import os
import threading
import time
import urllib.request
import urllib.error

import psutil

API_URL = "http://localhost:11434/api/generate"

MODELS = [
    ("FP16", "smollm2:135m-instruct-fp16"),
    ("4-bit", "smollm2:135m-instruct-q4_K_M"),
]

PROMPTS = [
    "What is photosynthesis? Explain in two sentences.",
    "Why do websites use HTTPS? Explain simply.",
    "What is the difference between a Python list and tuple?",
    "What is a primary key in SQL?",
    "What does a CPU do in a computer?",
    "Explain overfitting in machine learning.",
    "Calculate 12 multiplied by 8. Give only the answer.",
    'Reverse this string: "RAG". Give only the result.',
    "Summarize in one sentence: A server stores files and sends them to clients when requested.",
    "Give exactly three ways to reduce electricity consumption at home.",
]

def ollama_memory_mb():
    total = 0
    for process in psutil.process_iter(["name", "memory_info"]):
        try:
            name = (process.info["name"] or "").lower()
            if "ollama" in name:
                total += process.info["memory_info"].rss
        except (psutil.NoSuchProcess, psutil.AccessDenied, TypeError):
            pass
    return total / (1024 * 1024)

def call_ollama(model, prompt):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "keep_alive": "10m",
        "options": {
            "temperature": 0,
            "num_predict": 80,
            "num_ctx": 512
        }
    }

    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    with urllib.request.urlopen(request, timeout=600) as response:
        return json.loads(response.read().decode("utf-8"))

def benchmark_model(label, model):
    print("\n" + "=" * 55)
    print(f"Testing {label}: {model}")
    print("=" * 55)

    records = []
    peak_memory = [ollama_memory_mb()]
    monitoring = [True]

    def monitor_memory():
        while monitoring[0]:
            peak_memory[0] = max(peak_memory[0], ollama_memory_mb())
            time.sleep(0.25)

    available_before = psutil.virtual_memory().available / (1024 * 1024)

    monitor = threading.Thread(target=monitor_memory, daemon=True)
    monitor.start()

    try:
        for index, prompt in enumerate(PROMPTS, start=1):
            print(f"\nPrompt {index}/10...")
            result = call_ollama(model, prompt)

            eval_count = result.get("eval_count", 0)
            eval_duration = result.get("eval_duration", 0)
            tokens_per_second = (
                eval_count / (eval_duration / 1_000_000_000)
                if eval_duration else 0
            )

            record = {
                "model_type": label,
                "model": model,
                "prompt_number": index,
                "prompt": prompt,
                "answer": result.get("response", "").strip(),
                "output_tokens": eval_count,
                "generation_seconds": round(eval_duration / 1e9, 3),
                "tokens_per_second": round(tokens_per_second, 2),
                "load_seconds": round(result.get("load_duration", 0) / 1e9, 3),
                "total_seconds": round(result.get("total_duration", 0) / 1e9, 3),
            }
            records.append(record)

            print("Answer:", record["answer"])
            print("Tokens/sec:", record["tokens_per_second"])

    finally:
        monitoring[0] = False
        monitor.join(timeout=2)

    available_after = psutil.virtual_memory().available / (1024 * 1024)

    os.makedirs("results", exist_ok=True)

    output_file = f"results/{label.lower()}_results.json"
    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(records, file, indent=2, ensure_ascii=False)

    print(f"\n{label} results saved to {output_file}")
    print(f"Peak Ollama process memory: {peak_memory[0]:.1f} MB")
    print(f"Available RAM before: {available_before:.1f} MB")
    print(f"Available RAM after: {available_after:.1f} MB")

    return records, peak_memory[0], available_before, available_after

def main():
    all_records = []
    summary = []

    for label, model in MODELS:
        records, peak, before, after = benchmark_model(label, model)
        all_records.extend(records)

        speeds = [r["tokens_per_second"] for r in records]
        average_speed = sum(speeds) / len(speeds) if speeds else 0

        summary.append({
            "model_type": label,
            "model": model,
            "average_tokens_per_second": round(average_speed, 2),
            "peak_ollama_memory_mb": round(peak, 1),
            "available_ram_before_mb": round(before, 1),
            "available_ram_after_mb": round(after, 1),
        })

        # Unload this model before testing the next one.
        os.system(f'ollama stop "{model}"')

    with open("results/benchmark_summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    with open("results/all_answers.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=all_records[0].keys())
        writer.writeheader()
        writer.writerows(all_records)

    print("\n========== BENCHMARK SUMMARY ==========")
    for item in summary:
        print(item)

    print("\nSaved:")
    print("results/benchmark_summary.json")
    print("results/all_answers.csv")

if __name__ == "__main__":
    main()
