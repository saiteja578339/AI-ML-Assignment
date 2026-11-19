
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M-Instruct"

# Load existing training data
def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records

train_data = load_jsonl("dataset/train.jsonl")
valid_data = load_jsonl("dataset/validation.jsonl")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32
)

model.config.use_cache = False

# Configure LoRA
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

def preprocess(example):
    prompt = (
        "Instruction: " + example["instruction"] +
        "\nInput: " + example["input"] +
        "\nOutput: "
    )

    full_text = prompt + example["output"] + tokenizer.eos_token

    encoded = tokenizer(
        full_text,
        truncation=True,
        max_length=192,
        padding="max_length"
    )

    encoded["labels"] = [
        token if mask == 1 else -100
        for token, mask in zip(
            encoded["input_ids"],
            encoded["attention_mask"]
        )
    ]

    return encoded

train_dataset = Dataset.from_list(train_data)
valid_dataset = Dataset.from_list(valid_data)

train_dataset = train_dataset.map(
    preprocess,
    remove_columns=train_dataset.column_names
)

valid_dataset = valid_dataset.map(
    preprocess,
    remove_columns=valid_dataset.column_names
)

training_args = TrainingArguments(
    output_dir="results/cpu_checkpoints",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    report_to="none",
    use_cpu=True,
    dataloader_num_workers=0
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
)

trainer.train()

model.save_pretrained("adapters/smollm2-lora")
tokenizer.save_pretrained("adapters/smollm2-lora")

print("Training completed. LoRA adapter saved.")
