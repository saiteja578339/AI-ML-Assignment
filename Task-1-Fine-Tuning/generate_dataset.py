
import json
import random
import os

random.seed(42)

PRODUCTS = [
    "notebook", "pen", "pencil", "eraser",
    "book", "marker", "bag", "bottle",
    "calculator", "ruler", "folder", "charger",
    "keyboard", "mouse", "headphones", "diary"
]

TEMPLATES = [
    "I need {items} for ₹{price}.",
    "Please buy {items}. My budget is ₹{price}.",
    "Order {items} with a total cost of ₹{price}.",
    "I want {items} and the total should be ₹{price}.",
    "Get me {items}, costing ₹{price} altogether.",
    "Purchase {items} within ₹{price}.",
    "Can you arrange {items} for ₹{price}?"
]

def create_example():
    count = random.randint(1, 3)
    products = random.sample(PRODUCTS, count)

    items = []
    descriptions = []

    for product in products:
        quantity = random.randint(1, 5)

        items.append({
            "name": product,
            "quantity": quantity
        })

        descriptions.append(
            f"{quantity} {product}{'s' if quantity > 1 else ''}"
        )

    total_price = random.randint(50, 2000)

    item_text = ", ".join(descriptions)
    template = random.choice(TEMPLATES)

    user_text = template.format(
        items=item_text,
        price=total_price
    )

    output = {
        "items": items,
        "total_price": total_price
    }

    return {
        "instruction": (
            "Convert the shopping request into JSON. "
            "Return only valid JSON with items and total_price. "
            "Each item must contain name and quantity."
        ),
        "input": user_text,
        "output": json.dumps(output)
    }


def generate_unique(count):
    examples = []
    seen = set()

    while len(examples) < count:
        example = create_example()

        if example["input"] not in seen:
            seen.add(example["input"])
            examples.append(example)

    return examples


os.makedirs("dataset", exist_ok=True)

data = generate_unique(1200)

random.shuffle(data)

train_data = data[:1000]
validation_data = data[1000:1100]
test_data = data[1100:1200]


def save_jsonl(filename, records):
    with open(filename, "w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record) + "\n")


save_jsonl("dataset/train.jsonl", train_data)
save_jsonl("dataset/validation.jsonl", validation_data)
save_jsonl("dataset/test.jsonl", test_data)

print("Dataset generated successfully!")
print("Training examples:", len(train_data))
print("Validation examples:", len(validation_data))
print("Testing examples:", len(test_data))
