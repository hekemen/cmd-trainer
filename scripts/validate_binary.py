import json
import subprocess
import os
import sys

# Mapping from training data labels (int) to binary output labels (string)
LABEL_MAP = {
    0: "HARMLESS",
    1: "DISRUPTIVE",
    2: "MALICIOUS"
}

def validate():
    binary_path = "./run_3class_embedded"
    data_path = os.getenv("DATA_PATH", "data/linux_commands_3class_val.json")

    if not os.path.exists(binary_path):
        print(f"Error: Binary {binary_path} not found. Run 'make build-embedded-3class' first.")
        sys.exit(1)

    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found.")
        sys.exit(1)

    print(f"Loading data from {data_path}...")
    with open(data_path, "r") as f:
        data = json.load(f)

    # Allow limiting test size via environment variable
    sample_size = os.getenv("SAMPLE_SIZE")
    if sample_size:
        limit = int(sample_size)
        print(f"Sampling first {limit} items...")
        data = data[:limit]
    
    total = len(data)

    print(f"Starting validation on {total} items...")

    correct = 0
    mismatches = []

    # Start the binary once and use stdin/stdout for performance
    process = subprocess.Popen(
        [binary_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    try:
        for i, entry in enumerate(data):
            command = entry["text"]
            expected_int = entry["label"]
            expected_label = LABEL_MAP.get(expected_int, "UNKNOWN")

            # Send command to binary
            # Send command to binary
            process.stdin.write(command + "\n")
            process.stdin.flush()

            # Read prediction
            prediction = process.stdout.readline().strip()

            if i < 5:
                print(f"DEBUG: Cmd: {command[:30]}... | Exp: {expected_label} | Pred: {prediction}")

            if prediction == expected_label:
                correct += 1
            else:
                mismatches.append({
                    "command": command,
                    "expected": expected_label,
                    "predicted": prediction
                })

            if (i + 1) % 100 == 0:
                print(f"Progress: {i+1}/{total} (Accuracy: {correct/(i+1):.2%})", end="\r")

    finally:
        process.stdin.close()
        process.terminate()

    print(f"\n\nValidation Complete!")
    print(f"Total Items: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {correct/total:.2%}")

    if mismatches:
        print("\nTop 10 Mismatches:")
        for m in mismatches[:10]:
            print(f"Cmd: {m['command'][:50]}... | Exp: {m['expected']} | Pred: {m['predicted']}")

if __name__ == "__main__":
    validate()
