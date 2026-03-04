import json
import random
import os

def merge_datasets():
    real_data_path = "data/linux_commands_real_val.json"
    if not os.path.exists(real_data_path):
        # Try local path if not in data/
        real_data_path = "linux_commands_real_val.json"
        if not os.path.exists(real_data_path):
            print("Error: Real data not found. Run scripts/create_real_val.py first.")
            return
    
    with open(real_data_path, "r") as f:
        real_data = json.load(f)
    print(f"Loaded {len(real_data)} real-world samples.")

    # 2. Load Synthetic data
    synth_data_path = "data/linux_commands_3class.json"
    if not os.path.exists(synth_data_path):
        # Fallback to the 100k one if the original is gone
        synth_data_path = "data/linux_commands_3class_val.json"
        
    with open(synth_data_path, "r") as f:
        synth_data = json.load(f)
    print(f"Loaded {len(synth_data)} synthetic samples.")

    # 3. Combine and Balance
    # We want to give the real-world data a high weight.
    # Let's say we use all real-world data and a similar amount of synthetic data per class
    merged_data = real_data.copy()
    
    # Simple balancing: add 2000 synthetic samples to provide base patterns
    random.shuffle(synth_data)
    merged_data.extend(synth_data[:5000])

    random.shuffle(merged_data)
    
    output_path = "data/linux_commands_3class_mixed.json"
    with open(output_path, "w") as f:
        json.dump(merged_data, f, indent=4)
    
    print(f"Created mixed dataset with {len(merged_data)} samples at {output_path}")

if __name__ == "__main__":
    merge_datasets()
