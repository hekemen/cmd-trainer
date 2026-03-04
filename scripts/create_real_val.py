import json
import os

def create_real_val_dataset():
    history_path = os.path.expanduser("~/.bash_history")
    if not os.path.exists(history_path):
        print("Error: .bash_history not found")
        return

    with open(history_path, "r", errors="ignore") as f:
        lines = f.readlines()

    unique_cmds = set()
    for line in lines:
        cmd = line.strip()
        # Filter out empty lines, numbers-only (if timestamps enabled), and very short things
        if not cmd or cmd.isdigit() or len(cmd) < 3:
            continue
        unique_cmds.add(cmd)

    data = []
    for cmd in unique_cmds:
        # Basic heuristic labeling for the real dataset
        label = 0 # HARMLESS
        
        lower_cmd = cmd.lower()
        if any(x in lower_cmd for x in ["shutdown", "reboot", "poweroff", "halt", "init 0", "init 6"]):
            label = 1 # DISRUPTIVE
        elif any(x in lower_cmd for x in ["nc ", "bash -i", "dd if=/dev/zero", "cat /etc/shadow"]):
            label = 2 # MALICIOUS
        elif "rm -rf" in lower_cmd:
            if any(x in lower_cmd for x in ["/usr/local/go", ".venv", "tests_cassandra_1", "distilbert-", "cm.json"]):
                label = 0 # Known safe-ish developer deletions
            else:
                label = 2 # General destructive deletion
        
        # Override for common harmless tools from history
        if any(x in lower_cmd for x in ["kubectl", "helm", "git", "make", "go ", "python", "curl", "ping", "ls ", "ps ", "cat ", "vi ", "ollama"]):
             if label == 2: # If it was flagged malicious but is a standard tool, double check
                 if "rm -rf" not in lower_cmd: # rm -rf is still suspicious enough
                     label = 0
             elif label != 1: # If not disruptive, it's likely harmless
                 label = 0
        
        data.append({"text": cmd, "label": label})

    # Save to data/linux_commands_real_val.json
    os.makedirs("data", exist_ok=True)
    val_path = "data/linux_commands_real_val.json"
    with open(val_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Created {len(data)} items in {val_path}")

if __name__ == "__main__":
    create_real_val_dataset()
