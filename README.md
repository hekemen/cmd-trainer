# Linux Command Safety Classifier

An AI-powered tool for classifying Linux and Cisco commands into safety categories (HARMLESS, DISRUPTIVE, MALICIOUS) using a quantized DistilBERT model.

The project features a full lifecycle from training and ONNX export to high-performance inference in Go, with a hardened, self-contained Docker deployment.

## Key Features

- **BERT-based Classification**: Uses DistilBERT for fast and accurate link command analysis.
- **Quantized ONNX**: Models are quantized to `int8` for 4x smaller size and faster CPU inference.
- **Go Inference**: Blazing fast inference using the `Hugot` library and ONNX Runtime.
- **Self-Contained Binary**: The model is embedded directly into the Go binary using `go:embed`.
- **Hardened Docker**: Multi-stage `distroless` build with NO shell and non-root execution for maximum security.

## Project Structure

- `training/`: Python scripts for training and exporting models.
- `models/`: Exported ONNX models and tokenizers.
- `cmd/`: Go inference source code.
  - `run_3class/`: Main inference logic with embedded model support.
- `Makefile`: Orchestration for the entire project.
- `Dockerfile`: Multi-stage hardened container build.

## Prerequisites

- **Python 3.10+**: For training.
- **Go 1.25+**: For inference.
- **Docker**: For containerized deployment.
- **ONNX Runtime**: Shared libraries (`.so`) required for inference.

## 🐍 Python Environment Setup

It is recommended to use a virtual environment for training:

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Training & Exporting
Ensure your virtual environment is activated before training:
```bash
source .venv/bin/activate
make train-3class
```

### 2. Containerized Training (ROCm)
For a consistent and reproducible environment with GPU acceleration:

```bash
# Build the training image
make docker-train-build

# Train the 3-class model inside Docker
make docker-train-3class
```

The generated models will be saved to the `models/` directory on your host machine.

### 3. Local Go Inference
Build and run the 3-class inference locally:
```bash
# Standard run (requires models/ folder)
make run-go-3class

# Build self-contained binary (model embedded)
make build-embedded-3class
./run_3class_embedded
```

### 3. Docker Deployment (Hardened)
Build and run the ultra-secure distroless container:
```bash
# Build the image (everything happens inside Docker)
make docker-build

# Run the container (interactive)
make docker-run
```

## 🛠 Makefile Commands

| Command | Description |
|---------|-------------|
| `make train-all` | Run all training scripts sequentially |
| `make build-embedded-3class` | Build a Go binary with the 3-class model embedded |
| `make docker-build` | Build the hardened multi-stage Docker image |
| `make docker-run` | Run the Docker container interactively |
| `make pack-models` | Create `.tar.gz` archives of all model folders |
| `make clean` | Remove builds, models, and temporary files |

## Security Notes

The Docker container is built on `gcr.io/distroless/cc-debian12`, which means:
- **No Shell**: `/bin/sh` and `/bin/bash` are absent.
- **No Utilities**: No `ls`, `cat`, or package managers.
- **Non-Root**: Runs as the built-in `nonroot` user.
- **Immutable**: The image contains only what is strictly necessary to run the classifier.
