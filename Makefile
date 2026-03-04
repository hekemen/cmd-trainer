export HSA_OVERRIDE_GFX_VERSION=9.0.0
#export HIP_VISIBLE_DEVICES=1 # Uncomment and adjust if you have multiple GPUs

DOCKER_TRAIN_IMAGE = command-trainer

.PHONY: train-linux train-cisco train-3class train-all pack-models build-embedded-3class build-embedded-command docker-build docker-run docker-train-build docker-train-linux docker-train-cisco docker-train-3class docker-train-all download-lib validate-3class clean

# --- Local Training ---

# Train the Linux command classifier
train-linux:
	python training/train_linux.py

# Train the Cisco command classifier
train-cisco:
	python training/train_cisco.py

# Train the 3-class command classifier
train-3class:
	python training/train_3class.py

# Train all models sequentially
train-all: train-linux train-cisco train-3class

# --- Containerized Training (ROCm) ---

# Build the training Docker image
docker-train-build:
	docker build -t $(DOCKER_TRAIN_IMAGE) -f training.Dockerfile .

# Common docker run command for training
DOCKER_TRAIN_RUN = docker run --rm \
	--device=/dev/kfd --device=/dev/dri \
	--shm-size=16g \
	-v $(PWD):/workspace \
	-e HSA_OVERRIDE_GFX_VERSION=$(HSA_OVERRIDE_GFX_VERSION) \
	$(DOCKER_TRAIN_IMAGE)

docker-train-linux:
	$(DOCKER_TRAIN_RUN) python3 training/train_linux.py

docker-train-cisco:
	$(DOCKER_TRAIN_RUN) python3 training/train_cisco.py

docker-train-3class:
	$(DOCKER_TRAIN_RUN) python3 training/train_3class.py

docker-train-all: docker-train-linux docker-train-cisco docker-train-3class

# --- Inference ---

# Run the Go inference
run-go:
	CGO_LDFLAGS="-L$(PWD)/lib" go run -tags ORT cmd/run_cisco/main.go

# Run the 3-class Go inference
run-go-3class:
	CGO_LDFLAGS="-L$(PWD)/lib" go run -tags ORT cmd/run_3class/main.go

# Run the command Go inference
run-command:
	CGO_LDFLAGS="-L$(PWD)/lib" go run -tags ORT cmd/run_command/main.go

# Build the Go inference
build-go:
	CGO_LDFLAGS="-L$(PWD)/lib" go build -tags ORT -o run_inference cmd/run_cisco/main.go

# Build the 3-class Go binary with the model embedded
build-embedded-3class:
	@echo "Copying model files for embedding..."
	@mkdir -p cmd/run_3class/model_data
	@cp models/distilbert-3class-onnx-int8/model_quantized.onnx cmd/run_3class/model_data/
	@cp models/distilbert-3class-onnx-int8/config.json cmd/run_3class/model_data/
	@cp models/distilbert-3class-onnx-int8/tokenizer.json cmd/run_3class/model_data/
	@cp models/distilbert-3class-onnx-int8/tokenizer_config.json cmd/run_3class/model_data/
	@cp models/distilbert-3class-onnx-int8/special_tokens_map.json cmd/run_3class/model_data/
	@cp models/distilbert-3class-onnx-int8/vocab.txt cmd/run_3class/model_data/
	@echo "Building binary with embedded model..."
	CGO_LDFLAGS="-L$(PWD)/lib" go build -tags ORT -o run_3class_embedded cmd/run_3class/*.go
	@echo "Cleaning up copied model files..."
	@rm -rf cmd/run_3class/model_data
	@echo "Done! Binary: ./run_3class_embedded"

# Build the command Go binary with the model embedded
build-embedded-command:
	@echo "Copying model files for embedding..."
	@mkdir -p cmd/run_command/model_data
	@cp models/distilbert-command-onnx-int8/model_quantized.onnx cmd/run_command/model_data/
	@cp models/distilbert-command-onnx-int8/config.json cmd/run_command/model_data/
	@cp models/distilbert-command-onnx-int8/tokenizer.json cmd/run_command/model_data/
	@cp models/distilbert-command-onnx-int8/tokenizer_config.json cmd/run_command/model_data/
	@cp models/distilbert-command-onnx-int8/special_tokens_map.json cmd/run_command/model_data/
	@cp models/distilbert-command-onnx-int8/vocab.txt cmd/run_command/model_data/
	@echo "Building binary with embedded model..."
	CGO_LDFLAGS="-L$(PWD)/lib" go build -tags ORT -o run_command_embedded cmd/run_command/*.go
	@echo "Cleaning up copied model files..."
	@rm -rf cmd/run_command/model_data
	@echo "Done! Binary: ./run_command_embedded"

# Pack each model folder into a single .tar.gz file
pack-models:
	@for dir in models/distilbert-*; do \
		if [ -d "$$dir" ]; then \
			echo "Packing $$dir -> $$dir.tar.gz"; \
			tar -czf "$$dir.tar.gz" -C models "$$(basename $$dir)"; \
		fi; \
	done
	@echo "Done. Archives created in models/"

# Build Docker image (multi-stage: compiles Go + embeds model inside Docker)
docker-build:
	docker build -t command-classifier .
	@echo "Done! Image: command-classifier"

# Run the Docker container (interactive, reads from stdin)
docker-run:
	docker run -i --rm command-classifier

# Download the static tokenizer library for local development
download-lib:
	@mkdir -p lib
	@echo "Downloading libtokenizers.a..."
	@wget -q https://github.com/daulet/tokenizers/releases/download/v1.25.0/libtokenizers.linux-amd64.tar.gz
	@tar -xzf libtokenizers.linux-amd64.tar.gz
	@mv libtokenizers.a lib/
	@rm libtokenizers.linux-amd64.tar.gz
	@echo "Done! lib/libtokenizers.a is ready."

# Validate the 3-class embedded binary against training data
validate-3class: build-embedded-3class
	@python3 scripts/validate_binary.py

# Clean compiled binaries and model files
clean:
	rm -rf models/
	rm -f run run_3class_embedded run_3class_embeded run_command_embedded trainer run_inference
	rm -rf cmd/run_3class/model_data cmd/run_command/model_data

