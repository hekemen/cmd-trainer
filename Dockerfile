### Stage 1: Build the Go binary with embedded model ###
FROM golang:1.26-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Copy go module files first for caching
COPY go.mod go.sum ./
RUN go mod download

# Copy the source code
COPY cmd/ cmd/

# Download and install the static tokenizer library
RUN wget -q https://github.com/daulet/tokenizers/releases/download/v1.25.0/libtokenizers.linux-amd64.tar.gz \
    && tar -xzf libtokenizers.linux-amd64.tar.gz \
    && mv libtokenizers.a /usr/lib/ \
    && rm libtokenizers.linux-amd64.tar.gz

# Copy model files into the embed directory
COPY models/distilbert-3class-onnx-int8/model_quantized.onnx cmd/run_3class/model_data/
COPY models/distilbert-3class-onnx-int8/config.json cmd/run_3class/model_data/
COPY models/distilbert-3class-onnx-int8/tokenizer.json cmd/run_3class/model_data/
COPY models/distilbert-3class-onnx-int8/tokenizer_config.json cmd/run_3class/model_data/
COPY models/distilbert-3class-onnx-int8/special_tokens_map.json cmd/run_3class/model_data/
COPY models/distilbert-3class-onnx-int8/vocab.txt cmd/run_3class/model_data/

# Download ONNX Runtime for linking
RUN wget -q https://github.com/microsoft/onnxruntime/releases/download/v1.24.2/onnxruntime-linux-x64-1.24.2.tgz \
    && tar -xzf onnxruntime-linux-x64-1.24.2.tgz \
    && cp onnxruntime-linux-x64-1.24.2/lib/libonnxruntime.so /usr/lib/ \
    && ldconfig

# Build the binary
RUN CGO_LDFLAGS="-L/usr/lib" go build -tags ORT -o /run_3class_embedded cmd/run_3class/*.go

### Stage 2: Distroless runtime image (No Shell) ###
FROM gcr.io/distroless/cc-debian12

WORKDIR /app

# Copy the ONNX Runtime shared library from builder
COPY --from=builder /usr/lib/libonnxruntime.so /usr/lib/

# Copy the built binary
COPY --from=builder /run_3class_embedded /app/run_3class_embedded

# Use the built-in nonroot user for security (UID 65532)
# Since we can't RUN chown in distroless, we chown in the builder or just rely on the fact
# that embedded model extraction writes to /tmp, which is writable by non-root in distroless.
USER nonroot:nonroot

ENV ONNX_LIB_PATH=/usr/lib

ENTRYPOINT ["/app/run_3class_embedded"]
