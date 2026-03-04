# Training Dockerfile with ROCm support
FROM rocm/pytorch:rocm6.2_ubuntu22.04_py3.10_pytorch_release_2.3.0

# Set environment variables for ROCm
ENV HSA_OVERRIDE_GFX_VERSION=9.0.0
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The project files will be mounted via volume in the Makefile
# This ensures that generated models are persisted to the host.

CMD ["/bin/bash"]   



