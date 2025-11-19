FROM python:3.11-slim

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Install PyTorch CPU version
# -----------------------------
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# -----------------------------
# Copy requirements & install
# -----------------------------
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Copy project code
# -----------------------------
COPY . /app

# Default command
CMD ["python", "run_experiment.py", "--dataset", "mnist", "--clients", "3", "--rounds", "10", "--local-epochs", "1"]
