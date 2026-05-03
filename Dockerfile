FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    graphviz \
    git \
    curl \
    wget \
    unzip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------
# Create virtual environment
# (IMPORTANT: outside /workspace)
# ------------------------------
RUN python3 -m venv /opt/venv

# Add venv to PATH
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip inside venv
RUN pip install --upgrade pip

# Create working directory
WORKDIR /workspace

# Copy requirements first (for caching)
COPY requirements.txt /workspace/requirements.txt

# Install dependencies inside venv
RUN pip install -r /workspace/requirements.txt || true

# Default: open shell
CMD ["/bin/bash"]
