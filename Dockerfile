FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

LABEL maintainer="Marwan Dhifallah <marwan@mail.dlut.edu.cn>"
LABEL description="Depth-Stratified Relation-Query Scoring (DSRQS) - Paper Implementation"
LABEL repository="https://github.com/your-username/dsrqs"
LABEL paper="https://arxiv.org/abs/XXXX.XXXXX"

# Set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /dsrqs

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1
RUN pip3 install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Copy requirements first for better caching
COPY requirements.txt /dsrqs/

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . /dsrqs/

# Set environment variables
ENV PYTHONPATH=/dsrqs
ENV CUDA_VISIBLE_DEVICES=0
ENV HF_HOME=/dsrqs/.huggingface
ENV TORCH_HOME=/dsrqs/.torch

# Create necessary directories
RUN mkdir -p /dsrqs/data \
             /dsrqs/runs \
             /dsrqs/checkpoints \
             /dsrqs/results \
             /dsrqs/.huggingface \
             /dsrqs/.torch

# Default command
CMD ["python3", "-c", "print('DSRQS environment ready! Use docker exec to run experiments.')"]
