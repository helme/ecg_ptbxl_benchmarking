# Use NVIDIA CUDA base image for GPU support
# For CPU-only, use: FROM python:3.8-slim
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libhdf5-dev \
    libnetcdf-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set python3.8 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# Note: Some packages might have compatibility issues, adjust versions if needed
RUN pip install --no-cache-dir -r requirements.txt || \
    (echo "Warning: Some packages failed to install. Installing core packages..." && \
     pip install --no-cache-dir \
     torch==1.4.0 \
     torchvision==0.5.0 \
     numpy==1.19.4 \
     scipy==1.5.3 \
     pandas==1.1.4 \
     scikit-learn==0.23.2 \
     matplotlib==3.3.2 \
     wfdb==3.1.1 \
     tqdm==4.54.0 \
     pyyaml==5.3.1 \
     h5py==2.10.0 \
     && echo "Core packages installed successfully")

# Install TensorFlow separately (GPU version)
RUN pip install --no-cache-dir tensorflow-gpu==2.3.0 || \
    pip install --no-cache-dir tensorflow==2.3.0

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p data/ptbxl data/ICBEB output tmp_data

# Set permissions
RUN chmod +x get_datasets.sh

# Default command
CMD ["/bin/bash"]
