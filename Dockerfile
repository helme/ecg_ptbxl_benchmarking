# Use NVIDIA CUDA base image with CUDA 11.0 for better compatibility
FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install system dependencies and add PPA for Python 3.8
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
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

# Install pip for Python 3.8 (use Python 3.8 specific get-pip.py)
RUN curl https://bootstrap.pypa.io/pip/3.8/get-pip.py -o get-pip.py && \
    python3.8 get-pip.py && \
    rm get-pip.py

# Set python3.8 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace

# Install PyTorch 1.7.1 with CUDA 11.0 first
RUN pip install --no-cache-dir \
    torch==1.7.1+cu110 \
    torchvision==0.8.2+cu110 \
    torchaudio==0.7.2 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Copy requirements file
COPY requirements.txt .

# Install core Python dependencies
RUN pip install --no-cache-dir \
    numpy==1.19.4 \
    scipy==1.5.3 \
    pandas==1.1.4 \
    scikit-learn==0.23.2 \
    matplotlib==3.3.2 \
    tqdm==4.54.0 \
    pyyaml==5.3.1 \
    h5py==2.10.0

# Install additional packages
RUN pip install --no-cache-dir scikit-learn==0.23.2
RUN pip install --no-cache-dir --no-deps wfdb==3.1.1
RUN pip install --no-cache-dir mne
RUN pip install --no-cache-dir scikit-image

# Install fastai (compatible with PyTorch 1.7)
RUN pip install --no-cache-dir fastai==1.0.61

# Install TensorFlow separately (GPU version compatible with CUDA 11.0)
RUN pip install --no-cache-dir tensorflow-gpu==2.4.0 || \
    pip install --no-cache-dir tensorflow==2.4.0

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p data/ptbxl data/ICBEB output tmp_data

# Set permissions
RUN chmod +x get-datasets.sh

# Default command
CMD ["/bin/bash"]