# Use NVIDIA CUDA base image with Ubuntu
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables to prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and necessary system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/requirements.txt
COPY download_models_hf.py /app/download_models_hf.py

# Install Python dependencies
RUN pip3 install -U magic-pdf[full] --extra-index-url https://wheels.myhloli.com
RUN pip3 install huggingface_hub
RUN pip3 install faiss-gpu
RUN python3 download_models_hf.py
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . /app

# Expose the port that the application will use
EXPOSE 7860

# Set the default command to run the script
CMD ["python3", "demo_with_quant.py"]
