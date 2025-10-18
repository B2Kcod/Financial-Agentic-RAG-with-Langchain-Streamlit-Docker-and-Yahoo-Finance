# Base stage: Common setup for both CPU and GPU environments.
FROM python:3.11-slim AS base
# Ensure Python output is not buffered, for real-time logging.
ENV PYTHONUNBUFFERED=1
# Set the working directory inside the container to /app.
WORKDIR /app

# Install system dependencies:
# - Update package lists and install build tools, git, and curl.
# - Clean up to reduce image size.
RUN apt-get update --yes \
    && apt-get install --no-install-recommends -y build-essential git curl \
    && rm -rf /var/lib/apt/lists/*

# CPU stage: For systems without NVIDIA GPUs.
FROM base AS cpu
# Copy the requirements file for Python dependencies.
COPY requirements.txt /app/requirements.txt
# Upgrade pip and install Python dependencies from requirements.txt.
RUN pip install --upgrade pip && pip install -r requirements.txt
# Copy the entire application code into the container.
COPY . /app
# Run the Streamlit application on port 8000, accessible from all network interfaces.
CMD ["streamlit", "run", "ui.py", "--server.port", "8000", "--server.address", "0.0.0.0"]

# GPU stage: For systems with NVIDIA GPUs, using PyTorch with CUDA support.
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime AS gpu
# Ensure Python output is not buffered.
ENV PYTHONUNBUFFERED=1
# Set the working directory to /app.
WORKDIR /app
# Install system dependencies (same as base stage).
RUN apt-get update --yes \
    && apt-get install --no-install-recommends -y build-essential git curl \
    && rm -rf /var/lib/apt/lists/*
# Copy requirements files for both general and GPU-specific dependencies.
COPY requirements.txt /app/requirements.txt
COPY requirements-gpu.txt /app/requirements-gpu.txt
# Upgrade pip and install GPU-specific dependencies from PyTorch's index and requirements-gpu.txt.
RUN pip install --upgrade pip
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements-gpu.txt
# Copy the application code into the container.
COPY . /app
# Run the Streamlit application on port 8000, accessible from all network interfaces.
CMD ["streamlit", "run", "ui.py", "--server.port", "8000", "--server.address", "0.0.0.0"]