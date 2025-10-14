# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements_p39.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_p39.txt

# Copy the entire project
COPY . /app/

# Default command (can be overridden)
CMD ["python", "train.py"]


