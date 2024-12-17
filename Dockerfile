# Base image avec CUDA
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Éviter les prompts interactifs
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris

# Set working directory
WORKDIR /app

# Install system dependencies en une seule commande
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories with proper permissions
RUN mkdir -p models mlruns data runs static \
    && chmod -R 777 mlruns \
    && chown -R 1000:1000 mlruns

# Set environment variables
ENV TOKENIZERS_PARALLELISM=false
ENV MLFLOW_TRACKING_URI=file:///app/mlruns

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "online_serving:app", "--host", "0.0.0.0", "--port", "8000"]