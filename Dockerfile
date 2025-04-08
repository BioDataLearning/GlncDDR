FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libxml2-dev \
    libxslt1-dev \
    libz-dev \
    libffi-dev \
    libigraph-dev \
    pkg-config \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project code and folders
COPY . /app

# Set PYTHONPATH so modules are found
ENV PYTHONPATH="/app"

CMD ["python", "main.py", "--help"]
