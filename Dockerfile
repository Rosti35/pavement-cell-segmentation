FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/process_images.py .
COPY models/ models/

# Create directories for input/output
RUN mkdir -p /app/input /app/output

# Set default command
ENTRYPOINT ["python", "process_images.py"]
CMD ["/app/input", "/app/output", "--help"]
