# 1. Use lightweight Python base
FROM python:3.10-slim

# 2. Set working directory inside container
WORKDIR /app

# 3. Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy Python dependencies list
COPY requirements.txt .

# 5. Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --prefer-binary -r requirements.txt



# 6. Copy application code
COPY app/ .

# 7. Default command
CMD ["python", "predict.py"]


