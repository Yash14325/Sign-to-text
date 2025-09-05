# Use Python 3.10 slim image
FROM python:3.10-slim

# Install OS-level dependencies for OpenCV (and others)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
       libgl1 \
       libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Copy application code
COPY . .

# Expose the port Flask will run on
EXPOSE 10000

# Start the Flask app
CMD ["python", "sign-translator/app.py"]
