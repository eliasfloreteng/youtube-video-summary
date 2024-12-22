FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
  ffmpeg \
  && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create uploads directory
RUN mkdir -p uploads

# Set environment variables
ENV PORT=8000
ENV ANTHROPIC_API_KEY=""

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "app.py"]
