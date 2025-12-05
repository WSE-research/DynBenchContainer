# Use lightweight Python base image
FROM python:3.12-alpine

# Set work directory
WORKDIR /app

# Install system dependencies needed for pip packages
RUN apk add --no-cache gcc musl-dev

# Copy requirement file first (to leverage Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

RUN python3 setup.py

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI with Uvicorn
CMD ["uvicorn", "dynbench:app", "--host", "0.0.0.0", "--port", "8000"]
