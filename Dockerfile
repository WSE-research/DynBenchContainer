# Use lightweight Python base image
FROM python:3.12-alpine

# Set work directory
WORKDIR /app

# Copy the small pagerank file to the container
RUN mkdir -p /app/pagerank
COPY pagerank/allwiki.rank /app/pagerank/

# Copy requirement file first (to leverage Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK punkt modules
RUN python -m nltk.downloader punkt punkt_tab

# Copy utils
COPY utils/ /app/utils/

# Copy tests
COPY tests/ /app/tests/

# Copy the rest of the application
COPY dynbench.py __init__.py .env /app/

# Run tests and fail build if any test fails
RUN python -m pytest tests/ --tb=short || exit 1

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI with Uvicorn
CMD ["uvicorn", "dynbench:app", "--host", "0.0.0.0", "--port", "8000"]
