# Use lightweight Python base image
FROM python:3.12-alpine

# Set work directory
WORKDIR /app

# Install system dependencies needed for pip packages
RUN apk add --no-cache curl bzip2

RUN mkdir -p /app/pagerank

RUN curl -fSL "https://danker.s3.amazonaws.com/2025-11-05.allwiki.links.rank.bz2" -o "/app/pagerank/2025-11-05.allwiki.links.rank.bz2"
RUN bzip2 -df /app/pagerank/2025-11-05.allwiki.links.rank.bz2 > /app/pagerank/2025-11-05.allwiki.links.rank

# Copy requirement file first (to leverage Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader punkt punkt_tab

# Copy the rest of the application
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI with Uvicorn
CMD ["uvicorn", "dynbench:app", "--host", "0.0.0.0", "--port", "8000"]
