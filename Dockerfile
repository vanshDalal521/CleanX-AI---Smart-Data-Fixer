# CleanX AI - OpenEnv Environment
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir -r server/requirements.txt

# Metadata as required by the spec
LABEL openenv=true

# Expose the API port
EXPOSE 8000

# Start the OpenEnv server
CMD ["python", "-m", "server.app", "--port", "8000"]
