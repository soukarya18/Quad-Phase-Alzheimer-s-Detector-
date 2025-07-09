# # Use Python 3.10.11 as base image
# FROM python:3.10.11-slim

# # Set working directory in container
# WORKDIR /app

# # Copy all files to container
# COPY . .

# # Install dependencies
# RUN pip install --upgrade pip && \
#     pip install -r requirements.txt

# # Expose the Streamlit port
# EXPOSE 8501

# # Run Streamlit
# CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# Use Python 3.10.11 as base image
FROM python:3.10.11-slim

# Set working directory in container
WORKDIR /app

# Copy all files to container
COPY . .

# Install system dependencies (needed by numpy, pandas, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose the Streamlit default port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

