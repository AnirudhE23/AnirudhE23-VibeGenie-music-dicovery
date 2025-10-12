FROM python:3.11-slim 

# Setting up the working directory
WORKDIR /app

# Installing the dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copying the requirements.txt file
COPY requirements.txt .

# Installing python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copying the rest of the files
COPY . .

RUN mkdir -p models data cache

# Setting up the environment variables
ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Exposing the port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Running the streamlit application
CMD ["streamlit", "run", "new_main.py", "--server.port=8501", "--server.address=0.0.0.0"]