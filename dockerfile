## Use the slim Python 3.12 image
FROM python:3.12-slim

# Set environment variables
ENV APP_VERSION=0.0.0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /code

# Install system dependencies and update pip in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    libssl-dev gcc openssl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip

# Install Python dependencies separately for better caching
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Create and switch to a non-root user
RUN useradd -m user
USER user

# Set home environment and working directory
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app

# Copy application files and set ownership to the user
COPY --chown=user . $HOME/app

# Copy model files and SSL certificates
COPY ./llm /llm

# Generate self-signed SSL certificates
RUN mkdir -p $HOME/app/sec && \
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout $HOME/app/sec/key.pem \
    -out $HOME/app/sec/cert.pem \
    -subj "/C=US/ST=NV/L=Reno/O=NA/OU=NA/CN=localhost"

# Expose HTTPS port
EXPOSE 40443

# Command to run the app with SSL
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "40443", "--ssl-keyfile", "/home/user/app/sec/key.pem", "--ssl-certfile", "/home/user/app/sec/cert.pem"]
