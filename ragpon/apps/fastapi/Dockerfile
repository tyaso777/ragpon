# For Linux:
    # copy db and models to apps/fastapi folder manually.
    # copy Sudachi packages to user/ragpon/packages manually.
    # change compose.yml to use azure openai.

    # podman machine ssh
    # cd /home/user/ragpon
    # podman build -f ragpon/apps/fastapi/Dockerfile -t ragpon-fastapi:latest .

# create requirements.txt file with:
# poetry export --with dev --without-hashes -f requirements.txt -o requirements.txt

# Use lightweight Python base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy the entire ragpon project into the image
COPY ../../../../requirements.txt /app/requirements.txt

# For Linux:
# COPY ../../../../packages       ./packages

# For development, you can copy the entire ragpon project
# COPY ../../.. /app

# Set environment variables
ENV PYTHONPATH=/app \
    ANONYMIZED_TELEMETRY=False

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8006

# Run the FastAPI app
CMD ["uvicorn", "ragpon.apps.fastapi.fast_api_app:app", "--host", "0.0.0.0", "--port", "8006", "--workers", "4"]

# how to build:
# cd ragpon
# podman build -f ragpon/apps/fastapi/Dockerfile -t ragpon-fastapi .

# how to run:
# see how_to_run_memo.txt

# For deployment, you can delete the -v option by using Copy command when creating the image if necessary.


