# For Linux:
    # podman machine ssh
    # cd /home/user/ragpon/ragpon/apps/chroma
    # podman build -t chroma-app:latest .

# Use Python 3.11 base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Disable telemetry
ENV ANONYMIZED_TELEMETRY=False

# Install chromadb
RUN pip install chromadb==0.5.23

# Run chroma when container starts
CMD ["chroma", "run", "--path", "/app/ragpon/examples/db", "--host", "0.0.0.0", "--port", "8007"]

# how to run:
# see how_to_run_memo.txt