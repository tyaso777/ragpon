# For Linux:
    # podman machine ssh
    # cd /home/user/ragpon/ragpon/apps/streamlit
    # podman build -t ragpon-streamlit:latest .

FROM python:3.11-slim
WORKDIR /app
COPY ../../.. /app

ENV PYTHONPATH=/app \
    ANONYMIZED_TELEMETRY=False

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir streamlit==1.43.1 pydantic==2.7.3 spacy==3.7.5

CMD ["streamlit", "run", "ragpon/ui/streamlit_app.py", "--server.port=8005", "--server.address=0.0.0.0"]

# podman run -it --rm `
#   --name ragpon-streamlit `
#   --network ragnet `
#   -v "D:\Users\AtsushiSuzuki\OneDrive\デスクトップ\test\ragpon:/app:Z" `
#   # -v "/home/user/ragpon:/app:Z" `
#   --workdir /app `
#   -e PYTHONPATH=/app `
#   -e OPENAI_API_KEY=$Env:OPENAI_API_KEY `
#   -p 8005:8005 `
#   ragpon-streamlit bash -c "streamlit run streamlit_app.py --server.port 8005 --server.address 0.0.0.0"
