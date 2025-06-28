FROM python:3.11-slim

WORKDIR /mnt/locust

ENV ANONYMIZED_TELEMETRY=False

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir locust

EXPOSE 8089

CMD ["locust", "-f", "locustfile.py", "--host", "http://ragpon-fastapi:8006", "--web-host", "0.0.0.0"]
