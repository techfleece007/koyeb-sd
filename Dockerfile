FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py /app/app.py

EXPOSE 8000

CMD ["bash", "-lc", "uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
