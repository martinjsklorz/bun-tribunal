# CUDA image for Linux + NVIDIA GPU (needs ~20 GB VRAM in bf16).
# On Apple Silicon do NOT use Docker: containers cannot access MPS. Run natively (see README).
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/hf VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/* && python3 -m venv /opt/venv

WORKDIR /app
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt
COPY *.py clef_schema.json ./

# Mount a volume at /models so the ~19 GB download survives container restarts:
#   docker run --gpus all -p 8001:8001 -v clef-models:/models clef-server
VOLUME /models
EXPOSE 8001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
