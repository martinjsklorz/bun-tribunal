# CNN inference server. Build from cnn/:
#   docker build -t hotdog-cnn .
#   docker run -p 8002:8002 -v "$PWD/artifacts:/app/artifacts:ro" hotdog-cnn
#   docker run -p 8002:8002 -e MOCK=1 hotdog-cnn          # no weights needed
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    CNN_ARTIFACTS=/app/artifacts DEVICE=cpu
WORKDIR /app/server

COPY server/requirements.txt .
# CPU wheels keep the image small; drop --extra-index-url for a CUDA build.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY hotdog_model.py /app/hotdog_model.py
COPY server/app.py .
RUN mkdir -p /app/artifacts

EXPOSE 8002
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,json,sys; sys.exit(0 if json.load(urllib.request.urlopen('http://127.0.0.1:8002/health'))['ready'] else 1)"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
