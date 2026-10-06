FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py llm_client.py ./
EXPOSE 8003
# Pass LLM_BASE_URL / LLM_API_KEY / LLM_MODEL with -e or --env-file .env
# (for Ollama on the host use LLM_BASE_URL=http://host.docker.internal:11434/v1)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8003"]
