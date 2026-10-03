FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
# CPU-only torch/torchvision build -- much smaller and faster to pull than the
# default CUDA wheels, and this Space only ever runs on CPU.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY app.py treatment_db.py rag.py ./
COPY models ./models

# Render (and most Docker PaaS hosts) inject a $PORT env var the container must
# listen on; default to 7860 for local `docker run` testing when it's unset.
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
