FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (separate layer -> Docker cache is reused unless
# requirements.txt actually changes, so rebuilds during development are fast)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the application code and data
COPY src/ ./src/
COPY data/policies/ ./data/policies/

# The RAG index is built at container startup (see src/main.py lifespan handler),
# so it's regenerated fresh inside the container rather than shipped in the image.

EXPOSE 8000

WORKDIR /app/src

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
