FROM python:3.11-slim

# HF Spaces runs as non-root user 1000
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Give ownership to appuser
RUN chown -R appuser:appuser /app

USER appuser

# HF Spaces expects port 7860
EXPOSE 7860

CMD ["gunicorn", "app:app", \
     "--bind", "0.0.0.0:7860", \
     "--workers", "2", \
     "--threads", "4", \
     "--timeout", "60"]