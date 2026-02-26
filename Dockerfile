FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8501

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY src /app/src
COPY data /app/data

EXPOSE 8501

CMD ["sh", "-c", "streamlit run src/app.py --server.address 0.0.0.0 --server.port ${PORT} --server.headless true"]
