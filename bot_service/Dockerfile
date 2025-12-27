FROM python:3.10-slim
WORKDIR /app

# Минимальный набор зависимостей
RUN pip install --no-cache-dir \
    aiogram==3.10.0 \
    chromadb \
    sentence-transformers \
    requests

COPY bot.py .
COPY moderation.py .
CMD ["python", "bot.py"]
