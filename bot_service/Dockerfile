FROM python:3.10-slim
WORKDIR /app

# Только клиентские зависимости
RUN pip install --no-cache-dir \
    aiogram==3.10.0 \
    requests

COPY bot.py .

CMD ["python", "bot.py"]
