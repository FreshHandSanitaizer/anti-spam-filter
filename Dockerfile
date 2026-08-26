FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Сначала копируем только requirements_for_api.txt и устанавливаем зависимости
COPY ./requirements_for_api.txt /app
RUN pip3 install -r requirements_for_api.txt

# Затем копируем остальной код
COPY . /app

CMD ["python3", "app_api.py"]