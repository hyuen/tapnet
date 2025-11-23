FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git
COPY requirements.txt .
RUN pip install git+https://github.com/google-deepmind/tapnet.git
RUN pip install git+https://github.com/google-deepmind/recurrentgemma.git@main
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

CMD ["gunicorn", "app.main:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
