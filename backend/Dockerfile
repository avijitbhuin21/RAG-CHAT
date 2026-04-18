FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY . ./backend

CMD uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-8000}
