FROM python:3.11-slim

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

CMD uvicorn main:app --host 0.0.0.0 --port $PORT