apt-get update && apt-get install -y ffmpeg
uvicorn backend.main:app --host 0.0.0.0 --port $PORT --workers 1