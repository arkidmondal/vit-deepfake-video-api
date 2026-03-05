import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from backend.inference import predict_video

app = FastAPI(title="Deepfake Video Detection API")

UPLOAD_DIR = "temp_videos"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def home():
    return {"message": "Deepfake Video Detection API running"}


@app.post("/predict-video")
async def predict_video_api(file: UploadFile = File(...)):

    # Save uploaded file
    filename = os.path.basename(file.filename)
    video_path = os.path.join(UPLOAD_DIR, filename)

    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = predict_video(video_path)
        return JSONResponse(result)

    except Exception as e:
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)