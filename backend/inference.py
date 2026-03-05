import torch
import subprocess
import os
import requests

from backend.model_loader import load_model
from backend.video_processor import process_video

THRESHOLD = 0.40
AUDIO_API_URL = "https://ai-audio-detection-cp2l.onrender.com/predict-audio"
TIMEOUT = 15

device = torch.device("cpu")
model = load_model(device)


def extract_audio(video_path, audio_path):
    """
    Extract audio from video using FFmpeg
    """
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        audio_path,
        "-y"
    ]

    subprocess.run(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    return os.path.exists(audio_path)


def get_audio_score(audio_path):
    """
    Send audio to external API
    """
    try:
        with open(audio_path, "rb") as f:
            files = {"file": f}

            response = requests.post(
                AUDIO_API_URL,
                files=files,
                timeout=TIMEOUT
            )

        data = response.json()

        fake_percentage = data["fake_percentage"]

        return fake_percentage / 100

    except Exception:
        return "api_failed"


def predict_video(video_path):

    input_tensor = process_video(video_path)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        video_score = torch.sigmoid(logits).item()

    audio_path = video_path + ".wav"

    audio_score = None

    audio_exists = extract_audio(video_path, audio_path)

    if audio_exists:

        score = get_audio_score(audio_path)

        if score == "api_failed":
            audio_score = "api_failed"
            final_score = video_score
        else:
            audio_score = float(score)
            final_score = (video_score + audio_score) / 2

    else:
        audio_score = None
        final_score = video_score

    prediction = "FAKE" if final_score >= THRESHOLD else "REAL"

    if os.path.exists(audio_path):
        os.remove(audio_path)

    return {
        "video_score": float(video_score),
        "audio_score": audio_score,
        "final_score": float(final_score),
        "prediction": prediction
    }