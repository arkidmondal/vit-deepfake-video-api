# Multimodal Deepfake Video Detection API

A FastAPI-based multimodal deepfake detection system that combines **Vision Transformer (ViT) video analysis** with **audio deepfake detection** to identify manipulated videos.

This project analyzes both **visual artifacts** and **audio authenticity** to improve deepfake detection reliability.

---

## Features

- Vision Transformer based video deepfake detection
- Audio deepfake detection via external API
- Multimodal score fusion (video + audio)
- Automatic audio extraction using FFmpeg
- FastAPI backend for real-time inference
- Lightweight deployment-ready architecture
- Designed for cloud deployment (Render / Railway)

---

## System Architecture

Video Input  
↓  
Frame Extraction  
↓  
Vision Transformer Video Model  
↓  
Audio Extraction (FFmpeg)  
↓  
Audio Deepfake Detection API  
↓  
Score Fusion  
↓  
Final Deepfake Prediction

---

## API Response Example

```json
{
  "video_score": 0.42,
  "audio_score": 0.52,
  "final_score": 0.47,
  "prediction": "FAKE"
}