🎥 Video Deepfake Detection Model
Overview

The video deepfake detection module is designed to identify manipulated or AI-generated videos by analyzing facial and visual inconsistencies across frames. It uses a 
deep convolutional neural network based on EfficientNet to learn spatial features that distinguish real videos from synthetically altered ones. The system processes 
videos by extracting frames, detecting faces, and performing frame-level classification, followed by temporal aggregation to produce a final video-level decision.

Model Architecture

Backbone: EfficientNet (pretrained on ImageNet)

Input: RGB face frames resized to 224×224

Feature Extraction: Depthwise separable convolutions for efficient spatial representation

Classification Head: Fully connected layers with softmax/sigmoid output

Output: Probability score indicating whether a frame/video is real or fake

Processing Pipeline

Frame Extraction: Videos are sampled at fixed intervals.

Face Detection & Cropping: Faces are localized and aligned.

Preprocessing: Normalization and resizing.

Frame-Level Inference: Each frame is passed through EfficientNet.

Temporal Aggregation: Frame scores are averaged to obtain the final video prediction.

Key Features

Handles compressed and social-media videos

Works on both recorded and streamed video inputs

Lightweight and scalable with GPU acceleration

Robust to common deepfake artifacts such as texture mismatch, blending errors, and unnatural facial motion.
