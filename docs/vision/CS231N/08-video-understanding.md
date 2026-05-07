---
layout: default
title: "08 Video Understanding"
permalink: /vision/CS231N/08
#subtitle: 
use_math: true
parent: CS231N
grand_parent: vision
---

# 8. Video Understanding


### Introduction & Challenges

- **Definition of Video**: Video is treated as a 4D tensor (3×T×H×W), essentially adding a temporal dimension to 2D images.
    
- **Action Recognition**: Unlike image classification which focuses on objects, video understanding primarily focuses on identifying actions and activities.
    
- **Computational Constraints**: High-definition videos are too large for GPU memory (e.g., 10GB per minute). The solution is to process short **clips** (e.g., 16–32 frames) using a sliding window approach.
    

---

### Basic Fusion Strategies

- **Single Frame Baseline**: A simple but strong baseline where a 2D CNN is run on individual frames and their predictions are averaged.
    
- **Late Fusion**: Features are extracted from frames independently and concatenated or pooled only at the very last stage of the network.
    
- **Early Fusion**: All frames in a clip are concatenated at the input level (channel dimension), collapsing temporal information in the very first layer.
    
- **Slow Fusion (3D CNN)**: Balanced approach that gradually fuses spatial and temporal information throughout the network layers.
    

---

### 3D Convolutional Architectures

- **3D Convolution Operation**: Uses a 3D kernel that slides across both spatial (H,W) and temporal (T) dimensions, allowing for temporal shift invariance.
    
- **C3D (VGG for 3D)**: A VGG-style architecture extended to 3D. While effective as a feature extractor, it is computationally expensive (3x more FLOPs than 2D VGG).
    
- **I3D (Inflated 3D ConvNet)**: Architecture that "inflates" 2D filters from successful image models (like Inception) into 3D filters, allowing the reuse of ImageNet pre-trained weights.
    

---

### Motion & Two-Stream Networks

- **Optical Flow**: Explicitly measures the velocity of pixels between adjacent frames to capture low-level motion cues.
    
- **Two-Stream Networks**: Separate CNN streams for **Appearance** (RGB frames) and **Motion** (Optical Flow maps), with their predictions fused at the end.
    
- **Motion Importance**: Surprisingly, the motion stream often performs better than the appearance stream for many action categories.
    

---

### Long-Term Modeling & Attention

- **RNNs & LSTMs**: Used to aggregate features from 2D or 3D CNNs over longer sequences to capture extended temporal structures.
    
- **Non-Local Networks**: Implements **Self-Attention** in 3D to capture global interactions across space and time, which is more parallelizable than RNNs.
    
- **Video Transformers**: Modern approach using patches and attention (e.g., Video MAE) to achieve state-of-the-art results in action classification.
    

---

### Multimodal & Future Directions

- **Audio-Visual Understanding**: Integrating sound as a secondary modality for tasks like audio source separation or enhanced action recognition.
    
- **Egocentric Vision**: Processing video from first-person perspectives (e.g., smart glasses) to understand social interactions.
    
- **Video LLMs**: Connecting video encoders to Large Language Models to enable video-based Q&A and description.