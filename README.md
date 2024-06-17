# PoseControl
Pose recognition game controls

# PoseControl

PoseControl leverages MediaPipe Pose and OpenCV to create game controls based on real-time pose recognition. This project detects specific body movements to trigger game actions such as attacking, blocking, and dodging.

## Features
- **Real-time Pose Detection:** Uses MediaPipe Pose for accurate and efficient pose estimation.
- **Game Controls Integration:** Maps specific body movements to game actions like attack, block, and dodge.
- **Multi-Action Recognition:** Detects arm hits, blocks, and dodges based on pose angles and visibility.
- **Webcam Input:** Captures input from a webcam to track and process user movements.

## Installation

Clone the repository and install the required dependencies:
```bash
git clone https://github.com/yourusername/PoseControl.git
cd PoseControl
pip install -r requirements.txt
```
## Usage

To run the pose recognition and game control script, execute the following command:

bash

python posecontrol.py

Make sure your webcam is connected and properly configured.

## How It Works
The project uses the following libraries:

    cv2 (OpenCV) for capturing video input and processing images.
    mediapipe for pose detection and landmark recognition.
    numpy for numerical calculations.
    pydirectinput for simulating keyboard inputs.
    math for mathematical calculations.
