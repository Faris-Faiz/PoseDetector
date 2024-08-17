# Pose Detection Project

This project uses YOLOv8 and MediaPipe to detect poses and objects in real-time video streams.

## Prerequisites

- Windows 10 or higher
- Python 3.8 or higher
- Visual Studio Code
- Git (optional, for cloning the repository)

## Setup Guide

Follow these steps to set up and run the pose detection project using Visual Studio Code on Windows:

### 1. Open the Project in Visual Studio Code

1. Open Visual Studio Code.
2. Go to File > Open Folder and select the `person-detection` folder.

### 2. Open a Terminal in Visual Studio Code

1. In VSCode, go to View > Terminal or use the shortcut `` Ctrl + ` ``.
2. This will open a terminal at the bottom of your VSCode window.

### 3. Create a Virtual Environment

In the VSCode terminal, create a virtual environment by running:

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

After creating the virtual environment, activate it by running:

```bash
.\venv\Scripts\activate
```

You should see `(venv)` at the beginning of your terminal prompt, indicating that the virtual environment is active.

### 5. Install Requirements

With the virtual environment activated, install the required packages:

```bash
pip install -r requirements.txt
```

This may take a few minutes as it installs all necessary dependencies.

### 6. Run the Pose Detection Script

Now you're ready to run the pose detection script. In the VSCode terminal, enter:

```bash
python pose_detection.py
```

The script will open your default webcam and start detecting poses and objects in real-time.

### 7. Using the Application

- The application will open a window showing the video feed from your webcam.
- It will detect the main person in the frame and draw pose landmarks.
- The script will also detect backpacks and handbags.
- Colored circles represent virtual bags: red, green, and blue.
- The script will print messages when it detects an arm pointing towards one of these virtual bags.

### 8. Exiting the Application

To exit the application, press 'q' while the video window is in focus.

## Troubleshooting

1. If you encounter any issues with OpenCV, try reinstalling it in your virtual environment:
   ```bash
   pip uninstall opencv-python-headless opencv-contrib-python opencv-python
   pip install opencv-python opencv-contrib-python
   ```

2. Make sure your webcam is properly connected and not being used by another application.

3. If you face any GPU-related issues, ensure you have the correct CUDA version installed for your system.

4. If VSCode doesn't recognize your virtual environment, you may need to select the Python interpreter:
   - Press `Ctrl+Shift+P` to open the Command Palette.
   - Type "Python: Select Interpreter" and choose this option.
   - Select the interpreter in the `.\venv\Scripts\` directory.

## Notes

- The `yolov8n.pt` file should be in the same directory as `pose_detection.py`.
- Ensure you have adequate lighting for better pose detection results.
- The script uses YOLOv8 for object detection and MediaPipe for pose estimation.
- Always make sure your virtual environment is activated in the VSCode terminal before running the script or installing packages.