GazeSentinel uses a webcam to detect and track the direction of a user's gaze in real-time. The system is built using OpenCV for image processing, dlib for facial landmark detection, and Matplotlib for visualizing gaze data dynamically on a graph. Here's a detailed explanation of each component of the project:

Overview

The goal of the project is to detect whether a person is focused on a screen or looking elsewhere based on the direction of their eye gaze. It visualizes the tracking data on a live-updating graph and provides feedback on the screen when the person is either focused or distracted.

Key Features

1. Real-Time Gaze Detection: Tracks the user's gaze by identifying eye regions using facial landmarks.
2. Focus/Distraction Detection: The system detects if the user's eyes are centered on the screen (focused) or looking away (distracted).
3. Live Plot of Gaze Data: A dynamic graph shows the direction of the user's gaze in real-time using Matplotlib.
4. Webcam Interface: Uses a live video feed from the webcam to continuously analyze the user's gaze.
5. Modular Design: The code is structured into separate components for better organization and readability.

Components

1. Gaze Detection
   - Gaze detection is done by detecting the user's face and then locating the eye regions.
   - It determines whether the user is looking Left, Right, Up, Down, or Center by calculating the center of the eyes.
   - If the gaze is focused (center), the system displays "Focused" on the video frame. Otherwise, it shows "Look into Screen" and the plot spikes to indicate distraction.

2. Facial Landmark Detection
   - dlib is used for detecting 68 facial landmarks, which correspond to key points on the face (eyes, nose, mouth, etc.).
   - These landmarks allow the system to identify where the user's eyes are located and analyze their movement to determine gaze direction.

3. Real-Time Plotting
   - Matplotlib is used to create a live graph of the gaze direction data.
   - The graph shows the gaze history over time, with spikes representing moments when the user is looking away, and a flat line representing when the user is focused on the screen.

4. Video Feed with OpenCV
   - OpenCV handles capturing the webcam feed and displaying the video in a window.
   - It overlays bounding boxes around the user's eyes and messages indicating focus or distraction directly on the video feed.



Breakdown of the Code

 1. `main.py`

This is the main entry point of the program. It:
- Initializes the webcam and video feed.
- Detects faces and facial landmarks in real-time.
- Passes the detected eye regions to the gaze detection logic to determine where the user is looking.
- Displays the result on the video feed (focused/distracted).
- Updates a real-time graph with gaze direction data.

 2. `gaze_detector.py`

This file contains the logic to detect the gaze direction:
- The function `detect_gaze()` receives the coordinates of the left and right eyes.
- Based on the position of the eye center, it classifies the user's gaze into "Left", "Right", "Up", "Down", or "Center".

 3. `face_detection.py`

This file handles:
- Loading the dlib pre-trained models for detecting the face and its 68 facial landmarks.
- Detecting faces from the grayscale image captured from the webcam.
- Returning the detected landmarks so that the eyes can be isolated for gaze detection.

 4. `plotter.py`

This file is responsible for:
- Initializing a real-time plot using Matplotlib.
- Continuously updating the plot with the most recent gaze direction data (1 for focused, 3 for distracted).
- Keeping a fixed window of the last 500 frames in the graph to maintain a smooth real-time experience.

How the System Works

1. Webcam Video Feed:
   - The webcam continuously captures frames.
   - OpenCV converts these frames to grayscale for face detection.

2. Face Detection:
   - dlib detects the face within the frame.
   - 68 facial landmarks are identified, allowing the system to locate the eyes.

3. Gaze Detection:
   - The system calculates the center point of both eyes and compares their position relative to the width and height of the frame.
   - Based on this, it determines whether the user is looking left, right, up, down, or center.

4. Feedback:
   - If the gaze is centered, the system displays "Focused" on the video.
   - If the gaze is away, it displays "Look into Screen" to notify the user.

5. Real-Time Plot:
   - As the user’s gaze changes, the system updates a real-time plot that visually represents whether the user is focused or distracted.


Installation and Setup

1. Install Dependencies:
   - Install Python packages listed in `requirements.txt`:
     ```bash
     pip install -r requirements.txt
     ```

2. Download Pre-Trained Model:
   - Download the `shape_predictor_68_face_landmarks.dat` model from the [dlib model page](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2), decompress it, and place it inside the `models/` folder.

3. Run the Program:
   - Run the main script:
     ```bash
     python main.py
     ```

Requirements

- Python 3.x
- OpenCV: For capturing webcam video feed and drawing the visualizations.
- dlib: For face detection and facial landmark detection.
- imutils: To help in manipulating the coordinates of facial landmarks.
- Matplotlib: To display real-time graph tracking gaze data.
- scipy: Used for calculating distances between points, which helps in detecting gaze directions.

How to Extend the Project

- Gaze Tracking Calibration: Implement a calibration process to make gaze detection more accurate for different users.
- Focus Duration Analysis: Add functionality to analyze how long the user is focusing on the screen versus looking away.
- Alert System: Integrate an alert that notifies the user after prolonged distraction.



