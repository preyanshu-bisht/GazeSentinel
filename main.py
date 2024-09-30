import cv2
import matplotlib.pyplot as plt
from collections import deque
from utils.gaze_detector import detect_gaze
from utils.face_detection import detect_faces_and_landmarks
from utils.plotter import init_plot, update_plot

# Initialize constants and video capture
graph_window_size = 500
direction_history = deque([0] * graph_window_size, maxlen=graph_window_size)
cap = cv2.VideoCapture(0)

# Initialize real-time plot
fig, ax, line = init_plot(graph_window_size)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces and landmarks
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces, landmarks = detect_faces_and_landmarks(gray_frame)

    if faces:
        for face, landmark in zip(faces, landmarks):
            left_eye = landmark[42:48]
            right_eye = landmark[36:42]

            # Draw bounding boxes around eyes
            cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
            cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)

            # Detect gaze direction
            frame_height, frame_width = frame.shape[:2]
            left_eye_gaze, right_eye_gaze = detect_gaze(left_eye, right_eye, frame_width, frame_height)

            # Check focus status and append to direction history
            if left_eye_gaze != "Center" or right_eye_gaze != "Center":
                cv2.putText(frame, "Look into Screen", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                direction_history.append(3)  # Spikes for looking away
            else:
                cv2.putText(frame, "Focused", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                direction_history.append(1)  # Flat for focused

            # Update real-time plot
            update_plot(fig, ax, line, direction_history)

    # Display the frame with gaze detection
    cv2.imshow("Gaze Detection", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
