import cv2
import os
import numpy as np
from sklearn.metrics import accuracy_score
from collections import deque
from utils.gaze_detector import detect_gaze
from utils.face_detection import detect_faces_and_landmarks
from utils.plotter import init_plot, update_plot

# Resize the image to a maximum width or height while keeping the aspect ratio
def resize_image(image, max_width=800, max_height=600):
    height, width = image.shape[:2]

    if width > max_width:
        scaling_factor = max_width / float(width)
        image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    height, width = image.shape[:2]  # Update height and width after resizing
    if height > max_height:
        scaling_factor = max_height / float(height)
        image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    return image

# Function to process an uploaded image and return a prediction
def process_single_image(image_path, gaze_history):
    frame = cv2.imread(image_path)
    frame = resize_image(frame)  # Resize the image for display
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces, landmarks = detect_faces_and_landmarks(gray_frame)

    if faces:
        for face, landmark in zip(faces, landmarks):
            left_eye = landmark[42:48]
            right_eye = landmark[36:42]

            # Detect gaze direction
            frame_height, frame_width = frame.shape[:2]
            left_eye_gaze, right_eye_gaze = detect_gaze(left_eye, right_eye, frame_width, frame_height)

            # Accumulate gaze history for stability
            gaze_direction = (left_eye_gaze != "Center" or right_eye_gaze != "Center")
            gaze_history.append(gaze_direction)
            if len(gaze_history) > 5:  # Limit history length for performance
                gaze_history.popleft()

            # Return prediction: "Cheating" if the majority of gaze history indicates looking away
            return "Cheating" if sum(gaze_history) > len(gaze_history) // 2 else "Not Cheating"

    return "Not Cheating"  # Default to "Not Cheating" if no faces are detected

# Function to process a folder of images and calculate accuracy
def process_folder_and_evaluate(folder_path):
    true_labels = []
    predictions = []

    # Walk through the folder and subfolders (cheating, not_cheating)
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        if not os.path.isdir(subfolder_path):
            continue

        # Label based on the subfolder name
        label = subfolder.lower()
        gaze_history = deque(maxlen=5)  # Initialize gaze history for each image

        for image_name in os.listdir(subfolder_path):
            image_path = os.path.join(subfolder_path, image_name)

            # Get the model's prediction
            prediction = process_single_image(image_path, gaze_history)

            # Append true label and prediction to lists
            true_labels.append(1 if label == 'cheating' else 0)  # Convert labels to binary: 1 for cheating, 0 for not cheating
            predictions.append(1 if prediction == "Cheating" else 0)  # Convert predictions to binary

    # Calculate accuracy using sklearn
    accuracy = accuracy_score(true_labels, predictions)

    return accuracy

# Function for webcam-based gaze detection
def process_webcam():
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

                # Display "Look into Screen" or "Focus"
                if left_eye_gaze != "Center" or right_eye_gaze != "Center":
                    cv2.putText(frame, "Look into Screen", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    direction_history.append(3)  # Spikes for looking away
                else:
                    cv2.putText(frame, "Focus", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
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

# Main function to switch between webcam and image input
def main():
    while True:
        mode = input("Choose mode: 1. Webcam 2. Upload Image (1/2): ").strip()

        if mode == '1':
            process_webcam()
            break  # Exit after webcam processing

        elif mode == '2':
            upload_mode = input("Choose upload option: 1. Upload Single Image 2. Upload Folder (1/2): ").strip()

            if upload_mode == '1':
                image_path = input("Enter the path of the image: ")
                if os.path.exists(image_path):
                    gaze_history = deque(maxlen=5)  # Initialize gaze history for single image
                    prediction = process_single_image(image_path, gaze_history)
                    print(f"Prediction: {prediction}")
                else:
                    print("Invalid image path!")

            elif upload_mode == '2':
                folder_path = input("Enter the path to the folder containing 'cheating' and 'not_cheating' subfolders: ")

                if os.path.exists(folder_path):
                    accuracy = process_folder_and_evaluate(folder_path)
                    print(f"Accuracy: {accuracy * 100:.2f}%")
                else:
                    print("Invalid folder path!")
            break  # Exit after processing image or folder

        else:
            print("Invalid input. Please choose 1 or 2.")

if __name__ == '__main__':
    main()
