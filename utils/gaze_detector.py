import numpy as np

def detect_gaze(left_eye, right_eye, frame_width, frame_height):
    def eye_center(eye):
        return np.mean(eye, axis=0).astype(int)

    left_eye_center = eye_center(left_eye)
    right_eye_center = eye_center(right_eye)

    def determine_gaze(eye_center, frame_width, frame_height):
        if eye_center[0] < frame_width * 0.4:
            return "Left"
        elif eye_center[0] > frame_width * 0.6:
            return "Right"
        elif eye_center[1] < frame_height * 0.4:
            return "Up"
        elif eye_center[1] > frame_height * 0.6:
            return "Down"
        else:
            return "Center"

    left_eye_gaze = determine_gaze(left_eye_center, frame_width, frame_height)
    right_eye_gaze = determine_gaze(right_eye_center, frame_width, frame_height)

    return left_eye_gaze, right_eye_gaze
