import dlib
from imutils import face_utils

# Load dlib pre-trained face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

def detect_faces_and_landmarks(gray_frame):
    faces = detector(gray_frame)
    landmarks = [face_utils.shape_to_np(predictor(gray_frame, face)) for face in faces]
    return faces, landmarks
