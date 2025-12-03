import cv2
import mediapipe as mp
import numpy as np
import joblib
import os

from feature_extraction import extract_features   # your function

def main():
    # Load trained models
    model_path = os.path.join("..", "models", "emotion_model.joblib")
    encoder_path = os.path.join("..", "models", "label_encoder.joblib")

    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)

    # Initialize webcam
    cap = cv2.VideoCapture(1)   # change to 0 if needed

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb_frame)
        emotion_text = "No face"

        # If a face is detected
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Extract the numeric feature vector and feature names
            feat_values, _ = extract_features(landmarks)
            feat_values = feat_values.reshape(1, -1)

            # Predict emotion
            pred = model.predict(feat_values)[0]
            emotion_text = label_encoder.inverse_transform([pred])[0]


        # Overlay prediction on frame
        cv2.putText(
            frame,
            emotion_text,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

        # Display webcam feed
        cv2.imshow("Emotion Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
