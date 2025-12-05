import cv2
import mediapipe as mp
import numpy as np
import joblib
import os

from feature_extraction import extract_features   # your function

# Pastel colors for emotions
PASTEL_COLORS = {
    "happy":    (204, 153, 255),  # light purple
    "sad":      (255, 204, 204),  # light pink
    "angry":    (153, 204, 255),  # light blue
    "neutral":  (204, 255, 229),  # mint green
    "fear":     (255, 179, 102),  # light orange
    "disgust":  (229, 204, 255),  # lavender
    "surprise": (255, 255, 153),  # pale yellow
}

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
        color = (200, 200, 200)
        face_box = None

        # If a face is detected
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Bounding box
            xs = [lm.x * w for lm in landmarks]
            ys = [lm.y * h for lm in landmarks]
            x1, y1 = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))
            face_box = (x1, y1, x2, y2)
            
            # Extract the numeric feature vector and feature names
            feat_values, _ = extract_features(landmarks)
            feat_values = feat_values.reshape(1, -1)

            # Predict emotion
            pred = model.predict(feat_values)[0]
            emotion_text = label_encoder.inverse_transform([pred])[0]

            color = PASTEL_COLORS.get(emotion_text.lower(), (255, 255, 255))

        # Draw face box
        if face_box:
            x1, y1, x2, y2 = face_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)  # black box
            cv2.putText(
                frame,
                emotion_text,
                (x1, max(30, y1 - 10)),  # above box
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                color,
                3
            )
        else:
            cv2.putText(frame, emotion_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
     

        # Display webcam feed
        cv2.imshow("Emotion Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # 'q' or ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
