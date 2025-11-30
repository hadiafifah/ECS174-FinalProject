import cv2
from src.model.predict_emotion import predict_emotion
from realtime.webcam import get_features_from_webcam
import mediapipe as mp

def get_face_crop(frame):
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5 #detects faces with atleast 50% confidence
    )

    h, w, _ = frame.shape #frame dimension
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame) #run detection

    #if face is detected 
    if results.detections:
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box #extract the bounding box
        #convert to pixel coordinates
        x_min = max(0, int(bbox.xmin * w))
        y_min = max(0, int(bbox.ymin * h))
        x_max = min(w, x_min + int(bbox.width * w))
        y_max = min(h, y_min + int(bbox.height * h))
        #crop the face
        face_crop = frame[y_min:y_max, x_min:x_max]
        face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        face_crop = cv2.resize(face_crop, (48, 48)) #cnn trained on 48*48
        return face_crop
    return None

#use opencv to overlay the emotion string
def display_emotion_on_screen(frame, emotion):
    cv2.putText(frame, emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


def main():
    cap = cv2.VideoCapture(0) #open default webcam
    mp_face_detection = mp.solutions.face_detection

    while True:
        ret, frame = cap.read() #grab each frame from loop
        if not ret:
            break

        face_crop = get_face_crop(frame)

        if face_crop is not None: #if face exists
            emotion = predict_emotion(face_crop)
            display_emotion_on_screen(frame, emotion)

        cv2.imshow("Emotion Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release() #clean up
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()