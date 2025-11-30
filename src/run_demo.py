import cv2
from predict_emotion import predict_emotion
#from webcam import get_features_from_webcam
import mediapipe as mp

def get_face_crop(frame, face_detection):
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
        return face_crop, (x_min, y_min, x_max, y_max)
    return None, None

#use opencv to overlay the emotion string
def display_emotion_on_screen(frame, face_bbox, emotion_dict):
    x_min, y_min, x_max, y_max = face_bbox
    emotion = emotion_dict['emotion']
    confidence = emotion_dict['confidence']
    text = f"{emotion} ({confidence:.2f})"

    #color based on emotion
    color_dict = {
        'angry': (255, 102, 102),      # soft red
        'disgust': (102, 255, 102),    # mint green
        'fear': (153, 102, 255),       # lavender
        'happy': (255, 255, 102),      # pastel yellow
        'neutral': (200, 200, 200),    # light grey
        'sad': (102, 178, 255),        # sky blue
        'surprise': (255, 153, 255)    # pink
    }
    color = color_dict.get(emotion, (255, 255, 255))

    # Draw rectangle around the face
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

    # Position text above the face
    font = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    font_scale = 0.8
    thickness = 2
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = x_min
    text_y = max(0, y_min - 10)  # 10 pixels above face

    # Draw semi-transparent background rectangle for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (text_x - 5, text_y - text_height - 5),
                  (text_x + text_width + 5, text_y + 5), (50, 50, 50), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Put the emotion text
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)


def main():
    cap = cv2.VideoCapture(1) #open default webcam
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.5
    )
    while True:
        ret, frame = cap.read() #grab each frame from loop
        if not ret:
            break

        face_crop, bbox = get_face_crop(frame, face_detection)

        if face_crop is not None: #if face exists
            emotion_dict = predict_emotion(face_crop, model_path="models/emotion_net.pth")
            if emotion_dict['confidence'] > 0.6:
                display_emotion_on_screen(frame, bbox, emotion_dict)

        cv2.imshow("Emotion Demo", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: #to exit press q or esc key
            print("Exiting demo...")
            break

    cap.release() #clean up
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()