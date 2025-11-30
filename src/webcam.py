import cv2
import mediapipe as mp

def main():
    # Initialize the webcam
    cap = cv2.VideoCapture(1) # might need to change index based on your system

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=0,       
        min_detection_confidence=0.5
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        h, w, _ = frame.shape # Get frame dimensions
        face_crop = None
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # OpenCV uses BGR color order, but MediaPipe expects RGB
        results = face_detection.process(rgb_frame) # Process the frame with MediaPipe Face Mesh
        
        if results.detections:
            detection = results.detections[0]  
            bbox = detection.location_data.relative_bounding_box # get bounding box
            
            # Normalized coordinates to pixel coordinates
            x_min = int(bbox.xmin * w)
            y_min = int(bbox.ymin * h)
            box_width = int(bbox.width * w)
            box_height = int(bbox.height * h)

            x_max = x_min + box_width
            y_max = y_min + box_height

            # Clip to image boundaries
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)

            # Draw rectangle on the original frame
            cv2.rectangle(
                frame,
                (x_min, y_min),
                (x_max, y_max),
                (0, 255, 0),   # green box
                2
            )
            
            if x_max > x_min and y_max > y_min:
                face_crop = frame[y_min:y_max, x_min:x_max]
                face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                face_crop = cv2.resize(face_crop, (48, 48))
           
        cv2.imshow('Webcam Feed', frame)  
           
        if face_crop is not None:
            cv2.imshow('Model Input', face_crop)

        if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quit
            break
        


    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()
