import cv2
from deepface import DeepFace

video_capture = cv2.VideoCapture(0)  # استفاده از دوربین وب

try:
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        try:
            results = DeepFace.analyze(frame, actions=['age'], enforce_detection=False)
            
            for face in results:
                if 'facial_area' in face:  # برای نسخه‌های جدید DeepFace
                    x, y, w, h = face['facial_area']['x'], face['facial_area']['y'], face['facial_area']['w'], face['facial_area']['h']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    age = face['age']
                    cv2.putText(frame, f'Age: {age}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        except Exception as e:
            print(f"Error in face analysis: {e}")
        
        cv2.imshow('Face Age Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    video_capture.release()
    cv2.destroyAllWindows()