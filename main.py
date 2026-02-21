import cv2
import numpy as np
import winsound
import os
import time
from ultralytics import YOLO
import audio


# ________________________________________________
# LOAD MODELS
# ________________________________________________
def load_models():

    # YOLO
    yolo_model = YOLO("yolov8n.pt")

    # Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    # LBPH recognizer
    recognizer = train_recognizer(face_cascade)

    return yolo_model, face_cascade, recognizer


# ________________________________________________
# TRAIN FACE RECOGNIZER
# ________________________________________________
def train_recognizer(face_cascade, folder="faces"):

    if not os.path.exists(folder):
        print("Faces folder not found.")
        return None

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []

    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        detected = face_cascade.detectMultiScale(img, 1.3, 5)

        for (x, y, w, h) in detected:
            faces.append(img[y:y+h, x:x+w])
            labels.append(0)  # Sne label

    if len(faces) > 0:
        recognizer.train(faces, np.array(labels))
        print("Face recognizer trained.")
        return recognizer

    print("No valid face training data.")
    return None


# ________________________________________________
# OPEN CAMERA
# ________________________________________________
def open_camera(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise Exception("Could not open camera.")
    return cap


# ________________________________________________
# DETERMINE POSITION
# ________________________________________________
def get_position(x_center, frame_width):

    if x_center < frame_width / 3:
        return "left"
    elif x_center > 2 * frame_width / 3:
        return "right"
    else:
        return "center"


# ________________________________________________
# PROCESS FRAME
# ________________________________________________
def process_frame(frame, yolo_model, face_cascade, recognizer, last_announced):

    detected_labels = []
    frame_height, frame_width = frame.shape[:2]

    results = yolo_model(frame, verbose=False)

    for result in results:
        for box in result.boxes:

            confidence = float(box.conf[0])
            if confidence < 0.5:
                continue

            class_id = int(box.cls[0])
            label = yolo_model.names[class_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            height = y2 - y1

            # Distance alert
            if height > 0:
                distance_cm = 1000 / height
                if distance_cm < 50:
                    winsound.Beep(1000, 150)

            x_center = (x1 + x2) / 2
            position = get_position(x_center, frame_width)

            # If YOLO detects a person → run face recognition
            if label == "person" and recognizer is not None:

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                person_roi = gray[y1:y2, x1:x2]

                faces = face_cascade.detectMultiScale(person_roi, 1.3, 5)

                for (fx, fy, fw, fh) in faces:
                    face_img = person_roi[fy:fy+fh, fx:fx+fw]

                    try:
                        pred_label, conf = recognizer.predict(face_img)

                        if pred_label == 0 and conf < 80:
                            label = "Sne"
                        else:
                            label = "Unknown"

                    except:
                        label = "Unknown"

            detected_labels.append(label)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)

            cv2.putText(frame,
                        f"{label} {position}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)

            # Smart speech cooldown
            key = f"{label}_{position}"
            current_time = time.time()

            if key not in last_announced or current_time - last_announced[key] > 3:
                audio.speak(f"{label} on your {position}")
                last_announced[key] = current_time

    return frame, detected_labels


# ________________________________________________
# MAIN
# ________________________________________________
def main():

    yolo_model, face_cascade, recognizer = load_models()
    cap = open_camera()
    last_announced = {}

    print("Press V for voice command")
    print("Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, detected_objects = process_frame(
            frame,
            yolo_model,
            face_cascade,
            recognizer,
            last_announced
        )

        cv2.imshow("VisionAssist", processed_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("v"):
            command = audio.listen()
            if command:
                audio.find_item_in_screen(command, detected_objects)

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()