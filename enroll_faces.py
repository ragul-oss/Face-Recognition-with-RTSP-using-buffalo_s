import cv2
import os
import json
import numpy as np
from insightface.app import FaceAnalysis

# ---------------- CONFIG ----------------
RTSP_URL = "your_url"
DB_DIR = "face_db"
EMB_PATH = os.path.join(DB_DIR, "embeddings.npy")
LABEL_PATH = os.path.join(DB_DIR, "labels.json")

os.makedirs(DB_DIR, exist_ok=True)

# ---------------- LOAD EXISTING DB ----------------
if os.path.exists(EMB_PATH):
    embeddings = np.load(EMB_PATH).tolist()
    with open(LABEL_PATH, "r") as f:
        labels = json.load(f)
else:
    embeddings = []
    labels = []

# ---------------- MODEL ----------------
app = FaceAnalysis(
    name="buffalo_s",
    providers=["CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

# ---------------- RTSP ----------------
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("[ERROR] RTSP stream not opened")
    exit(1)

print(">>> RTSP FACE ENROLLMENT <<<")
print("Press 's' to save face | 'q' to quit")

person_name = input("Enter person name / ID: ").strip()

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    faces = app.get(frame)

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Enroll Face (RTSP)", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        if len(faces) == 0:
            print("[WARN] No face detected")
            continue

        embeddings.append(faces[0].embedding.tolist())
        labels.append(person_name)

        np.save(EMB_PATH, np.array(embeddings))
        with open(LABEL_PATH, "w") as f:
            json.dump(labels, f)

        print(f"[OK] Face enrolled: {person_name}")
        break

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
