import cv2
import json
import numpy as np
from insightface.app import FaceAnalysis

# ---------------- CONFIG ----------------
RTSP_URL = "your_url"
EMB_PATH = "face_db/embeddings.npy"
LABEL_PATH = "face_db/labels.json"
THRESHOLD = 0.45

FRAME_SKIP = 3          # 🔥 inference every N frames
RESIZE_W = 640
RESIZE_H = 360

# ---------------- LOAD DATABASE ----------------
known_embeddings = np.load(EMB_PATH)
with open(LABEL_PATH, "r") as f:
    labels = json.load(f)

# ---------------- MODEL ----------------
app = FaceAnalysis(
    name="buffalo_s",
    providers=["CPUExecutionProvider"]  # GPU later when TensorRT is ready
)
app.prepare(ctx_id=0, det_size=(320, 320))

# ---------------- RTSP STREAM ----------------
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FPS, 15)

if not cap.isOpened():
    print("[ERROR] Cannot open RTSP stream")
    exit(1)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(">>> RTSP FACE RECOGNITION STARTED <<<")

frame_id = 0
last_faces = []   # store last inference result

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # 🔹 Resize frame to control latency
    frame = cv2.resize(frame, (RESIZE_W, RESIZE_H))

    frame_id += 1

    # 🔥 Run inference only every FRAME_SKIP frames
    if frame_id % FRAME_SKIP == 0:
        last_faces = app.get(frame)

    # 🔹 Draw results from last inference
    for face in last_faces:
        emb = face.embedding
        sims = [cosine_similarity(emb, e) for e in known_embeddings]

        best_idx = np.argmax(sims)
        best_score = sims[best_idx]

        x1, y1, x2, y2 = map(int, face.bbox)

        if best_score > THRESHOLD:
            name = labels[best_idx]
            color = (0, 255, 0)
        else:
            name = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            frame,
            f"{name} {best_score:.2f}",
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    cv2.imshow("Jetson RTSP Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

