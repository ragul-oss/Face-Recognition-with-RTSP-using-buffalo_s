#!/usr/bin/env python3
"""
Optimized GPU Face Enrollment
- TensorRT + FP16
- Async frame capture
- Batch-safe database save
"""

import os
import cv2
import json
import time
import numpy as np
import threading
from queue import Queue
from insightface.app import FaceAnalysis

# ---------------- CONFIG ----------------
RTSP_URL = "your_url"
DB_DIR = "face_db"
EMB_PATH = os.path.join(DB_DIR,"embeddings.npy")
LABEL_PATH = os.path.join(DB_DIR,"labels.json")
DET_SIZE=(320,320)
USE_FP16=True
os.makedirs(DB_DIR, exist_ok=True)

# ---------------- DATABASE ----------------
def load_db():
    if os.path.exists(EMB_PATH):
        embeddings = np.load(EMB_PATH).tolist()
        with open(LABEL_PATH,"r") as f:
            labels = json.load(f)
        return embeddings, labels
    return [], []

embeddings, labels = load_db()

# ---------------- MODEL ----------------
def init_model():
    providers = [
        ('TensorrtExecutionProvider', {'trt_fp16_enable': USE_FP16, 'trt_engine_cache_enable': True, 'trt_engine_cache_path':'./trt_cache'}),
        'CUDAExecutionProvider',
        'CPUExecutionProvider'
    ]
    app = FaceAnalysis(name="buffalo_s", providers=providers, det_thresh=0.5)
    app.prepare(ctx_id=0, det_size=DET_SIZE)
    dummy = np.zeros((DET_SIZE[1],DET_SIZE[0],3),dtype=np.uint8)
    for _ in range(3): _ = app.get(dummy)
    return app

app = init_model()
print("[MODEL] Ready")

# ---------------- ASYNC CAPTURE ----------------
class AsyncCapture:
    def __init__(self,url):
        self.cap=cv2.VideoCapture(url,cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        self.queue=Queue(maxsize=2)
        self.stopped=False
        self.thread=threading.Thread(target=self.update,daemon=True)
    def start(self):
        self.thread.start()
        return self
    def update(self):
        while not self.stopped:
            ret,frame=self.cap.read()
            if ret:
                if self.queue.full():
                    try: self.queue.get_nowait()
                    except: pass
                self.queue.put(frame)
    def read(self):
        return self.queue.get()
    def stop(self):
        self.stopped=True
        self.thread.join()
        self.cap.release()

cap = AsyncCapture(RTSP_URL).start()

# ---------------- ENROLLMENT ----------------
print(">>> GPU FACE ENROLLMENT <<<")
person_name = input("Enter person name / ID: ").strip()

while True:
    frame = cap.read()
    display_frame = cv2.resize(frame,(960,540))
    faces = app.get(frame)

    # Draw detections
    for f in faces:
        x1,y1,x2,y2=map(int,f.bbox)
        conf = f.det_score
        color = (0,255,0) if conf>0.8 else (0,255,255)
        cv2.rectangle(display_frame,(x1//2,y1//2),(x2//2,y2//2),color,2)
        cv2.putText(display_frame,f"Conf:{conf:.2f}",(x1//2,y1//2-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)

    cv2.imshow("Enrollment - Press 's' to save",display_frame)
    key = cv2.waitKey(1) & 0xFF

    if key==ord('s'):
        if not faces:
            print("[WARN] No face detected")
            continue
        if len(faces)>1: faces=[max(faces,key=lambda x:x.det_score)]
        face = faces[0]
        if face.det_score<0.8:
            print(f"[WARN] Low confidence {face.det_score:.2f}")
            continue

        embeddings.append(face.embedding.astype(np.float16).tolist())
        labels.append(person_name)
        np.save(EMB_PATH,np.array(embeddings,dtype=np.float16))
        with open(LABEL_PATH,"w") as f: json.dump(labels,f)
        print(f"[OK] Face enrolled: {person_name}")
        break

    elif key==ord('q'):
        print("[INFO] Enrollment cancelled")
        break

cap.stop()
cv2.destroyAllWindows()
print("[EXIT] Enrollment complete")
