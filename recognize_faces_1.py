#!/usr/bin/env python3
"""
Optimized Real-Time Face Recognition for Jetson Orin Nano
Features:
- TensorRT (ONNX engine) acceleration
- FP16 embeddings
- Async multi-threaded frame capture + inference
- Batch similarity computation
- Minimal CPU overhead
"""

import os
import cv2
import json
import time
import threading
import numpy as np
from queue import Queue
from dataclasses import dataclass
from typing import List, Tuple, Optional
from insightface.app import FaceAnalysis

# ---------------- CONFIG ----------------
RTSP_URL = "your_url"
DB_DIR = "face_db"
EMB_PATH = os.path.join(DB_DIR, "embeddings.npy")
LABEL_PATH = os.path.join(DB_DIR, "labels.json")

DET_SIZE = (320, 320)         # smaller for faster detection
RESIZE_W, RESIZE_H = 640, 480
DISPLAY_W, DISPLAY_H = 1280, 720
INFERENCE_EVERY_N_FRAMES = 2
CONFIDENCE_THRESHOLD = 0.6
SIMILARITY_THRESHOLD = 0.45
USE_FP16 = True
MAX_BATCH_SIZE = 4

os.environ['CUDA_CACHE_DISABLE'] = '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.makedirs("./trt_cache", exist_ok=True)

# ---------------- DATA STRUCTURES ----------------
@dataclass
class TrackedFace:
    face_id: int
    bbox: Tuple[int, int, int, int]
    embedding: Optional[np.ndarray] = None
    name: str = "Unknown"
    confidence: float = 0.0
    last_seen: float = 0.0

# ---------------- DATABASE ----------------
class GPUEmbeddingDatabase:
    def __init__(self, emb_path, label_path):
        self.labels = []
        self.embeddings = None
        self.load(emb_path, label_path)

    def load(self, emb_path, label_path):
        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"Database not found: {emb_path}")
        self.embeddings = np.load(emb_path).astype(np.float16 if USE_FP16 else np.float32)
        with open(label_path, "r") as f:
            self.labels = json.load(f)
        self.norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings_normalized = self.embeddings / self.norms
        print(f"[DB] Loaded {len(self.labels)} faces.")

    def batch_similarity(self, query_embeddings: np.ndarray):
        query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        queries_normalized = query_embeddings / query_norms
        sims = np.dot(queries_normalized, self.embeddings_normalized.T)
        best_idx = np.argmax(sims, axis=1)
        best_scores = sims[np.arange(len(best_idx)), best_idx]
        return best_idx.tolist(), best_scores.tolist()

# ---------------- MODEL ----------------
def initialize_model():
    providers = [
        ('TensorrtExecutionProvider', {
            'trt_fp16_enable': USE_FP16,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': './trt_cache',
        }),
        'CUDAExecutionProvider',
        'CPUExecutionProvider'
    ]
    app = FaceAnalysis(name="buffalo_s", providers=providers, det_thresh=CONFIDENCE_THRESHOLD)
    app.prepare(ctx_id=0, det_size=DET_SIZE)

    # Warmup
    dummy = np.zeros((DET_SIZE[1], DET_SIZE[0], 3), dtype=np.uint8)
    for _ in range(3):
        _ = app.get(dummy)
    print("[MODEL] GPU warmed up")
    return app

# ---------------- ASYNC FRAME CAPTURE ----------------
class AsyncCapture:
    def __init__(self, url):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.queue = Queue(maxsize=2)
        self.stopped = False
        self.thread = threading.Thread(target=self.update, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                if self.queue.full():
                    try: self.queue.get_nowait()
                    except: pass
                self.queue.put(frame)

    def read(self):
        return self.queue.get()

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()

# ---------------- FACE RECOGNITION ----------------
class JetsonFaceRecognition:
    def __init__(self):
        print("="*60)
        print("JETSON ORIN NANO - GPU FACE RECOGNITION")
        print("="*60)

        self.db = GPUEmbeddingDatabase(EMB_PATH, LABEL_PATH)
        self.app = initialize_model()
        self.capture = AsyncCapture(RTSP_URL).start()

        self.last_faces: List[TrackedFace] = []
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()
        self.inference_times = []

    def preprocess(self, frame):
        return cv2.resize(frame, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_LINEAR)

    def draw_faces(self, frame, faces: List[TrackedFace]):
        for face in faces:
            x1, y1, x2, y2 = face.bbox
            scale_x = DISPLAY_W / RESIZE_W
            scale_y = DISPLAY_H / RESIZE_H
            x1, x2 = int(x1*scale_x), int(x2*scale_x)
            y1, y2 = int(y1*scale_y), int(y2*scale_y)
            color = (0,255,0) if face.name!="Unknown" else (0,0,255)
            label = f"{face.name} ({face.confidence:.2f})"
            cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1-th-5),(x1+tw,y1), color,-1)
            cv2.putText(frame,label,(x1,y1-5),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        return frame

    def run(self):
        cv2.namedWindow("Jetson GPU Face Recognition", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Jetson GPU Face Recognition", DISPLAY_W, DISPLAY_H)

        try:
            while True:
                frame = self.capture.read()
                display_frame = cv2.resize(frame,(DISPLAY_W,DISPLAY_H))
                process_frame = self.preprocess(frame)
                self.frame_count += 1

                # Run inference
                if self.frame_count % INFERENCE_EVERY_N_FRAMES == 0:
                    t0 = time.time()
                    faces = self.app.get(process_frame)
                    self.inference_times.append((time.time()-t0)*1000)
                    if len(self.inference_times)>30: self.inference_times.pop(0)

                    if faces:
                        embeddings = np.array([f.embedding for f in faces], dtype=np.float16)
                        indices, scores = self.db.batch_similarity(embeddings)
                        self.last_faces=[]
                        for i, f in enumerate(faces):
                            name = self.db.labels[indices[i]] if scores[i]>SIMILARITY_THRESHOLD else "Unknown"
                            self.last_faces.append(TrackedFace(face_id=i, bbox=tuple(map(int,f.bbox)),
                                                               embedding=f.embedding,
                                                               name=name, confidence=scores[i],
                                                               last_seen=time.time()))
                    else:
                        self.last_faces=[]

                display_frame = self.draw_faces(display_frame, self.last_faces)

                # FPS
                if time.time() - self.last_fps_time > 1.0:
                    self.fps = self.frame_count
                    self.frame_count=0
                    self.last_fps_time=time.time()
                    avg_inf = np.mean(self.inference_times) if self.inference_times else 0
                    print(f"[PERF] FPS:{self.fps} | Inference:{avg_inf:.1f}ms | Faces:{len(self.last_faces)}")

                cv2.putText(display_frame,f"FPS: {self.fps}",(10,30),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
                cv2.imshow("Jetson GPU Face Recognition", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key==27 or key==ord('q'):
                    break

        finally:
            self.capture.stop()
            cv2.destroyAllWindows()
            print("[EXIT] Stopped")

# ---------------- MAIN ----------------
if __name__=="__main__":
    system = JetsonFaceRecognition()
    system.run()
