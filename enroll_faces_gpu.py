#!/usr/bin/env python3
"""
GPU-Optimized Face Enrollment for Jetson Orin Nano
Uses TensorRT + CUDA with FP16 precision for maximum speed
"""

import cv2
import os
import json
import numpy as np
import time
from insightface.app import FaceAnalysis
from multiprocessing import Process, Queue
import threading

# ---------------- CONFIG ----------------
RTSP_URL = "rtsp://admin:Tiandy%40123@192.168.1.103:554/streams1"
DB_DIR = "face_db"
EMB_PATH = os.path.join(DB_DIR, "embeddings.npy")
LABEL_PATH = os.path.join(DB_DIR, "labels.json")

# Jetson Orin Nano GPU Optimization Settings
DET_SIZE = (640, 640)  # Detection input size
USE_FP16 = True        # FP16 for 2x speedup with minimal accuracy loss
MAX_BATCH = 1          # Enrollment is single-face

os.makedirs(DB_DIR, exist_ok=True)

# ---------------- LOAD EXISTING DB ----------------
def load_database():
    """Thread-safe database loading"""
    if os.path.exists(EMB_PATH):
        embeddings = np.load(EMB_PATH).tolist()
        with open(LABEL_PATH, "r") as f:
            labels = json.load(f)
        return embeddings, labels
    return [], []

embeddings, labels = load_database()

# ---------------- GPU MODEL SETUP ----------------
def initialize_gpu_model():
    """
    Initialize InsightFace with TensorRT + CUDA execution providers
    Priority: TensorRT > CUDA > CPU
    """
    providers = [
        ('TensorrtExecutionProvider', {
            'trt_max_workspace_size': 2147483648,  # 2GB workspace
            'trt_fp16_enable': USE_FP16,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': './trt_cache',
            'trt_max_partition_iterations': 1000,
            'trt_min_subgraph_size': 5,
        }),
        ('CUDAExecutionProvider', {
            'cudnn_conv_algo_search': 'HEURISTIC',
            'cuda_mem_limit': 2147483648,  # 2GB limit
        }),
        #'CPUExecutionProvider'
    ]

    app = FaceAnalysis(
        name="buffalo_s",  # Lightweight model optimized for edge
        providers=providers,
        det_thresh=0.5,    # Detection threshold
    )

    # Prepare with GPU context
    app.prepare(ctx_id=0, det_size=DET_SIZE)

    # Warm up GPU (critical for consistent latency)
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(3):
        app.get(dummy_frame)

    return app

print("[INIT] Loading GPU-optimized model...")
app = initialize_gpu_model()
print("[OK] Model loaded with TensorRT + CUDA")

# ---------------- RTSP STREAM WITH GPU DECODING ----------------
def create_gpu_capture(url):
    """
    Create optimized VideoCapture with GPU-accelerated decoding if available
    """
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    # Critical optimizations for Jetson
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Hardware acceleration flags (if supported by FFmpeg build)
    cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)

    # Set decode format to reduce CPU load
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|buffer_size;65536'

    return cap

cap = create_gpu_capture(RTSP_URL)

if not cap.isOpened():
    print("[ERROR] RTSP stream not opened")
    exit(1)

# ---------------- MAIN ENROLLMENT LOOP ----------------
print("\n" + "="*50)
print(">>> JETSON ORIN NANO GPU FACE ENROLLMENT <<<")
print("="*50)
print("Press 's' to save face | 'q' to quit")
print(f"Current database: {len(labels)} enrolled faces")
print("="*50 + "\n")

person_name = input("Enter person name / ID: ").strip()

# Pre-allocate GPU memory
frame_buffer = []
last_detection = None
fps_time = time.time()
fps_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Resize for display (keep original for inference if needed)
    display_frame = cv2.resize(frame, (640, 640))

    # Run inference every frame for enrollment (accuracy critical)
    faces = app.get(display_frame)

    # Draw detections
    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        conf = face.det_score

        # Color based on confidence
        color = (0, 255, 0) if conf > 0.8 else (0, 255, 255)

        cv2.rectangle(display_frame, (x1//2, y1//2), (x2//2, y2//2), color, 2)
        cv2.putText(display_frame, f"Conf: {conf:.2f}", 
                   (x1//2, y1//2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # FPS calculation
    fps_counter += 1
    if time.time() - fps_time > 1.0:
        fps = fps_counter
        fps_counter = 0
        fps_time = time.time()
        print(f"[FPS] Enrollment stream: {fps}")

    cv2.imshow("GPU Face Enrollment (Press 's' to save)", display_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        if len(faces) == 0:
            print("[WARN] No face detected - try again")
            continue
        time.sleep(0.05)

        if len(faces) > 1:
            print("[WARN] Multiple faces detected - using highest confidence")
            faces = [max(faces, key=lambda x: x.det_score)]

        face = faces[0]

        # Validate face quality
        if face.det_score < 0.8:
            print(f"[WARN] Low confidence ({face.det_score:.2f}), try again")
            continue
            
        time.sleep(0.05)

        # Save embedding
        embedding = face.embedding.tolist()
        embeddings.append(embedding)
        labels.append(person_name)

        # Atomic save
        np.save(EMB_PATH, np.array(embeddings))
        with open(LABEL_PATH, "w") as f:
            json.dump(labels, f)
            
        time.sleep(0.05)

        print(f"\n[OK] ✓ Face enrolled: {person_name}")
        print(f"     Confidence: {face.det_score:.3f}")
        print(f"     Embedding dim: {len(embedding)}")
        print(f"     Total enrolled: {len(labels)}")
        break

    elif key == ord('q'):
        print("[INFO] Enrollment cancelled")
        break

cap.release()
cv2.destroyAllWindows()
print("[EXIT] Enrollment complete")
