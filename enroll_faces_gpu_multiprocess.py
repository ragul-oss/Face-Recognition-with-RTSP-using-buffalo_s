#!/usr/bin/env python3
"""
GPU-Optimized Face Enrollment for Jetson Orin Nano
Multiprocessing Version (Logic Unchanged)
"""

import cv2
import os
import json
import numpy as np
import time
from insightface.app import FaceAnalysis
from multiprocessing import Process, Queue

# ---------------- CONFIG ----------------
RTSP_URL = "rtsp://admin:Tiandy%40123@192.168.1.103:554/streams1"
DB_DIR = "face_db"
EMB_PATH = os.path.join(DB_DIR, "embeddings.npy")
LABEL_PATH = os.path.join(DB_DIR, "labels.json")

DET_SIZE = (640, 640)
USE_FP16 = True

os.makedirs(DB_DIR, exist_ok=True)

# ---------------- LOAD EXISTING DB ----------------
def load_database():
    if os.path.exists(EMB_PATH):
        embeddings = np.load(EMB_PATH).tolist()
        with open(LABEL_PATH, "r") as f:
            labels = json.load(f)
        return embeddings, labels
    return [], []

embeddings, labels = load_database()

# ---------------- GPU MODEL SETUP ----------------
def initialize_gpu_model():
    providers = [
        ('TensorrtExecutionProvider', {
            'trt_max_workspace_size': 2147483648,
            'trt_fp16_enable': USE_FP16,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': './trt_cache',
            'trt_max_partition_iterations': 1000,
            'trt_min_subgraph_size': 5,
        }),
        ('CUDAExecutionProvider', {
            'cudnn_conv_algo_search': 'HEURISTIC',
            'cuda_mem_limit': 2147483648,
        }),
    ]

    app = FaceAnalysis(
        name="buffalo_s",
        providers=providers,
        det_thresh=0.5,
    )

    app.prepare(ctx_id=0, det_size=DET_SIZE)

    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(3):
        app.get(dummy_frame)

    return app

# ---------------- RTSP PROCESS ----------------
def rtsp_frame_reader(url, queue):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)

    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp|buffer_size;65536'

    if not cap.isOpened():
        print("[ERROR] RTSP stream not opened")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        if not queue.full():
            queue.put(frame)

# ---------------- MAIN ----------------
if __name__ == "__main__":

    print("[INIT] Loading GPU-optimized model...")
    app = initialize_gpu_model()
    print("[OK] Model loaded with TensorRT + CUDA")

    frame_queue = Queue(maxsize=5)

    rtsp_process = Process(
        target=rtsp_frame_reader,
        args=(RTSP_URL, frame_queue),
        daemon=True
    )
    rtsp_process.start()

    print("\n" + "="*50)
    print(">>> JETSON ORIN NANO GPU FACE ENROLLMENT <<<")
    print("="*50)
    print("Press 's' to save face | 'q' to quit")
    print(f"Current database: {len(labels)} enrolled faces")
    print("="*50 + "\n")

    person_name = input("Enter person name / ID: ").strip()

    fps_time = time.time()
    fps_counter = 0

    while True:

        if frame_queue.empty():
            continue

        frame = frame_queue.get()

        display_frame = cv2.resize(frame, (640, 640))

        faces = app.get(display_frame)

        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            conf = face.det_score

            color = (0, 255, 0) if conf > 0.8 else (0, 255, 255)

            cv2.rectangle(display_frame, (x1//2, y1//2),
                          (x2//2, y2//2), color, 2)

            cv2.putText(display_frame,
                        f"Conf: {conf:.2f}",
                        (x1//2, y1//2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2)

        fps_counter += 1
        if time.time() - fps_time > 1.0:
            print(f"[FPS] Enrollment stream: {fps_counter}")
            fps_counter = 0
            fps_time = time.time()

        cv2.imshow("GPU Face Enrollment (Press 's' to save)", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):

            if len(faces) == 0:
                print("[WARN] No face detected - try again")
                continue

            if len(faces) > 1:
                print("[WARN] Multiple faces detected - using highest confidence")
                faces = [max(faces, key=lambda x: x.det_score)]

            face = faces[0]

            if face.det_score < 0.8:
                print(f"[WARN] Low confidence ({face.det_score:.2f}), try again")
                continue

            embedding = face.embedding.tolist()
            embeddings.append(embedding)
            labels.append(person_name)

            np.save(EMB_PATH, np.array(embeddings))
            with open(LABEL_PATH, "w") as f:
                json.dump(labels, f)

            print(f"\n[OK] ✓ Face enrolled: {person_name}")
            print(f"     Confidence: {face.det_score:.3f}")
            print(f"     Embedding dim: {len(embedding)}")
            print(f"     Total enrolled: {len(labels)}")
            break

        elif key == ord('q'):
            print("[INFO] Enrollment cancelled")
            break

    cv2.destroyAllWindows()
    print("[EXIT] Enrollment complete")
 
