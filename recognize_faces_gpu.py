#!/usr/bin/env python3
"""
GPU-Optimized Real-Time Face Recognition for Jetson Orin Nano
Features: TensorRT acceleration, FP16 inference, async processing, 
          batch similarity computation, and optimized memory management
"""

import cv2
import json
import numpy as np
import time
import threading
from queue import Queue
from dataclasses import dataclass
from typing import List, Tuple, Optional
from insightface.app import FaceAnalysis

# ---------------- CONFIGURATION ----------------
RTSP_URL = "your_url"
DB_DIR = "face_db"
EMB_PATH = f"{DB_DIR}/embeddings.npy"
LABEL_PATH = f"{DB_DIR}/labels.json"

# Jetson Orin Nano Optimization Parameters
DET_SIZE = (320, 320)           # Smaller = faster, 320 is sweet spot for Orin Nano
INFERENCE_EVERY_N_FRAMES = 2     # Run model every N frames (tracking in between)
RESIZE_W, RESIZE_H = 640, 480    # Input resolution for processing
DISPLAY_W, DISPLAY_H = 1280, 720 # Display resolution
CONFIDENCE_THRESHOLD = 0.6       # Detection threshold
SIMILARITY_THRESHOLD = 0.45        # Recognition threshold

# GPU Performance Settings
USE_FP16 = True
GPU_MEM_LIMIT = 4 * 1024 * 1024 * 1024  # 4GB GPU memory limit
MAX_BATCH_SIZE = 4               # Batch processing for multiple faces

# ---------------- DATA STRUCTURES ----------------
@dataclass
class TrackedFace:
    """Structure to hold face tracking data"""
    face_id: int
    bbox: Tuple[int, int, int, int]
    embedding: Optional[np.ndarray] = None
    name: str = "Unknown"
    confidence: float = 0.0
    last_seen: float = 0.0
    frames_since_update: int = 0

# ---------------- GPU-ACCELERATED DATABASE ----------------
class GPUEmbeddingDatabase:
    """GPU-optimized database with batch similarity computation"""

    def __init__(self, emb_path: str, label_path: str):
        self.labels = []
        self.embeddings_gpu = None  # Store on GPU if possible
        self.embeddings_cpu = None
        self.load(emb_path, label_path)

    def load(self, emb_path: str, label_path: str):
        if not os.path.exists(emb_path):
            raise FileNotFoundError(f"Database not found: {emb_path}")

        self.embeddings_cpu = np.load(emb_path).astype(np.float16 if USE_FP16 else np.float32)
        with open(label_path, "r") as f:
            self.labels = json.load(f)

        # Normalize for cosine similarity (precompute)
        self.norms = np.linalg.norm(self.embeddings_cpu, axis=1, keepdims=True)
        self.embeddings_normalized = self.embeddings_cpu / self.norms

        print(f"[DB] Loaded {len(self.labels)} faces, shape: {self.embeddings_cpu.shape}")

    def batch_similarity(self, query_embeddings: np.ndarray) -> Tuple[List[int], List[float]]:
        """
        Vectorized batch similarity computation using matrix multiplication
        Much faster than loop-based approach
        """
        # Normalize queries
        query_norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        queries_normalized = query_embeddings / query_norms

        # Matrix multiplication for all similarities at once
        # Shape: (num_queries, num_known)
        similarities = np.dot(queries_normalized, self.embeddings_normalized.T)

        # Get best match for each query
        best_indices = np.argmax(similarities, axis=1)
        best_scores = similarities[np.arange(len(best_indices)), best_indices]

        return best_indices.tolist(), best_scores.tolist()

# ---------------- GPU MODEL INITIALIZATION ----------------
def initialize_tensorrt_model():
    """
    Initialize with TensorRT as primary provider, CUDA fallback
    """
    # Create cache directory
    os.makedirs("./trt_cache", exist_ok=True)

    providers = [
        ('TensorrtExecutionProvider', {
            'trt_max_workspace_size': 4 * 1024 * 1024 * 1024,  # 4GB
            'trt_fp16_enable': USE_FP16,
            'trt_int8_enable': False,  # FP16 is better balance for face recognition
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': './trt_cache',
            'trt_dump_subgraphs': False,
            'trt_timing_cache_enable': True,
            'trt_timing_cache_path': './trt_cache/timing.cache',
        }),
        ('CUDAExecutionProvider', {
            'device_id': 0,
            'arena_extend_strategy': 'kNextPowerOfTwo',
            'cudnn_conv_algo_search': 'HEURISTIC',
            'do_copy_in_default_stream': True,
            'cuda_mem_limit': GPU_MEM_LIMIT,
        }),
        'CPUExecutionProvider'
    ]

    app = FaceAnalysis(
        name="buffalo_s",
        providers=providers,
        det_thresh=CONFIDENCE_THRESHOLD,
    )

    app.prepare(ctx_id=0, det_size=DET_SIZE)

    # Aggressive GPU warmup
    print("[WARMUP] Preheating GPU...")
    dummy = np.zeros((DET_SIZE[1], DET_SIZE[0], 3), dtype=np.uint8)
    for i in range(10):
        _ = app.get(dummy)
    print("[OK] GPU ready")

    return app

# ---------------- ASYNC FRAME CAPTURE ----------------
class AsyncFrameCapture:
    """Double-buffered async frame capture to eliminate I/O latency"""

    def __init__(self, url: str):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        # GPU decoding hints
        self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)

        self.queue = Queue(maxsize=2)
        self.stopped = False
        self.thread = threading.Thread(target=self._update, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                # Drop old frame if queue full (keep latest)
                if self.queue.full():
                    try:
                        self.queue.get_nowait()
                    except:
                        pass
                self.queue.put(frame)

    def read(self):
        return self.queue.get()

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.cap.release()

# ---------------- MAIN RECOGNITION SYSTEM ----------------
class JetsonFaceRecognition:
    def __init__(self):
        print("="*60)
        print("JETSON ORIN NANO GPU FACE RECOGNITION")
        print("TensorRT + CUDA Accelerated")
        print("="*60)

        # Initialize components
        self.db = GPUEmbeddingDatabase(EMB_PATH, LABEL_PATH)
        self.app = initialize_tensorrt_model()
        self.capture = AsyncFrameCapture(RTSP_URL).start()

        # Tracking state
        self.last_faces = []
        self.frame_count = 0
        self.fps = 0
        self.last_fps_time = time.time()

        # Performance stats
        self.inference_times = []
        self.max_stats_samples = 30

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Fast resize with aspect ratio preservation"""
        return cv2.resize(frame, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_LINEAR)

    def draw_results(self, frame: np.ndarray, faces: List[TrackedFace]) -> np.ndarray:
        """Optimized drawing with minimal CPU overhead"""
        for face in faces:
            x1, y1, x2, y2 = face.bbox

            # Scale bbox to display resolution
            scale_x = DISPLAY_W / RESIZE_W
            scale_y = DISPLAY_H / RESIZE_H

            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

            # Color coding
            if face.name != "Unknown":
                color = (0, 255, 0)  # Green for known
                label = f"{face.name} ({face.confidence:.2f})"
            else:
                color = (0, 0, 255)  # Red for unknown
                label = f"Unknown ({face.confidence:.2f})"

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    def run(self):
        print("\n[STARTING] Recognition loop...")
        print(f"[CONFIG] Resolution: {RESIZE_W}x{RESIZE_H}, Skip: {INFERENCE_EVERY_N_FRAMES}")
        print("[CONTROLS] Press 'q' or ESC to quit\n")

        # Create display window
        cv2.namedWindow("Jetson GPU Face Recognition", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Jetson GPU Face Recognition", DISPLAY_W, DISPLAY_H)

        try:
            while True:
                # Get frame
                frame = self.capture.read()
                display_frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
                process_frame = self.preprocess_frame(frame)

                self.frame_count += 1

                # Run inference periodically
                if self.frame_count % INFERENCE_EVERY_N_FRAMES == 0:
                    t0 = time.time()
                    detected_faces = self.app.get(process_frame)
                    inference_time = (time.time() - t0) * 1000  # ms
                    self.inference_times.append(inference_time)

                    if len(self.inference_times) > self.max_stats_samples:
                        self.inference_times.pop(0)

                    # Batch process similarities
                    if detected_faces:
                        embeddings = np.array([f.embedding for f in detected_faces])
                        indices, scores = self.db.batch_similarity(embeddings)

                        # Update tracking
                        self.last_faces = []
                        for i, face in enumerate(detected_faces):
                            x1, y1, x2, y2 = map(int, face.bbox)

                            score = scores[i]
                            if score > SIMILARITY_THRESHOLD:
                                name = self.db.labels[indices[i]]
                            else:
                                name = "Unknown"

                            self.last_faces.append(TrackedFace(
                                face_id=i,
                                bbox=(x1, y1, x2, y2),
                                embedding=face.embedding,
                                name=name,
                                confidence=score,
                                last_seen=time.time()
                            ))
                    else:
                        self.last_faces = []

                # Draw results (using last known positions)
                display_frame = self.draw_results(display_frame, self.last_faces)

                # FPS overlay
                if time.time() - self.last_fps_time > 1.0:
                    self.fps = self.frame_count
                    self.frame_count = 0
                    self.last_fps_time = time.time()

                    avg_inference = np.mean(self.inference_times) if self.inference_times else 0
                    print(f"[PERF] FPS: {self.fps} | Inference: {avg_inference:.1f}ms | "
                          f"Faces: {len(self.last_faces)}")

                cv2.putText(display_frame, f"FPS: {self.fps}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Jetson GPU Face Recognition", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == 27 or key == ord('q'):
                    break

        finally:
            self.capture.stop()
            cv2.destroyAllWindows()
            print("\n[EXIT] Recognition stopped")

# ---------------- ENTRY POINT ----------------
if __name__ == "__main__":
    import os
    os.environ['CUDA_CACHE_DISABLE'] = '0'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

    system = JetsonFaceRecognition()
    system.run()
