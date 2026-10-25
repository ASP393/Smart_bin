"""
Day 11: Train YOLOv8n Object Detection Model for Smart Dustbin
Usage:
    python train_yolo.py
"""

import os
import shutil
import time
import numpy as np
import torch
from ultralytics import YOLO

import os
import shutil
import time
import numpy as np
import torch
from ultralytics import YOLO


def main():
    # Detect hardware cleanly
    has_cuda = torch.cuda.is_available()
    device = 0 if has_cuda else "cpu"
    print(f"🚀 Training Device: {'GPU (CUDA)' if has_cuda else 'CPU'}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_yaml = os.path.join(base_dir, "dataset", "data.yaml")
    train_img_dir = os.path.join(base_dir, "dataset", "images", "train")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # 1. Check dataset availability
    if not os.path.exists(train_img_dir):
        print("Error: Dataset directory not found!")
        return

    num_images = len([f for f in os.listdir(train_img_dir) if f.endswith(".jpg")])
    print(f"=== Starting YOLOv8n Training ===")
    print(f"Training images found: {num_images}")
    if num_images < 20:
        print("Warning: Dataset has fewer than 20 images. Run Webots with MODE='COLLECT' first to collect frames!")

    # 2. Load Pretrained YOLOv8 Nano
    print("Loading YOLOv8n base model...")
    model = YOLO("yolov8n.pt")

    # 3. Train on dataset
    # imgsz=320 matches our camera width (320x240)
    print("Training on custom trash dataset (25 epochs)...")
    results = model.train(
        data=dataset_yaml,
        epochs=25,
        imgsz=320,
        batch=16,
        workers=0,  # Single-process data loading to prevent Windows multiprocessing freezes
        device=device,
        project=os.path.join(base_dir, "runs"),
        name="trash_yolov8n",
        exist_ok=True,
        verbose=True
    )

    # 4. Save best weights to models/best.pt
    best_weights = os.path.join(base_dir, "runs", "trash_yolov8n", "weights", "best.pt")
    target_weights = os.path.join(models_dir, "best.pt")

    if os.path.exists(best_weights):
        shutil.copy(best_weights, target_weights)
        print(f"\nSuccessfully saved trained model to: {target_weights}")
    else:
        print(f"Warning: {best_weights} not found. Check training runs directory.")

    # 5. Measure Average Inference Latency
    print("\nBenchmarking Model Latency...")
    eval_model = YOLO(target_weights if os.path.exists(target_weights) else "yolov8n.pt")
    dummy_frame = np.zeros((240, 320, 3), dtype=np.uint8)

    latencies = []
    for _ in range(30):
        t0 = time.perf_counter()
        eval_model.predict(dummy_frame, device=device, verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_latency = np.mean(latencies[5:])  # ignore first 5 warmup iterations
    print(f"Average Inference Speed: {avg_latency:.2f} ms/frame ({1000.0/avg_latency:.1f} FPS)")
    print("\n=== Day 11 YOLO Training Complete! ===")
    print("To use YOLO in simulation, set MODE = 'YOLO' in bin_controller.py.")


if __name__ == '__main__':
    main()
