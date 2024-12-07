+++
title = "YOLO11 with CUDA and TensorRT"
date = "2024-12-01"
github = "https://github.com/RTae/yolov11-cuda"
+++

This project provides a high-performance implementation of YOLOv11 object detection using TensorRT for inference acceleration. The pipeline processes images and videos in batches, leveraging CUDA for preprocessing and inference. Non-Maximum Suppression (NMS) and postprocessing are performed on the CPU to optimize results.

<!--more-->
[GitHub Repository](https://github.com/RTae/yolov11-cuda)

# Feature

- CUDA-accelerated preprocessing and inference for faster performance.
- Batch processing for images and videos, supporting multiple inputs simultaneously.
- Postprocessing with NMS to refine detections.
- Threaded Execution to handle multiple input streams concurrently, with each stream processed on separate CUDA streams for maximum GPU utilization.
- Scalable pipeline: Process multiple files (images or videos) in parallel.
- Outputs detections with bounding boxes, class labels, and confidence scores.

# Report

| **Configuration**                   | **Inference Time (ms)** | **Preprocessing Time (ms)** | **Total Latency (ms)** | **FPS** |
|-------------------------------------|--------------------------|-----------------------------|-------------------------|---------|
| Baseline (CUDA Only)                | 80                      | -                           | 80                     | 12.5    |
| TensorRT + CUDA Streams (CPU Preprocessing) | 30               | Depends on CPU (~0-10)      | 30                     | 33.3    |
| TensorRT + CUDA Streams (CUDA Preprocessing) | 20             | Optimal                     | 20                     | 50      |