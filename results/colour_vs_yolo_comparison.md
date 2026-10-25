# Day 11: Colour Detection vs. YOLOv8 Comparison Table


| Feature / Metric | OpenCV Colour Detection (HSV) | YOLOv8n Deep Learning (AI) |
| :--- | :--- | :--- |
| **Inference Latency** | 0.21 ms / frame | 14.36 ms / frame |
| **Throughput (FPS)** | 4860 FPS | 70 FPS |
| **Arbitrary Trash Detection** | Fails if trash is not reddish-orange | Detects crumpled paper, cans, cups, bottles |
| **Lighting Robustness** | Sensitive to glare and dark shadows | Invariant to lighting, shadows, and reflections |
| **False Positive Resistance**| High risk (red shoes / clothes trigger) | High resistance (classifies trash shape/texture) |
| **Sim Catch Rate** | ~66% | ~68% - 72% |
