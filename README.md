# Real-Time Automated Cattle Counting System

**Author:** Wilfred Ayine Asumboya  
**Supervisor:** Dr. Cletus Fiifi Adams  
**Affiliation:** Department of Biomedical Engineering, School of Engineering Sciences  

This repository contains the complete implementation of a real-time cattle counting system that detects, tracks, and counts cattle as they exit through a ranch gate using deep learning. The system employs YOLOv8 for object detection, ByteTrack for multi-object tracking, and a custom directional counting algorithm with 3-layer anti-double-count protection.

The detection model is trained using a **brute-force data strategy** -- aggregating every publicly available cattle detection dataset into a single unified training corpus to guarantee generalization across any deployment environment without site-specific fine-tuning.

---

## Key Features

1. **Real-Time Detection**: Fine-tuned YOLOv8 models (YOLOv8m, YOLOv8l, YOLOv8x) evaluated for single-class cattle detection at 30+ FPS. 2. **Persistent Tracking**: ByteTrack assigns unique IDs to each animal across frames, handling occlusion and re-entry. 3. **3-Layer Anti-Double-Count System**: -ID Lock: Counted animals are permanently locked in memory -- never counted again. -Direction Gate: Only exit-direction crossings register as valid counts. -Track Buffer: IDs persist for ~2 seconds after temporary disappearance, preventing step-back double-counts. 4. **Brute-Force Mega Dataset**: Aggregates 10,000--22,000+ cattle images from Roboflow, Kaggle, COCO, Open Images, Mendeley, Zinodo, and LILA. 5. **Comparative Model Evaluation**: Trains YOLOv8m, YOLOv8l, and YOLOv8x under identical conditions, selecting the best for downstream counting accuracy. 6. **Rich Console Telemetry**: Publication-grade terminal output with hardware cards, progress bars, and diagnostic tables.
