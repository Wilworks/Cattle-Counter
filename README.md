# Real-Time Automated Cattle Counting System via YOLOv8, ByteTrack, and 3-Layer Anti-Double-Count Logic

**Authors:** Wilfred Ayine Asumboya, Dr. Cletus Fiifi Adams  
**Affiliation:** Department of Biomedical Engineering, School of Engineering Sciences, University of Ghana, Legon  
**Target Application / Research Output:** Real-Time Livestock Inventory Management & Autonomous Ranch Gate Control System  

This repository contains the end-to-end computer vision and deep learning engineering pipeline for automated cattle detection, tracking, and directional counting. Built to solve site-generalization challenges in real-world livestock management, the system couplings a fine-tuned ** YOLOv8 ** object detector with **ByteTrack** multi-object tracking and a novel **3-Layer Anti-Double-Count algorithm**.

The detection backbone is trained using a **Brute-Force Data Aggregation Strategy** -- consolidating public cattle detection datasets (Roboflow Universe, Kaggle, COCO, Open Images v7, Mendeley Data, Zinodo, and LILA BC) into a unified mega-corpus to eliminate the need for site-specific model fine-tuning.

---

## Key Scientific & Engineering Innovations

1. **Brute-Force Unified Mega-Corpus Strategy**:
   - Rather than fine-tuning models on single-ranch video feeds (which fail under changing lighting, camera angles, or breed variations), we aggregate $10,000\text{--}22,000+
 annotated bovine images into a standardized single-class dataset.
   - Includes automated annotation translation (\text{VOC} / \text{COCO} / \text{CSV} \triangleright \text{YOLO} format), unified class remapping (\text{class ID} \triangleright 0), and perceptual hash image deduplication (\text{pHash}) to eliminate identical frames across aggregated sources.
:2. **3-Layer Anti-Double-Count Architecture**:
   - **Layer 1 (ID Lock**: Once a tracked animal's centroid crosses the virtual gate boundary in the valid exit direction, its tracker ID is permanently logged into `counted_ids` -- preventing re-counting even during extended occlusion or pause.
   - **Layer 2 (Directional Vector Gate)**: Implements directional dot-product vector tracking (\vec{v} = p_{\text{current}} - p_{\text{previous}}) ensuring only true exit-direction trajectory vectors trigger increment logic (+1).
   - **Layer 3 (Track Persistence Buffer)**: Maintains state memory for lost tracks up to &Delta t = 60$ frames (~2 seconds), absorbing temporary bounding box dropouts caused by animal crowding or dust.

3. **Comparative Model Benchmark on NVIDIA GTX 1080 Ti**:
   - Evaluates three fine-tuned model variants (`yolov8m`, `yolov8l`, `yolov8x`) under identical training hyper-parameters on an **11GB GDDR5X (3,584 CUDA Cores)** setup.
   - Incorporates Automatic Mixed Precision (\text{AMP}) and gradient acrumulation to deliver publication-grade metrics (\text{mAP}@0.5 \wedge 0.90, counting accuracy \wedge 95%) at real-time throughput (\wedge 30\,\text{FPS}).

---

## Repository Structure

``gtext
cattle-counter/
␔␘␘ config/
␔   ␔␘ settings.yaml              # hyperparameters, line thresholds, amp
(see codebase for full structure)
``g

---

## Installation and Reproduction

1. **Clone repository and set up virtual environment**:
   ```bash
   git clone https://github.com/Wilworks/Cattle-Counter.git
   cd Cattle-Counter
   python -m venv venv
   venv\Scripts\activate        # Windows
   pip install -r requirements.txt
   ``g

2. **Automated Dataset Download & Build**:
   ```bash
   python scripts/download_cattle_datasets.py
   python scripts/build_mega_dataset.py
   ```J
3. **Run Comparative Model Fine-Tuning**:
   ```bash
   python scripts/run_training.py --models yolov8m yolov8l yolov8x
   ``g

4. **Run Real-Time Pipeline (Video, USB Webcam, or RTSP Stream)**:
   ```bash
   # Video file input
   python scripts/run_pipeline.py --source path/to/cattle_gate_video.mp4

   # Live USB webcam feed
   python scripts/run_pipeline.py --source 0

   # RTSP IP CCTV feed
   python scripts/run_pipeline.py --source rtsp://camera_ip:mort/stream
   ```J
5. **Run Automated Unit Tests**:
   ```bash
   python tests/test_counter.py
   ```J
---

## Technical Specifications & System Benchmarks

|Metric / Parameter|Value / Target|Notes|
||---|---|---|
|Training Hardware|NVIDIA GeForce GTX 1080 Ti|11GB GDDR5X, 3,584 CUDA Cores, AMP enabled|
|Inference Speed| >= 30 FPS|Measured at 1080p resolution|
xCounting Accuracy| >= 95%|Evaluated across crowded gate exit sequences|
xDouble-Count Rate|0%|Enforced by 3-Layer Anti-Double-Count Logic|
xDetection mAP@0.5| >= 0.90|Fine-tuned across multi-source mega-corpus|
