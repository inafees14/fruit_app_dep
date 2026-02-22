🍎 Robust Image Classification for Edge AI

From Statistical Validation to Hardware-Constrained Deployment










---

1. Overview

This repository presents a production-grade image classification system engineered for resource-constrained edge environments.

The objective was not to maximize notebook accuracy, but to design a system that:

Maintains statistical reliability across seeds

Achieves low-latency inference on constrained hardware

Uses a minimal runtime footprint

Survives real-world deployment constraints (memory limits, cold starts, CI/CD constraints)


The final deployed system runs a fine-tuned MobileNetV2 model through a pure TensorFlow Lite runtime stack, served via FastAPI and deployed on Heroku.


---

2. Problem Framing

Standard academic pipelines optimize for validation accuracy.
Edge systems optimize for:

Deterministic latency

Memory ceiling constraints

Binary size

Runtime stability

Predictive confidence calibration


This project explicitly models the accuracy–latency–footprint trade-off.

Dataset scale:
~31,000 curated images across 11 fruit classes.


---

3. Architecture Selection: Measured Trade-offs

Three architectures were evaluated under identical training conditions.

Model	Fine-Tuned Accuracy	TFLite Latency	Model Size	Edge Suitability

Custom CNN	74%	—	Small	High variance
EfficientNetB0	95%	40 ms	Heavy	Moderate
MobileNetV2	94%	19 ms	Compact	Selected


Decision Rationale

Although EfficientNetB0 achieved marginally higher accuracy, MobileNetV2:

Delivered >2× faster inference

Reduced memory overhead

Maintained strong macro-F1 stability

Showed superior generalization consistency across seeds


A 1% accuracy sacrifice yielded a significant improvement in deployment viability.


---

4. Edge Validation: Physical Device Testing

All benchmarking was repeated on a physical Raspberry Pi 3 Model B running headless Linux.

Observed Results

Stable inference latency: ~24 ms

Minimal frame-to-frame jitter

Sustained accuracy: 84–86% across physical test cycles

Per-class precision/recall validated via confusion matrices


Quantization and runtime isolation ensured deterministic behavior under constrained CPU conditions.


---

5. Production Stack Design

5.1 Dependency-Free Inference

Training and deployment environments were strictly separated.

Training Stack

Full TensorFlow

Data augmentation

Seed-controlled experiments


Deployment Stack

.h5 → FP16-quantized .tflite

tflite-runtime (<10MB)

No TensorFlow dependency

Uvicorn + FastAPI


This eliminated:

Heroku memory crashes

Cold-start latency spikes

Excess container size



---

5.2 Confidence-Aware Guardrails

Implemented calibrated probability thresholding:

< 70% confidence → flagged as unknown

Prevents forced misclassification

Improves real-world trustworthiness


This transforms the system from a naive classifier into a confidence-aware inference service.


---

5.3 Telemetry & Monitoring

Integrated:

PostgreSQL for prediction logging

Cloudinary for image storage

Structured metadata capture (confidence, timestamp, label)


Enables:

Misclassification analysis

Drift inspection

Active learning loop preparation



---

5.4 Repository Engineering

Resolved severe Git history bloat (1.1GB) caused by large model binaries.

Actions:

Aggressive .gitignore

History rewriting

Binary decoupling from training repo


Final production repository footprint: ~15MB

This ensures:

Clean CI/CD

Fast cloning

Stable deployment cycles



---

6. Explainability Audit

Used Input-Gradient CAM for architectural comparison.

Findings:

MobileNetV2 focused on fruit texture and geometry

Baseline CNN exhibited background leakage

Explainability supported deployment decision


The selected architecture demonstrated biologically meaningful feature extraction.


---

7. System Capabilities

11 fruit classes

Confidence-aware unknown detection

REST API architecture

Responsive minimal frontend

Edge-ready inference engine

Production telemetry logging



---

8. Local Execution

git clone <repo-url>
cd fruit-classification-app
pip install -r requirements.txt
uvicorn main:app --reload

Access:

http://127.0.0.1:8000


---

9. Version Milestones

Version	Date	Engineering Milestone

v38	2026-01-03	Finalized edge-optimized deployment pipeline
v29	2025-10-10	Integrated Cloudinary production image storage
v24	2025-10-07	Provisioned PostgreSQL telemetry logging
v13	2025-10-07	Implemented calibrated unknown-image threshold
v1	2025-10-01	Initial inference deployment



---

10. Future Directions

Quantization-Aware Training (QAT)

Real-time video stream inference

Automated active learning ingestion

Edge hardware benchmarking across ARM variants



---

11. Project Positioning

This is not a template-based classifier demo.

It is a controlled study in deployment-aware model design, integrating:

Statistical rigor

Edge hardware validation

Runtime minimization

MLOps discipline


It demonstrates the full pipeline:

> Research → Calibration → Quantization → Hardware Validation → Production Deployment




---

If you'd like, I can now:

🔹 Refine it further for a research-lab audience (more formal tone)

🔹 Optimize it for recruiters (impact-focused)

🔹 Or compress it into a one-page high-density README


Tell me the target audience.