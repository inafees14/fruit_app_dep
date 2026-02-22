Robust Image Classification for Edge AI
From Statistical Validation to Hardware-Constrained Deployment










---

Overview

This project presents a production-oriented image classification system engineered for resource-constrained edge environments.

Rather than optimizing purely for notebook accuracy, this system was designed around:

Statistical reliability across seeds

Low-latency deterministic inference

Minimal runtime footprint

Deployment stability under memory limits

Confidence-aware prediction behavior


The final production stack runs a fine-tuned MobileNetV2 model converted to TensorFlow Lite, served through FastAPI, and deployed on Heroku using a lightweight runtime environment.


---

Problem Framing

Academic pipelines typically optimize validation accuracy.

Edge systems require optimization across:

Inference latency

Memory ceiling constraints

Binary size

Cold-start behavior

Confidence calibration


This project explicitly models the accuracy–latency–footprint trade-off.

Dataset size: ~31,000 curated images across 11 fruit classes.


---

Model Benchmarking

Three architectural approaches were evaluated under identical training protocols.

Model	Fine-Tuned Accuracy	TFLite Latency	Deployment Suitability

Custom CNN	74%	—	High variance
EfficientNetB0	95%	40 ms	Moderate
MobileNetV2	94%	19 ms	Selected


Architectural Decision

EfficientNetB0 achieved the highest accuracy (95%).
However, MobileNetV2 achieved comparable accuracy (94%) while delivering more than 2× faster inference.

The 1% drop in accuracy was deliberately traded for:

Lower memory usage

Reduced latency

Improved edge stability

Smaller deployment footprint



---

Edge Hardware Validation

The selected MobileNetV2 model was deployed and stress-tested on a headless Raspberry Pi 3 running Linux.

Observed Results

Stable average inference latency: ~24 ms

Minimal prediction jitter

Sustained physical accuracy: 84–86%

Verified via confusion matrices and per-class precision/recall


Hardware-specific quantization significantly improved runtime determinism.


---

Production Engineering

1. Dependency-Free Inference Stack

Training and deployment environments were strictly separated.

Training Environment

Full TensorFlow stack

Data augmentation

Multi-seed experiments


Deployment Environment

FP16-quantized .tflite model

tflite-runtime (<10MB)

FastAPI + Uvicorn

No TensorFlow dependency


This prevented:

Heroku memory crashes

Cold-start lag

Large container builds



---

2. Confidence-Aware Guardrails

A calibrated probability threshold (70%) was implemented.

Predictions below threshold → flagged as unknown

Prevents forced misclassification

Improves real-world reliability



---

3. Telemetry & Logging

Integrated:

PostgreSQL for prediction logging

Cloudinary for image storage

Structured metadata (confidence, timestamp, label)


This enables:

Misclassification analysis

Drift inspection

Future active learning integration



---

4. Repository Optimization

The repository originally suffered from Git history bloat (1.1GB) due to model binaries.

Mitigation steps:

History rewriting

Strict .gitignore policies

Binary separation from training repo


Final production repository size: ~15MB.

This ensures clean CI/CD and stable deployments.


---

Explainability

Input-Gradient Class Activation Mapping (CAM) was used to validate learned representations.

Findings:

MobileNetV2 focused on fruit texture and structural regions

Baseline CNN showed background leakage

Explainability analysis supported architectural selection



---

Application Features

11 fruit classes

Confidence-based unknown detection

RESTful API architecture

Responsive frontend (HTML5 / CSS3 / JavaScript)

Edge-ready inference engine

Production telemetry logging



---

Local Development

Clone the repository:

git clone <your-repo-url>
cd fruit-classification-app

Install dependencies:

pip install -r requirements.txt

Run the server:

uvicorn main:app --reload

Access:

http://127.0.0.1:8000


---

Version History

Version	Date	Key Engineering Milestone

v38	2026-01-03	Finalized edge-optimized deployment pipeline
v29	2025-10-10	Integrated Cloudinary production image storage
v24	2025-10-07	Provisioned PostgreSQL telemetry logging
v13	2025-10-07	Implemented calibrated unknown threshold
v1	2025-10-01	Initial inference deployment



---

Future Roadmap

Quantization-Aware Training (QAT)

Real-time video inference

Automated active learning loop

Benchmarking across ARM hardware variants



---

License

MIT License.


---

If you want, I can now:

Make a short elite recruiter version (very sharp, 60% shorter)

Or make a research-lab formal version (more mathematical framing)

Or help you align this with your M.Sc Data Science positioning**


Just tell me which audience this README is targeting.