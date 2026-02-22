# Robust Image Classification for Edge AI  
### Research → Edge Validation → Production

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95-blue?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![TensorFlow Lite](https://img.shields.io/badge/TensorFlow%20Lite-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/lite)
[![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-3B-C51A4A?logo=raspberry-pi&logoColor=white)](https://www.raspberrypi.com/)
[![Linux](https://img.shields.io/badge/Linux-Ready-FCC624?logo=linux&logoColor=black)](https://www.linux.org/)
[![Heroku](https://img.shields.io/badge/Heroku-Deployed-purple?logo=heroku&logoColor=white)](https://www.heroku.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This project is a deployment-aware fruit classification system engineered for real-world edge and cloud-constrained environments.

Unlike template-based classifiers, this system emphasizes:

- Multi-seed statistical validation  
- Architecture benchmarking  
- Hardware-level inference validation  
- Quantized TFLite deployment  
- Confidence-calibrated predictions  
- Repository and CI/CD integrity  

Dataset size: ~31,000 curated images across 11 fruit classes.

The final production model is a fine-tuned **MobileNetV2**, converted to **FP16 TensorFlow Lite**, served via **FastAPI**, and deployed on **Heroku** with a lean runtime stack.

---

# Experimental Pipeline & Research

## Model Benchmarking

| Model | Input Dim | Strategy | Strength | Role |
|--------|-----------|-----------|-----------|-----------|
| Custom CNN | 128x128 | From Scratch | Ultra-lightweight | CPU Baseline |
| MobileNetV2 | 224x224 | Fine-tuned | High Accuracy / Speed | Deployment Candidate |
| EfficientNetB0 | 224x224 | Fine-tuned | Feature Richness | Benchmark |

---

## Statistical Performance (Test Set)

Evaluated on ~31,000 images across 11 classes using **Macro-F1**.

| Model | Accuracy (Mean) | Macro-F1 | Inference Stability |
|--------|----------------|-----------|--------------------|
| Custom CNN | 74.2% | 0.734 | High Variance |
| MobileNetV2 | 89.1% | 0.885 | High Stability |
| EfficientNetB0 | 85.3% | 0.849 | Moderate (Heavy) |

**Insight:**  
MobileNetV2 was selected because it significantly outperformed the CNN (p < 0.05) while maintaining a lower memory footprint than EfficientNet.

---

## Explainability & Robustness

Input-Gradient Class Activation Mapping (CAM) was used to validate model focus.

Findings:

- MobileNetV2 focused on fruit texture and body  
- Custom CNN showed background bias  
- Explainability supported deployment decision  

---

# Edge Hardware Validation

The selected MobileNetV2 model was stress-tested on a headless Raspberry Pi 3 (Linux, CPU-only).

**Results:**

- Stable average inference latency: ~24 ms  
- Minimal prediction jitter  
- Sustained physical accuracy: ~84–86%  
- Verified via confusion matrices and per-class precision/recall  

Quantization significantly improved runtime determinism.

---

# Production Engineering

## The “No-TensorFlow” Inference Stack

To prevent Heroku memory crashes and cold-start lag:

- Model Format: `.h5` → `.tflite` (FP16 Quantization)  
- Runtime: `tensorflow` (400MB+) replaced with `tflite-runtime` (<10MB)  
- Backend: FastAPI + Uvicorn  
- Strict training/deployment separation  

---

## Confidence-Aware Threshold

A calibrated **70% probability threshold** was implemented.

Predictions below threshold are flagged as **Unknown**, preventing forced misclassification.

---

## Telemetry & Logging

Integrated:

- PostgreSQL for structured prediction logs  
- Cloudinary for image hosting  
- Confidence score tracking  

Enables:

- Misclassification pattern analysis  
- Drift monitoring  
- Future active learning  

---

## Repository Optimization

The repository previously reached 1.1GB due to historical model binaries.

Actions:

- Git history rewrite (`git filter-repo`)  
- Strict `.gitignore`  
- Training/deployment repo separation  

Final production repository size: ~15MB.

---

# Application Features

- 11 Fruit Classes  
- Confidence-based Unknown Detection  
- RESTful API  
- Responsive Frontend (HTML5 / CSS3 / JS)  
- Edge-Compatible Runtime  
- Production Telemetry Logging  

---

# Live Demo

 Launch Fruit Classifier App  
https://fruit-classification-1-7c2a30615392.herokuapp.com/

---

# Local Development Setup

Clone the repository:

```bash
git clone https://github.com/inafees14/Fruits_classification
cd fruit_app_dep
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the server:

```bash
uvicorn main:app --reload
```

Open your browser and navigate to:

```
http://127.0.0.1:8000
```

---

# 📂 Project Structure

```bash
├── research-repo/
│   ├── notebooks/
│   ├── statistical_analysis/
│   └── xai/
│
└── deployment-repo/
    ├── main.py
    ├── model/
    │   └── model.tflite
    ├── class_names.txt
    └── requirements.txt
```

---

# 📜 Version History

| Version | Date | Engineering Milestone |
|----------|------------|----------------------------|
| v38 | 2026-01-03 | Finalized edge-optimized deployment pipeline |
| v29 | 2025-10-10 | Integrated Cloudinary API |
| v24 | 2025-10-07 | Provisioned Heroku PostgreSQL |
| v19 | 2025-10-07 | Google Drive integration |
| v13 | 2025-10-07 | Probability threshold + dynamic fun facts |
| v8 | 2025-10-02 | Two-page UI deployment |
| v1 | 2025-10-01 | Initial inference release |

---

# 🛣️ Future Roadmap

- Quantization-Aware Training (QAT)  
- Real-time Video Inference  
- Automated Active Learning Loop  
- Cross-ARM Benchmarking  

---

# 📄 License

MIT License