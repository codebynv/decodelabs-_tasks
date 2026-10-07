# DecodeLabs AI Internship

<p align="center">
  <strong>AI Internship Projects • Python • Machine Learning • Recommendation Systems • OCR</strong>
</p>

<p align="center">
  <a href="https://www.decodelabs.tech/">DecodeLabs</a>
  ·
  <a href="https://github.com/codebynv/decodelabs-_tasks">Repository</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/AI-Internship-111827?style=for-the-badge" alt="AI Internship">
  <img src="https://img.shields.io/badge/Machine%20Learning-scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Machine Learning">
  <img src="https://img.shields.io/badge/Computer%20Vision-OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="Computer Vision">
  <img src="https://img.shields.io/badge/Tests-79%20Passing-16A34A?style=for-the-badge" alt="79 tests passing">
</p>

---

## About

This repository contains my **AI internship project work at DecodeLabs**, organized as four progressively broader implementations — from deterministic rule-based logic to supervised learning, content-based recommendations, and OCR-based image/text recognition.

The focus throughout the internship work is on **understanding the fundamentals, building working implementations, validating behavior with tests, and keeping each project reproducible and explainable**.

> **Built by Nirav Vala • DecodeLabs AI Internship**

---

## The Project Journey

    01  Rule-Based Logic
            ↓
    02  Supervised Learning
            ↓
    03  Recommendation Systems
            ↓
    04  Computer Vision + OCR

Each project introduces a different layer of practical AI development while keeping the implementation focused on the requirements of the internship.

---

## Project Showcase

| # | Project | What it demonstrates | Stack |
|---|---|---|---|
| **01** | [Rule-Based Chatbot](./project-01-rule-based-chatbot) | Deterministic conversation logic, input normalization, greetings, exits and fallback handling | Python |
| **02** | [Data Classification](./project-02-data-classification) | Dataset handling, train/test splitting, scaling, KNN classification and evaluation | Python · scikit-learn |
| **03** | [AI Recommendation System](./project-03-ai-recommendation) | Content-based recommendation using TF-IDF and cosine similarity | Python · TF-IDF |
| **04** | [Image/Text Recognition](./project-04-image-text-recognition) | Image preprocessing, OCR, confidence filtering and visual confirmation | Python · OpenCV · Tesseract |

---

## 01 — Rule-Based Chatbot

A lightweight deterministic chatbot built around rule-based decision logic.

**Highlights**

- Greeting and exit command handling
- Normalized user input
- Dictionary-based response lookup
- Fallback responses for unknown input
- Continuous CLI interaction
- Separated chatbot logic and input/output
- Unit-tested behavior

[**Explore Project 01 →**](./project-01-rule-based-chatbot)

---

## 02 — Data Classification

A supervised classification pipeline built with the **Iris dataset**.

**Pipeline**

    Iris Dataset
        ↓
    Train / Test Split
        ↓
    StandardScaler
        ↓
    K-Nearest Neighbors (K=5)
        ↓
    Prediction
        ↓
    Accuracy + F1 + Confusion Matrix

**Highlights**

- Stratified 80/20 train/test split
- Feature scaling without data leakage
- KNN classifier
- Accuracy and macro F1 evaluation
- Confusion matrix
- Reproducible training configuration

[**Explore Project 02 →**](./project-02-data-classification)

---

## 03 — AI Recommendation System

A **content-based Tech Stack Recommender** that matches user skills/interests against a structured technology-role catalogue.

**Pipeline**

    User Skills
        ↓
    Normalize Input
        ↓
    TF-IDF Representation
        ↓
    Cosine Similarity
        ↓
    Rank Matches
        ↓
    Top-3 Recommendations

**Highlights**

- Content-based recommendation
- TF-IDF weighting
- Cosine similarity scoring
- Top-3 recommendations
- Unknown-input handling
- Cold-start handling
- Deterministic ranking and tie-breaking
- Human-readable skill catalogue

[**Explore Project 03 →**](./project-03-ai-recommendation)

---

## 04 — Image/Text Recognition

An OCR pipeline using **OpenCV + Tesseract** to extract readable text from images.

**Pipeline**

    Input Image
        ↓
    Grayscale
        ↓
    Gaussian Blur
        ↓
    Deskew
        ↓
    Adaptive Threshold
        ↓
    Tesseract OCR
        ↓
    Confidence Filtering
        ↓
    Annotated Visual Output

**Highlights**

- pytesseract / Tesseract OCR integration
- OpenCV preprocessing
- Adaptive thresholding
- Deskewing
- Configurable Tesseract PSM modes
- 80% minimum confidence acceptance threshold
- Bounding-box visual confirmation
- Invalid-input handling

> **Important:** The 80% requirement is an OCR confidence threshold, not a claim of 80% model accuracy. The implementation reports actual OCR confidence values.

[**Explore Project 04 →**](./project-04-image-text-recognition)

---

## Validation & Testing

Every project includes its own test suite.

| Project | Tests | Result |
|---|---:|---|
| Rule-Based Chatbot | **4** | PASS |
| Data Classification | **5** | PASS |
| AI Recommendation | **27** | PASS |
| Image/Text Recognition | **43** | PASS |
| **Total** | **79** | **PASS** |

The projects were individually verified before being added to this repository.

---

## Technology Stack

### Core

- **Python**
- **Git / GitHub**

### Machine Learning

- **scikit-learn**
- **K-Nearest Neighbors**
- **StandardScaler**
- **TF-IDF**
- **Cosine Similarity**

### Computer Vision & OCR

- **OpenCV**
- **pytesseract**
- **Tesseract OCR**

### Testing

- **Python unittest**

The projects intentionally keep their dependencies focused on what each implementation actually needs.

---

## Repository Structure

    decodelabs-_tasks/
    │
    ├── project-01-rule-based-chatbot/
    │   ├── src/
    │   ├── tests/
    │   ├── README.md
    │   ├── requirements.txt
    │   └── .gitignore
    │
    ├── project-02-data-classification/
    │   ├── src/
    │   ├── tests/
    │   ├── README.md
    │   ├── requirements.txt
    │   └── .gitignore
    │
    ├── project-03-ai-recommendation/
    │   ├── data/
    │   ├── src/
    │   ├── tests/
    │   ├── README.md
    │   └── requirements.txt
    │
    └── project-04-image-text-recognition/
        ├── data/
        ├── outputs/
        ├── src/
        ├── tests/
        ├── README.md
        ├── requirements.txt
        └── .gitignore

Each project is intentionally self-contained with its own implementation, dependencies, documentation, and tests.

---

## Running a Project

Each project has its own setup instructions.

General workflow:

    cd project-XX-name
    pip install -r requirements.txt

Then follow the project's individual README for its execution and test commands.

Project-specific requirements — especially the external **Tesseract OCR engine** for Project 04 — are documented inside the relevant project.

---

## What This Internship Work Covers

Through these four implementations, the repository demonstrates practical exposure to:

- Python programming for AI
- Rule-based decision systems
- Data loading and preprocessing
- Supervised machine learning
- Model evaluation
- Feature scaling
- Similarity-based recommendation
- TF-IDF representations
- Cosine similarity
- Computer vision preprocessing
- OCR integration
- Confidence-based filtering
- Automated testing
- Reproducible project structure
- Technical documentation

The progression is deliberately practical:

**logic → learning → recommendation → perception**

---

## DecodeLabs

This work was completed as part of the **DecodeLabs AI Internship**.

**DecodeLabs:**  
https://www.decodelabs.tech/

The repository is a personal internship project showcase and is not an official DecodeLabs corporate repository.

---

## Author

### Nirav Vala

B.Sc. Information Technology student focused on **AI/ML, software development, full-stack engineering, and practical problem solving**.

- GitHub: [@codebynv](https://github.com/codebynv)
- Repository: [decodelabs-_tasks](https://github.com/codebynv/decodelabs-_tasks)

---

<p align="center">
  <strong>DecodeLabs AI Internship</strong><br>
  From fundamentals to working AI systems.
</p>
