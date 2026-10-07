# Project 2 — Data Classification Using AI

## Overview
This is the predictive phase of the Decode Labs Industrial Training Kit (Batch 2026), focusing on Data Classification using Artificial Intelligence. Building upon the logic skeleton of Project 1, this project introduces Supervised Learning, enabling the machine to recognize patterns in historical data and categorize new information rather than relying on explicit hardcoded rules.

## Objective
The goal is to build a basic classification model using a small benchmark dataset, mastering the fundamental machine learning pipeline from raw data to intelligent decision-making.

## Decode Labs Requirements
Following the Master Blueprint (IPO Framework) specified in the training material:
1. Load and understand the Iris benchmark dataset.
2. Split the data into training and validation sets (with shuffling to prevent order bias).
3. Apply standard feature scaling (Mean = 0, Variance = 1).
4. Implement a K-Nearest Neighbors (KNN) classification algorithm.
5. Evaluate the model beyond simple accuracy, using a Confusion Matrix and F1 Score.

## Dataset
The project utilizes the classic **Iris benchmark dataset**, obtained directly via `scikit-learn`:
- **Samples**: 150 (Balanced across classes)
- **Features (Dimensions)**: 4 (Sepal Length, Sepal Width, Petal Length, Petal Width)
- **Classes**: 3 (Setosa, Versicolor, Virginica)

## Machine Learning Approach

### Why KNN
The K-Nearest Neighbors algorithm was chosen as the baseline model. It operates on the "Proximity Principle"—the idea that similar data points exist in close proximity to one another. For an unknown sample, the algorithm observes the 'K' closest data points and assigns a class based on a majority vote. In this implementation, $K=5$ is used as the optimal "elbow" value between overfitting (noise) and underfitting (generic).

### Preprocessing
Before measuring distance between neighbors, it is critical to balance the data. The "Gatekeeper Rule" requires feature scaling to ensure features with larger numerical ranges don't disproportionately dominate distance calculations. We use `StandardScaler` to transform features so they have a mean of 0 and a variance of 1.

### Train/Test Split
To ensure structural integrity and prevent our model from simply memorizing the answers, the data is shuffled and split:
- **Training Set (80%)**: Used by the algorithm to recognize patterns (memorize the map).
- **Test Set (20%)**: Locked away during training and used strictly for validation.

### Model Training
The project uses a `scikit-learn` Pipeline to chain the standard scaler and the KNN classifier. This ensures that the scaler only learns parameters (mean/variance) from the training data, applying those exact parameters to the test data, thus completely preventing data leakage.

### Evaluation
As taught in the material, in some datasets, "accuracy is a lie" (the accuracy mirage). Therefore, we rely on diagnostic tools:
- **Confusion Matrix**: A diagnostic matrix showing True Positives (TP), False Positives (FP - False Alarms), True Negatives (TN), and False Negatives (FN - Missed Detections).
- **F1 Score**: The harmonic mean of precision and recall, balancing the trade-offs between trustworthiness and sensitivity. We use `macro` averaging to treat all 3 classes equally in the final score.

## Project Structure
```
project-02-data-classification/
├── src/
│   └── classifier.py         # Main ML pipeline execution
├── tests/
│   └── test_classifier.py    # Automated test suite
├── .gitignore                # Git exclusions
├── requirements.txt          # Minimal Python dependencies
└── README.md                 # Project documentation
```

## Technologies Used
- Python
- `numpy` (Numerical operations)
- `scikit-learn` (Machine learning library & dataset)
- `unittest` (Testing framework)

## Installation
Ensure you have the required dependencies installed:
```bash
pip install -r requirements.txt
```

## How to Run
Execute the classification pipeline from the root directory:
```bash
python src/classifier.py
```

## Example Output
```text
Dataset information
-------------------
Samples: 150
Features: 4 (sepal length (cm), sepal width (cm), petal length (cm), petal width (cm))
Classes: 3 (setosa, versicolor, virginica)

Model
-----
Algorithm: K-Nearest Neighbors
K: 5

Evaluation
----------
Accuracy: 0.9333
F1 Score: 0.9327

Confusion Matrix:
[[10  0  0]
 [ 0 10  0]
 [ 0  2  8]]
```

## Testing
A comprehensive test suite validates the integrity of the pipeline, confirming:
- Dataset structural correctness (150x4 shape, 3 classes).
- Correct train/test separation lengths (120/30 split).
- Valid prediction output formats.
- Correct calculation and dimensions of evaluation metrics.
- Deterministic reproducibility given a fixed random state.

Run the tests via:
```bash
python tests/test_classifier.py
```
Tests passed: `5/5`

## What I Learned
- **The danger of data leakage:** Scaling must be part of a `Pipeline` or strictly `.fit()` on the training set and only `.transform()` on the test set.
- **Why KNN requires scaling:** Because it calculates Euclidean distance, features with larger scales will overshadow smaller ones if left unscaled.
- **Beyond Accuracy:** The Confusion Matrix provides significantly more transparency into *where* the model is failing (e.g., confusing Versicolor and Virginica) than a single accuracy percentage.

## Limitations
- KNN becomes computationally expensive as the dataset grows (linear scan for neighbors).
- KNN suffers from the "curse of dimensionality" on datasets with many features.
- The model is currently static; it does not save/export its trained state to disk (no `joblib` or `pickle` implementation).

## Possible Future Improvements
- Implement Cross-Validation (K-Fold) rather than a single train/test split.
- Export the trained model pipeline for persistence and deployment.
- Perform Grid Search to dynamically find the mathematically optimal $K$ value for the given data.
