import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline

def load_data():
    """Loads the Iris dataset."""
    iris = load_iris()
    return iris.data, iris.target, iris.target_names, iris.feature_names

def build_pipeline(n_neighbors: int = 5):
    """
    Builds a scikit-learn pipeline for scaling and classification.
    Using a pipeline ensures scaling is fit only on training data, preventing data leakage.
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=n_neighbors))
    ])
    return pipeline

def run_ml_pipeline():
    """Runs the complete supervised learning pipeline."""
    # 1. Load Iris Data
    X, y, target_names, feature_names = load_data()
    
    print("Dataset information")
    print("-------------------")
    print(f"Samples: {X.shape[0]}")
    print(f"Features: {X.shape[1]} ({', '.join(feature_names)})")
    print(f"Classes: {len(target_names)} ({', '.join(target_names)})\n")
    
    # 2. Train / Test Split
    # Project 2 PDF specifies an 80/20 split and emphasizes removing order bias (shuffle).
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Model Definition (Pipeline)
    k_value = 5
    model = build_pipeline(n_neighbors=k_value)
    
    print("Model")
    print("-----")
    print(f"Algorithm: K-Nearest Neighbors")
    print(f"K: {k_value}\n")
    
    # 4. Train the Model
    model.fit(X_train, y_train)
    
    # 5. Predict on Test Set
    predictions = model.predict(X_test)
    
    # 6. Evaluate
    acc = accuracy_score(y_test, predictions)
    # Use macro F1 since we have 3 classes and want an unweighted average of their F1 scores.
    f1 = f1_score(y_test, predictions, average='macro')
    cm = confusion_matrix(y_test, predictions)
    
    print("Evaluation")
    print("----------")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}\n")
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    run_ml_pipeline()
