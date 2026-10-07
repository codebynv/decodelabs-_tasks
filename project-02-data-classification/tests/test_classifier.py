import unittest
import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

class TestClassifier(unittest.TestCase):
    def setUp(self):
        self.X, self.y, self.target_names, self.feature_names = classifier.load_data()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        self.model = classifier.build_pipeline(n_neighbors=5)
        self.model.fit(self.X_train, self.y_train)

    def test_dataset_characteristics(self):
        # 1. Dataset loads successfully
        self.assertIsNotNone(self.X)
        self.assertIsNotNone(self.y)
        # 2. Expected number of samples
        self.assertEqual(self.X.shape[0], 150)
        # 3. Four input features
        self.assertEqual(self.X.shape[1], 4)
        # 4. Three target classes
        self.assertEqual(len(self.target_names), 3)

    def test_train_test_split(self):
        # 5. Non-empty datasets
        self.assertGreater(len(self.X_train), 0)
        self.assertGreater(len(self.X_test), 0)
        # Verify 80/20 split (150 * 0.2 = 30)
        self.assertEqual(len(self.X_test), 30)
        self.assertEqual(len(self.X_train), 120)

    def test_training_and_prediction(self):
        # 6. Training completed successfully (model has been fitted)
        # 7. Predictions have the expected number of test samples
        predictions = self.model.predict(self.X_test)
        self.assertEqual(len(predictions), 30)
        # 8. Predictions contain valid class labels
        self.assertTrue(all(pred in [0, 1, 2] for pred in predictions))

    def test_evaluation_metrics(self):
        predictions = self.model.predict(self.X_test)
        # 9. Evaluation metrics generated
        acc = accuracy_score(self.y_test, predictions)
        f1 = f1_score(self.y_test, predictions, average='macro')
        cm = confusion_matrix(self.y_test, predictions)
        
        self.assertTrue(isinstance(acc, float))
        self.assertTrue(isinstance(f1, float))
        
        # 10. Confusion matrix has expected dimensions (3x3 for 3 classes)
        self.assertEqual(cm.shape, (3, 3))

    def test_reproducibility(self):
        # 11. Re-running with same random_state yields same splits and predictions
        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        self.assertTrue(np.array_equal(self.X_train, X_train2))
        
        model2 = classifier.build_pipeline(n_neighbors=5)
        model2.fit(X_train2, y_train2)
        predictions1 = self.model.predict(self.X_test)
        predictions2 = model2.predict(X_test2)
        self.assertTrue(np.array_equal(predictions1, predictions2))

if __name__ == '__main__':
    unittest.main()
