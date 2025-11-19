import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt

train_path = 'data/features/processed/train_data_processed.csv'
test_path = 'data/features/processed/test_data_processed.csv'

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

X_train = train_data.drop(columns=['label', 'label_encoded'])
y_train = train_data['label_encoded']
X_test = test_data.drop(columns=['label', 'label_encoded'])
y_test = test_data['label_encoded']

rf = RandomForestClassifier(random_state=42)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X_train, y_train, cv=skf, scoring='accuracy')

print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest accuracy: {test_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=train_data['label'].unique()))
