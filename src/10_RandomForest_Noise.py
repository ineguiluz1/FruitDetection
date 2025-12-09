import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

train_path = 'data/augmented/features/processed/train_features_processed.csv'
test_path = 'data/augmented/features/processed/test_features_processed.csv'

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

df = pd.concat([train_data, test_data], ignore_index=True)

X = df.drop(columns=['label', 'label_encoded'])
y = df['label_encoded']

rf = joblib.load('src/models/random_forest_classifier.pkl')

y_pred = rf.predict(X)

accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, average='weighted')
recall = recall_score(y, y_pred, average='weighted')
f1 = f1_score(y, y_pred, average='weighted')

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')
print('\nClassification Report:')
print(classification_report(y, y_pred))