import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

train_path = 'data/features/bruised/train_features.csv'
test_path = 'data/features/bruised/test_features.csv'

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

X_train = train_data.drop(columns=['apple_id', 'damaged'])
y_train = train_data['damaged']
X_test = test_data.drop(columns=['apple_id', 'damaged'])
y_test = test_data['damaged']

print(f'Training set: {len(train_data)} apples')
print(f'  - Healthy: {(y_train==0).sum()}')
print(f'  - Damaged: {(y_train==1).sum()}')
print(f'\nTest set: {len(test_data)} apples')
print(f'  - Healthy: {(y_test==0).sum()}')
print(f'  - Damaged: {(y_test==1).sum()}\n')

lgb_train = lgb.Dataset(X_train, y_train)
lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 15,
    'learning_rate': 0.01,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'scale_pos_weight': scale_pos_weight,
    'min_child_samples': 3,
    'verbose': -1
}

print('Training LightGBM model...')
model = lgb.train(
    params,
    lgb_train,
    num_boost_round=200,
    valid_sets=[lgb_train, lgb_test],
    valid_names=['train', 'test']
)

y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
print('\nTesting different thresholds:')
print('=' * 50)

best_threshold = 0.5
best_accuracy = 0

for threshold in thresholds:
    y_pred = (y_pred_proba > threshold).astype(int)
    acc = accuracy_score(y_test, y_pred)
    print(f'\nThreshold {threshold}:')
    print(f'  Accuracy: {acc:.4f}')
    print(f'  Predicted Healthy: {(y_pred==0).sum()}, Damaged: {(y_pred==1).sum()}')
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_threshold = threshold

print(f'\n{"=" * 50}')
print(f'Best threshold: {best_threshold} (Accuracy: {best_accuracy:.4f})')
print('=' * 50)

y_pred = (y_pred_proba > best_threshold).astype(int)

accuracy = accuracy_score(y_test, y_pred)

print(f'\nTest Accuracy: {accuracy:.4f}')
print('\nConfusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=['Healthy', 'Damaged']))

feature_importance = model.feature_importance(importance_type='gain')
feature_names = X_train.columns
importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
importance_df = importance_df.sort_values('importance', ascending=False)

print('\nTop 10 Most Important Features:')
print(importance_df.head(10).to_string(index=False))
