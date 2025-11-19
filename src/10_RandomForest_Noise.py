import cv2
import numpy as np
import os
import joblib
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from skimage.measure import shannon_entropy
from skimage.feature import graycomatrix, graycoprops


def remove_white_background(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 55, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    fruit_mask = cv2.bitwise_not(white_mask)
    kernel = np.ones((3, 3), np.uint8)
    fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return fruit_mask


def extract_features(img, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    fruit_pixels = hsv[mask > 0]
    
    hsv_mean = np.mean(fruit_pixels, axis=0)
    hsv_std = np.std(fruit_pixels, axis=0)
    
    hist_features = []
    for i in range(3):
        hist, _ = np.histogram(fruit_pixels[:, i], bins=8, range=(0, 256))
        hist = hist / (np.sum(hist) + 1e-6)
        hist_features.extend(hist)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_fruit = gray[mask > 0].reshape(-1, 1)
    
    if len(gray_fruit) < 4:
        haralick = [0, 0, 0, 0]
        entropy = 0
    else:
        gray_norm = ((gray_fruit - gray_fruit.min()) / (gray_fruit.max() - gray_fruit.min() + 1e-6) * 7).astype(np.uint8)
        
        h, w = mask.shape
        gray_img = np.zeros((h, w), dtype=np.uint8)
        gray_img[mask > 0] = gray_norm.flatten()
        
        glcm = graycomatrix(gray_img, distances=[1], angles=[0], levels=8, symmetric=True, normed=True)
        haralick = [
            graycoprops(glcm, 'contrast')[0, 0],
            graycoprops(glcm, 'dissimilarity')[0, 0],
            graycoprops(glcm, 'homogeneity')[0, 0],
            graycoprops(glcm, 'energy')[0, 0]
        ]
        entropy = shannon_entropy(gray_norm)
    
    features = [circularity, area] + list(hsv_mean) + list(hsv_std) + hist_features + haralick + [entropy]
    return features


def extract_features_from_directory(directory):
    data = []
    subdirs = [d for d in Path(directory).iterdir() if d.is_dir()]
    
    for subdir in tqdm(subdirs, desc='Processing'):
        label = subdir.name
        images = list(subdir.glob('*.jpg')) + list(subdir.glob('*.png'))
        
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            mask = remove_white_background(img)
            features = extract_features(img, mask)
            
            if features is not None:
                data.append(features + [label])
    
    columns = ['circularity', 'area'] + \
              ['h_mean', 's_mean', 'v_mean', 'h_std', 's_std', 'v_std'] + \
              [f'hist_h_{i}' for i in range(8)] + \
              [f'hist_s_{i}' for i in range(8)] + \
              [f'hist_v_{i}' for i in range(8)] + \
              ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'entropy'] + \
              ['label']
    
    return pd.DataFrame(data, columns=columns)


rf = joblib.load('src/models/random_forest_classifier.pkl')
label_mapping = joblib.load('src/models/label_mapping.pkl')

print('Extracting features from augmented test data...')
test_df = extract_features_from_directory('data/augmented/Test')

X_test = test_df.drop(columns=['label'])
y_test = test_df['label']

label_encoder = {v: k for k, v in label_mapping.items()}
y_test_encoded = y_test.map(label_encoder)

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test_encoded, y_pred)
print(f'\nAccuracy: {accuracy:.4f}')
print('\nClassification Report:')
print(classification_report(y_test_encoded, y_pred, target_names=sorted(label_mapping.values())))
