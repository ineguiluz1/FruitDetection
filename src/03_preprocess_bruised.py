import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from skimage.measure import shannon_entropy
from skimage.feature import graycomatrix, graycoprops


def remove_white_background(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    fruit_mask = cv2.bitwise_not(white_mask)
    kernel = np.ones((5, 5), np.uint8)
    fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_CLOSE, kernel)
    fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_OPEN, kernel)
    return fruit_mask


def extract_geometric_features(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {'circularity': 0.0, 'area': 0}
    
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)
    
    if perimeter == 0:
        circularity = 0.0
    else:
        circularity = (4 * np.pi * area) / (perimeter ** 2)
    
    return {'circularity': circularity, 'area': area}


def extract_color_features(img, mask, bins=8):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    h_channel = hsv[:, :, 0][mask > 0]
    s_channel = hsv[:, :, 1][mask > 0]
    v_channel = hsv[:, :, 2][mask > 0]
    
    h_mean = np.mean(h_channel) if len(h_channel) > 0 else 0.0
    s_mean = np.mean(s_channel) if len(s_channel) > 0 else 0.0
    v_mean = np.mean(v_channel) if len(v_channel) > 0 else 0.0
    
    h_std = np.std(h_channel) if len(h_channel) > 0 else 0.0
    s_std = np.std(s_channel) if len(s_channel) > 0 else 0.0
    v_std = np.std(v_channel) if len(v_channel) > 0 else 0.0
    
    h_hist, _ = np.histogram(h_channel, bins=bins, range=(0, 180))
    h_hist = h_hist.astype(float) / (np.sum(h_hist) + 1e-7)
    
    s_hist, _ = np.histogram(s_channel, bins=bins, range=(0, 256))
    s_hist = s_hist.astype(float) / (np.sum(s_hist) + 1e-7)
    
    v_hist, _ = np.histogram(v_channel, bins=bins, range=(0, 256))
    v_hist = v_hist.astype(float) / (np.sum(v_hist) + 1e-7)
    
    features = {
        'h_mean': h_mean, 's_mean': s_mean, 'v_mean': v_mean,
        'h_std': h_std, 's_std': s_std, 'v_std': v_std
    }
    
    for i, val in enumerate(h_hist):
        features[f'h_hist_bin{i}'] = val
    for i, val in enumerate(s_hist):
        features[f's_hist_bin{i}'] = val
    for i, val in enumerate(v_hist):
        features[f'v_hist_bin{i}'] = val
    
    return features


def extract_texture_features(img, mask):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_masked = gray.copy()
    gray_masked[mask == 0] = 0
    
    entropy = shannon_entropy(gray_masked[mask > 0]) if np.sum(mask) > 0 else 0.0
    
    glcm = graycomatrix(gray_masked, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    
    return {
        'haralick_contrast': contrast,
        'haralick_dissimilarity': dissimilarity,
        'haralick_homogeneity': homogeneity,
        'haralick_energy': energy,
        'entropy': entropy
    }


def extract_all_features_from_folder(folder_path):
    features_list = []
    
    for img_file in os.listdir(folder_path):
        if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        img_path = os.path.join(folder_path, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        mask = remove_white_background(img)
        
        geom_features = extract_geometric_features(mask)
        color_features = extract_color_features(img, mask)
        texture_features = extract_texture_features(img, mask)
        
        features = {**geom_features, **color_features, **texture_features}
        features_list.append(features)
    
    if not features_list:
        return {}
    
    avg_features = {}
    for key in features_list[0].keys():
        avg_features[key] = np.mean([f[key] for f in features_list])
    
    return avg_features


train_labels = pd.read_csv('data/labels/apple_damage_labels_train.csv')
test_labels = pd.read_csv('data/labels/apple_damage_labels_test.csv')

train_records = []
test_records = []

print('Processing training apples...')
for _, row in tqdm(train_labels.iterrows(), total=len(train_labels)):
    apple_id = row['apple_id']
    folder_path = os.path.join('data/raw/Training', apple_id)
    
    if not os.path.exists(folder_path):
        continue
    
    features = extract_all_features_from_folder(folder_path)
    if not features:
        continue
    
    features['apple_id'] = apple_id
    features['damaged'] = row['damaged']
    train_records.append(features)

print('\nProcessing testing apples...')
for _, row in tqdm(test_labels.iterrows(), total=len(test_labels)):
    apple_id = row['apple_id']
    folder_path = os.path.join('data/raw/Test', apple_id)
    
    if not os.path.exists(folder_path):
        continue
    
    features = extract_all_features_from_folder(folder_path)
    if not features:
        continue
    
    features['apple_id'] = apple_id
    features['damaged'] = row['damaged']
    test_records.append(features)

os.makedirs('data/features/bruised', exist_ok=True)

train_df = pd.DataFrame(train_records)
test_df = pd.DataFrame(test_records)

train_df.to_csv('data/features/bruised/train_features.csv', index=False)
test_df.to_csv('data/features/bruised/test_features.csv', index=False)

print(f'\nTrain features saved: {len(train_df)} apples')
print(f'Test features saved: {len(test_df)} apples')
print(f'Total features per apple: {len(train_df.columns) - 2}')  # -2 for apple_id and damaged
