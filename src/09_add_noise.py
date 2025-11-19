import cv2
import numpy as np
import os
import random
from pathlib import Path
from tqdm import tqdm


def replace_background(img, new_color=(200, 200, 200)):
    """Replace white background with a new color"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    img_copy = img.copy()
    img_copy[mask > 0] = new_color
    return img_copy


def change_saturation_hue(img, sat_factor=1.2, hue_shift=10):
    """Change saturation and hue of the image"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_factor, 0, 255)
    hsv = hsv.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def apply_gaussian_blur(img, kernel_size=(5, 5)):
    """Apply Gaussian blur"""
    return cv2.GaussianBlur(img, kernel_size, 0)


def apply_median_blur(img, kernel_size=5):
    """Apply median blur"""
    return cv2.medianBlur(img, kernel_size)


def add_gaussian_noise(img, mean=0, sigma=25):
    """Add Gaussian noise to the image"""
    noise = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    noisy_img = img.astype(np.float32) + noise
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    return noisy_img


def augment_image(img, augmentation_type):
    """Apply specific augmentation to image"""
    if augmentation_type == 'replace_bg':
        return replace_background(img, new_color=(220, 220, 220))
    elif augmentation_type == 'sat_hue':
        return change_saturation_hue(img, sat_factor=1.3, hue_shift=15)
    elif augmentation_type == 'gaussian_blur':
        return apply_gaussian_blur(img, kernel_size=(5, 5))
    elif augmentation_type == 'median_blur':
        return apply_median_blur(img, kernel_size=5)
    elif augmentation_type == 'gaussian_noise':
        return add_gaussian_noise(img, mean=0, sigma=20)
    return img


def process_directory(input_dir, output_dir, augmentations):
    """Process all images in directory with augmentations"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Get all subdirectories
    subdirs = [d for d in input_path.iterdir() if d.is_dir()]
    
    print(f'Processing {len(subdirs)} categories...')
    
    for subdir in tqdm(subdirs, desc='Categories'):
        category_name = subdir.name
        
        # Create output subdirectory
        output_subdir = output_path / category_name
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        # Get all images in subdirectory
        images = list(subdir.glob('*.jpg')) + list(subdir.glob('*.png'))
        
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Save original
            original_name = img_path.stem + img_path.suffix
            cv2.imwrite(str(output_subdir / original_name), img)
            
            # Apply ONE random augmentation per image
            aug_name = random.choice(augmentations)
            aug_img = augment_image(img, aug_name)
            aug_filename = f"{img_path.stem}_{aug_name}{img_path.suffix}"
            cv2.imwrite(str(output_subdir / aug_filename), aug_img)


if __name__ == '__main__':
    # Define augmentations to apply
    augmentations = [
        'replace_bg',
        'sat_hue',
        'gaussian_blur',
        'median_blur',
        'gaussian_noise'
    ]
    
    # Process Training set
    train_input = 'data/processed/Training'
    train_output = 'data/augmented/Training'
    print('\n=== Processing Training Set ===')
    process_directory(train_input, train_output, augmentations)
    
    # Process Test set
    test_input = 'data/processed/Test'
    test_output = 'data/augmented/Test'
    print('\n=== Processing Test Set ===')
    process_directory(test_input, test_output, augmentations)
    
    print('\nData augmentation completed!')

