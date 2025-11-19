import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.measure import shannon_entropy
from skimage.feature import graycomatrix, graycoprops
import pandas as pd
from tqdm import tqdm


def load_image(img_path):
    """
    Load an image from path and convert to RGB.
    
    Args:
        img_path: str - Path to the image file
    
    Returns:
        img: numpy array - RGB image
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def remove_white_background(img):
    """
    Remove white background from Fruits 360 images.
    
    Args:
        img: numpy array - Input image (RGB format)
    
    Returns:
        mask: numpy array - Binary mask of the fruit (foreground)
    """
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Define range for white color
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 55, 255])
    
    # Create mask for white pixels (background)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Invert mask to get the fruit (foreground)
    fruit_mask = cv2.bitwise_not(white_mask)
    
    # Clean up the mask with morphological operations
    kernel = np.ones((3, 3), np.uint8)
    fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    fruit_mask = cv2.morphologyEx(fruit_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return fruit_mask


def extract_features(img, mask=None):
    """
    Extract geometric features from a fruit image: circularity and area.
    
    Features:
    - Circularity: 4π × Area / Perimeter² (perfect circle = 1.0)
    - Area: Number of pixels in the fruit region
    
    Args:
        img: numpy array - Input image (RGB format)
        mask: numpy array - Binary mask of the fruit (optional, will be computed if None)
    
    Returns:
        features: dict - Dictionary with 'circularity' and 'area'
        contour: numpy array - Largest contour found
        mask: numpy array - Binary mask used
    """
    # If no mask provided, compute it by removing white background
    if mask is None:
        mask = remove_white_background(img)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Warning: No contours found")
        return {'circularity': 0.0, 'area': 0.0}, None, mask
    
    # Get the largest contour (should be the fruit)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate area
    area = cv2.contourArea(largest_contour)
    
    # Calculate perimeter
    perimeter = cv2.arcLength(largest_contour, True)
    
    # Calculate circularity (roundness)
    # Formula: 4π × Area / Perimeter²
    if perimeter == 0:
        circularity = 0.0
    else:
        circularity = (4 * np.pi * area) / (perimeter ** 2)
    
    features = {
        'circularity': circularity,
        'area': area
    }
    
    return features, largest_contour, mask


def extract_color_features(img, mask=None, bins=8):
    """
    Extract color features from a fruit image in HSV color space.
    
    Features extracted:
    - Mean values for H, S, V channels
    - Standard deviation for H, S, V channels
    - Histograms for H, S, V channels (bins values per channel)
    
    Total features: 3 means + 3 stds + (bins * 3 histogram values) = 6 + 24 = 30 features (with bins=8)
    
    Args:
        img: numpy array - Input image (RGB format)
        mask: numpy array - Binary mask of the fruit (optional, will be computed if None)
        bins: int - Number of bins for histogram (default: 8)
    
    Returns:
        color_features: dict - Dictionary with color features
    """
    # If no mask provided, compute it by removing white background
    if mask is None:
        mask = remove_white_background(img)
    
    # Convert to HSV color space
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Extract only the fruit pixels using the mask
    h_channel = hsv[:, :, 0][mask > 0]
    s_channel = hsv[:, :, 1][mask > 0]
    v_channel = hsv[:, :, 2][mask > 0]
    
    # Calculate mean values
    h_mean = np.mean(h_channel) if len(h_channel) > 0 else 0.0
    s_mean = np.mean(s_channel) if len(s_channel) > 0 else 0.0
    v_mean = np.mean(v_channel) if len(v_channel) > 0 else 0.0
    
    # Calculate standard deviation
    h_std = np.std(h_channel) if len(h_channel) > 0 else 0.0
    s_std = np.std(s_channel) if len(s_channel) > 0 else 0.0
    v_std = np.std(v_channel) if len(v_channel) > 0 else 0.0
    
    # Calculate histograms (normalized)
    # H channel: 0-179 in OpenCV
    h_hist, _ = np.histogram(h_channel, bins=bins, range=(0, 180))
    h_hist = h_hist.astype(float) / (np.sum(h_hist) + 1e-7)  # Normalize
    
    # S and V channels: 0-255
    s_hist, _ = np.histogram(s_channel, bins=bins, range=(0, 256))
    s_hist = s_hist.astype(float) / (np.sum(s_hist) + 1e-7)  # Normalize
    
    v_hist, _ = np.histogram(v_channel, bins=bins, range=(0, 256))
    v_hist = v_hist.astype(float) / (np.sum(v_hist) + 1e-7)  # Normalize
    
    # Combine all features
    color_features = {
        'h_mean': h_mean,
        's_mean': s_mean,
        'v_mean': v_mean,
        'h_std': h_std,
        's_std': s_std,
        'v_std': v_std,
        'h_hist': h_hist.tolist(),
        's_hist': s_hist.tolist(),
        'v_hist': v_hist.tolist()
    }
    
    return color_features


def extract_texture_features(img, mask=None, distance=1):
    """
    Extract texture features from a fruit image using Haralick features and entropy.
    
    Features extracted:
    - 4 Haralick properties (contrast, dissimilarity, homogeneity, energy) at 1 distance
    - 1 Entropy value
    
    Total features: 4 + 1 = 5 texture features
    
    Args:
        img: numpy array - Input image (RGB format)
        mask: numpy array - Binary mask of the fruit (optional, will be computed if None)
        distance: int - Distance for GLCM calculation (default: 1)
    
    Returns:
        texture_features: dict - Dictionary with texture features
    """
    # If no mask provided, compute it by removing white background
    if mask is None:
        mask = remove_white_background(img)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Apply mask to grayscale image
    gray_masked = gray.copy()
    gray_masked[mask == 0] = 0
    
    # Calculate entropy on masked region
    fruit_pixels = gray[mask > 0]
    if len(fruit_pixels) > 0:
        entropy = shannon_entropy(fruit_pixels)
    else:
        entropy = 0.0
    
    # Calculate GLCM (Gray-Level Co-occurrence Matrix)
    # Using 4 directions: 0°, 45°, 90°, 135°
    distances = [distance]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    # Crop to bounding box of mask to focus on fruit region
    coords = np.where(mask > 0)
    if len(coords[0]) > 0:
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        gray_crop = gray[y_min:y_max+1, x_min:x_max+1]
    else:
        gray_crop = gray
    
    # Calculate GLCM
    glcm = graycomatrix(gray_crop, distances=distances, angles=angles, 
                        levels=256, symmetric=True, normed=True)
    
    # Calculate Haralick features (4 properties)
    # Average over all 4 directions
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    
    texture_features = {
        'haralick_contrast': contrast,
        'haralick_dissimilarity': dissimilarity,
        'haralick_homogeneity': homogeneity,
        'haralick_energy': energy,
        'entropy': entropy
    }
    
    return texture_features


def extract_all_features(img, mask=None, bins=8, distance=1):
    """
    Extract all features (geometric + color + texture) from a fruit image.
    
    Args:
        img: numpy array - Input image (RGB format)
        mask: numpy array - Binary mask of the fruit (optional, will be computed if None)
        bins: int - Number of bins for color histograms (default: 8)
        distance: int - Distance for texture GLCM calculation (default: 1)
    
    Returns:
        all_features: dict - Dictionary with all extracted features
        mask: numpy array - Binary mask used
    """
    # Extract geometric features
    geom_features, contour, mask = extract_features(img, mask)
    
    # Extract color features
    color_features = extract_color_features(img, mask, bins)
    
    # Extract texture features
    texture_features = extract_texture_features(img, mask, distance)
    
    # Combine all features
    all_features = {**geom_features, **color_features, **texture_features}
    
    return all_features, contour, mask


def visualize_all_features(img_path, save_path=None, bins=8, distance=1):
    """
    Visualize all feature extraction (geometric + color + texture) for a fruit image.
    
    Args:
        img_path: str - Path to input image
        save_path: str - Path to save the visualization (optional)
        bins: int - Number of bins for histograms (default: 8)
        distance: int - Distance for texture GLCM calculation (default: 1)
    
    Returns:
        all_features: dict - Dictionary with all extracted features
    """
    # Load image
    img = load_image(img_path)
    
    # Extract all features
    all_features, contour, mask = extract_all_features(img, bins=bins, distance=distance)
    
    # Create visualization
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Row 1: Original image, mask, contour
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img)
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(mask, cmap='gray')
    ax2.set_title('Binary Mask')
    ax2.axis('off')
    
    img_with_contour = img.copy()
    if contour is not None:
        cv2.drawContours(img_with_contour, [contour], -1, (255, 0, 0), 2)
    ax3 = fig.add_subplot(gs[0, 2:4])
    ax3.imshow(img_with_contour)
    title = f"Circularity = {all_features['circularity']:.4f}\n"
    title += f"Area = {all_features['area']:.0f} px²"
    ax3.set_title(title)
    ax3.axis('off')
    
    # Row 2 & 3: HSV histograms
    # H histogram
    ax4 = fig.add_subplot(gs[1, 0:2])
    ax4.bar(range(bins), all_features['h_hist'], color='red', alpha=0.7)
    ax4.set_title(f"H Histogram (Mean={all_features['h_mean']:.2f}, Std={all_features['h_std']:.2f})")
    ax4.set_xlabel('Bin')
    ax4.set_ylabel('Normalized Frequency')
    ax4.grid(alpha=0.3)
    
    # S histogram
    ax5 = fig.add_subplot(gs[1, 2:4])
    ax5.bar(range(bins), all_features['s_hist'], color='green', alpha=0.7)
    ax5.set_title(f"S Histogram (Mean={all_features['s_mean']:.2f}, Std={all_features['s_std']:.2f})")
    ax5.set_xlabel('Bin')
    ax5.set_ylabel('Normalized Frequency')
    ax5.grid(alpha=0.3)
    
    # V histogram
    ax6 = fig.add_subplot(gs[2, 1:3])
    ax6.bar(range(bins), all_features['v_hist'], color='blue', alpha=0.7)
    ax6.set_title(f"V Histogram (Mean={all_features['v_mean']:.2f}, Std={all_features['v_std']:.2f})")
    ax6.set_xlabel('Bin')
    ax6.set_ylabel('Normalized Frequency')
    ax6.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()
    
    # Print results
    print(f"\n{'='*60}")
    print("EXTRACTED FEATURES")
    print(f"{'='*60}")
    print("\nGeometric Features:")
    print(f"  Circularity: {all_features['circularity']:.4f}")
    print(f"  Area: {all_features['area']:.2f} px²")
    
    print("\nColor Features (HSV):")
    print(f"  H - Mean: {all_features['h_mean']:.2f}, Std: {all_features['h_std']:.2f}")
    print(f"  S - Mean: {all_features['s_mean']:.2f}, Std: {all_features['s_std']:.2f}")
    print(f"  V - Mean: {all_features['v_mean']:.2f}, Std: {all_features['v_std']:.2f}")
    
    print(f"\nHistogram Features ({bins} bins per channel):")
    print(f"  H histogram: {[f'{x:.3f}' for x in all_features['h_hist']]}")
    print(f"  S histogram: {[f'{x:.3f}' for x in all_features['s_hist']]}")
    print(f"  V histogram: {[f'{x:.3f}' for x in all_features['v_hist']]}")
    
    print("\nTexture Features:")
    print(f"  Haralick Contrast: {all_features['haralick_contrast']:.4f}")
    print(f"  Haralick Dissimilarity: {all_features['haralick_dissimilarity']:.4f}")
    print(f"  Haralick Homogeneity: {all_features['haralick_homogeneity']:.4f}")
    print(f"  Haralick Energy: {all_features['haralick_energy']:.4f}")
    print(f"  Entropy: {all_features['entropy']:.4f}")
    
    total_features = 2 + 6 + (bins * 3) + 5  # geometric + color stats + histograms + texture
    print(f"\nTotal feature vector size: {total_features}")
    print(f"  - Geometric: 2 (circularity, area)")
    print(f"  - Color statistics: 6 (3 means + 3 stds)")
    print(f"  - Histograms: {bins * 3} ({bins} bins × 3 channels)")
    print(f"  - Texture: 5 (4 Haralick + 1 entropy)")
    print(f"{'='*60}\n")
    
    return all_features


def features_to_vector(features, bins=8):
    """
    Convert features dictionary to a flat feature vector.
    
    Args:
        features: dict - Dictionary with all features
        bins: int - Number of bins used for histograms
    
    Returns:
        feature_vector: list - Flat list of all feature values
        feature_names: list - Names of features in the same order
    """
    feature_vector = []
    feature_names = []
    
    # Geometric features
    feature_vector.append(features['circularity'])
    feature_names.append('circularity')
    
    feature_vector.append(features['area'])
    feature_names.append('area')
    
    # Color statistics
    feature_vector.append(features['h_mean'])
    feature_names.append('h_mean')
    feature_vector.append(features['s_mean'])
    feature_names.append('s_mean')
    feature_vector.append(features['v_mean'])
    feature_names.append('v_mean')
    
    feature_vector.append(features['h_std'])
    feature_names.append('h_std')
    feature_vector.append(features['s_std'])
    feature_names.append('s_std')
    feature_vector.append(features['v_std'])
    feature_names.append('v_std')
    
    # Histograms
    for i, val in enumerate(features['h_hist']):
        feature_vector.append(val)
        feature_names.append(f'h_hist_bin{i}')
    
    for i, val in enumerate(features['s_hist']):
        feature_vector.append(val)
        feature_names.append(f's_hist_bin{i}')
    
    for i, val in enumerate(features['v_hist']):
        feature_vector.append(val)
        feature_names.append(f'v_hist_bin{i}')
    
    # Texture features
    feature_vector.append(features['haralick_contrast'])
    feature_names.append('haralick_contrast')
    feature_vector.append(features['haralick_dissimilarity'])
    feature_names.append('haralick_dissimilarity')
    feature_vector.append(features['haralick_homogeneity'])
    feature_names.append('haralick_homogeneity')
    feature_vector.append(features['haralick_energy'])
    feature_names.append('haralick_energy')
    feature_vector.append(features['entropy'])
    feature_names.append('entropy')
    
    return feature_vector, feature_names


def process_dataset_to_csv(input_dir, output_csv, bins=8, distance=1):
    """
    Process entire dataset and save features to CSV with labels.
    
    Args:
        input_dir: str - Path to input directory (e.g., data/raw/Training)
        output_csv: str - Path to save CSV file
        bins: int - Number of bins for color histograms (default: 8)
        distance: int - Distance for texture GLCM calculation (default: 1)
    
    Returns:
        df: DataFrame - DataFrame with all features and labels
    """
    # Get all fruit categories (subdirectories)
    categories = sorted([d for d in os.listdir(input_dir) 
                        if os.path.isdir(os.path.join(input_dir, d))])
    
    print(f"\nProcessing {len(categories)} categories from {input_dir}")
    print("="*60)
    
    all_data = []
    feature_names = None
    total_images = 0
    failed_images = 0
    
    for category in tqdm(categories, desc="Processing categories"):
        category_path = os.path.join(input_dir, category)
        images = [f for f in os.listdir(category_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_name in images:
            img_path = os.path.join(category_path, img_name)
            
            try:
                # Load image
                img = load_image(img_path)
                
                # Extract all features
                features, _, _ = extract_all_features(img, bins=bins, distance=distance)
                
                # Convert to feature vector
                feature_vector, feat_names = features_to_vector(features, bins=bins)
                
                # Store feature names (only once)
                if feature_names is None:
                    feature_names = feat_names
                
                # Create row with label and features
                row = [category] + feature_vector
                all_data.append(row)
                
                total_images += 1
                
            except Exception as e:
                failed_images += 1
                if failed_images <= 5:  # Print first 5 errors
                    print(f"\nError processing {img_path}: {str(e)}")
    
    # Create DataFrame
    columns = ['label'] + feature_names
    df = pd.DataFrame(all_data, columns=columns)
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
    print(f"Total images processed: {total_images}")
    print(f"Failed images: {failed_images}")
    print(f"Number of categories: {len(categories)}")
    print(f"Number of features: {len(feature_names)}")
    print(f"CSV saved to: {output_csv}")
    print(f"DataFrame shape: {df.shape}")
    print("="*60 + "\n")
    
    return df


if __name__ == "__main__":
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_train_dir = os.path.join(base_dir, '..', 'data', 'augmented', 'Training')
    raw_test_dir = os.path.join(base_dir, '..', 'data', 'augmented', 'Test')
    features_dir = os.path.join(base_dir, '..', 'data','augmented' ,'features')
    
    # Create features directory if it doesn't exist
    os.makedirs(features_dir, exist_ok=True)
    
    # Define output CSV paths
    train_csv = os.path.join(features_dir, 'train_features.csv')
    test_csv = os.path.join(features_dir, 'test_features.csv')
    
    # Configuration
    BINS = 8
    DISTANCE = 1
    
    # Ask user what to do
    print("\n" + "="*60)
    print("FRUIT FEATURE EXTRACTION")
    print("="*60)
    print("\nOptions:")
    print("1. Process Training dataset")
    print("2. Process Test dataset")
    print("3. Process both datasets")
    print("4. Visualize single image")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        # Process Training dataset
        print("\nProcessing Training dataset...")
        df_train = process_dataset_to_csv(raw_train_dir, train_csv, bins=BINS, distance=DISTANCE)
        print(f"\nTraining features saved to: {train_csv}")
        print(f"Sample of data:\n{df_train.head()}")
        
    elif choice == '2':
        # Process Test dataset
        print("\nProcessing Test dataset...")
        df_test = process_dataset_to_csv(raw_test_dir, test_csv, bins=BINS, distance=DISTANCE)
        print(f"\nTest features saved to: {test_csv}")
        print(f"Sample of data:\n{df_test.head()}")
        
    elif choice == '3':
        # Process both datasets
        print("\nProcessing Training dataset...")
        df_train = process_dataset_to_csv(raw_train_dir, train_csv, bins=BINS, distance=DISTANCE)
        
        print("\nProcessing Test dataset...")
        df_test = process_dataset_to_csv(raw_test_dir, test_csv, bins=BINS, distance=DISTANCE)
        
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Training features saved to: {train_csv}")
        print(f"  Shape: {df_train.shape}")
        print(f"  Categories: {df_train['label'].nunique()}")
        print(f"\nTest features saved to: {test_csv}")
        print(f"  Shape: {df_test.shape}")
        print(f"  Categories: {df_test['label'].nunique()}")
        print(f"{'='*60}\n")
        
    elif choice == '4':
        # Visualize single image
        apple_img_path = os.path.join(raw_train_dir, 'Apple Red 1', '12_100_gaussian_noise.jpg')
        
        if os.path.exists(apple_img_path):
            print("\nVisualizing features for an apple image...")
            features = visualize_all_features(apple_img_path, bins=BINS, distance=DISTANCE)
        else:
            print(f"Image not found: {apple_img_path}")
            print("\nAvailable categories:")
            if os.path.exists(raw_train_dir):
                categories = [d for d in os.listdir(raw_train_dir) 
                             if os.path.isdir(os.path.join(raw_train_dir, d))]
                for cat in categories[:10]:  # Show first 10
                    print(f"  - {cat}")
    else:
        print("Invalid choice. Exiting.")
