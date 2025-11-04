import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import shutil

# Paths configuration
base_dir = os.path.dirname(os.path.abspath(__file__))
raw_train_dir = os.path.join(base_dir, '..', 'data', 'raw', 'Training')
raw_test_dir = os.path.join(base_dir, '..', 'data', 'raw', 'Test')
processed_train_dir = os.path.join(base_dir, '..', 'data', 'processed', 'Training')
processed_test_dir = os.path.join(base_dir, '..', 'data', 'processed', 'Test')


def load_image(img_path):
    """
    Load an image from path and convert to RGB.
    
    Args:
        img_path: str - Path to the image file
    
    Returns:
        img: numpy array - RGB image (0-255)
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def normalize_image(img):
    """
    Normalize image to [0, 1] range.
    
    Args:
        img: numpy array - Input image (0-255)
    
    Returns:
        normalized_img: numpy array - Normalized image (0-1)
    """
    normalized_img = img.astype(np.float32) / 255.0
    return normalized_img


def preprocess_pipeline(img_path):
    """
    Simple preprocessing pipeline: load and normalize image.
    
    Args:
        img_path: str - Path to input image
    
    Returns:
        processed_img: numpy array - Normalized image (0-1)
    """
    # Load image
    img = load_image(img_path)
    
    # Normalize
    img = normalize_image(img)
    
    return img


def process_dataset(input_dir, output_dir):
    """
    Process entire dataset (Training or Test folder) - normalize images.
    
    Args:
        input_dir: str - Path to input directory (e.g., data/raw/Training)
        output_dir: str - Path to output directory (e.g., data/processed/Training)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all fruit categories (subdirectories)
    categories = [d for d in os.listdir(input_dir) 
                 if os.path.isdir(os.path.join(input_dir, d))]
    
    print(f"Found {len(categories)} categories")
    print(f"Processing images from {input_dir} to {output_dir}")
    
    total_images = 0
    
    for category in tqdm(categories, desc="Processing categories"):
        category_input_path = os.path.join(input_dir, category)
        category_output_path = os.path.join(output_dir, category)
        
        # Create category output directory
        os.makedirs(category_output_path, exist_ok=True)
        
        # Get all images in this category
        images = [f for f in os.listdir(category_input_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_name in images:
            img_path = os.path.join(category_input_path, img_name)
            
            try:
                # Load and normalize image
                img = load_image(img_path)
                normalized_img = normalize_image(img)
                
                # Save processed image
                output_path = os.path.join(category_output_path, img_name)
                
                # Convert back to uint8 for saving
                save_img = (normalized_img * 255).astype(np.uint8)
                
                # Convert RGB to BGR for cv2.imwrite
                save_img_bgr = cv2.cvtColor(save_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, save_img_bgr)
                
                total_images += 1
                
            except Exception as e:
                print(f"\nError processing {img_path}: {str(e)}")
    
    print(f"\nProcessing complete! Total images processed: {total_images}")


def visualize_preprocessing(img_path, save_path=None):
    """
    Visualize the preprocessing (normalization) for a single image.
    
    Args:
        img_path: str - Path to input image
        save_path: str - Path to save the visualization (optional)
    """
    # Load original image
    original = load_image(img_path)
    
    # Normalize
    normalized = normalize_image(original)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original)
    axes[0].set_title('Original Image (0-255)')
    axes[0].axis('off')
    
    axes[1].imshow(normalized)
    axes[1].set_title('Normalized Image (0-1)')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()


if __name__ == "__main__":
    # Example: Visualize preprocessing for a single image
    sample_img = os.path.join(raw_train_dir, 'Apple 5', 'r0_100_100.jpg')
    if os.path.exists(sample_img):
        print("Visualizing preprocessing pipeline...")
        visualize_preprocessing(sample_img)
    
    # Process entire training dataset
    print("\n" + "="*50)
    print("Processing Training Dataset")
    print("="*50)
    process_dataset(
        input_dir=raw_train_dir,
        output_dir=processed_train_dir
    )
    
    # Process entire test dataset
    print("\n" + "="*50)
    print("Processing Test Dataset")
    print("="*50)
    process_dataset(
        input_dir=raw_test_dir,
        output_dir=processed_test_dir
    )
    
    print("\n" + "="*50)
    print("All preprocessing completed!")
    print("="*50)






