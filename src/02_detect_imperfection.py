import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm


def is_apple_folder(name):
    return name.lower().startswith('apple')


def analyze_image(path, return_images=False):
    img = cv2.imread(path)
    if img is None:
        return None
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    mask_white = cv2.inRange(img_hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
    mask_fruit = cv2.bitwise_not(mask_white)
    
    kernel = np.ones((5, 5), np.uint8)
    mask_fruit = cv2.morphologyEx(mask_fruit, cv2.MORPH_CLOSE, kernel)
    
    mask_bruise = cv2.inRange(img_hsv, np.array([0, 20, 20]), np.array([17, 200, 220]))
    mask_bruise = cv2.bitwise_and(mask_bruise, mask_fruit)
    mask_bruise = cv2.morphologyEx(mask_bruise, cv2.MORPH_CLOSE, kernel)
    mask_bruise = cv2.morphologyEx(mask_bruise, cv2.MORPH_OPEN, kernel)
    
    bruise_area = int(cv2.countNonZero(mask_bruise))
    fruit_area = int(cv2.countNonZero(mask_fruit))
    bruise_pct = (bruise_area / fruit_area) * 100.0 if fruit_area > 0 else 0.0
    
    result = {
        'bruise_area': bruise_area,
        'fruit_area': fruit_area,
        'bruise_percentage': bruise_pct
    }
    
    if return_images:
        result['image'] = img
        result['mask_bruise'] = mask_bruise
        result['mask_fruit'] = mask_fruit
    
    return result


def display_bruised_images(bruised_data):
    """Display images of bruised fruits with their masks"""
    if not bruised_data:
        return
    
    for apple_id, data in bruised_data.items():
        img_path = data['image_path']
        bruise_pct = data['bruise_percentage']
        
        # Re-analyze image to get masks
        result = analyze_image(img_path, return_images=True)
        if result is None:
            continue
        
        img = result['image']
        mask_bruise = result['mask_bruise']
        
        # Create visualization
        # Convert mask to 3 channels
        mask_bruise_colored = cv2.cvtColor(mask_bruise, cv2.COLOR_GRAY2BGR)
        
        # Highlight bruised areas in red
        img_with_bruise = img.copy()
        img_with_bruise[mask_bruise > 0] = [0, 0, 255]  # Red color for bruises
        
        # Blend original image with bruise overlay
        alpha = 0.6
        img_overlay = cv2.addWeighted(img, alpha, img_with_bruise, 1 - alpha, 0)
        
        # Stack images horizontally
        display = np.hstack([img, mask_bruise_colored, img_overlay])
        
        # Add text
        title = f"{apple_id} - Bruise: {bruise_pct:.2f}%"
        cv2.putText(display, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Resize for better display
        height, width = display.shape[:2]
        max_width = 1800
        if width > max_width:
            scale = max_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            display = cv2.resize(display, (new_width, new_height))
        
        # Display
        cv2.imshow('Bruised Fruit Detection', display)
        print(f"\nShowing: {title}")
        print("Press any key to see next image, or 'q' to skip remaining...")
        
        key = cv2.waitKey(0)
        if key == ord('q') or key == ord('Q'):
            cv2.destroyAllWindows()
            break
    
    cv2.destroyAllWindows()


data_train_root = 'data/processed/Training'
data_test_root = 'data/processed/Testing'
out_train_csv = 'data/labels/apple_damage_labels_train.csv'
out_test_csv = 'data/labels/apple_damage_labels_test.csv'
threshold = 30.0
allowed_folders = ['Apple 5', 'Apple 6', 'Apple 7', 'Apple 8', 'Apple 9', 'Apple 10', 'Apple 11', 'Apple 12','Apple 13', 'Apple 14', 'Apple 17', 'Apple 18', 'Apple 19', 
                   'Apple Braeburn 1', 'Apple Core 1', 'Apple Crimson Snow 1', 'Apple Golden 1', 'Apple Golden 2', 'Apple Granny Smith 1', 'Apple Red 1', 'Apple Red 2', 'Apple Red Delicious 1',
                   'Apple worm 1', 'Apple Red Yellow 1', 'Apple Red Yellow 2']

apple_data = {}
bruised_images_train = {}

print('Processing training data...')
print('=' * 30)

for root, dirs, files in os.walk(data_train_root):
    folder = os.path.basename(root)
    if folder not in allowed_folders:
        continue
    
    bruise_percentages = []
    image_paths = []
    
    for f in tqdm(files, desc=f'Processing {folder}'):
        if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(root, f)
        res = analyze_image(img_path)
        if res is None:
            continue
        bruise_percentages.append(res['bruise_percentage'])
        image_paths.append(img_path)
    
    if bruise_percentages:
        avg_bruise = np.mean(bruise_percentages)
        max_bruise = np.max(bruise_percentages)
        max_idx = np.argmax(bruise_percentages)
        max_bruise_img_path = image_paths[max_idx]
        damaged = 1 if max_bruise > threshold else 0
        
        apple_data[folder] = {
            'apple_id': folder,
            'num_images': len(bruise_percentages),
            'avg_bruise_percentage': avg_bruise,
            'max_bruise_percentage': max_bruise,
            'damaged': damaged
        }
        
        # Store image path for bruised fruits
        if damaged == 1:
            bruised_images_train[folder] = {
                'image_path': max_bruise_img_path,
                'bruise_percentage': max_bruise
            }

records = list(apple_data.values())

if not records:
    print('No apple images found')
else:
    out_dir = os.path.dirname(out_train_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(out_train_csv, index=False)
    print(f'Wrote {len(df)} rows to {out_train_csv}')
    print(f'\nDamaged: {df["damaged"].sum()} ({df["damaged"].sum()/len(df)*100:.1f}%)')
    print(f'Healthy: {(df["damaged"]==0).sum()} ({(df["damaged"]==0).sum()/len(df)*100:.1f}%)')
    
    # Display bruised fruits
    bruised_fruits = df[df['damaged'] == 1]
    if len(bruised_fruits) > 0:
        print(f'\n{"="*30}')
        print('BRUISED FRUITS DETECTED (Training):')
        print(f'{"="*30}')
        for idx, row in bruised_fruits.iterrows():
            print(f"  - {row['apple_id']}: {row['max_bruise_percentage']:.2f}% bruise")
        
        # Display images
        print(f'\n{"="*30}')
        print('Displaying bruised fruit images...')
        print(f'{"="*30}')
        display_bruised_images(bruised_images_train)
    else:
        print('\nNo bruised fruits detected in training data.')

print('\nProcessing testing data...')
print('=' * 30)

bruised_images_test = {}

for root, dirs, files in os.walk(data_test_root):
    folder = os.path.basename(root)
    if folder not in allowed_folders:
        continue
    
    bruise_percentages = []
    image_paths = []
    
    for f in tqdm(files, desc=f'Processing {folder}'):
        if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        img_path = os.path.join(root, f)
        res = analyze_image(img_path)
        if res is None:
            continue
        bruise_percentages.append(res['bruise_percentage'])
        image_paths.append(img_path)
    
    if bruise_percentages:
        avg_bruise = np.mean(bruise_percentages)
        max_bruise = np.max(bruise_percentages)
        max_idx = np.argmax(bruise_percentages)
        max_bruise_img_path = image_paths[max_idx]
        damaged = 1 if max_bruise > threshold else 0
        
        apple_data[folder] = {
            'apple_id': folder,
            'num_images': len(bruise_percentages),
            'avg_bruise_percentage': avg_bruise,
            'max_bruise_percentage': max_bruise,
            'damaged': damaged
        }
        
        # Store image path for bruised fruits
        if damaged == 1:
            bruised_images_test[folder] = {
                'image_path': max_bruise_img_path,
                'bruise_percentage': max_bruise
            }

records = list(apple_data.values())

if not records:
    print('No apple images found')
else:
    out_dir = os.path.dirname(out_test_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(out_test_csv, index=False)
    print(f'Wrote {len(df)} rows to {out_test_csv}')
    print(f'\nDamaged: {df["damaged"].sum()} ({df["damaged"].sum()/len(df)*100:.1f}%)')
    print(f'Healthy: {(df["damaged"]==0).sum()} ({(df["damaged"]==0).sum()/len(df)*100:.1f}%)')
    
    # Display bruised fruits
    bruised_fruits = df[df['damaged'] == 1]
    if len(bruised_fruits) > 0:
        print(f'\n{"="*30}')
        print('BRUISED FRUITS DETECTED (Testing):')
        print(f'{"="*30}')
        for idx, row in bruised_fruits.iterrows():
            print(f"  - {row['apple_id']}: {row['max_bruise_percentage']:.2f}% bruise")
        
        # Display images
        print(f'\n{"="*30}')
        print('Displaying bruised fruit images...')
        print(f'{"="*30}')
        display_bruised_images(bruised_images_test)
    else:
        print('\nNo bruised fruits detected in testing data.')

