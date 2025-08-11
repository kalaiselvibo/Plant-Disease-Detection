import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def edge_aware_clahe_image(img, clip_limit=2.0, tile_grid_size=(8, 8), edge_threshold1=100, edge_threshold2=200):
    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)

    # Detect edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, edge_threshold1, edge_threshold2)
    edge_mask = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    # Combine L-channel with edge-aware mask
    l_combined = np.where(edge_mask > 0, l, l_clahe)

    # Merge and convert back to BGR
    lab_result = cv2.merge((l_combined.astype(np.uint8), a, b))
    result = cv2.cvtColor(lab_result, cv2.COLOR_LAB2BGR)

    return result

def process_folder_display_and_save(input_root, output_root):
    input_root = Path(input_root)
    output_root = Path(output_root)

    image_paths = list(input_root.rglob("*.*"))
    image_paths = [p for p in image_paths if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]]

    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to read image: {img_path}")
            continue

        processed = edge_aware_clahe_image(img)

        # Show input and output
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img_rgb)
        plt.title("Original")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(processed_rgb)
        plt.title("Edge-Aware CLAHE")
        plt.axis('off')

        plt.suptitle(str(img_path.name), fontsize=12)
        plt.show()

        # Create corresponding output path and save
        rel_path = img_path.relative_to(input_root)
        out_path = output_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), processed)
        print(f"Saved: {out_path}")

# === USAGE ===
input_main_folder = "/content/drive/MyDrive/Colab Notebooks/Plant Dieases/Test/Test"
output_main_folder = "/content/drive/MyDrive/Colab Notebooks/Plant Dieases/Test/OutputImages"
process_folder_display_and_save(input_main_folder, output_main_folder)
