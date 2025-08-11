import cv2
import numpy as np
import random
from pathlib import Path
import os
import itertools
import matplotlib.pyplot as plt

def cutmix(image1, image2, beta=1.0, target_size=(224, 224)):
    image1 = cv2.resize(image1, target_size)
    image2 = cv2.resize(image2, target_size)

    h, w, _ = image1.shape
    lam = np.random.beta(beta, beta)

    cut_rat = np.sqrt(1. - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)

    cx = np.random.randint(w)
    cy = np.random.randint(h)

    x1 = np.clip(cx - cut_w // 2, 0, w)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    y2 = np.clip(cy + cut_h // 2, 0, h)

    mixed = image1.copy()
    mixed[y1:y2, x1:x2] = image2[y1:y2, x1:x2]

    return mixed

def load_images_from_folder(folder_path):
    folder = Path(folder_path)
    image_paths = list(folder.glob("*.*"))  # Only current folder, no recursion here
    images = []
    filenames = []
    for p in image_paths:
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            img = cv2.imread(str(p))
            if img is not None:
                images.append(img)
                filenames.append(p.name)
    return images, filenames

def display_images(img1, img2, mixed, target_size=(224,224)):
    plt.figure(figsize=(12, 4))

    plt.subplot(1,3,1)
    plt.imshow(cv2.cvtColor(cv2.resize(img1, target_size), cv2.COLOR_BGR2RGB))
    plt.title("Image 1")
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.imshow(cv2.cvtColor(cv2.resize(img2, target_size), cv2.COLOR_BGR2RGB))
    plt.title("Image 2")
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.imshow(cv2.cvtColor(mixed, cv2.COLOR_BGR2RGB))
    plt.title("CutMix Output")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def apply_cutmix_folder_wise_display(input_root_folder, output_root_folder, target_size=(224, 224)):
    input_root = Path(input_root_folder)
    output_root = Path(output_root_folder)

    # Find subfolders with images
    subfolders = [f for f in input_root.glob("**/") if any(f.glob("*.*"))]

    total_saved = 0
    for subfolder in subfolders:
        images, filenames = load_images_from_folder(subfolder)
        if len(images) < 2:
            print(f"❌ Not enough images in {subfolder} for CutMix. Need at least 2 images.")
            continue

        relative_subfolder = subfolder.relative_to(input_root)
        output_subfolder = output_root / relative_subfolder
        os.makedirs(output_subfolder, exist_ok=True)

        print(f"Processing folder: {relative_subfolder} with {len(images)} images")

        count = 0
        for i, j in itertools.combinations(range(len(images)), 2):
            img1 = images[i]
            img2 = images[j]
            mixed = cutmix(img1, img2, target_size=target_size)

            save_name = f"cutmix_{filenames[i].split('.')[0]}_{filenames[j].split('.')[0]}.jpg"
            save_path = output_subfolder / save_name

            cv2.imwrite(str(save_path), mixed)
            count += 1
            total_saved += 1

            # Display images
            display_images(img1, img2, mixed, target_size=target_size)

        print(f"✅ Saved {count} augmented images in {output_subfolder}")

    print(f"✅ Total augmented images saved: {total_saved}")

# === USAGE ===
input_folder = "/content/drive/MyDrive/Colab Notebooks/Plant Dieases/Test/OutputImages"
output_folder = "/content/drive/MyDrive/Colab Notebooks/Plant Dieases/Test/AugmentedImages"

apply_cutmix_folder_wise_display(input_folder, output_folder, target_size=(224, 224))
