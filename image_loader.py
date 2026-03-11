import os
import numpy as np
import pandas as pd
import medmnist
from medmnist import INFO
from torchvision import transforms
from PIL import Image

DATASETS = [
    "pathmnist",
    "octmnist",
    "tissuemnist"
]

def load_dataset(data_flag):
    info = INFO[data_flag]
    DataClass = getattr(medmnist, info["python_class"])

    train = DataClass(split="train", download=True)
    val   = DataClass(split="val",   download=True)
    test  = DataClass(split="test",  download=True)

    return train, val, test


def build_image_pool(train, val, test):
    return np.concatenate([
        train.imgs,   
        val.imgs,    
        test.imgs,    
    ], axis=0)


def save_images_for_split(data_flag, split_name, df, image_pool):
    out_dir = f"datasets/images/{data_flag}/{split_name}"
    os.makedirs(out_dir, exist_ok=True)

    for _, row in df.iterrows():
        img_idx = int(row["image_index"])

        # Retrieve the image from the pool using image_index
        img_array = image_pool[img_idx]  

        # Convert to PIL image
        if img_array.ndim == 2:
            # Grayscale (e.g. tissuemnist, octmnist)
            img = Image.fromarray(img_array.astype(np.uint8), mode="L")
        else:
            # RGB (e.g. pathmnist)
            img = Image.fromarray(img_array.astype(np.uint8), mode="RGB")

        # Save as {image_index}.png so it directly maps back to the CSV
        img.save(f"{out_dir}/{img_idx}.png")

    print(f"  Saved {len(df)} images → {out_dir}")


def process_dataset(data_flag):
    print(f"\nProcessing {data_flag}")

    # Load raw datasets
    train, val, test = load_dataset(data_flag)

    # Build full image pool in same order as create_full_dataframe()
    image_pool = build_image_pool(train, val, test)
    print(f"  Image pool shape: {image_pool.shape}")

    # Load each split CSV and save its images
    splits = ["calibration", "evaluation", "train", "val", "test"]

    for split_name in splits:
        csv_path = f"datasets/{data_flag}_{split_name}_labels.csv"

        if not os.path.exists(csv_path):
            print(f" CSV not found, skipping: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        save_images_for_split(data_flag, split_name, df, image_pool)


def main():
    for data_flag in DATASETS:
        process_dataset(data_flag)


if __name__ == "__main__":
    main()
