import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Downloads the PlantDoc dataset from HuggingFace (agyaatcoder/PlantDoc).
    Contains ~2,598 images across 27 classes of real-world plant disease photos.
    """
    dest_dir = Path(os.getenv("PLANTDOC_DIR", "data/raw/plantdoc"))
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing_images = list(dest_dir.rglob("*.jpg")) + list(dest_dir.rglob("*.jpeg")) + list(dest_dir.rglob("*.png"))
    if len(existing_images) > 100:
        logging.info(f"PlantDoc already downloaded ({len(existing_images)} images found in {dest_dir}). Skipping.")
        return

    logging.info(f"Downloading PlantDoc dataset to {dest_dir}")
    logging.info("Source: HuggingFace (agyaatcoder/PlantDoc)")

    try:
        from datasets import load_dataset
    except ImportError:
        logging.error("'datasets' library not found. Install it: pip install datasets")
        sys.exit(1)

    hf_token = os.getenv("HF_TOKEN")

    logging.info("Loading dataset from HuggingFace Hub (this may take a few minutes)...")
    dataset = load_dataset("agyaatcoder/PlantDoc", token=hf_token)

    # Detect label feature
    first_split = list(dataset.keys())[0]
    features = dataset[first_split].features

    # PlantDoc may have 'label' as ClassLabel or as a string field
    label_key = None
    label_names = None
    for key in ["label", "labels", "class"]:
        if key in features:
            label_key = key
            if hasattr(features[key], "names"):
                label_names = features[key].names
            break

    if label_key is None:
        logging.error(f"Could not find label column. Available features: {list(features.keys())}")
        sys.exit(1)

    total_saved = 0
    for split_name in dataset:
        split = dataset[split_name]
        logging.info(f"Processing split '{split_name}' with {len(split)} samples...")

        for i, sample in enumerate(split):
            image = sample["image"]
            label_val = sample[label_key]

            # Resolve label name
            if label_names is not None and isinstance(label_val, int):
                label_name = label_names[label_val]
            else:
                label_name = str(label_val)

            # Create class directory
            class_dir_name = label_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
            class_dir = dest_dir / class_dir_name
            class_dir.mkdir(parents=True, exist_ok=True)

            # Save image
            img_filename = f"{split_name}_{i:06d}.jpg"
            img_path = class_dir / img_filename

            if not img_path.exists():
                # Convert to RGB if needed (some images may be RGBA)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image.save(img_path, "JPEG", quality=95)

            total_saved += 1
            if total_saved % 500 == 0:
                logging.info(f"  Saved {total_saved} images so far...")

    logging.info(f"Download complete. Total images saved: {total_saved}")

    # Count images per class
    for class_dir in sorted(dest_dir.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*.*")))
            logging.info(f"  {class_dir.name}: {count} images")

if __name__ == "__main__":
    main()
