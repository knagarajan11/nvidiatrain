import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Downloads the PlantVillage dataset from HuggingFace (mohanty/PlantVillage).
    Contains ~54,000 images across 38 classes of healthy and diseased plant leaves.
    """
    dest_dir = Path(os.getenv("PLANTVILLAGE_DIR", "data/raw/plantvillage"))
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing_images = list(dest_dir.rglob("*.jpg")) + list(dest_dir.rglob("*.jpeg")) + list(dest_dir.rglob("*.png"))
    if len(existing_images) > 100:
        logging.info(f"PlantVillage already downloaded ({len(existing_images)} images found in {dest_dir}). Skipping.")
        return

    logging.info(f"Downloading PlantVillage dataset to {dest_dir}")
    logging.info("Source: HuggingFace (mohanty/PlantVillage, config=color)")

    try:
        from datasets import load_dataset
    except ImportError:
        logging.error("'datasets' library not found. Install it: pip install datasets")
        sys.exit(1)

    # Set HF token if available
    hf_token = os.getenv("HF_TOKEN")

    logging.info("Loading dataset from HuggingFace Hub (this may take several minutes)...")

    # Clear any stale HuggingFace cache for this dataset to avoid builder config errors
    import shutil
    cache_dir = Path.home() / ".cache" / "huggingface" / "datasets" / "mohanty___plant_village"
    if cache_dir.exists():
        logging.info(f"Clearing stale HuggingFace cache at {cache_dir}...")
        shutil.rmtree(cache_dir, ignore_errors=True)

    dataset = load_dataset("mohanty/PlantVillage", name="default", trust_remote_code=True, token=hf_token)

    # Auto-detect the label column and image column
    first_split = list(dataset.keys())[0]
    features = dataset[first_split].features
    logging.info(f"Dataset features: {list(features.keys())}")

    # Find label column
    label_key = None
    label_names = None
    for key in ["label", "labels", "class", "category", "disease"]:
        if key in features:
            label_key = key
            if hasattr(features[key], "names"):
                label_names = features[key].names
            break

    if label_key is None:
        logging.error(f"Could not find label column. Available features: {list(features.keys())}")
        sys.exit(1)

    logging.info(f"Using label column: '{label_key}' with {len(label_names) if label_names else '?'} classes")

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

            # Create class directory (sanitize name for filesystem)
            class_dir_name = label_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
            class_dir = dest_dir / class_dir_name
            class_dir.mkdir(parents=True, exist_ok=True)

            # Save image
            img_filename = f"{split_name}_{i:06d}.jpg"
            img_path = class_dir / img_filename

            if not img_path.exists():
                image.save(img_path, "JPEG", quality=95)

            total_saved += 1
            if total_saved % 1000 == 0:
                logging.info(f"  Saved {total_saved} images so far...")

    logging.info(f"Download complete. Total images saved: {total_saved}")
    logging.info(f"Number of classes: {len(label_names)}")

    # Count images per class
    for class_dir in sorted(dest_dir.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*.*")))
            logging.info(f"  {class_dir.name}: {count} images")

if __name__ == "__main__":
    main()
