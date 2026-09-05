import os
import sys
import logging
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Downloads a rice disease dataset from HuggingFace.
    Uses 'masoudnickparvar/brain-tumor-mri-dataset' style approach to find a suitable
    rice leaf disease dataset. We use Kaggle's rice-diseases-image-dataset via kagglehub
    if available, otherwise fall back to a HuggingFace mirror.
    """
    dest_dir = Path(os.getenv("RICE1426_DIR", "data/raw/rice1426"))
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    existing_images = list(dest_dir.rglob("*.jpg")) + list(dest_dir.rglob("*.jpeg")) + list(dest_dir.rglob("*.png"))
    if len(existing_images) > 100:
        logging.info(f"Rice dataset already downloaded ({len(existing_images)} images found in {dest_dir}). Skipping.")
        return

    logging.info(f"Downloading Rice disease dataset to {dest_dir}")

    # Try kagglehub first (most reliable for rice disease datasets)
    downloaded = False
    try:
        import kagglehub
        logging.info("Using kagglehub to download rice disease dataset from Kaggle...")
        logging.info("Source: Kaggle (teddevriern/rice-leaf-diseases)")
        path = kagglehub.dataset_download("teddevriern/rice-leaf-diseases")
        logging.info(f"Kaggle download complete to: {path}")

        # Copy images from kaggle cache to our dest_dir
        import shutil
        src = Path(path)
        total_copied = 0
        for class_dir in src.rglob("*"):
            if class_dir.is_dir():
                # Only copy directories that contain images
                images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.jpeg")) + list(class_dir.glob("*.png")) + list(class_dir.glob("*.JPG"))
                if images:
                    # Use the directory name as the class name
                    target_class_dir = dest_dir / class_dir.name.replace(" ", "_")
                    target_class_dir.mkdir(parents=True, exist_ok=True)
                    for img in images:
                        target = target_class_dir / img.name
                        if not target.exists():
                            shutil.copy2(img, target)
                            total_copied += 1

        if total_copied > 0:
            logging.info(f"Copied {total_copied} rice disease images to {dest_dir}")
            downloaded = True
        else:
            logging.warning("Kaggle download succeeded but no images found in expected structure.")

    except ImportError:
        logging.info("kagglehub not installed. Trying HuggingFace fallback...")
    except Exception as e:
        logging.warning(f"Kaggle download failed: {e}. Trying HuggingFace fallback...")

    # Fallback: Try HuggingFace datasets
    if not downloaded:
        try:
            from datasets import load_dataset
            hf_token = os.getenv("HF_TOKEN")

            logging.info("Loading rice disease dataset from HuggingFace Hub...")
            # Try common rice disease datasets on HuggingFace
            dataset_candidates = [
                "Prottoy001/Rice_Disease_Dataset",
                "smaranjitghose/rice-disease-dataset",
            ]

            dataset = None
            for candidate in dataset_candidates:
                try:
                    logging.info(f"Trying HuggingFace dataset: {candidate}")
                    dataset = load_dataset(candidate, token=hf_token)
                    logging.info(f"Successfully loaded: {candidate}")
                    break
                except Exception as e:
                    logging.warning(f"  Failed to load {candidate}: {e}")
                    continue

            if dataset is None:
                logging.error("Could not download rice disease dataset from any source.")
                logging.error("Please manually download a rice disease dataset and place images in:")
                logging.error(f"  {dest_dir}/<class_name>/<image_files>")
                sys.exit(1)

            # Save images
            first_split = list(dataset.keys())[0]
            features = dataset[first_split].features

            # Find label column
            label_key = None
            label_names = None
            for key in ["label", "labels", "class"]:
                if key in features:
                    label_key = key
                    if hasattr(features[key], "names"):
                        label_names = features[key].names
                    break

            total_saved = 0
            for split_name in dataset:
                split = dataset[split_name]
                logging.info(f"Processing split '{split_name}' with {len(split)} samples...")

                for i, sample in enumerate(split):
                    image = sample["image"]

                    if label_key and label_names and isinstance(sample[label_key], int):
                        label_name = label_names[sample[label_key]]
                    elif label_key:
                        label_name = str(sample[label_key])
                    else:
                        label_name = "unknown"

                    class_dir_name = label_name.replace(" ", "_").replace("/", "_").replace("\\", "_")
                    class_dir = dest_dir / class_dir_name
                    class_dir.mkdir(parents=True, exist_ok=True)

                    img_filename = f"{split_name}_{i:06d}.jpg"
                    img_path = class_dir / img_filename

                    if not img_path.exists():
                        if image.mode != "RGB":
                            image = image.convert("RGB")
                        image.save(img_path, "JPEG", quality=95)

                    total_saved += 1
                    if total_saved % 500 == 0:
                        logging.info(f"  Saved {total_saved} images so far...")

            logging.info(f"Download complete. Total images saved: {total_saved}")

        except ImportError:
            logging.error("'datasets' library not found. Install it: pip install datasets")
            sys.exit(1)

    # Count images per class
    for class_dir in sorted(dest_dir.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*.*")))
            logging.info(f"  {class_dir.name}: {count} images")

if __name__ == "__main__":
    main()
