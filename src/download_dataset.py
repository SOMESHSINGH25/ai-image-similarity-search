import tensorflow as tf
import os
import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

OUTPUT_DIR = "../data/raw/cifar10_images"
NUM_IMAGES = 2000   # Meets 1000+ requirement safely


def sharpen_image(img_pil):
    """
    Apply sharpening to compensate for CIFAR-10's low 32x32 resolution.
    UnsharpMask is more controlled than a basic SHARPEN filter:
      - radius: how far out to look for edges
      - percent: strength of sharpening
      - threshold: how different a pixel must be before it's sharpened
    """
    img_pil = img_pil.filter(ImageFilter.UnsharpMask(radius=1, percent=120, threshold=2))
    return img_pil


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Downloading CIFAR-10 via Keras...")
    (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()

    print(f"Saving {NUM_IMAGES} images to disk with sharpening applied...")
    count = 0

    for i in tqdm(range(len(x_train))):
        img_array = x_train[i]          # shape: (32, 32, 3), dtype: uint8
        label = int(y_train[i][0])

        class_dir = os.path.join(OUTPUT_DIR, str(label))
        os.makedirs(class_dir, exist_ok=True)

        img_pil = Image.fromarray(img_array)    # still 32x32 here
        img_pil = sharpen_image(img_pil)        # sharpen before saving

        img_pil.save(os.path.join(class_dir, f"{count}.png"))
        count += 1

        if count >= NUM_IMAGES:
            break

    print(f"\n✅ Done. {count} images saved to: {OUTPUT_DIR}")
    print("Note: Images saved at native 32x32 with sharpening.")
    print("Resizing to 64x64 happens during training and embedding extraction.")


if __name__ == "__main__":
    main()