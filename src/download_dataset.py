import tensorflow as tf
import os
import numpy as np
from PIL import Image
from tqdm import tqdm

OUTPUT_DIR = "../data/raw/cifar10_images"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Downloading CIFAR-10...")
    (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()

    print("Saving images to disk...")
    count = 0
    for i in tqdm(range(len(x_train))):
        img = x_train[i]
        label = int(y_train[i][0])

        class_dir = os.path.join(OUTPUT_DIR, str(label))
        os.makedirs(class_dir, exist_ok=True)

        img_pil = Image.fromarray(img)
        img_pil.save(os.path.join(class_dir, f"{count}.png"))

        count += 1

        if count >= 2000:   # We take 2000 images (more than required 1000)
            break

    print("Done. Images saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()
