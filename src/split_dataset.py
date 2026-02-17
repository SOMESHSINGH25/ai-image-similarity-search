import os
import random
import shutil

SRC_DIR    = "../data/raw/cifar10_images"
OUT_DIR    = "../data/raw"
TRAIN_RATIO = 0.8   # 80% train, 20% test

random.seed(42)

def main():
    # Confirm source exists
    if not os.path.exists(SRC_DIR):
        print(f"❌ Source folder not found: {SRC_DIR}")
        print("   Please run download_dataset.py first.")
        return

    train_dir = os.path.join(OUT_DIR, "train")
    test_dir  = os.path.join(OUT_DIR, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir,  exist_ok=True)

    print(f"Reading from : {SRC_DIR}")
    print(f"Train output : {train_dir}")
    print(f"Test  output : {test_dir}")
    print("-" * 40)

    total_train = 0
    total_test  = 0

    for cls in sorted(os.listdir(SRC_DIR)):
        cls_path = os.path.join(SRC_DIR, cls)
        if not os.path.isdir(cls_path):
            continue

        images = os.listdir(cls_path)
        random.shuffle(images)

        split_idx  = int(len(images) * TRAIN_RATIO)
        train_imgs = images[:split_idx]
        test_imgs  = images[split_idx:]

        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(test_dir,  cls), exist_ok=True)

        for img in train_imgs:
            shutil.copy(
                os.path.join(cls_path, img),
                os.path.join(train_dir, cls, img)
            )

        for img in test_imgs:
            shutil.copy(
                os.path.join(cls_path, img),
                os.path.join(test_dir, cls, img)
            )

        print(f"Class {cls}: {len(train_imgs)} train, {len(test_imgs)} test")
        total_train += len(train_imgs)
        total_test  += len(test_imgs)

    print("-" * 40)
    print(f"Total train images : {total_train}")
    print(f"Total test  images : {total_test}")
    print(f"Total images       : {total_train + total_test}")
    print("\n✅ Dataset split completed!")

if __name__ == "__main__":
    main()