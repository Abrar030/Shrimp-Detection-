import os
import shutil
from sklearn.model_selection import train_test_split

SOURCE_IMAGES = "data/raw_images"  

TRAIN_DIR = "data/images/train"
VAL_DIR = "data/images/val"
TEST_DIR = "data/images/test"

TEST_SIZE = 0.2
VAL_SIZE = 0.5  

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

all_images = [f for f in os.listdir(SOURCE_IMAGES) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

train_files, temp_files = train_test_split(all_images, test_size=TEST_SIZE, random_state=42)

val_files, test_files = train_test_split(temp_files, test_size=VAL_SIZE, random_state=42)

def copy_files(files, dest_dir):
    for f in files:
        src = os.path.join(SOURCE_IMAGES, f)
        dst = os.path.join(dest_dir, f)
        shutil.copy(src, dst)

copy_files(train_files, TRAIN_DIR)
copy_files(val_files, VAL_DIR)
copy_files(test_files, TEST_DIR)

print(f"Split completed!\n"
      f"- Training images: {len(train_files)}\n"
      f"- Validation images: {len(val_files)}\n"
      f"- Test images: {len(test_files)}")