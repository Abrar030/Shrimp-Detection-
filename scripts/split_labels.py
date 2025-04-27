import os
import shutil
from sklearn.model_selection import train_test_split

SOURCE_LABELS = "data/labels"
TRAIN_DIR = "data/labels/train"
VAL_DIR = "data/labels/val"
TEST_DIR = "data/labels/test"

shutil.rmtree(TRAIN_DIR, ignore_errors=True)  
shutil.rmtree(VAL_DIR, ignore_errors=True)
shutil.rmtree(TEST_DIR, ignore_errors=True)
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)


all_labels = [f for f in os.listdir(SOURCE_LABELS) if f.endswith(".txt")]

train_labels, temp_labels = train_test_split(all_labels, test_size=0.2, random_state=42)
val_labels, test_labels = train_test_split(temp_labels, test_size=0.5, random_state=42)

def copy_labels(files, dest_dir):
    for f in files:
        src = os.path.join(SOURCE_LABELS, f)
        dst = os.path.join(dest_dir, f)
        if os.path.isfile(src):  
            shutil.copy(src, dst)

copy_labels(train_labels, TRAIN_DIR)
copy_labels(val_labels, VAL_DIR)
copy_labels(test_labels, TEST_DIR)

print("Labels split successfully!")