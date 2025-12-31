import os
import random
import shutil

# -------- CONFIG --------
SOURCE_DIR = "Training & Evaluation/training_data"
TEST_DIR = "testing"
CLASSES = ["Violence", "NonViolence"]
SAMPLES_PER_CLASS = 100
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# -------- CREATE TEST FOLDERS --------
os.makedirs(TEST_DIR, exist_ok=True)

for cls in CLASSES:
    os.makedirs(os.path.join(TEST_DIR, cls), exist_ok=True)

# -------- MOVE FILES --------
for cls in CLASSES:
    class_path = os.path.join(SOURCE_DIR, cls)
    files = os.listdir(class_path)

    if len(files) < SAMPLES_PER_CLASS:
        raise ValueError(f"Not enough videos in {cls} folder")

    selected_files = random.sample(files, SAMPLES_PER_CLASS)

    for file in selected_files:
        src = os.path.join(class_path, file)
        dst = os.path.join(TEST_DIR, cls, file)
        shutil.move(src, dst)

    print(f"Moved {SAMPLES_PER_CLASS} videos from {cls}")

print("\n✅ Testing dataset created successfully!")
