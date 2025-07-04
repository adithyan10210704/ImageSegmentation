import os
import shutil

# Source folders
src_images = "/Users/adithyans/Downloads/archive/Chest-X-Ray/Chest-X-Ray/image"
src_masks = "/Users/adithyans/Downloads/archive/Chest-X-Ray/Chest-X-Ray/mask"

# Destination folders
dst_images = "train_images"
dst_masks = "train_masks"

# Make sure destination folders exist
os.makedirs(dst_images, exist_ok=True)
os.makedirs(dst_masks, exist_ok=True)

# Move image files
for filename in os.listdir(src_images):
    src_path = os.path.join(src_images, filename)
    dst_path = os.path.join(dst_images, filename)
    if os.path.isfile(src_path):
        shutil.copy(src_path, dst_path)

# Move mask files
for filename in os.listdir(src_masks):
    src_path = os.path.join(src_masks, filename)
    dst_path = os.path.join(dst_masks, filename)
    if os.path.isfile(src_path):
        shutil.copy(src_path, dst_path)

print("✅ All images and masks have been moved!")