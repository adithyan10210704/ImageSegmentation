from unet_model import unet
from utils import ImageMaskGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import os

# Define paths
image_dir = 'train_images'
mask_dir = 'train_masks'

all_filenames = os.listdir(image_dir)
train_files, val_files = train_test_split(all_filenames, test_size=0.1, random_state=42)

# Data generator
train_gen = ImageMaskGenerator(image_dir, mask_dir, batch_size=4, image_size=(128, 128), file_list=train_files)
val_gen = ImageMaskGenerator(image_dir, mask_dir, batch_size=4, image_size=(128, 128), file_list=val_files)

# Model
model = unet(input_shape=(128, 128, 3))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Save best model
checkpoint = ModelCheckpoint("unet_best.keras", save_best_only=True)

# Train
model.fit(train_gen, validation_data=val_gen, epochs=10, callbacks=[checkpoint])