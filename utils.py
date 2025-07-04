import os
import numpy as np
import cv2
from tensorflow.keras.utils import Sequence

class ImageMaskGenerator(Sequence):
    def __init__(self, image_dir, mask_dir, batch_size=8, image_size=(128, 128), file_list=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_filenames = file_list or os.listdir(image_dir)

    def __len__(self):
        return len(self.image_filenames) // self.batch_size

    def __getitem__(self, idx):
        batch_filenames = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        images = []
        masks = []

        for filename in batch_filenames:
            img = cv2.imread(os.path.join(self.image_dir, filename))
            img = cv2.resize(img, self.image_size)
            img = img / 255.0

            mask = cv2.imread(os.path.join(self.mask_dir, filename), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, self.image_size)
            mask = np.expand_dims(mask, axis=-1) / 255.0

            images.append(img)
            masks.append(mask)

        return np.array(images), np.array(masks)