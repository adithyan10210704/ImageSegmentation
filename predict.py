import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load model
model = load_model("unet_best.keras")
# Load and preprocess input
img = cv2.imread("train_images/1100.png")  # replace with your image
img = cv2.resize(img, (128, 128))
input_img = img / 255.0
input_img = np.expand_dims(input_img, axis=0)

# Predict
pred_mask = model.predict(input_img)[0]
pred_mask = (pred_mask > 0.5).astype(np.uint8)

# Show results
plt.subplot(1, 2, 1)
plt.title("Image")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

plt.subplot(1, 2, 2)
plt.title("Prediction")
plt.imshow(pred_mask.squeeze(), cmap='gray')
plt.show()