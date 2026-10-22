import json
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

model = tf.keras.models.load_model("brain_tumor_model.h5")

with open("evaluation_results.json") as f:
    threshold = float(json.load(f).get("threshold", 0.5))

img = image.load_img("test_image.jpg", target_size=(150,150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

prediction = model.predict(img_array)

if prediction[0][0] > threshold:
    print(f"Tumor detected (prob={prediction[0][0]:.3f}, threshold={threshold:.2f})")
else:
    print(f"No tumor (prob={prediction[0][0]:.3f}, threshold={threshold:.2f})")
