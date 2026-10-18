import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, confusion_matrix

# dataset path
data_dir = "dataset/train"

# image preprocessing
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary',
    subset='training',
    seed=42
)

val_data = datagen.flow_from_directory(
    data_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    seed=42
)

# CNN Model
model = models.Sequential([
    layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128,(3,3),activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dense(1,activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

# ---- Save training history ----
hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
hist["config"] = {
    "epochs": 5,
    "batch_size": 32,
    "optimizer": "adam",
    "loss": "binary_crossentropy",
    "input_size": [150, 150],
    "train_images": train_data.samples,
    "val_images": val_data.samples,
    "class_indices": train_data.class_indices,
    "model_params": int(model.count_params()),
}
with open("training_history.json", "w") as f:
    json.dump(hist, f, indent=2)

# ---- Test / evaluation metrics ----
val_data.reset()
preds = model.predict(val_data)
pred_labels = (preds > 0.5).astype(int).ravel()
true_labels = val_data.classes

report = classification_report(
    true_labels, pred_labels,
    target_names=list(val_data.class_indices.keys()),
    output_dict=True
)
cm = confusion_matrix(true_labels, pred_labels)

eval_results = {
    "classification_report": report,
    "confusion_matrix": cm.tolist(),
    "val_loss": float(hist["val_loss"][-1]),
    "val_accuracy": float(hist["val_accuracy"][-1]),
}
with open("evaluation_results.json", "w") as f:
    json.dump(eval_results, f, indent=2)

model.save("brain_tumor_model.h5")

print("Model trained and saved!")
print("History saved to training_history.json")
print("Evaluation saved to evaluation_results.json")
print(classification_report(true_labels, pred_labels,
      target_names=list(val_data.class_indices.keys())))
