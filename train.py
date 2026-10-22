import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import classification_report, confusion_matrix

# dataset path
data_dir = "dataset/train"

# ── Augmentation on train (fixes overfitting) ──────────────────────
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
)

# No augmentation on validation — only rescale
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_data = train_datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
    subset="training",
    seed=42,
)

val_data = val_datagen.flow_from_directory(
    data_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode="binary",
    subset="validation",
    seed=42,
    shuffle=False,
)

# ── Improved CNN: GlobalAveragePooling + Dropout (cuts params 48x) ─
model = models.Sequential([
    layers.Input(shape=(150, 150, 3)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.GlobalAveragePooling2D(),   # replaces Flatten → ~100k params total
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.5),               # regularisation
    layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

print(f"Model params: {model.count_params():,} (was 4,828,481)")

# ── Early stopping to prevent overfitting ─────────────────────────
es = callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,
    callbacks=[es],
    verbose=2,
)

# ── Save training history ─────────────────────────────────────────
hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
hist["config"] = {
    "epochs_run": len(hist["loss"]),
    "epochs_max": 30,
    "batch_size": 32,
    "optimizer": "adam",
    "loss": "binary_crossentropy",
    "input_size": [150, 150],
    "train_images": train_data.samples,
    "val_images": val_data.samples,
    "class_indices": train_data.class_indices,
    "model_params": int(model.count_params()),
    "augmentation": True,
    "dropout": 0.5,
    "early_stopping": True,
}
with open("training_history.json", "w") as f:
    json.dump(hist, f, indent=2)

# ── Threshold tuning: maximize specificity while sensitivity ≥ 90% ─
val_data.reset()
probs = model.predict(val_data, verbose=0).ravel()
true_labels = val_data.classes

best_t, best_spec, best_sens = 0.5, 0.0, 0.0
for t in np.arange(0.05, 0.95, 0.01):
    preds = (probs > t).astype(int)
    tn, fp, fn, tp = confusion_matrix(true_labels, preds).ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    if sens >= 0.90 and spec > best_spec:
        best_t, best_spec, best_sens = float(t), spec, sens

print(f"Best threshold: {best_t:.2f} (sens={best_sens:.2%}, spec={best_spec:.2%})")

# ── Evaluation at tuned threshold ─────────────────────────────────
final_preds = (probs > best_t).astype(int)
report = classification_report(
    true_labels, final_preds,
    target_names=list(val_data.class_indices.keys()),
    output_dict=True,
)
cm = confusion_matrix(true_labels, final_preds)

eval_results = {
    "threshold": best_t,
    "sensitivity": best_sens,
    "specificity": best_spec,
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
print(classification_report(true_labels, final_preds,
      target_names=list(val_data.class_indices.keys())))
print("Confusion matrix:\n", cm)
