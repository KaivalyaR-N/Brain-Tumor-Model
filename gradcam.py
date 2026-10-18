import numpy as np
import tensorflow as tf
import cv2


def get_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found in model")


def grad_cam(model, img_array, last_conv_layer_name=None):
    """Compute Grad-CAM heatmap for a single preprocessed image (1,H,W,3)."""
    if last_conv_layer_name is None:
        last_conv_layer_name = get_last_conv_layer(model)

    # Keras 3: rebuild a Functional clone so gradients flow through conv layers
    if not isinstance(model, tf.keras.Model):
        raise ValueError("model must be a tf.keras.Model")
    if not model.built:
        model.build(input_shape=(None,) + tuple(img_array.shape[1:]))

    inputs = tf.keras.Input(shape=tuple(img_array.shape[1:]))
    x = inputs
    last_conv_output = None
    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_layer_name:
            last_conv_output = x
    outputs = x

    grad_model = tf.keras.Model(inputs=inputs, outputs=[last_conv_output, outputs])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        tape.watch(conv_outputs)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise RuntimeError("Gradient is None; Grad-CAM failed on this model.")
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), float(predictions[0][0])


def overlay_heatmap(original_img, heatmap, alpha=0.45, colormap=cv2.COLORMAP_JET):
    """Overlay heatmap on the original image. original_img: uint8 RGB (H,W,3)."""
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_resized = np.uint8(255 * heatmap_resized)
    colored = cv2.applyColorMap(heatmap_resized, colormap)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original_img, 1 - alpha, colored, alpha, 0)
    return overlay


def tumor_region(heatmap, threshold=0.5):
    """Bounding box (x1,y1,x2,y2) of the region above threshold, or None."""
    mask = heatmap >= threshold
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def region_size(heatmap, threshold=0.5):
    """Fraction of heatmap area (0-1) above threshold."""
    mask = heatmap >= threshold
    return float(mask.mean())
