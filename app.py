import json
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from gradcam import grad_cam, overlay_heatmap, tumor_region, region_size

st.set_page_config(
    page_title="Brain Tumor Detection AI",
    page_icon="🧠",
    layout="wide",
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("brain_tumor_model.h5")

model = load_model()

st.title("🧠 Brain Tumor Detection AI")
st.markdown(
    "Upload a brain MRI scan. The CNN model analyzes it and highlights the "
    "region (via Grad-CAM) that most influenced the prediction."
)

uploaded_file = st.file_uploader(
    "Upload MRI Image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    original = np.array(img)
    resized = img.resize((150, 150))
    arr = np.array(resized)
    x = np.expand_dims(arr / 255.0, axis=0)

    heatmap, tumor_prob = grad_cam(model, x)
    tumor_prob = float(model.predict(x, verbose=0)[0][0])
    no_tumor_prob = 1.0 - tumor_prob

    if tumor_prob > 0.5:
        label = "Tumor Detected"
        confidence = tumor_prob * 100
        color = "error"
    else:
        label = "No Tumor Detected"
        confidence = no_tumor_prob * 100
        color = "success"

    st.subheader("Scan Result")
    col1, col2 = st.columns(2)

    with col1:
        st.image(original, caption="Uploaded MRI", use_container_width=True)

    with col2:
        overlay = overlay_heatmap(original, heatmap)
        st.image(
            overlay,
            caption="Grad-CAM Heatmap — warm areas (red/yellow) drove the decision",
            use_container_width=True,
        )

    if color == "error":
        st.error(f"### {label} — {confidence:.1f}% confidence")
    else:
        st.success(f"### {label} — {confidence:.1f}% confidence")

    st.progress(min(int(confidence), 100) / 100.0)

    st.subheader("Why this reading?")
    st.markdown(
        f"""
        | Component | Value |
        |---|---|
        | Tumor probability | **{tumor_prob * 100:.1f}%** |
        | No-tumor probability | **{no_tumor_prob * 100:.1f}%** |
        | Decision threshold | 50% |
        | Final prediction | **{label}** |
        """
    )

    region = tumor_region(heatmap, threshold=0.3)
    area_frac = region_size(heatmap, threshold=0.3)

    if region is not None and area_frac > 0.01:
        x1, y1, x2, y2 = region
        st.markdown(
            f"""
            **Tumor location (approx.):** bounding box at
            `x={x1}–{x2}, y={y1}–{y2}` (on the 150×150 scan grid).
            The Grad-CAM heatmap shows the CNN focused on
            **{area_frac * 100:.1f}%** of the image — the bright/red zones in the
            overlay — when making this decision. Those zones correspond to
            abnormal tissue patterns (irregular texture, intensity variation)
            the model learned during training.
            """
        )
    else:
        st.markdown(
            """
            **Tumor location:** no concentrated hot region was found on the
            heatmap. The decision was driven by diffuse, low-saliency features
            across the scan rather than a single focused abnormality — which
            is typical when the model predicts **no tumor**.
            """
        )

    with st.expander("How the confidence % is computed", expanded=True):
        st.markdown(
            f"""
            1. The image is resized to **150×150** pixels and normalized to `[0,1]`.
            2. The CNN outputs a raw **sigmoid score** `p` between 0 and 1
               (here **{tumor_prob:.4f}**).
            3. If `p > 0.5` → **Tumor** with confidence `p×100`;
               if `p ≤ 0.5` → **No Tumor** with confidence `(1−p)×100`.
            4. So this scan's **{confidence:.1f}%** means the model's internal
               score landed {abs(tumor_prob - 0.5):.4f} away from the 0.5
               decision boundary — larger distance ⇒ more confident the model is.
            5. The heatmap is produced by **Grad-CAM**: gradients of the output
               w.r.t. the last convolutional layer are pooled and weighted,
               producing a saliency map of which pixels mattered most.
            """
        )

    st.caption(
        "⚠️ Educational / research use only — not a substitute for professional medical diagnosis."
    )

else:
    st.info("Upload an MRI image (JPG / PNG) to run a scan.")
    st.markdown(
        """
        **Pages:**
        - **Upload & Scan** (this page) — prediction, confidence, tumor heatmap
        - **Training & Test Results** — epoch-by-epoch training history,
          validation curves, confusion matrix, and per-epoch analysis
        """
    )
