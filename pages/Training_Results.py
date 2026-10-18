import json
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Training & Test Results", page_icon="📊", layout="wide")
st.title("📊 Training & Testing Results")
st.markdown("Detailed epoch-by-epoch breakdown of how the CNN learned on the brain MRI dataset.")

try:
    with open("training_history.json") as f:
        hist = json.load(f)
    with open("evaluation_results.json") as f:
        eval_res = json.load(f)
except FileNotFoundError:
    st.error("training_history.json / evaluation_results.json not found. Run `python train.py` first.")
    st.stop()

cfg = hist.get("config", {})
epochs = list(range(1, len(hist["accuracy"]) + 1))

# ---------- Summary metrics ----------
st.subheader("Run Summary")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Epochs", cfg.get("epochs", len(epochs)))
c2.metric("Train images", cfg.get("train_images", "—"))
c3.metric("Val images", cfg.get("val_images", "—"))
c4.metric("Model params", f"{cfg.get('model_params', 0):,}")
c5.metric("Final Val Accuracy", f"{hist['val_accuracy'][-1] * 100:.1f}%")

st.caption(
    f"Optimizer: {cfg.get('optimizer')} | Loss: {cfg.get('loss')} | "
    f"Batch size: {cfg.get('batch_size')} | Input: {cfg.get('input_size')}"
)

# ---------- Epoch table ----------
st.subheader("Epoch-by-Epoch Metrics")
df = pd.DataFrame({
    "Epoch": epochs,
    "Train Accuracy": [f"{a * 100:.2f}%" for a in hist["accuracy"]],
    "Train Loss": [f"{l:.4f}" for l in hist["loss"]],
    "Val Accuracy": [f"{a * 100:.2f}%" for a in hist["val_accuracy"]],
    "Val Loss": [f"{l:.4f}" for l in hist["val_loss"]],
    "Δ Train Acc": ["—"] + [f"{(hist['accuracy'][i] - hist['accuracy'][i-1]) * 100:+.2f}%" for i in range(1, len(epochs))],
    "Δ Val Acc": ["—"] + [f"{(hist['val_accuracy'][i] - hist['val_accuracy'][i-1]) * 100:+.2f}%" for i in range(1, len(epochs))],
    "Δ Val Loss": ["—"] + [f"{hist['val_loss'][i] - hist['val_loss'][i-1]:+.4f}" for i in range(1, len(epochs))],
})
st.dataframe(df, use_container_width=True, hide_index=True)

# ---------- Curves ----------
st.subheader("Training Curves")
tab1, tab2 = st.tabs(["Accuracy", "Loss"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=hist["accuracy"], mode="lines+markers", name="Train Accuracy"))
    fig.add_trace(go.Scatter(x=epochs, y=hist["val_accuracy"], mode="lines+markers", name="Val Accuracy"))
    fig.update_layout(xaxis_title="Epoch", yaxis_title="Accuracy", yaxis_tickformat=".0%", height=400)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=hist["loss"], mode="lines+markers", name="Train Loss"))
    fig.add_trace(go.Scatter(x=epochs, y=hist["val_loss"], mode="lines+markers", name="Val Loss"))
    fig.update_layout(xaxis_title="Epoch", yaxis_title="Loss", height=400)
    st.plotly_chart(fig, use_container_width=True)

# ---------- Detailed per-epoch explanation ----------
st.subheader("Detailed Epoch Analysis — what changed and why")

def analyze_epoch(i):
    lines = []
    ta, tl = hist["accuracy"][i], hist["loss"][i]
    va, vl = hist["val_accuracy"][i], hist["val_loss"][i]
    if i > 0:
        d_ta = (ta - hist["accuracy"][i-1]) * 100
        d_va = (va - hist["val_accuracy"][i-1]) * 100
        d_vl = vl - hist["val_loss"][i-1]
        lines.append(
            f"Train accuracy moved **{d_ta:+.2f}%** "
            f"({hist['accuracy'][i-1]*100:.2f}% → {ta*100:.2f}%), "
            f"val accuracy **{d_va:+.2f}%**, val loss **{d_vl:+.4f}**."
        )
        if d_ta > 1 and d_va <= 0:
            lines.append(
                "⚠️ **Gap widening:** training improved but validation did not — "
                "the model is starting to memorise training images rather than "
                "generalise. Common with small datasets (~200 images) and no augmentation."
            )
        if d_vl < 0:
            lines.append(
                "✅ Validation loss **decreased**, so the model still extracts "
                "real generalisable signal at this epoch."
            )
        else:
            lines.append(
                "⚠️ Validation loss **increased** — predictions on unseen data "
                "got worse even if training loss fell."
            )
    else:
        lines.append(
            f"**First epoch.** The CNN starts with random weights; after one pass "
            f"over ~{cfg.get('train_images', 203)} images it already reaches "
            f"**{ta*100:.2f}%** train / **{va*100:.2f}%** val accuracy. "
            "Early filters learn edges, intensity blobs and texture patterns "
            "typical of MRI scans."
        )
    lines.append(
        f"Final numbers this epoch — train loss **{tl:.4f}**, val loss **{vl:.4f}**, "
        f"generalisation gap (val − train acc) **{(va - ta)*100:+.2f}%**."
    )
    return "\n\n".join(lines)

for i, ep in enumerate(epochs):
    with st.expander(
        f"Epoch {ep} — train {hist['accuracy'][i]*100:.2f}% / "
        f"val {hist['val_accuracy'][i]*100:.2f}% (val loss {hist['val_loss'][i]:.4f})",
        expanded=(i == len(epochs) - 1),
    ):
        st.markdown(analyze_epoch(i))

# ---------- Overall diagnosis ----------
st.subheader("Overall Training Diagnosis")
gap = (hist["val_accuracy"][-1] - hist["accuracy"][-1]) * 100
st.markdown(
    f"""
    - Train accuracy climbed **{hist['accuracy'][0]*100:.1f}% → {hist['accuracy'][-1]*100:.1f}%** over {len(epochs)} epochs
      (loss {hist['loss'][0]:.4f} → {hist['loss'][-1]:.4f}).
    - Validation accuracy stayed roughly flat at **{min(hist['val_accuracy'])*100:.0f}–{max(hist['val_accuracy'])*100:.0f}%**
      while validation loss steadily improved {hist['val_loss'][0]:.4f} → {hist['val_loss'][-1]:.4f}.
    - Final generalisation gap: **{gap:+.1f}%** (val − train). Negative gap ⇒ classic **overfitting**.
    - **Cause:** only ~{cfg.get('train_images', 203)} training images, no data augmentation,
      4.8M parameters, and just 5 epochs of Adam optimisation.
    - **Fixes to try:** add `ImageDataGenerator` augmentation (rotation, shift, zoom),
      Dropout layers, more epochs with early stopping, or a pretrained backbone
      (MobileNetV2 / ResNet50) via transfer learning.
    """
)

# ---------- Test / evaluation metrics ----------
st.subheader("Testing Results (held-out validation set)")
rep = eval_res["classification_report"]
cm = eval_res["confusion_matrix"]

m1, m2, m3 = st.columns(3)
m1.metric("Val Accuracy", f"{rep['accuracy']*100:.1f}%")
m2.metric("Val Loss", f"{eval_res['val_loss']:.4f}")
m3.metric("Val images", int(sum(sum(r) for r in cm)))

st.markdown("**Classification report**")
report_df = pd.DataFrame(rep).T
st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

st.markdown("**Confusion matrix** (rows = actual, cols = predicted)")
cm_df = pd.DataFrame(
    cm,
    index=["Actual: no_tumor", "Actual: tumor"],
    columns=["Pred: no_tumor", "Pred: tumor"],
)
st.dataframe(cm_df, use_container_width=True)

fig = go.Figure(data=go.Heatmap(
    z=cm, x=["no_tumor", "tumor"], y=["no_tumor", "tumor"],
    colorscale="Blues", text=cm, texttemplate="%{text}",
))
fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual", height=350)
st.plotly_chart(fig, use_container_width=True)

tn, fp = cm[0]
fn, tp = cm[1]
st.markdown(
    f"""
    | Metric | Value | Meaning |
    |---|---|---|
    | True Negatives (TN) | {tn} | no_tumor scans correctly rejected |
    | False Positives (FP) | {fp} | no_tumor wrongly flagged as tumor |
    | False Negatives (FN) | {fn} | **tumor missed** — most dangerous error |
    | True Positives (TP) | {tp} | tumors correctly detected |
    | Sensitivity (Recall tumor) | {tp/(tp+fn)*100:.1f}% | of actual tumors, how many caught |
    | Specificity | {tn/(tn+fp)*100:.1f}% | of healthy scans, how many correctly cleared |
    """
)

st.caption("⚠️ Educational / research use only — not a substitute for professional medical diagnosis.")
