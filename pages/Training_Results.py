import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="NeuroScan AI — Training Results",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── Professional CSS ──────────────────────
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0f1729 40%, #131c33 100%);
        font-family: 'Inter', sans-serif;
    }
    .stApp > header { background: transparent; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1120 0%, #101828 100%);
        border-right: 1px solid rgba(56, 189, 248, 0.12);
    }

    h1, h2, h3 { font-family: 'Inter', sans-serif !important; color: #e2e8f0 !important; }
    h1 {
        font-weight: 800 !important; font-size: 2.1rem !important;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    h2 { font-weight: 700 !important; font-size: 1.35rem !important; color: #cbd5e1 !important; }
    h3 { font-weight: 600 !important; font-size: 1.05rem !important; }
    p, li, label { color: #94a3b8 !important; font-size: 0.95rem; }

    .hero-sub { color: #64748b; font-size: 1rem; margin-bottom: 1.6rem; }

    .info-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 16px; padding: 1.4rem 1.6rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
        height: 100%;
    }
    .card-title {
        font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1.5px;
        color: #64748b; font-weight: 600; margin-bottom: 0.4rem;
    }
    .card-value { font-size: 1.9rem; font-weight: 800; color: #f1f5f9; line-height: 1.1; }
    .card-value.good { color: #34d399; }
    .card-value.warn { color: #fbbf24; }
    .card-value.bad  { color: #f87171; }
    .card-sub { font-size: 0.8rem; color: #64748b; margin-top: 0.3rem; }

    [data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(56, 189, 248, 0.12);
        border-radius: 14px; padding: 1rem 1.2rem;
    }
    [data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 1px; }
    [data-testid="stMetricValue"] { color: #f1f5f9 !important; font-weight: 700 !important; }

    details {
        background: rgba(15, 23, 42, 0.65) !important;
        border: 1px solid rgba(56, 189, 248, 0.12) !important;
        border-radius: 14px !important; padding: 0.4rem 1rem !important;
    }
    details summary { color: #cbd5e1 !important; font-weight: 600; }

    .stButton > button {
        border-radius: 12px; font-weight: 600;
        border: 1px solid rgba(56,189,248,0.3);
        background: rgba(56,189,248,0.1); color: #e2e8f0;
    }
    .stButton > button:hover { background: rgba(56,189,248,0.22); }

    .footer {
        margin-top: 3rem; padding: 1.2rem 0;
        border-top: 1px solid rgba(56, 189, 248, 0.1);
        text-align: center; font-size: 0.78rem; color: #475569;
    }
    hr { border: none; border-top: 1px solid rgba(56, 189, 248, 0.1); }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────── Sidebar branding ──────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 0.5rem 0 1.2rem 0;">
            <div style="font-size: 2.6rem;">🧠</div>
            <div style="
                font-size: 1.35rem; font-weight: 800;
                background: linear-gradient(90deg,#38bdf8,#818cf8);
                -webkit-background-clip:text; -webkit-text-fill-color:transparent;
            ">NeuroScan AI</div>
            <div style="font-size:0.72rem; color:#64748b; letter-spacing:2px; text-transform:uppercase; margin-top:0.2rem;">
                Brain Tumor Detection
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.page_link("app.py", label="🔬  Upload & Scan", icon="🔬")
    st.page_link("pages/Training_Results.py", label="📊  Training Results", icon="📊")
    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:0.75rem; color:#475569; line-height:1.6; padding:0.5rem;">
            <b style="color:#64748b;">This page</b><br>
            Epoch-by-epoch training history,<br>
            validation curves, confusion<br>
            matrix & per-epoch analysis.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────── Load data ─────────────────────────────
st.markdown("## Training & Testing Results")
st.markdown(
    '<div class="hero-sub">Detailed epoch-by-epoch breakdown of how the CNN learned on the brain MRI dataset.</div>',
    unsafe_allow_html=True,
)

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

# ─────────────────────────── Summary cards ─────────────────────────
c1, c2, c3, c4, c5 = st.columns(5, gap="medium")
c1.markdown(
    f'<div class="info-card"><div class="card-title">Epochs</div>'
    f'<div class="card-value">{cfg.get("epochs_run", cfg.get("epochs", len(epochs)))}</div></div>',
    unsafe_allow_html=True,
)
c2.markdown(
    f'<div class="info-card"><div class="card-title">Train Images</div>'
    f'<div class="card-value">{cfg.get("train_images", "—")}</div></div>',
    unsafe_allow_html=True,
)
c3.markdown(
    f'<div class="info-card"><div class="card-title">Val Images</div>'
    f'<div class="card-value">{cfg.get("val_images", "—")}</div></div>',
    unsafe_allow_html=True,
)
c4.markdown(
    f'<div class="info-card"><div class="card-title">Model Params</div>'
    f'<div class="card-value">{cfg.get("model_params", 0):,}</div></div>',
    unsafe_allow_html=True,
)
c5.markdown(
    f'<div class="info-card"><div class="card-title">Final Val Acc</div>'
    f'<div class="card-value good">{hist["val_accuracy"][-1]*100:.1f}%</div></div>',
    unsafe_allow_html=True,
)
st.caption(
    f"Optimizer: {cfg.get('optimizer')}  ·  Loss: {cfg.get('loss')}  ·  "
    f"Batch: {cfg.get('batch_size')}  ·  Input: {cfg.get('input_size')}"
)

# ─────────────────────────── Epoch table ───────────────────────────
st.markdown("### Epoch-by-Epoch Metrics")
df = pd.DataFrame({
    "Epoch": epochs,
    "Train Acc": [f"{a*100:.2f}%" for a in hist["accuracy"]],
    "Train Loss": [f"{l:.4f}" for l in hist["loss"]],
    "Val Acc": [f"{a*100:.2f}%" for a in hist["val_accuracy"]],
    "Val Loss": [f"{l:.4f}" for l in hist["val_loss"]],
    "Δ Train Acc": ["—"] + [f"{(hist['accuracy'][i]-hist['accuracy'][i-1])*100:+.2f}%" for i in range(1, len(epochs))],
    "Δ Val Acc": ["—"] + [f"{(hist['val_accuracy'][i]-hist['val_accuracy'][i-1])*100:+.2f}%" for i in range(1, len(epochs))],
    "Δ Val Loss": ["—"] + [f"{hist['val_loss'][i]-hist['val_loss'][i-1]:+.4f}" for i in range(1, len(epochs))],
})
st.dataframe(df, use_container_width=True, hide_index=True)

# ─────────────────────────── Curves ────────────────────────────────
st.markdown("### Training Curves")
tab1, tab2 = st.tabs(["📈 Accuracy", "📉 Loss"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=hist["accuracy"], mode="lines+markers",
        name="Train Accuracy", line=dict(color="#38bdf8", width=3),
        marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=hist["val_accuracy"], mode="lines+markers",
        name="Val Accuracy", line=dict(color="#fbbf24", width=3, dash="dash"),
        marker=dict(size=8),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.5)",
        font=dict(color="#94a3b8"), xaxis_title="Epoch", yaxis_title="Accuracy",
        yaxis_tickformat=".0%", height=420,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=20, t=20, b=40),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.08)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.08)")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=hist["loss"], mode="lines+markers",
        name="Train Loss", line=dict(color="#38bdf8", width=3),
        marker=dict(size=8),
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=hist["val_loss"], mode="lines+markers",
        name="Val Loss", line=dict(color="#f87171", width=3, dash="dash"),
        marker=dict(size=8),
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(15,23,42,0.5)",
        font=dict(color="#94a3b8"), xaxis_title="Epoch", yaxis_title="Loss", height=420,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=40, r=20, t=20, b=40),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,0.08)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.08)")
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────── Per-epoch analysis ────────────────────
st.markdown("### Detailed Epoch Analysis — what changed and why")

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

# ─────────────────────────── Diagnosis ─────────────────────────────
st.markdown("### Overall Training Diagnosis")
gap = (hist["val_accuracy"][-1] - hist["accuracy"][-1]) * 100
st.markdown(
    f"""
    <div class="info-card" style="margin:0.5rem 0;">
        <div style="color:#cbd5e1; line-height:1.8; font-size:0.93rem;">
            • Train accuracy climbed <b style="color:#38bdf8;">{hist['accuracy'][0]*100:.1f}% → {hist['accuracy'][-1]*100:.1f}%</b>
            over {len(epochs)} epochs (loss {hist['loss'][0]:.4f} → {hist['loss'][-1]:.4f}).<br>
            • Validation accuracy stayed roughly flat at
            <b style="color:#fbbf24;">{min(hist['val_accuracy'])*100:.0f}–{max(hist['val_accuracy'])*100:.0f}%</b>
            while validation loss steadily improved {hist['val_loss'][0]:.4f} → {hist['val_loss'][-1]:.4f}.<br>
            • Final generalisation gap: <b style="color:#f87171;">{gap:+.1f}%</b> (val − train) ⇒ classic <b>overfitting</b>.<br>
            • <b>Cause:</b> only ~{cfg.get('train_images', 203)} training images, no augmentation,
            4.8M parameters, 5 epochs of Adam.<br>
            • <b>Fixes:</b> add image augmentation (rotation/shift/zoom), Dropout layers,
            early stopping, or transfer learning (MobileNetV2 / ResNet50).
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────── Test metrics ──────────────────────────
st.markdown("### Testing Results (held-out validation set)")
rep = eval_res["classification_report"]
cm = eval_res["confusion_matrix"]
threshold = eval_res.get("threshold", 0.5)
sens = eval_res.get("sensitivity", rep.get("tumor", {}).get("recall", 0))
spec = eval_res.get("specificity", rep.get("no_tumor", {}).get("recall", 0))

m1, m2, m3, m4, m5 = st.columns(5, gap="medium")
m1.markdown(
    f'<div class="info-card"><div class="card-title">Val Accuracy</div>'
    f'<div class="card-value good">{rep["accuracy"]*100:.1f}%</div></div>',
    unsafe_allow_html=True,
)
m2.markdown(
    f'<div class="info-card"><div class="card-title">Val Loss</div>'
    f'<div class="card-value">{eval_res["val_loss"]:.4f}</div></div>',
    unsafe_allow_html=True,
)
m3.markdown(
    f'<div class="info-card"><div class="card-title">Threshold</div>'
    f'<div class="card-value warn">{threshold:.2f}</div></div>',
    unsafe_allow_html=True,
)
m4.markdown(
    f'<div class="info-card"><div class="card-title">Sensitivity</div>'
    f'<div class="card-value good">{sens*100:.1f}%</div>'
    f'<div class="card-sub">tumors caught</div></div>',
    unsafe_allow_html=True,
)
m5.markdown(
    f'<div class="info-card"><div class="card-title">Specificity</div>'
    f'<div class="card-value">{spec*100:.1f}%</div>'
    f'<div class="card-sub">healthy cleared</div></div>',
    unsafe_allow_html=True,
)

st.markdown("#### Classification Report")
report_df = pd.DataFrame(rep).T
st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

st.markdown("#### Confusion Matrix")
colA, colB = st.columns([1, 1], gap="large")
with colA:
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=["no_tumor", "tumor"], y=["no_tumor", "tumor"],
        colorscale=[[0, "#0f1729"], [0.5, "#1d4ed8"], [1, "#38bdf8"]],
        text=cm, texttemplate="%{text}", textfont=dict(color="#f1f5f9", size=16),
        showscale=False,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"), xaxis_title="Predicted", yaxis_title="Actual",
        height=350, margin=dict(l=40, r=20, t=10, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

with colB:
    tn, fp = cm[0]
    fn, tp = cm[1]
    st.markdown(
        f"""
        <div class="info-card">
            <div class="card-title">Error Breakdown</div>
            <div style="color:#cbd5e1; line-height:2; font-size:0.92rem; margin-top:0.4rem;">
                <b style="color:#34d399;">True Negatives</b> — {tn} · healthy correctly cleared<br>
                <b style="color:#fbbf24;">False Positives</b> — {fp} · healthy wrongly flagged<br>
                <b style="color:#f87171;">False Negatives</b> — {fn} · <b>tumor missed</b> (most dangerous)<br>
                <b style="color:#38bdf8;">True Positives</b> — {tp} · tumors correctly detected<br><br>
                <b style="color:#e2e8f0;">Sensitivity (tumor recall)</b><br>
                <span style="font-size:1.5rem;font-weight:800;color:#38bdf8;">{tp/(tp+fn)*100:.1f}%</span>
                &nbsp;of actual tumors caught<br><br>
                <b style="color:#e2e8f0;">Specificity</b><br>
                <span style="font-size:1.5rem;font-weight:800;color:#818cf8;">{tn/(tn+fp)*100:.1f}%</span>
                &nbsp;of healthy scans correctly cleared
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="footer">⚠️ Educational / research use only — not a substitute for professional medical diagnosis.</div>',
    unsafe_allow_html=True,
)
