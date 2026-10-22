import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from gradcam import grad_cam, overlay_heatmap, tumor_region, region_size

# ─────────────────────────── Page config ───────────────────────────
st.set_page_config(
    page_title="NeuroScan AI — Brain Tumor Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────── Professional CSS ──────────────────────
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global */
    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0f1729 40%, #131c33 100%);
        font-family: 'Inter', sans-serif;
    }
    .stApp > header { background: transparent; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1120 0%, #101828 100%);
        border-right: 1px solid rgba(56, 189, 248, 0.12);
    }
    section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

    /* Headings */
    h1, h2, h3 { font-family: 'Inter', sans-serif !important; color: #e2e8f0 !important; }
    h1 {
        font-weight: 800 !important;
        font-size: 2.1rem !important;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem !important;
    }
    h2 { font-weight: 700 !important; font-size: 1.35rem !important; color: #cbd5e1 !important; }
    h3 { font-weight: 600 !important; font-size: 1.1rem !important; }

    p, li, label { color: #94a3b8 !important; font-size: 0.95rem; }

    /* Hero subtitle */
    .hero-sub {
        color: #64748b; font-size: 1rem; font-weight: 400;
        margin-bottom: 1.8rem; letter-spacing: 0.2px;
    }

    /* Metric / info cards */
    .info-card {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(56, 189, 248, 0.15);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
        height: 100%;
        transition: border 0.2s;
    }
    .info-card:hover { border-color: rgba(56, 189, 248, 0.35); }

    .card-title {
        font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1.5px;
        color: #64748b; font-weight: 600; margin-bottom: 0.4rem;
    }
    .card-value {
        font-size: 1.9rem; font-weight: 800; color: #f1f5f9; line-height: 1.1;
    }
    .card-value.good { color: #34d399; }
    .card-value.bad  { color: #f87171; }
    .card-value.warn { color: #fbbf24; }
    .card-sub { font-size: 0.8rem; color: #64748b; margin-top: 0.3rem; }

    /* Verdict banner */
    .verdict {
        border-radius: 18px; padding: 1.3rem 1.8rem; margin: 1rem 0 1.5rem 0;
        display: flex; align-items: center; gap: 1.2rem;
        backdrop-filter: blur(14px);
    }
    .verdict-tumor {
        background: linear-gradient(135deg, rgba(127,29,29,0.55), rgba(190,18,60,0.35));
        border: 1px solid rgba(248,113,113,0.35);
        box-shadow: 0 0 40px rgba(239,68,68,0.12);
    }
    .verdict-safe {
        background: linear-gradient(135deg, rgba(6,78,59,0.55), rgba(16,185,129,0.3));
        border: 1px solid rgba(52,211,153,0.35);
        box-shadow: 0 0 40px rgba(16,185,129,0.1);
    }
    .verdict-icon { font-size: 2.6rem; line-height: 1; }
    .verdict-title { font-size: 1.45rem; font-weight: 800; color: #f8fafc; }
    .verdict-sub { font-size: 0.9rem; color: #cbd5e1; margin-top: 0.15rem; }
    .verdict-pct {
        font-size: 2.4rem; font-weight: 800; margin-left: auto; text-align: right;
    }
    .verdict-tumor .verdict-pct { color: #fca5a5; }
    .verdict-safe  .verdict-pct { color: #6ee7b7; }

    /* Image frame */
    .img-frame {
        border-radius: 16px; overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.15);
        box-shadow: 0 10px 40px rgba(0,0,0,0.45);
        background: #0b1120;
        padding: 6px;
    }
    .img-frame img { border-radius: 12px; width: 100%; display: block; }
    .img-caption {
        text-align: center; font-size: 0.78rem; color: #64748b;
        padding: 0.6rem 0 0.2rem 0; letter-spacing: 0.5px; font-weight: 500;
    }

    /* Upload zone */
    [data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.6);
        border: 1.5px dashed rgba(56, 189, 248, 0.35) !important;
        border-radius: 16px !important;
        padding: 1.5rem !important;
        transition: border-color 0.2s;
    }
    [data-testid="stFileUploader"]:hover { border-color: #38bdf8 !important; }
    [data-testid="stFileUploaderDropzone"] div { color: #94a3b8 !important; }

    /* Progress bar */
    div[data-testid="stProgress"] > div > div > div > div {
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        border-radius: 8px;
    }

    /* Metric blocks */
    [data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(56, 189, 248, 0.12);
        border-radius: 14px; padding: 1rem 1.2rem;
    }
    [data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 1px; }
    [data-testid="stMetricValue"] { color: #f1f5f9 !important; font-weight: 700 !important; }

    /* Tables */
    .stDataFrame { border-radius: 12px !important; }

    /* Expander */
    details {
        background: rgba(15, 23, 42, 0.65) !important;
        border: 1px solid rgba(56, 189, 248, 0.12) !important;
        border-radius: 14px !important;
        padding: 0.4rem 1rem !important;
    }
    details summary { color: #cbd5e1 !important; font-weight: 600; }

    /* Footer */
    .footer {
        margin-top: 3rem; padding: 1.2rem 0;
        border-top: 1px solid rgba(56, 189, 248, 0.1);
        text-align: center; font-size: 0.78rem; color: #475569;
    }

    /* Divider */
    hr { border: none; border-top: 1px solid rgba(56, 189, 248, 0.1); }

    /* Buttons */
    .stButton > button {
        border-radius: 12px; font-weight: 600; border: 1px solid rgba(56,189,248,0.3);
        background: rgba(56,189,248,0.1); color: #e2e8f0; transition: all 0.2s;
    }
    .stButton > button:hover {
        background: rgba(56,189,248,0.22); border-color: #38bdf8; color: #fff;
    }
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
            <b style="color:#64748b;">Model</b><br>
            CNN · 3 conv blocks · 150×150 input<br>
            Binary: Tumor / No Tumor<br><br>
            <b style="color:#64748b;">Heatmap</b><br>
            Grad-CAM saliency overlay
        </div>
        """,
        unsafe_allow_html=True,
    )

# ─────────────────────────── Cached model ──────────────────────────
@st.cache_resource(show_spinner="Loading CNN model…")
def load_model():
    return tf.keras.models.load_model("brain_tumor_model.h5")

@st.cache_data
def load_threshold():
    try:
        with open("evaluation_results.json") as f:
            return float(json.load(f).get("threshold", 0.5))
    except Exception:
        return 0.5

try:
    model = load_model()
    THRESHOLD = load_threshold()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# ─────────────────────────── Main content ──────────────────────────
st.markdown("## Brain Tumor Detection")
st.markdown(
    '<div class="hero-sub">Upload a brain MRI scan — the CNN analyzes it and Grad-CAM highlights the region that drove the decision.</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader("Drop MRI image here", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with st.spinner("Analyzing scan…"):
        img = Image.open(uploaded_file).convert("RGB")
        original = np.array(img)
        resized = img.resize((150, 150))
        arr = np.array(resized)
        x = np.expand_dims(arr / 255.0, axis=0)

        heatmap, _ = grad_cam(model, x)
        tumor_prob = float(model.predict(x, verbose=0)[0][0])
        no_tumor_prob = 1.0 - tumor_prob
        is_tumor = tumor_prob > THRESHOLD
        confidence = (tumor_prob if is_tumor else no_tumor_prob) * 100

    # ── Verdict banner ──
    cls = "verdict-tumor" if is_tumor else "verdict-safe"
    icon = "🚨" if is_tumor else "✅"
    title = "Tumor Detected" if is_tumor else "No Tumor Detected"
    sub = (
        "Abnormal tissue patterns identified — review heatmap below."
        if is_tumor
        else "No significant abnormality detected in this scan."
    )
    st.markdown(
        f"""
        <div class="verdict {cls}">
            <div class="verdict-icon">{icon}</div>
            <div>
                <div class="verdict-title">{title}</div>
                <div class="verdict-sub">{sub}</div>
            </div>
            <div class="verdict-pct">{confidence:.1f}%<div style="font-size:0.7rem;font-weight:500;color:#94a3b8;letter-spacing:1px;">CONFIDENCE</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(int(confidence), 100) / 100.0)

    # ── Images side by side ──
    overlay = overlay_heatmap(original, heatmap)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.image(original, caption="ORIGINAL MRI", use_container_width=True)
    with c2:
        st.image(
            overlay,
            caption="GRAD-CAM HEATMAP — warm zones drove the decision",
            use_container_width=True,
        )

    # ── Metric cards ──
    st.markdown("")
    m1, m2, m3, m4 = st.columns(4, gap="medium")
    m1.markdown(
        f'<div class="info-card"><div class="card-title">Tumor Probability</div>'
        f'<div class="card-value {"bad" if is_tumor else "good"}">{tumor_prob*100:.1f}%</div>'
        f'<div class="card-sub">raw sigmoid output</div></div>',
        unsafe_allow_html=True,
    )
    m2.markdown(
        f'<div class="info-card"><div class="card-title">No-Tumor Probability</div>'
        f'<div class="card-value {"good" if not is_tumor else "warn"}">{no_tumor_prob*100:.1f}%</div>'
        f'<div class="card-sub">complement of tumor score</div></div>',
        unsafe_allow_html=True,
    )
    m3.markdown(
        f'<div class="info-card"><div class="card-title">Decision Threshold</div>'
        f'<div class="card-value">{THRESHOLD*100:.0f}%</div>'
        f'<div class="card-sub">tuned for high sensitivity</div></div>',
        unsafe_allow_html=True,
    )
    region = tumor_region(heatmap, threshold=0.3)
    area_frac = region_size(heatmap, threshold=0.3)
    loc_txt = f"{area_frac*100:.1f}%" if region is not None and area_frac > 0.01 else "None"
    loc_cls = "warn" if loc_txt != "None" else "good"
    m4.markdown(
        f'<div class="info-card"><div class="card-title">Hot Region</div>'
        f'<div class="card-value {loc_cls}">{loc_txt}</div>'
        f'<div class="card-sub">image area highlighted</div></div>',
        unsafe_allow_html=True,
    )

    # ── Why this reading ──
    st.markdown("")
    st.markdown("### Why this reading?")
    if region is not None and area_frac > 0.01:
        x1, y1, x2, y2 = region
        st.markdown(
            f"""
            <div class="info-card" style="margin:0.5rem 0;">
                <div style="color:#cbd5e1; line-height:1.75; font-size:0.93rem;">
                    <b style="color:#e2e8f0;">Tumor location:</b> bounding box at
                    <code style="color:#38bdf8;">x={x1}–{x2}, y={y1}–{y2}</code> on the 150×150 grid.<br><br>
                    The Grad-CAM heatmap shows the CNN focused on
                    <b style="color:#fbbf24;">{area_frac*100:.1f}%</b> of the image — the bright red/yellow
                    zones in the overlay. Those zones correspond to abnormal tissue patterns
                    (irregular texture, intensity variation) the model learned during training.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="info-card" style="margin:0.5rem 0;">
                <div style="color:#cbd5e1; line-height:1.75; font-size:0.93rem;">
                    <b style="color:#e2e8f0;">Tumor location:</b> no concentrated hot region found.
                    The decision was driven by diffuse, low-saliency features across the scan
                    rather than a single focused abnormality — typical when the model predicts
                    <b>no tumor</b>.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── How confidence is computed ──
    with st.expander("How the confidence % is computed", expanded=True):
        st.markdown(
            f"""
            1. Image resized to **150×150** and normalized to `[0,1]`.
            2. The CNN outputs a raw **sigmoid score** `p` ∈ (0,1) — here **{tumor_prob:.4f}**.
            3. Decision threshold is **{THRESHOLD:.2f}** (tuned for high sensitivity —
               missing a tumor is worse than a false alarm):
               if `p > {THRESHOLD:.2f}` → **Tumor**, confidence = `p×100`;
               if `p ≤ {THRESHOLD:.2f}` → **No Tumor**, confidence = `(1−p)×100`.
            4. This scan's **{confidence:.1f}%** means the score landed
               **{abs(tumor_prob - THRESHOLD):.4f}** away from the {THRESHOLD:.2f} boundary —
               larger distance ⇒ more confident the model is.
            5. The heatmap comes from **Grad-CAM**: gradients of the output w.r.t.
               the last conv layer, pooled and weighted into a saliency map of
               which pixels mattered most for this decision.
            """
        )

    st.markdown(
        '<div class="footer">⚠️ Educational / research use only — not a substitute for professional medical diagnosis.</div>',
        unsafe_allow_html=True,
    )

else:
    # ── Empty state ──
    st.markdown("")
    st.markdown(
        """
        <div class="info-card" style="text-align:center; padding:3rem 2rem;">
            <div style="font-size:3rem; margin-bottom:0.8rem;">🧠</div>
            <div style="font-size:1.2rem; font-weight:700; color:#e2e8f0;">Drop an MRI scan above to begin</div>
            <div style="color:#64748b; margin-top:0.5rem; font-size:0.9rem;">
                Supports JPG, JPEG, PNG · analysis runs in seconds
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(
            """
            <div class="info-card">
                <div class="card-title">🔬 Upload & Scan</div>
                <div style="color:#94a3b8; font-size:0.9rem; line-height:1.7; margin-top:0.5rem;">
                    Upload an MRI → get tumor / no-tumor prediction with a confidence
                    percentage and a Grad-CAM heatmap showing <i>where</i> the model looked.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="info-card">
                <div class="card-title">📊 Training Results</div>
                <div style="color:#94a3b8; font-size:0.9rem; line-height:1.7; margin-top:0.5rem;">
                    Epoch-by-epoch training history, validation curves, confusion matrix
                    and a detailed analysis of every metric change.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div class="footer">⚠️ Educational / research use only — not a substitute for professional medical diagnosis.</div>',
        unsafe_allow_html=True,
    )
