"""
=============================================================================
File        : gui/interface.py
Project     : PCB Inspection System
Description : Professional Streamlit interface - Module 1 active.
=============================================================================
"""

import os
import sys
import time
import tempfile
import io

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# --- Project root ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.preprocessing import preprocess_image, get_preprocessing_stages

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "preprocessing")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Page config
# ============================================================

st.set_page_config(
    page_title="PCB Defect Inspection System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0b0f19; }

/* Hero */
.hero {
    background: linear-gradient(135deg, #0d1b3e 0%, #0b1629 50%, #0a1220 100%);
    border: 1px solid #1e3a5f; border-radius: 16px;
    padding: 32px 40px 26px 40px; margin-bottom: 24px;
    position: relative; overflow: hidden;
}
.hero::before {
    content:''; position:absolute; top:-60px; right:-60px;
    width:220px; height:220px; border-radius:50%;
    background:radial-gradient(circle,rgba(58,134,255,0.12) 0%,transparent 70%);
}
.hero-title { font-size:2rem; font-weight:700; color:#fff; margin:0 0 6px 0; }
.hero-subtitle { font-size:0.95rem; color:#6b85b0; margin:0; }
.hero-badge {
    display:inline-block; background:rgba(58,134,255,0.15); color:#5fa8ff;
    border:1px solid rgba(58,134,255,0.3); border-radius:20px;
    padding:3px 12px; font-size:0.7rem; font-weight:600;
    letter-spacing:0.08em; text-transform:uppercase; margin-bottom:12px;
}

/* Pipeline bar */
.pipeline-row { display:flex; align-items:center; gap:0; margin-bottom:24px; }
.pipe-step {
    flex:1; background:#111827; border:1px solid #1f2d45;
    border-radius:12px; padding:12px 8px; text-align:center;
}
.pipe-step.active {
    background:linear-gradient(135deg,#0f2347,#112040);
    border-color:#3a86ff; box-shadow:0 0 16px rgba(58,134,255,0.15);
}
.pipe-step .icon { font-size:1.4rem; }
.pipe-step .plabel {
    font-size:0.68rem; font-weight:600; color:#4a5e80;
    margin-top:5px; text-transform:uppercase; letter-spacing:0.06em;
}
.pipe-step.active .plabel { color:#7ab3ff; }
.pipe-arrow { color:#1e3050; font-size:1rem; padding:0 4px; flex-shrink:0; }

/* Section label */
.section-label {
    font-size:0.68rem; font-weight:700; letter-spacing:0.12em;
    text-transform:uppercase; color:#3a86ff; margin-bottom:10px;
}

/* Upload hint */
.upload-hint {
    background:#0d1829; border:2px dashed #1e3a5f; border-radius:14px;
    padding:18px; text-align:center; color:#4a6890; font-size:0.88rem;
    margin-bottom:14px;
}

/* Image info card */
.img-info-card {
    background:#111827; border:1px solid #1f2d45; border-radius:12px;
    padding:16px; font-size:0.82rem; color:#7a90b0;
}
.img-info-card .ril { font-size:0.65rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.1em; color:#3a5070; margin-bottom:2px; margin-top:10px; }
.img-info-card .ril:first-child { margin-top:0; }
.img-info-card .riv { color:#c5d4e8; font-weight:500; }

/* Success banner */
.success-banner {
    background:linear-gradient(90deg,#0c2e1a,#0a1f12);
    border:1px solid #1a5c35; border-left:4px solid #22c55e;
    border-radius:10px; padding:12px 16px; color:#6ee89a;
    font-size:0.88rem; font-weight:500; margin:16px 0;
}

/* Stage strip */
.stage-card {
    background:#0e1520; border:1px solid #1a2840;
    border-radius:12px; padding:12px; text-align:center;
}
.stage-card .stage-num {
    display:inline-block; background:rgba(58,134,255,0.12);
    color:#4a86d0; border-radius:12px; padding:1px 10px;
    font-size:0.65rem; font-weight:700; letter-spacing:0.08em;
    text-transform:uppercase; margin-bottom:8px;
}
.stage-card.final { border-color:#3a86ff; background:#0d1829; }
.stage-card.final .stage-num { background:rgba(58,134,255,0.25); color:#7ab3ff; }
.stage-name { font-size:0.8rem; font-weight:600; color:#7a90b0; margin-bottom:2px; }
.stage-desc { font-size:0.7rem; color:#3a5070; }

/* Quality metric row */
.qm-row { display:flex; gap:12px; margin:16px 0; }
.qm-card {
    flex:1; background:#111827; border:1px solid #1f2d45;
    border-radius:12px; padding:14px 16px;
}
.qm-label { font-size:0.65rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.1em; color:#3a5070; margin-bottom:5px; }
.qm-value { font-size:1.25rem; font-weight:700; color:#e0eaff; }
.qm-delta { font-size:0.72rem; margin-top:3px; }
.qm-delta.pos { color:#22c55e; }
.qm-delta.neg { color:#ef4444; }
.qm-delta.neu { color:#3a5070; }

/* Step explain card */
.explain-card {
    background:#0e1520; border:1px solid #1a2840;
    border-left:3px solid #3a86ff; border-radius:10px;
    padding:14px 16px; margin-bottom:10px;
}
.ec-step { font-size:0.65rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.1em; color:#3a86ff; margin-bottom:4px; }
.ec-title { font-size:0.88rem; font-weight:600; color:#c5d4e8; margin-bottom:4px; }
.ec-desc { font-size:0.78rem; color:#4a6890; line-height:1.6; }
.ec-result { font-size:0.75rem; color:#22c55e; margin-top:6px; }

/* Sidebar */
.sb-title { font-size:1rem; font-weight:700; color:#d0dff5; margin-bottom:4px; }
.sb-tag { font-size:0.68rem; color:#3a5070; margin-bottom:16px; }
.sb-section { font-size:0.65rem; font-weight:700; letter-spacing:0.13em;
    text-transform:uppercase; color:#2a3d5a; margin:16px 0 9px 0; }
.sb-step-on {
    background:linear-gradient(135deg,#0f2044,#0d1a38);
    border:1px solid rgba(58,134,255,0.4); border-left:3px solid #3a86ff;
    border-radius:8px; padding:9px 12px; margin-bottom:6px;
}
.sb-step-on .sn { font-size:0.82rem; font-weight:600; color:#7ab3ff; }
.sb-step-on .sd { font-size:0.7rem; color:#3a6099; margin-top:2px; }
.sb-step-off {
    background:#0d1218; border:1px solid #151e2e;
    border-left:3px solid #1a2540; border-radius:8px;
    padding:9px 12px; margin-bottom:6px;
}
.sb-step-off .sn { font-size:0.82rem; font-weight:500; color:#243050; }
.sb-step-off .sd { font-size:0.7rem; color:#182030; margin-top:2px; }
.sb-info {
    background:#0d1218; border:1px solid #151e2e;
    border-radius:10px; padding:13px; font-size:0.78rem;
}
.sb-info-label { font-size:0.64rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.1em; color:#3a86ff; margin-top:11px; margin-bottom:2px; }
.sb-info-label:first-child { margin-top:0; }
.sb-info-val { color:#7a90b0; font-size:0.77rem; }
.sb-pipe-step { color:#4a6890; font-size:0.74rem; padding:1px 0; }
.sb-pipe-arr { color:#1a2a40; font-size:0.7rem; padding-left:6px; }

hr { border-color:#131e30 !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("<div class='sb-title'>🔬 PCB Inspection</div>", unsafe_allow_html=True)
    st.markdown("<div class='sb-tag'>Automated Quality Control System</div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div class='sb-section'>📋 Inspection Workflow</div>", unsafe_allow_html=True)
    st.markdown("""
        <div class='sb-step-on'>
            <div class='sn'>✦ Image Pre-processing</div>
            <div class='sd'>Contrast enhancement &amp; noise reduction</div>
        </div>
        <div class='sb-step-off'>
            <div class='sn'>○ Defect Segmentation</div>
            <div class='sd'>PCB defect region extraction</div>
        </div>
        <div class='sb-step-off'>
            <div class='sn'>○ Feature Analysis</div>
            <div class='sd'>Defect characteristic analysis</div>
        </div>
        <div class='sb-step-off'>
            <div class='sn'>○ Inspection Report</div>
            <div class='sd'>Result visualisation &amp; summary</div>
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div class='sb-section'>⚙️ System Information</div>", unsafe_allow_html=True)
    st.markdown("""
        <div class='sb-info'>
            <div class='sb-info-label'>Dataset</div>
            <div class='sb-info-val'>DeepPCB</div>
            <div class='sb-info-label'>Current Pipeline</div>
            <div class='sb-pipe-step'>📥 Input Image</div>
            <div class='sb-pipe-arr'>↓</div>
            <div class='sb-pipe-step'>🔲 Grayscale Conversion</div>
            <div class='sb-pipe-arr'>↓</div>
            <div class='sb-pipe-step'>🔉 Median Filtering</div>
            <div class='sb-pipe-arr'>↓</div>
            <div class='sb-pipe-step'>🔆 CLAHE Enhancement</div>
            <div class='sb-info-label'>Architecture</div>
            <div class='sb-info-val'>Modular Image Processing</div>
            <div class='sb-info-label'>Parameters</div>
            <div class='sb-info-val'>Median kernel: 5×5<br>CLAHE clip: 2.0<br>Tile grid: 8×8</div>
        </div>
    """, unsafe_allow_html=True)

# ============================================================
# Hero
# ============================================================

st.markdown("""
<div class='hero'>
    <div class='hero-badge'>Module 1 — Image Pre-processing &amp; Calibration</div>
    <div class='hero-title'>🔬 PCB Defect Inspection System</div>
    <div class='hero-subtitle'>
        Upload a PCB board image to run the complete pre-processing pipeline.
        Each stage is visualised with quality metrics to demonstrate the enhancement applied.
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Pipeline bar
# ============================================================

st.markdown("""
<div class='pipeline-row'>
    <div class='pipe-step active'>
        <div class='icon'>🖼️</div>
        <div class='plabel'>Pre-processing</div>
    </div>
    <div class='pipe-arrow'>→</div>
    <div class='pipe-step'>
        <div class='icon'>🔍</div>
        <div class='plabel'>Segmentation</div>
    </div>
    <div class='pipe-arrow'>→</div>
    <div class='pipe-step'>
        <div class='icon'>📊</div>
        <div class='plabel'>Analysis</div>
    </div>
    <div class='pipe-arrow'>→</div>
    <div class='pipe-step'>
        <div class='icon'>📄</div>
        <div class='plabel'>Report</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Upload
# ============================================================

st.markdown("<div class='section-label'>📂 Step 1 — Upload PCB Image</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a PCB image",
    type=["jpg", "jpeg", "png"],
    help="Supported: JPG, JPEG, PNG",
    label_visibility="collapsed",
)

if uploaded_file is None:
    st.markdown("""
        <div class='upload-hint'>
            ⬆️ &nbsp; Drag and drop or browse to upload a PCB image<br>
            <span style='font-size:0.78rem;'>Supported formats: JPG · JPEG · PNG</span>
        </div>
    """, unsafe_allow_html=True)

# ============================================================
# Main flow
# ============================================================

if uploaded_file is not None:

    # Decode
    file_bytes   = np.frombuffer(uploaded_file.read(), np.uint8)
    original_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    file_kb      = uploaded_file.size / 1024

    st.markdown("---")

    # --- Original image ---
    st.markdown("<div class='section-label'>📷 Step 2 — Review Uploaded Image</div>", unsafe_allow_html=True)

    col_img, col_info = st.columns([3, 1], gap="medium")

    with col_img:
        st.image(original_rgb, caption=f"📄 {uploaded_file.name}", use_container_width=True)

    with col_info:
        st.markdown(f"""
        <div class='img-info-card'>
            <div class='ril'>Filename</div>
            <div class='riv'>{uploaded_file.name}</div>
            <div class='ril'>Resolution</div>
            <div class='riv'>{original_bgr.shape[1]} × {original_bgr.shape[0]} px</div>
            <div class='ril'>Colour Space</div>
            <div class='riv'>BGR (3 channels)</div>
            <div class='ril'>File Size</div>
            <div class='riv'>{file_kb:.1f} KB</div>
            <div class='ril'>Bit Depth</div>
            <div class='riv'>8-bit per channel</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Run button ---
    st.markdown("<div class='section-label'>⚡ Step 3 — Run Pre-processing</div>", unsafe_allow_html=True)

    col_btn, col_hint = st.columns([1, 4], gap="small")
    with col_btn:
        run_btn = st.button("▶ Run Inspection", type="primary", use_container_width=True)
    with col_hint:
        st.markdown(
            "<p style='color:#2a4060;font-size:0.84rem;margin-top:8px;'>"
            "Pipeline: Grayscale Conversion → Median Filter (5×5) → CLAHE (clip=2.0, tile=8×8)</p>",
            unsafe_allow_html=True,
        )

    # --- Processing ---
    if run_btn:

        t_start = time.time()

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        try:
            with st.spinner("🔧 Running Image Pre-processing Pipeline..."):
                stages, metrics = get_preprocessing_stages(tmp_path)
                processed = stages["enhanced"]
                time.sleep(0.2)

            proc_time = time.time() - t_start

            # Save output
            out_name = f"preprocessed_{uploaded_file.name}"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            cv2.imwrite(out_path, processed)

            # ── Success banner ──────────────────────────────────────
            st.markdown(
                f"<div class='success-banner'>"
                f"✅ &nbsp; Pre-processing completed in <strong>{proc_time:.2f}s</strong> &nbsp;|&nbsp; "
                f"Contrast improved by <strong>{metrics['contrast_gain']:+.1f}%</strong> &nbsp;|&nbsp; "
                f"Saved → <code>outputs/preprocessing/{out_name}</code>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # ── Section: Pipeline Stage Visualisation ───────────────
            st.markdown("<div class='section-label'>🔬 Step 4 — Pipeline Stage Visualisation</div>",
                        unsafe_allow_html=True)

            sc1, sc2, sc3, sc4 = st.columns(4, gap="small")
            stage_defs = [
                (sc1, "original",  "Stage 1",  "Original",         "Raw BGR → Grayscale",   False),
                (sc2, "grayscale", "Stage 2",  "Grayscale",        "Luminance only",         False),
                (sc3, "filtered",  "Stage 3",  "Median Filtered",  "Salt-and-pepper removed",False),
                (sc4, "enhanced",  "Stage 4",  "CLAHE Enhanced",   "Local contrast boosted", True),
            ]

            for col, key, num, name, desc, is_final in stage_defs:
                with col:
                    cls = "stage-card final" if is_final else "stage-card"
                    st.markdown(f"""
                        <div class='{cls}'>
                            <div class='stage-num'>{num}</div>
                            <div class='stage-name'>{name}</div>
                            <div class='stage-desc'>{desc}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    st.image(stages[key], clamp=True, channels="GRAY",
                             use_container_width=True)

            # ── Section: Quality Metrics ────────────────────────────
            st.markdown("---")
            st.markdown("<div class='section-label'>📊 Image Quality Metrics</div>", unsafe_allow_html=True)

            m_orig = metrics["original"]
            m_enh  = metrics["enhanced"]
            m_filt = metrics["filtered"]

            contrast_gain_str = (
                f"<span class='qm-delta pos'>▲ +{metrics['contrast_gain']}%</span>"
                if metrics["contrast_gain"] >= 0
                else f"<span class='qm-delta neg'>▼ {metrics['contrast_gain']}%</span>"
            )

            noise_str = (
                f"<span class='qm-delta pos'>▼ {metrics['noise_reduction']:.1f}% reduced</span>"
                if metrics["noise_reduction"] > 0
                else f"<span class='qm-delta neu'>— No change</span>"
            )

            dr_gain = metrics["dynamic_range_gain"]
            dr_str  = (
                f"<span class='qm-delta pos'>▲ +{dr_gain} levels</span>"
                if dr_gain > 0
                else f"<span class='qm-delta neu'>— {dr_gain} levels</span>"
            )

            st.markdown(f"""
            <div class='qm-row'>
                <div class='qm-card'>
                    <div class='qm-label'>Mean Brightness</div>
                    <div class='qm-value'>{m_enh['mean_brightness']:.1f}</div>
                    <div class='qm-delta neu'>Original: {m_orig['mean_brightness']:.1f} / 255</div>
                </div>
                <div class='qm-card'>
                    <div class='qm-label'>Contrast (Std Dev)</div>
                    <div class='qm-value'>{m_enh['contrast']:.1f}</div>
                    {contrast_gain_str}
                </div>
                <div class='qm-card'>
                    <div class='qm-label'>Noise Reduction</div>
                    <div class='qm-value'>{m_orig['noise_estimate']:.0f} → {m_filt['noise_estimate']:.0f}</div>
                    {noise_str}
                </div>
                <div class='qm-card'>
                    <div class='qm-label'>Dynamic Range</div>
                    <div class='qm-value'>{m_enh['dynamic_range']}</div>
                    {dr_str}
                </div>
                <div class='qm-card'>
                    <div class='qm-label'>Processing Time</div>
                    <div class='qm-value'>{proc_time:.2f}s</div>
                    <div class='qm-delta neu'>End-to-end</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Section: Histogram Comparison ──────────────────────
            st.markdown("---")
            st.markdown("<div class='section-label'>📈 Pixel Intensity Histogram — Before vs After</div>",
                        unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(12, 3.2))
            fig.patch.set_facecolor("#0b0f19")
            ax.set_facecolor("#0e1520")

            ax.hist(stages["original"].ravel(), bins=256, range=(0,255),
                    color="#5fa8ff", alpha=0.55, label="Original (Grayscale)", density=True)
            ax.hist(stages["enhanced"].ravel(), bins=256, range=(0,255),
                    color="#22c55e", alpha=0.65, label="After CLAHE (Final)", density=True)

            ax.set_xlabel("Pixel Intensity (0 = black  ·  255 = white)",
                          color="#4a6890", fontsize=9)
            ax.set_ylabel("Normalised Frequency", color="#4a6890", fontsize=9)
            ax.tick_params(colors="#2a4060", labelsize=8)
            ax.spines[["top","right","left","bottom"]].set_color("#1a2840")
            ax.set_xlim(0, 255)
            ax.set_title("Histogram comparison shows CLAHE redistributes pixel intensities "
                         "for improved local contrast",
                         color="#3a5070", fontsize=8.5, pad=8)

            legend = ax.legend(fontsize=8.5, framealpha=0.15,
                               labelcolor="white", facecolor="#0e1520")
            for line in legend.get_lines():
                line.set_linewidth(2)

            plt.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                        facecolor="#0b0f19")
            buf.seek(0)
            st.image(buf, use_container_width=True)
            plt.close(fig)

            # ── Section: Step-by-Step Explanation ──────────────────
            st.markdown("---")
            st.markdown("<div class='section-label'>📖 What Each Pipeline Step Does</div>",
                        unsafe_allow_html=True)

            exp_col1, exp_col2 = st.columns(2, gap="medium")

            with exp_col1:
                st.markdown(f"""
                <div class='explain-card'>
                    <div class='ec-step'>Step 1</div>
                    <div class='ec-title'>🔲 Grayscale Conversion</div>
                    <div class='ec-desc'>
                        Converts the 3-channel BGR colour image into a single-channel
                        luminance image using the weighted formula:<br>
                        <code>Y = 0.299R + 0.587G + 0.114B</code><br>
                        This reduces data dimensionality from 3 channels to 1,
                        focusing subsequent processing on brightness information
                        relevant to surface defect detection.
                    </div>
                    <div class='ec-result'>
                        ✓ Mean brightness: {m_orig['mean_brightness']:.1f} &nbsp;|&nbsp;
                        Contrast (std): {m_orig['contrast']:.1f}
                    </div>
                </div>

                <div class='explain-card'>
                    <div class='ec-step'>Step 2</div>
                    <div class='ec-title'>🔉 Median Filtering (5×5 kernel)</div>
                    <div class='ec-desc'>
                        Applies a 5×5 median filter to suppress salt-and-pepper noise
                        and sensor artefacts. Unlike Gaussian blur, median filtering
                        is a non-linear operation that preserves sharp PCB trace edges
                        while eliminating isolated bright/dark pixels caused by
                        imaging noise or dust particles.
                    </div>
                    <div class='ec-result'>
                        ✓ Noise estimate: {m_orig['noise_estimate']:.0f} → {m_filt['noise_estimate']:.0f}
                        ({metrics['noise_reduction']:.1f}% reduction)
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with exp_col2:
                st.markdown(f"""
                <div class='explain-card'>
                    <div class='ec-step'>Step 3</div>
                    <div class='ec-title'>🔆 CLAHE Enhancement</div>
                    <div class='ec-desc'>
                        Contrast Limited Adaptive Histogram Equalization (CLAHE)
                        enhances local contrast by equalising histograms within
                        small tile regions (8×8 grid) rather than globally.
                        The clip limit (2.0) prevents over-amplification of noise.
                        This is critical for PCB inspection where defects appear
                        under uneven illumination conditions.
                    </div>
                    <div class='ec-result'>
                        ✓ Contrast: {m_orig['contrast']:.1f} → {m_enh['contrast']:.1f}
                        ({metrics['contrast_gain']:+.1f}%) &nbsp;|&nbsp;
                        Dynamic range: {m_enh['dynamic_range']} levels
                    </div>
                </div>

                <div class='explain-card'>
                    <div class='ec-step'>Pipeline Summary</div>
                    <div class='ec-title'>✅ Pre-processing Calibration Result</div>
                    <div class='ec-desc'>
                        The pre-processed image is now ready for downstream modules.
                        Colour information has been removed, noise has been suppressed,
                        and local contrast has been enhanced — producing a calibrated
                        grayscale image where PCB traces and potential defect regions
                        are clearly distinguishable.
                    </div>
                    <div class='ec-result'>
                        ✓ Output: {processed.shape[1]}×{processed.shape[0]} px
                        &nbsp;|&nbsp; 1-channel grayscale
                        &nbsp;|&nbsp; Range: {int(processed.min())}–{int(processed.max())}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Final comparison ────────────────────────────────────
            st.markdown("---")
            st.markdown("<div class='section-label'>🖼️ Final Comparison — Original vs Preprocessed</div>",
                        unsafe_allow_html=True)

            fc1, fc2 = st.columns(2, gap="medium")
            with fc1:
                st.markdown("<p style='color:#3a5070;font-size:0.75rem;margin-bottom:5px;'>INPUT — ORIGINAL BGR</p>",
                            unsafe_allow_html=True)
                st.image(original_rgb, use_container_width=True)
            with fc2:
                st.markdown("<p style='color:#3a86ff;font-size:0.75rem;margin-bottom:5px;'>OUTPUT — PREPROCESSED GRAYSCALE</p>",
                            unsafe_allow_html=True)
                st.image(processed, clamp=True, channels="GRAY", use_container_width=True)

            # ── Download ────────────────────────────────────────────
            st.markdown("---")
            _, dl_col, _ = st.columns([2, 1, 2])
            with dl_col:
                ok, buf = cv2.imencode(".png", processed)
                if ok:
                    st.download_button(
                        label="⬇ Download Preprocessed Image",
                        data=buf.tobytes(),
                        file_name=f"preprocessed_{os.path.splitext(uploaded_file.name)[0]}.png",
                        mime="image/png",
                        use_container_width=True,
                    )

        except Exception as err:
            st.error(f"❌ Pre-processing failed: {err}")

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

else:
    st.info("👆 Upload a PCB image above to get started.")
