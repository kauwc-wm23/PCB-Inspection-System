"""
=============================================================================
File        : gui/interface.py
Project     : PCB Inspection System
Description : Professional Streamlit interface - Modules 1 & 2 active (Tabbed Results).
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
from modules.segmentation import get_segmentation_stages
from modules.feature_analysis import analyse_features
from modules.reporting import generate_inspection_summary, format_defect_table_data

OUTPUT_DIR_PRE = os.path.join(PROJECT_ROOT, "outputs", "preprocessing")
OUTPUT_DIR_SEG = os.path.join(PROJECT_ROOT, "outputs", "segmentation")
os.makedirs(OUTPUT_DIR_PRE, exist_ok=True)
os.makedirs(OUTPUT_DIR_SEG, exist_ok=True)

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
# CSS  (ORIGINAL DESIGN KEPT)
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

/* Tab styling - ADDED, same colour language as original design */
.stTabs [data-baseweb="tab-list"] {
    gap:8px;
    background:#0d1522;
    border:1px solid #1f2d45;
    border-radius:12px;
    padding:8px 10px;
}
.stTabs [data-baseweb="tab"] {
    border-radius:8px;
    color:#6b85b0;
    font-size:0.84rem;
    font-weight:600;
    padding:10px 18px;
}
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,#0f2347,#112040) !important;
    color:#7ab3ff !important;
    border:1px solid #3a86ff !important;
}

hr { border-color:#131e30 !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown(
        "<div class='sb-title'>🔬 PCB Inspection</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<div class='sb-tag'>Automated Quality Control System</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    # ========================================================
    # Inspection Workflow
    # ========================================================

    st.markdown(
        "<div class='sb-section'>📋 Inspection Workflow</div>",
        unsafe_allow_html=True
    )

    st.markdown("""
        <div class='sb-step-on'>
            <div class='sn'>✦ Image Pre-processing</div>
            <div class='sd'>Contrast enhancement &amp; noise reduction</div>
        </div>

        <div class='sb-step-on'>
            <div class='sn'>✦ Defect Segmentation</div>
            <div class='sd'>Difference + Otsu + Morphology + Contours</div>
        </div>

        <div class='sb-step-on'>
            <div class='sn'>✦ Feature Analysis</div>
            <div class='sd'>Defect characteristic analysis</div>
        </div>

        <div class='sb-step-on'>
            <div class='sn'>✦ Inspection Report</div>
            <div class='sd'>Result visualisation &amp; summary</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ========================================================
    # System Information
    # ========================================================

    st.markdown(
        "<div class='sb-section'>⚙️ System Information</div>",
        unsafe_allow_html=True
    )

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
    <div class='hero-badge'>Modules 1, 2, 3 &amp; 4 — Integrated Inspection Pipeline</div>
    <div class='hero-title'>🔬 PCB Defect Inspection System</div>
    <div class='hero-subtitle'>
        Upload a defective PCB image and its matching template.
        Run the integrated pipeline, then use the tabs to inspect each module result.
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
    <div class='pipe-step active'>
        <div class='icon'>🔍</div>
        <div class='plabel'>Segmentation</div>
    </div>
    <div class='pipe-arrow'>→</div>
    <div class='pipe-step active'>
        <div class='icon'>📊</div>
        <div class='plabel'>Analysis</div>
    </div>
    <div class='pipe-arrow'>→</div>
    <div class='pipe-step active'>
        <div class='icon'>📄</div>
        <div class='plabel'>Report</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Upload
# ============================================================

st.markdown("<div class='section-label'>📂 Step 1 — Upload PCB Image Pair</div>", unsafe_allow_html=True)

up1, up2 = st.columns(2, gap="medium")

with up1:
    defective_file = st.file_uploader(
        "Choose defective / test PCB image",
        type=["jpg", "jpeg", "png"],
        help="Upload the defective/test PCB image",
        key="defective_upload",
    )

with up2:
    template_file = st.file_uploader(
        "Choose matching defect-free template image",
        type=["jpg", "jpeg", "png"],
        help="Upload the matching PCB type and orientation template",
        key="template_upload",
    )

if defective_file is None or template_file is None:
    st.markdown("""
        <div class='upload-hint'>
            ⬆️ &nbsp; Upload both the defective/test image and its matching template image.<br>
            <span style='font-size:0.78rem;'>Use the same PCB type and orientation for both images.</span>
        </div>
    """, unsafe_allow_html=True)

# ============================================================
# Main flow
# ============================================================

if defective_file is not None and template_file is not None:

    # Decode both images
    test_bytes = np.frombuffer(defective_file.getvalue(), np.uint8)
    tmpl_bytes = np.frombuffer(template_file.getvalue(), np.uint8)

    test_bgr = cv2.imdecode(test_bytes, cv2.IMREAD_COLOR)
    template_bgr = cv2.imdecode(tmpl_bytes, cv2.IMREAD_COLOR)

    if test_bgr is None or template_bgr is None:
        st.error("❌ Unable to decode one of the uploaded images.")
        st.stop()

    test_rgb = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2RGB)
    template_rgb = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2RGB)

    st.markdown("---")
    st.markdown("<div class='section-label'>📷 Step 2 — Review Uploaded Image Pair</div>", unsafe_allow_html=True)

    rv1, rv2 = st.columns(2, gap="medium")

    with rv1:
        st.image(test_rgb, caption=f"Defective/Test — {defective_file.name}", use_container_width=True)
        st.markdown(f"""
        <div class='img-info-card'>
            <div class='ril'>Filename</div>
            <div class='riv'>{defective_file.name}</div>
            <div class='ril'>Image Type</div>
            <div class='riv'>Defective / Test PCB</div>
            <div class='ril'>Resolution</div>
            <div class='riv'>{test_bgr.shape[1]} × {test_bgr.shape[0]} px</div>
            <div class='ril'>Colour Space</div>
            <div class='riv'>BGR (3 channels)</div>
            <div class='ril'>File Size</div>
            <div class='riv'>{defective_file.size / 1024:.1f} KB</div>
            <div class='ril'>Bit Depth</div>
            <div class='riv'>8-bit per channel</div>
        </div>
        """, unsafe_allow_html=True)

    with rv2:
        st.image(template_rgb, caption=f"Template — {template_file.name}", use_container_width=True)
        st.markdown(f"""
        <div class='img-info-card'>
            <div class='ril'>Filename</div>
            <div class='riv'>{template_file.name}</div>
            <div class='ril'>Image Type</div>
            <div class='riv'>Defect-Free Template</div>
            <div class='ril'>Resolution</div>
            <div class='riv'>{template_bgr.shape[1]} × {template_bgr.shape[0]} px</div>
            <div class='ril'>Colour Space</div>
            <div class='riv'>BGR (3 channels)</div>
            <div class='ril'>File Size</div>
            <div class='riv'>{template_file.size / 1024:.1f} KB</div>
            <div class='ril'>Bit Depth</div>
            <div class='riv'>8-bit per channel</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='section-label'>⚡ Step 3 — Run Inspection Pipeline</div>", unsafe_allow_html=True)

    col_btn, col_hint = st.columns([1, 4], gap="small")
    with col_btn:
        run_btn = st.button("▶ Run Inspection", type="primary", use_container_width=True)
    with col_hint:
        st.markdown(
            "<p style='color:#2a4060;font-size:0.84rem;margin-top:8px;'>"
            "Pipeline: Pre-processing → Template Difference → Otsu → Opening → Closing → Contour Detection</p>",
            unsafe_allow_html=True,
        )

    if run_btn:

        t_start = time.time()

        test_tmp_path = None
        tmpl_tmp_path = None

        try:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_test:
                tmp_test.write(defective_file.getvalue())
                test_tmp_path = tmp_test.name

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_tmpl:
                tmp_tmpl.write(template_file.getvalue())
                tmpl_tmp_path = tmp_tmpl.name

            with st.spinner("🔧 Running PCB Inspection Pipeline..."):

                # ------------------------------
                # Module 1: preprocess TEST
                # ------------------------------
                test_stages, test_metrics = get_preprocessing_stages(test_tmp_path)
                processed_test = test_stages["enhanced"]

                # ------------------------------
                # Module 1: preprocess TEMPLATE
                # ------------------------------
                template_stages, template_metrics = get_preprocessing_stages(tmpl_tmp_path)
                processed_template = template_stages["enhanced"]

                # ------------------------------
                # Module 2: template-based segmentation
                # ------------------------------
                seg_stages, seg_metrics = get_segmentation_stages(
                    processed_test,
                    processed_template
                )

                # ------------------------------
                # Module 3: feature analysis
                # ------------------------------
                defects, analysis_metrics = analyse_features(
                    seg_stages["morphology"],
                    seg_metrics["contours"]
                )

            proc_time = time.time() - t_start

            # Store in session state so tabs remain stable
            st.session_state["pcb_result"] = {
                "test_stages": test_stages,
                "test_metrics": test_metrics,
                "template_stages": template_stages,
                "template_metrics": template_metrics,
                "processed_test": processed_test,
                "processed_template": processed_template,
                "seg_stages": seg_stages,
                "seg_metrics": seg_metrics,
                "defects": defects,
                "analysis_metrics": analysis_metrics,
                "proc_time": proc_time,
                "test_name": defective_file.name,
                "template_name": template_file.name,
            }

        except Exception as err:
            st.error(f"❌ Inspection failed: {err}")

        finally:
            if test_tmp_path and os.path.exists(test_tmp_path):
                os.remove(test_tmp_path)
            if tmpl_tmp_path and os.path.exists(tmpl_tmp_path):
                os.remove(tmpl_tmp_path)

    # ============================================================
    # Result tabs
    # ============================================================

    if "pcb_result" in st.session_state:

        result = st.session_state["pcb_result"]

        test_stages = result["test_stages"]
        test_metrics = result["test_metrics"]
        template_stages = result["template_stages"]
        template_metrics = result["template_metrics"]
        processed_test = result["processed_test"]
        processed_template = result["processed_template"]
        seg_stages = result["seg_stages"]
        seg_metrics = result["seg_metrics"]
        proc_time = result["proc_time"]

        out_name_pre = f"preprocessed_{result['test_name']}"
        out_path_pre = os.path.join(OUTPUT_DIR_PRE, out_name_pre)
        cv2.imwrite(out_path_pre, processed_test)

        out_name_seg = f"segmented_{os.path.splitext(result['test_name'])[0]}.png"
        out_path_seg = os.path.join(OUTPUT_DIR_SEG, out_name_seg)
        if "overlay" in seg_stages:
            cv2.imwrite(out_path_seg, seg_stages["overlay"])

        st.markdown(
            f"<div class='success-banner'>"
            f"✅ &nbsp; Inspection completed in <strong>{proc_time:.2f}s</strong> &nbsp;|&nbsp; "
            f"Detected <strong>{seg_metrics.get('defect_count', 0)}</strong> potential defect region(s)"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ------------------------------------------------------------
        # TOP TABS - user can click each module result
        # ------------------------------------------------------------
        tab1, tab2, tab3, tab4 = st.tabs([
            "🖼️ Module 1 — Pre-processing",
            "🔍 Module 2 — Defect Segmentation",
            "📊 Module 3 — Feature Analysis",
            "📄 Module 4 — Inspection Report",
        ])

        # ============================================================
        # TAB 1 — MODULE 1
        # ============================================================
        with tab1:

            st.markdown("<div class='section-label'>🔬 Module 1 — Test Image Pre-processing Stages</div>",
                        unsafe_allow_html=True)

            sc1, sc2, sc3, sc4 = st.columns(4, gap="small")
            stage_defs = [
                (sc1, "original",  "Stage 1", "Original",        "Raw BGR → Grayscale", False),
                (sc2, "grayscale", "Stage 2", "Grayscale",       "Luminance only", False),
                (sc3, "filtered",  "Stage 3", "Median Filtered", "Noise suppressed", False),
                (sc4, "enhanced",  "Stage 4", "CLAHE Enhanced",  "Local contrast boosted", True),
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
                    st.image(test_stages[key], clamp=True, channels="GRAY",
                             use_container_width=True)

            st.markdown("---")
            st.markdown("<div class='section-label'>📊 Image Quality Metrics</div>",
                        unsafe_allow_html=True)

            m_orig = test_metrics["original"]
            m_enh = test_metrics["enhanced"]
            m_filt = test_metrics["filtered"]

            contrast_gain_str = (
                f"<span class='qm-delta pos'>▲ +{test_metrics['contrast_gain']}%</span>"
                if test_metrics["contrast_gain"] >= 0
                else f"<span class='qm-delta neg'>▼ {test_metrics['contrast_gain']}%</span>"
            )

            noise_str = (
                f"<span class='qm-delta pos'>▼ {test_metrics['noise_reduction']:.1f}% reduced</span>"
                if test_metrics["noise_reduction"] > 0
                else f"<span class='qm-delta neu'>— No change</span>"
            )

            dr_gain = test_metrics["dynamic_range_gain"]
            dr_str = (
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
                    <div class='qm-delta neu'>Modules 1 + 2 total</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("<div class='section-label'>📈 Pixel Intensity Histogram — Before vs After</div>",
                        unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(12, 3.2))
            fig.patch.set_facecolor("#0b0f19")
            ax.set_facecolor("#0e1520")

            ax.hist(test_stages["original"].ravel(), bins=256, range=(0,255),
                    color="#5fa8ff", alpha=0.55, label="Original (Grayscale)", density=True)
            ax.hist(test_stages["enhanced"].ravel(), bins=256, range=(0,255),
                    color="#22c55e", alpha=0.65, label="After CLAHE (Final)", density=True)

            ax.set_xlabel("Pixel Intensity (0 = black · 255 = white)", color="#4a6890", fontsize=9)
            ax.set_ylabel("Normalised Frequency", color="#4a6890", fontsize=9)
            ax.tick_params(colors="#2a4060", labelsize=8)
            ax.spines[["top","right","left","bottom"]].set_color("#1a2840")
            ax.set_xlim(0, 255)
            ax.set_title(
                "Histogram comparison shows CLAHE redistributes pixel intensities "
                "for improved local contrast",
                color="#3a5070", fontsize=8.5, pad=8
            )
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
                        relevant to PCB structural analysis.
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
                        and sensor artefacts. Median filtering preserves PCB trace edges
                        while removing isolated bright or dark noise pixels.
                    </div>
                    <div class='ec-result'>
                        ✓ Noise estimate: {m_orig['noise_estimate']:.0f} → {m_filt['noise_estimate']:.0f}
                        ({test_metrics['noise_reduction']:.1f}% reduction)
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
                        enhances local contrast within small image regions.
                        The clip limit prevents excessive noise amplification and
                        improves the visibility of PCB traces and defect-sensitive areas.
                    </div>
                    <div class='ec-result'>
                        ✓ Contrast: {m_orig['contrast']:.1f} → {m_enh['contrast']:.1f}
                        ({test_metrics['contrast_gain']:+.1f}%) &nbsp;|&nbsp;
                        Dynamic range: {m_enh['dynamic_range']} levels
                    </div>
                </div>

                <div class='explain-card'>
                    <div class='ec-step'>Pipeline Summary</div>
                    <div class='ec-title'>✅ Pre-processing Calibration Result</div>
                    <div class='ec-desc'>
                        The preprocessed image is ready for Module 2.
                        Colour information has been removed, noise has been suppressed,
                        and local contrast has been enhanced before template comparison
                        and defect segmentation.
                    </div>
                    <div class='ec-result'>
                        ✓ Output: {processed_test.shape[1]}×{processed_test.shape[0]} px
                        &nbsp;|&nbsp; 1-channel grayscale
                        &nbsp;|&nbsp; Range: {int(processed_test.min())}–{int(processed_test.max())}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("<div class='section-label'>🖼️ Final Comparison — Original vs Preprocessed</div>",
                        unsafe_allow_html=True)

            fc1, fc2 = st.columns(2, gap="medium")
            with fc1:
                st.markdown(
                    "<p style='color:#3a5070;font-size:0.75rem;margin-bottom:5px;'>INPUT — ORIGINAL PCB</p>",
                    unsafe_allow_html=True)
                st.image(test_rgb, use_container_width=True)

            with fc2:
                st.markdown(
                    "<p style='color:#3a86ff;font-size:0.75rem;margin-bottom:5px;'>OUTPUT — PREPROCESSED GRAYSCALE</p>",
                    unsafe_allow_html=True)
                st.image(processed_test, clamp=True, channels="GRAY", use_container_width=True)

            st.markdown("---")
            _, dl_col, _ = st.columns([2, 1, 2])
            with dl_col:
                ok, enc = cv2.imencode(".png", processed_test)
                if ok:
                    st.download_button(
                        label="⬇ Download Preprocessed Image",
                        data=enc.tobytes(),
                        file_name=f"preprocessed_{os.path.splitext(result['test_name'])[0]}.png",
                        mime="image/png",
                        use_container_width=True,
                    )

        # ============================================================
        # TAB 2 — MODULE 2
        # ============================================================
        with tab2:

            st.markdown("<div class='section-label'>🔬 Module 2 — Defect Segmentation Stages</div>",
                        unsafe_allow_html=True)

            # 7 stages, shown in 4 + 3 layout
            stage_order = [
                ("test_image", "Stage 1", "Preprocessed Test", "Module 1 output"),
                ("template_image", "Stage 2", "Preprocessed Template", "Reference input"),
                ("difference", "Stage 3", "Absolute Difference", "Test vs template"),
                ("otsu_binary", "Stage 4", "Otsu Binary", "Automatic thresholding"),
                ("opening", "Stage 5", "Opening", "Small-noise removal"),
                ("morphology", "Stage 6", "Closing", "Structural refinement"),
                ("overlay", "Stage 7", "Defect Overlay", "Final bounding boxes"),
            ]

            row1 = st.columns(4, gap="small")
            for col, item in zip(row1, stage_order[:4]):
                key, num, name, desc = item
                with col:
                    st.markdown(f"""
                        <div class='stage-card'>
                            <div class='stage-num'>{num}</div>
                            <div class='stage-name'>{name}</div>
                            <div class='stage-desc'>{desc}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    img = seg_stages.get(key)
                    if img is None:
                        st.info("Stage output is not available from segmentation.py.")
                    elif len(img.shape) == 3:
                        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
                    else:
                        st.image(img, clamp=True, channels="GRAY", use_container_width=True)

            row2 = st.columns(3, gap="small")
            for idx, (col, item) in enumerate(zip(row2, stage_order[4:])):
                key, num, name, desc = item
                with col:
                    cls = "stage-card final" if idx == 2 else "stage-card"
                    st.markdown(f"""
                        <div class='{cls}'>
                            <div class='stage-num'>{num}</div>
                            <div class='stage-name'>{name}</div>
                            <div class='stage-desc'>{desc}</div>
                        </div>
                    """, unsafe_allow_html=True)
                    img = seg_stages.get(key)
                    if img is None:
                        st.info("Stage output is not available from segmentation.py.")
                    elif len(img.shape) == 3:
                        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
                    else:
                        st.image(img, clamp=True, channels="GRAY", use_container_width=True)

            st.markdown("---")
            st.markdown("<div class='section-label'>📊 Segmentation Metrics</div>",
                        unsafe_allow_html=True)

            thresh = seg_metrics.get("threshold", "Auto")
            count = seg_metrics.get("defect_count", seg_metrics.get("detected_regions", 0))
            area_px = seg_metrics.get("defect_area_px", 0)
            area_pct = seg_metrics.get("defect_area_pct", 0.0)

            st.markdown(f"""
            <div class='qm-row'>
                <div class='qm-card'>
                    <div class='qm-label'>Otsu Threshold</div>
                    <div class='qm-value'>{thresh}</div>
                    <div class='qm-delta neu'>Automatically selected</div>
                </div>
                <div class='qm-card'>
                    <div class='qm-label'>Defect Regions</div>
                    <div class='qm-value'>{count}</div>
                    <div class='qm-delta pos'>Valid contours detected</div>
                </div>
                <div class='qm-card'>
                    <div class='qm-label'>Defect Area</div>
                    <div class='qm-value'>{area_px} px</div>
                    <div class='qm-delta neu'>Total contour area</div>
                </div>
                <div class='qm-card'>
                    <div class='qm-label'>Surface Coverage</div>
                    <div class='qm-value'>{area_pct:.4f}%</div>
                    <div class='qm-delta neg'>Defect ratio</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("<div class='section-label'>📖 Segmentation Pipeline Breakdown</div>",
                        unsafe_allow_html=True)

            exp1, exp2 = st.columns(2, gap="medium")

            with exp1:
                st.markdown(f"""
                <div class='explain-card'>
                    <div class='ec-step'>Step 1</div>
                    <div class='ec-title'>↔ Absolute Difference</div>
                    <div class='ec-desc'>
                        Compares the preprocessed test PCB with its matching defect-free template.
                        Unchanged PCB structures are suppressed while abnormal regions remain visible.
                    </div>
                </div>

                <div class='explain-card'>
                    <div class='ec-step'>Step 2</div>
                    <div class='ec-title'>◐ Otsu Thresholding</div>
                    <div class='ec-desc'>
                        Automatically selects a global threshold from the difference image and converts
                        candidate defect pixels into a binary mask.
                    </div>
                    <div class='ec-result'>✓ Threshold: {thresh}</div>
                </div>
                """, unsafe_allow_html=True)

            with exp2:
                st.markdown(f"""
                <div class='explain-card'>
                    <div class='ec-step'>Step 3</div>
                    <div class='ec-title'>◼ Morphological Opening &amp; Closing</div>
                    <div class='ec-desc'>
                        Opening removes small isolated noise. Closing reconnects small discontinuities
                        and improves the continuity of candidate defect regions.
                    </div>
                </div>

                <div class='explain-card'>
                    <div class='ec-step'>Step 4</div>
                    <div class='ec-title'>▣ Contour Detection</div>
                    <div class='ec-desc'>
                        Extracts independent defect candidates from the refined mask and draws
                        bounding boxes for visual localisation.
                    </div>
                    <div class='ec-result'>✓ Detected regions: {count}</div>
                </div>
                """, unsafe_allow_html=True)


            st.markdown("---")
            _, dl_col2, _ = st.columns([2, 1, 2])
            with dl_col2:
                overlay = seg_stages.get("overlay")
                ok, enc2 = (False, None) if overlay is None else cv2.imencode(".png", overlay)
                if ok:
                    st.download_button(
                        label="⬇ Download Defect Detection Result",
                        data=enc2.tobytes(),
                        file_name=out_name_seg,
                        mime="image/png",
                        use_container_width=True,
                    )

        # ============================================================
        # TAB 3 — MODULE 3
        # ============================================================
        with tab3:

            st.markdown("<div class='section-label'>🔬 Module 3 — Feature Analysis Results</div>",
                        unsafe_allow_html=True)

            defects = result.get("defects", [])
            analysis_metrics = result.get("analysis_metrics", {})

            if not defects or analysis_metrics.get("total_defects", 0) == 0:
                st.markdown("""
                <div class='explain-card'>
                    <div class='ec-title'>✓ No valid defect regions detected.</div>
                    <div class='ec-desc'>
                        The PCB inspection found no defects that match the segmentation criteria.
                        The defective/test image appears to be structurally identical to the template.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:

                st.markdown("---")
                st.markdown("<div class='section-label'>📊 Inspection-Level Metrics</div>",
                            unsafe_allow_html=True)

                total_defects = analysis_metrics.get("total_defects", 0)
                total_area = analysis_metrics.get("total_defect_area", 0)
                avg_area = analysis_metrics.get("average_area", 0.0)
                largest = analysis_metrics.get("largest_defect", {})
                largest_area = largest.get("area", 0) if largest else 0

                st.markdown(f"""
                <div class='qm-row'>
                    <div class='qm-card'>
                        <div class='qm-label'>Total Defects</div>
                        <div class='qm-value'>{total_defects}</div>
                        <div class='qm-delta pos'>Connected components identified</div>
                    </div>
                    <div class='qm-card'>
                        <div class='qm-label'>Total Defect Area</div>
                        <div class='qm-value'>{total_area} px²</div>
                        <div class='qm-delta neu'>Sum of all defect regions</div>
                    </div>
                    <div class='qm-card'>
                        <div class='qm-label'>Average Defect Area</div>
                        <div class='qm-value'>{avg_area:.1f} px²</div>
                        <div class='qm-delta neu'>Mean area per defect</div>
                    </div>
                    <div class='qm-card'>
                        <div class='qm-label'>Largest Defect Area</div>
                        <div class='qm-value'>{largest_area} px²</div>
                        <div class='qm-delta pos'>Maximum defect size</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("---")
                st.markdown("<div class='section-label'>📋 Individual Defect Features</div>",
                            unsafe_allow_html=True)

                # Build table data
                table_data = []
                for defect in defects:
                    bbox = defect.get("bounding_box", {})
                    loc = defect.get("location", {})
                    table_data.append({
                        "Defect ID": defect.get("id", ""),
                        "Area (px²)": defect.get("area", 0),
                        "Width (px)": defect.get("width", 0),
                        "Height (px)": defect.get("height", 0),
                        "Centroid X": f"{loc.get('x', 0):.1f}",
                        "Centroid Y": f"{loc.get('y', 0):.1f}",
                        "BBox X": bbox.get("x", 0),
                        "BBox Y": bbox.get("y", 0),
                        "BBox W": bbox.get("width", 0),
                        "BBox H": bbox.get("height", 0),
                    })

                # Display table using st.dataframe
                st.dataframe(
                    table_data,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Defect ID": st.column_config.NumberColumn(format="%d"),
                        "Area (px²)": st.column_config.NumberColumn(format="%d"),
                        "Width (px)": st.column_config.NumberColumn(format="%d"),
                        "Height (px)": st.column_config.NumberColumn(format="%d"),
                        "Centroid X": st.column_config.TextColumn(),
                        "Centroid Y": st.column_config.TextColumn(),
                        "BBox X": st.column_config.NumberColumn(format="%d"),
                        "BBox Y": st.column_config.NumberColumn(format="%d"),
                        "BBox W": st.column_config.NumberColumn(format="%d"),
                        "BBox H": st.column_config.NumberColumn(format="%d"),
                    }
                )

                st.markdown("---")
                st.markdown("<div class='section-label'>📖 Module 3 — Feature Analysis Overview</div>",
                            unsafe_allow_html=True)

                st.markdown("""
                <div class='explain-card'>
                    <div class='ec-step'>Connected Component Analysis</div>
                    <div class='ec-title'>✓ Defect Region Identification</div>
                    <div class='ec-desc'>
                        Module 3 performs connected component analysis on the morphologically-refined
                        binary mask from Module 2. Each isolated defect region is identified and
                        assigned a unique ID. Centroids, bounding boxes, and area measurements are
                        extracted for each connected component using OpenCV's connectedComponentsWithStats().
                    </div>
                </div>

                <div class='explain-card'>
                    <div class='ec-step'>Feature Extraction</div>
                    <div class='ec-title'>✓ Quantitative Defect Characterization</div>
                    <div class='ec-desc'>
                        For each defect:
                        <br>• <strong>Area:</strong> Number of pixels in the defect region
                        <br>• <strong>Width / Height:</strong> Dimensions of the bounding box
                        <br>• <strong>Centroid:</strong> (X, Y) coordinates of the defect centre
                        <br>• <strong>Bounding Box:</strong> (x, y, width, height) of the region
                        <br><br>
                        These measurements enable automated grading and statistical analysis
                        for quality control workflows.
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ============================================================
        # TAB 4 — MODULE 4
        # ============================================================
        with tab4:

            st.markdown("<div class='section-label'>🔬 Module 4 — Inspection Report Summary</div>",
                        unsafe_allow_html=True)

            defects = result.get("defects", [])
            analysis_metrics = result.get("analysis_metrics", {})
            test_name = result.get("test_name", "test_image.jpg")
            template_name = result.get("template_name", "template_image.jpg")
            proc_time = result.get("proc_time", 0.0)

            # Generate inspection summary report
            inspection_report = generate_inspection_summary(
                test_filename=test_name,
                template_filename=template_name,
                processing_time=proc_time,
                defects=defects,
                analysis_metrics=analysis_metrics
            )

            st.markdown("---")
            st.markdown("<div class='section-label'>📊 Inspection Summary Metrics</div>",
                        unsafe_allow_html=True)

            # Display high-level metrics
            total_defects = inspection_report["total_defects"]
            total_area = inspection_report["total_defect_area"]
            avg_area = inspection_report["average_defect_area"]
            largest_area = inspection_report["largest_defect_area"]
            status = inspection_report["inspection_status"]

            status_color = "#22c55e" if status == "No Defect Detected" else "#ef4444"
            status_icon = "✓" if status == "No Defect Detected" else "⚠"

            st.markdown(f"""
            <div class='qm-row'>
                <div class='qm-card'>
                    <div class='qm-label'>Inspection Status</div>
                    <div class='qm-value' style='color:{status_color};'>{status_icon} {status}</div>
                    <div class='qm-delta neu'>Report generated at {inspection_report['timestamp']}</div>
                </div>
                <div class='qm-card'>
                    <div class='qm-label'>Total Defects</div>
                    <div class='qm-value'>{total_defects}</div>
                    <div class='qm-delta neu'>Regions identified</div>
                </div>
                <div class='qm-card'>
                    <div class='qm-label'>Total Defect Area</div>
                    <div class='qm-value'>{total_area} px²</div>
                    <div class='qm-delta neu'>Combined region size</div>
                </div>
                <div class='qm-card'>
                    <div class='qm-label'>Processing Time</div>
                    <div class='qm-value'>{proc_time:.2f}s</div>
                    <div class='qm-delta neu'>Full pipeline execution</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("<div class='section-label'>📝 Automated Inspection Summary</div>",
                        unsafe_allow_html=True)

            st.markdown(f"""
            <div class='explain-card'>
                <div class='ec-desc'>
                    {inspection_report['summary_text']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("<div class='section-label'>📸 Final Inspection Visualisation</div>",
                        unsafe_allow_html=True)

            seg_stages = result.get("seg_stages", {})
            overlay = seg_stages.get("overlay")

            if overlay is not None:
                # Display the overlay image
                if len(overlay.shape) == 3:
                    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                    st.image(overlay_rgb, use_container_width=True,
                            caption="Final inspection result — detected defect regions highlighted by bounding boxes.")
                else:
                    st.image(overlay, clamp=True, channels="GRAY", use_container_width=True,
                            caption="Final inspection result — detected defect regions highlighted by bounding boxes.")
            else:
                st.info("Overlay image is not available from segmentation.py.")

            st.markdown("---")
            st.markdown("<div class='section-label'>📋 Defect Detail Summary</div>",
                        unsafe_allow_html=True)

            if not defects or total_defects == 0:
                st.markdown("""
                <div class='explain-card'>
                    <div class='ec-desc'>
                        No individual defect measurements are available because no valid defect regions were detected.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Build and display defect table
                table_data = format_defect_table_data(defects)

                st.dataframe(
                    table_data,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Defect ID": st.column_config.NumberColumn(format="%d"),
                        "Area (px²)": st.column_config.NumberColumn(format="%d"),
                        "Width (px)": st.column_config.NumberColumn(format="%d"),
                        "Height (px)": st.column_config.NumberColumn(format="%d"),
                        "Centroid X": st.column_config.TextColumn(),
                        "Centroid Y": st.column_config.TextColumn(),
                        "BBox (x, y, w, h)": st.column_config.TextColumn(),
                    }
                )

            st.markdown("---")
            st.markdown("<div class='section-label'>📊 Inspection Information</div>",
                        unsafe_allow_html=True)

            st.markdown(f"""
            <div class='explain-card'>
                <div class='ec-step'>Test Image</div>
                <div class='ec-result'>{inspection_report['test_filename']}</div>

                <div class='ec-step' style='margin-top:10px;'>Template Image</div>
                <div class='ec-result'>{inspection_report['template_filename']}</div>

                <div class='ec-step' style='margin-top:10px;'>Processing Time</div>
                <div class='ec-result'>{inspection_report['processing_time']:.2f} seconds</div>

                <div class='ec-step' style='margin-top:10px;'>Report Generated</div>
                <div class='ec-result'>{inspection_report['timestamp']}</div>
            </div>
            """, unsafe_allow_html=True)

else:
    st.info("👆 Upload both a defective/test PCB image and its matching template image to get started.")
