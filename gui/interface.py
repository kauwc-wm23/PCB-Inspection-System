"""
=============================================================================
File        : gui/interface.py
Project     : PCB Inspection System
Description : Streamlit integration interface for the PCB inspection pipeline.
=============================================================================
"""

import os
import sys
import time
import tempfile
import io
import importlib

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

from modules.dataset_paths import find_reference_image
from modules.preprocessing import get_preprocessing_stages
from modules.segmentation import get_segmentation_stages
from modules.feature_analysis import analyse_features
from modules.inspection_evaluation import assess_defect_severity
from modules.reporting import generate_inspection_summary, format_defect_table_data
from gui import preprocessing_presentation as _preprocessing_presentation

# Streamlit may retain the previous helper module during a hot reload while
# rerunning this interface. Reload only that stale module when the newly added
# change-map helper is absent; normal application starts do not reload it.
if not hasattr(_preprocessing_presentation, "create_filtering_change_map"):
    _preprocessing_presentation = importlib.reload(_preprocessing_presentation)

create_filtering_change_map = (
    _preprocessing_presentation.create_filtering_change_map
)
extract_matching_center_rois = (
    _preprocessing_presentation.extract_matching_center_rois
)

OUTPUT_DIR_PRE = os.path.join(PROJECT_ROOT, "outputs", "preprocessing")
OUTPUT_DIR_SEG = os.path.join(PROJECT_ROOT, "outputs", "segmentation")
RESULT_SCHEMA_VERSION = 2
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
            <div class='sd'>Contrast enhancement &amp; potential impulse-noise suppression</div>
        </div>

        <div class='sb-step-on'>
            <div class='sn'>✦ Defect Segmentation</div>
            <div class='sd'>Defect region detection</div>
        </div>

        <div class='sb-step-on'>
            <div class='sn'>✦ Feature Analysis</div>
            <div class='sd'>Geometric defect measurements</div>
        </div>

        <div class='sb-step-on'>
            <div class='sn'>✦ Severity &amp; Spatial Analysis</div>
            <div class='sd'>Severity, priority &amp; spatial distribution</div>
        </div>

        <div class='sb-step-on'>
            <div class='sn'>✦ Inspection Report</div>
            <div class='sd'>Results, visualisation &amp; summary</div>
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
<div class='sb-info-val'>Ironbrotherstyle PCB-DATASET</div>

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
    <div class='hero-badge'>Integrated PCB Inspection Pipeline</div>
    <div class='hero-title'>🔬 PCB Defect Inspection System</div>
    <div class='hero-subtitle'>
        Upload a defective PCB image and its matching template.
        Run the automated workflow, then review each inspection stage in the result tabs.
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
        <div class='plabel'>Image Pre-processing</div>
    </div>
    <div class='pipe-arrow'>→</div>
    <div class='pipe-step active'>
        <div class='icon'>🔍</div>
        <div class='plabel'>Defect Segmentation</div>
    </div>
    <div class='pipe-arrow'>→</div>
    <div class='pipe-step active'>
        <div class='icon'>📊</div>
        <div class='plabel'>Feature Analysis</div>
    </div>
    <div class='pipe-arrow'>→</div>
    <div class='pipe-step active'>
        <div class='icon'>✅</div>
        <div class='plabel'>Severity Assessment</div>
    </div>
    <div class='pipe-arrow'>→</div>
    <div class='pipe-step active'>
        <div class='icon'>📄</div>
        <div class='plabel'>Inspection Report</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Upload
# ============================================================

st.markdown("<div class='section-label'>📂 Step 1 — Upload PCB Image</div>", unsafe_allow_html=True)

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
        "Matching defect-free template (optional for PCB-DATASET)",
        type=["jpg", "jpeg", "png"],
        help="A PCB_USED reference is selected automatically when the dataset filename has a known board ID.",
        key="template_upload",
    )

auto_template_path = find_reference_image(defective_file.name) if defective_file else None
template_available = template_file is not None or auto_template_path is not None

if defective_file is None:
    st.markdown("""
        <div class='upload-hint'>
            ⬆️ &nbsp; Upload a defective/test PCB image.<br>
            <span style='font-size:0.78rem;'>A matching PCB_USED reference is selected automatically for dataset images.</span>
        </div>
    """, unsafe_allow_html=True)
elif not template_available:
    st.warning("No matching PCB_USED reference was found. Upload the corresponding clean template.")
elif template_file is None:
    st.info(f"Using matching dataset reference automatically: {auto_template_path.name}")

# ============================================================
# Main flow
# ============================================================

if defective_file is not None and template_available:

    # Decode the test image and either the uploaded or auto-matched template.
    test_bytes = np.frombuffer(defective_file.getvalue(), np.uint8)
    test_bgr = cv2.imdecode(test_bytes, cv2.IMREAD_COLOR)

    if template_file is not None:
        tmpl_bytes = np.frombuffer(template_file.getvalue(), np.uint8)
        template_bgr = cv2.imdecode(tmpl_bytes, cv2.IMREAD_COLOR)
        template_name = template_file.name
        template_size = template_file.size
        template_source = "Uploaded clean template"
    else:
        template_bgr = cv2.imread(str(auto_template_path), cv2.IMREAD_COLOR)
        template_name = auto_template_path.name
        template_size = auto_template_path.stat().st_size
        template_source = "Auto-matched from dataset/PCB_USED"

    if test_bgr is None or template_bgr is None:
        st.error("❌ Unable to decode one of the uploaded images.")
        st.stop()

    test_rgb = cv2.cvtColor(test_bgr, cv2.COLOR_BGR2RGB)
    template_rgb = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2RGB)

    st.markdown("---")
    st.markdown("<div class='section-label'>📷 Step 2 — Review Uploaded Image Pair</div>", unsafe_allow_html=True)

    rv1, rv2 = st.columns(2, gap="medium")

    with rv1:
        st.image(test_rgb, caption=f"Defective/Test — {defective_file.name}", width="stretch")
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
        st.image(template_rgb, caption=f"Template — {template_name}", width="stretch")
        st.markdown(f"""
        <div class='img-info-card'>
            <div class='ril'>Filename</div>
            <div class='riv'>{template_name}</div>
            <div class='ril'>Image Type</div>
            <div class='riv'>{template_source}</div>
            <div class='ril'>Resolution</div>
            <div class='riv'>{template_bgr.shape[1]} × {template_bgr.shape[0]} px</div>
            <div class='ril'>Colour Space</div>
            <div class='riv'>BGR (3 channels)</div>
            <div class='ril'>File Size</div>
            <div class='riv'>{template_size / 1024:.1f} KB</div>
            <div class='ril'>Bit Depth</div>
            <div class='riv'>8-bit per channel</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='section-label'>⚡ Step 3 — Run Inspection Pipeline</div>", unsafe_allow_html=True)

    input_signature = (
        defective_file.name,
        defective_file.size,
        template_name,
        template_size,
    )

    col_btn, col_reset, col_hint = st.columns([1, 1, 4], gap="small")
    with col_btn:
        run_btn = st.button("▶ Run Inspection", type="primary", width="stretch")
    with col_reset:
        reset_btn = st.button("↺ Reset", width="stretch")
        if reset_btn:
            st.session_state.pop("pcb_result", None)
            st.rerun()
    with col_hint:
        st.markdown(
            "<p style='color:#2a4060;font-size:0.84rem;margin-top:8px;'>"
            "Pipeline: Image Pre-processing → Defect Segmentation → Feature Analysis → Severity Assessment → Inspection Report</p>",
            unsafe_allow_html=True,
        )

    if run_btn:

        t_start = time.time()

        test_tmp_path = None
        tmpl_tmp_path = None

        try:
            test_suffix = os.path.splitext(defective_file.name)[1] or ".jpg"
            with tempfile.NamedTemporaryFile(suffix=test_suffix, delete=False) as tmp_test:
                tmp_test.write(defective_file.getvalue())
                test_tmp_path = tmp_test.name

            if template_file is not None:
                template_suffix = os.path.splitext(template_file.name)[1] or ".jpg"
                with tempfile.NamedTemporaryFile(suffix=template_suffix, delete=False) as tmp_tmpl:
                    tmp_tmpl.write(template_file.getvalue())
                    tmpl_tmp_path = tmp_tmpl.name
                template_input_path = tmpl_tmp_path
            else:
                template_input_path = str(auto_template_path)

            with st.spinner("🔧 Running PCB Inspection Pipeline..."):

                # ------------------------------
                # Preprocess the test image
                # ------------------------------
                test_stages, test_metrics = get_preprocessing_stages(test_tmp_path)
                processed_test = test_stages["enhanced"]

                # ------------------------------
                # Preprocess the reference image
                # ------------------------------
                template_stages, template_metrics = get_preprocessing_stages(template_input_path)
                processed_template = template_stages["enhanced"]

                # ------------------------------
                # Template-based defect segmentation
                # ------------------------------
                seg_stages, seg_metrics = get_segmentation_stages(
                    processed_test,
                    processed_template
                )

                # ------------------------------
                # Geometric feature analysis
                # ------------------------------
                defects, analysis_metrics = analyse_features(
                    seg_stages["morphology"],
                    seg_metrics["contours"]
                )

                # ------------------------------
                # Per-defect severity and spatial assessment
                # ------------------------------
                evaluation = assess_defect_severity(
                    defects,
                    seg_stages["morphology"].shape,
                )

            proc_time = time.time() - t_start
            evaluation["processing_time"] = proc_time

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
                "evaluation": evaluation,
                "proc_time": proc_time,
                "test_name": defective_file.name,
                "template_name": template_name,
                "input_signature": input_signature,
                "result_schema_version": RESULT_SCHEMA_VERSION,
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

    if (
        "pcb_result" in st.session_state
        and st.session_state["pcb_result"].get("input_signature") == input_signature
        and st.session_state["pcb_result"].get("result_schema_version")
        == RESULT_SCHEMA_VERSION
    ):

        result = st.session_state["pcb_result"]

        test_stages = result["test_stages"]
        test_metrics = result["test_metrics"]
        template_stages = result["template_stages"]
        template_metrics = result["template_metrics"]
        processed_test = result["processed_test"]
        processed_template = result["processed_template"]
        seg_stages = result["seg_stages"]
        seg_metrics = result["seg_metrics"]
        evaluation = result["evaluation"]
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
            f"Result: <strong>{evaluation['status_label']}</strong> &nbsp;|&nbsp; "
            f"Detected <strong>{evaluation['total_defect_count']}</strong> potential defect region(s)"
            f"</div>",
            unsafe_allow_html=True,
        )

        # ------------------------------------------------------------
        # Result tabs
        # ------------------------------------------------------------
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🖼️ Image Pre-processing",
            "🔍 Defect Segmentation",
            "📊 Feature Analysis",
            "✅ Severity & Spatial Analysis",
            "📄 Inspection Report",
        ])

        # ============================================================
        # IMAGE PRE-PROCESSING
        # ============================================================
        with tab1:

            st.markdown("<div class='section-label'>🔬 Image Pre-processing Pipeline</div>",
                        unsafe_allow_html=True)

            sc1, sc2, sc3, sc4 = st.columns(4, gap="small")
            stage_defs = [
                (sc1, "original",  "Stage 1", "Original PCB",    "Raw colour PCB image.", False),
                (sc2, "grayscale", "Stage 2", "Grayscale",       "Single-channel intensity image.", False),
                (sc3, "filtered",  "Stage 3", "Median Filtered", "Small local variations suppressed; structures generally retained.", False),
                (sc4, "enhanced",  "Stage 4", "CLAHE Enhanced",  "Local contrast enhanced; segmentation input.", True),
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
                    if key == "original":
                        st.image(test_rgb, width="stretch")
                    else:
                        st.image(
                            test_stages[key], clamp=True, channels="GRAY", width="stretch"
                        )

            st.markdown("---")
            st.markdown("<div class='section-label'>📊 Image Quality Improvement</div>",
                        unsafe_allow_html=True)

            m_gray = test_metrics["grayscale"]
            m_enh = test_metrics["enhanced"]
            m_filt = test_metrics["filtered"]
            preprocessing_time = test_metrics.get("processing_time_seconds")
            preprocessing_time_available = (
                isinstance(
                    preprocessing_time,
                    (int, float, np.integer, np.floating),
                )
                and not isinstance(preprocessing_time, (bool, np.bool_))
                and np.isfinite(preprocessing_time)
                and preprocessing_time >= 0
            )
            preprocessing_time_display = (
                f"{float(preprocessing_time):.3f}s"
                if preprocessing_time_available
                else "N/A"
            )
            preprocessing_time_summary = (
                f"{float(preprocessing_time):.3f} seconds"
                if preprocessing_time_available
                else "N/A"
            )

            brightness_delta = m_enh["mean_brightness"] - m_gray["mean_brightness"]
            contrast_gain = 0.0
            if m_filt["contrast"] > 0:
                contrast_gain = (
                    (m_enh["contrast"] - m_filt["contrast"])
                    / m_filt["contrast"]
                ) * 100
            if contrast_gain > 0:
                contrast_gain_str = f"<span class='qm-delta pos'>▲ {contrast_gain:.1f}%</span>"
            elif contrast_gain < 0:
                contrast_gain_str = f"<span class='qm-delta neg'>▼ {abs(contrast_gain):.1f}%</span>"
            else:
                contrast_gain_str = "<span class='qm-delta neu'>— No change</span>"

            high_frequency_reduction = test_metrics.get(
                "high_frequency_reduction", test_metrics["noise_reduction"]
            )
            gray_high_frequency = m_gray.get(
                "high_frequency_estimate", m_gray["noise_estimate"]
            )
            filtered_high_frequency = m_filt.get(
                "high_frequency_estimate", m_filt["noise_estimate"]
            )
            if high_frequency_reduction > 0:
                noise_str = f"<span class='qm-delta pos'>▼ {high_frequency_reduction:.1f}% high-frequency reduction</span>"
                high_frequency_summary = (
                    f"High-frequency estimate reduced by {high_frequency_reduction:.1f}%."
                )
            elif high_frequency_reduction < 0:
                noise_str = f"<span class='qm-delta neg'>▲ {abs(high_frequency_reduction):.1f}% high-frequency increase</span>"
                high_frequency_summary = (
                    f"High-frequency estimate increased by {abs(high_frequency_reduction):.1f}%."
                )
            else:
                noise_str = "<span class='qm-delta neu'>— No change</span>"
                high_frequency_summary = "High-frequency estimate was unchanged."

            dynamic_range_change = m_enh["dynamic_range"] - m_filt["dynamic_range"]
            dynamic_range_str = (
                f"<span class='qm-delta pos'>▲ {dynamic_range_change} levels</span>"
                if dynamic_range_change > 0
                else (
                    f"<span class='qm-delta neg'>▼ {abs(dynamic_range_change)} levels</span>"
                    if dynamic_range_change < 0
                    else "<span class='qm-delta neu'>— No change</span>"
                )
            )

            st.markdown(f"""
            <div class='qm-row'>
                <div class='qm-card'>
                    <div class='qm-label'>Mean Intensity · Grayscale → Final</div>
                    <div class='qm-value'>{m_gray['mean_brightness']:.1f} → {m_enh['mean_brightness']:.1f}</div>
                    <div class='qm-delta neu'>Δ {brightness_delta:+.1f} intensity levels</div>
                </div>
                <div class='qm-card'>
                    <div class='qm-label'>Contrast · Filtered → CLAHE</div>
                    <div class='qm-value'>{m_filt['contrast']:.1f} → {m_enh['contrast']:.1f}</div>
                    {contrast_gain_str}
                </div>
                <div class='qm-card'>
                    <div class='qm-label'>High-Frequency Estimate · Gray → Median</div>
                    <div class='qm-value'>{gray_high_frequency:.0f} → {filtered_high_frequency:.0f}</div>
                    {noise_str}
                </div>
                <div class='qm-card'>
                    <div class='qm-label'>Dynamic Range · Filtered → CLAHE</div>
                    <div class='qm-value'>{m_filt['dynamic_range']} → {m_enh['dynamic_range']}</div>
                    {dynamic_range_str}
                </div>
                <div class='qm-card'>
                    <div class='qm-label'>Pre-processing Time</div>
                    <div class='qm-value'>{preprocessing_time_display}</div>
                    <div class='qm-delta neu'>Uploaded test image only</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("<div class='section-label'>🔎 Local Processing Comparison</div>",
                        unsafe_allow_html=True)

            filtering_change_raw, filtering_change_display = (
                create_filtering_change_map(
                    test_stages["grayscale"], test_stages["filtered"]
                )
            )
            roi_views, roi_bounds = extract_matching_center_rois(
                {
                    "grayscale": test_stages["grayscale"],
                    "filtered": test_stages["filtered"],
                    "change_map": filtering_change_display,
                    "enhanced": test_stages["enhanced"],
                }
            )
            roi_x1, roi_y1, roi_x2, roi_y2 = roi_bounds
            st.markdown(
                "<div class='ec-step'>A. Median Filtering Effect</div>",
                unsafe_allow_html=True,
            )
            roi_col1, roi_col2, roi_col3 = st.columns(3, gap="medium")
            with roi_col1:
                st.markdown(
                    "<p style='color:#3a5070;font-size:0.75rem;margin-bottom:5px;'>GRAYSCALE INPUT — BEFORE MEDIAN FILTERING</p>",
                    unsafe_allow_html=True,
                )
                st.image(
                    roi_views["grayscale"], clamp=True, channels="GRAY", width="stretch"
                )
            with roi_col2:
                st.markdown(
                    "<p style='color:#7a90b0;font-size:0.75rem;margin-bottom:5px;'>MEDIAN FILTERED — AFTER 5×5 FILTERING</p>",
                    unsafe_allow_html=True,
                )
                st.image(
                    roi_views["filtered"], clamp=True, channels="GRAY", width="stretch"
                )
            with roi_col3:
                st.markdown(
                    "<p style='color:#3a86ff;font-size:0.75rem;margin-bottom:5px;'>FILTERING CHANGE MAP — MODIFIED PIXELS</p>",
                    unsafe_allow_html=True,
                )
                st.image(
                    roi_views["change_map"], clamp=True, channels="GRAY", width="stretch"
                )
            st.caption(
                f"All local views use the identical central ROI: x={roi_x1}:{roi_x2}, "
                f"y={roi_y1}:{roi_y2} ({roi_x2 - roi_x1}×{roi_y2 - roi_y1} px). "
                f"The change map is display-normalized |Grayscale − Median|; its raw "
                f"maximum change is {int(filtering_change_raw.max())} intensity levels and "
                "it is not used by the processing pipeline. Bright regions indicate stronger "
                "filtering changes and may include potential noise, compression artefacts, "
                "real PCB edges, or fine detail; they should not be interpreted directly as noise."
            )

            st.markdown(f"""
            <div class='qm-row'>
                <div class='qm-card'>
                    <div class='qm-label'>High-Frequency Estimate · Before → After</div>
                    <div class='qm-value'>{gray_high_frequency:.0f} → {filtered_high_frequency:.0f}</div>
                    <div class='qm-delta neu'>{high_frequency_summary}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(
                "<div class='ec-step' style='margin-top:18px;'>B. Contrast Enhancement</div>",
                unsafe_allow_html=True,
            )
            clahe_col1, clahe_col2 = st.columns(2, gap="medium")
            with clahe_col1:
                st.markdown(
                    "<p style='color:#7a90b0;font-size:0.75rem;margin-bottom:5px;'>MEDIAN FILTERED — BEFORE CLAHE</p>",
                    unsafe_allow_html=True,
                )
                st.image(
                    roi_views["filtered"], clamp=True, channels="GRAY", width="stretch"
                )
            with clahe_col2:
                st.markdown(
                    "<p style='color:#3a86ff;font-size:0.75rem;margin-bottom:5px;'>CLAHE ENHANCED — AFTER LOCAL CONTRAST ENHANCEMENT</p>",
                    unsafe_allow_html=True,
                )
                st.image(
                    roi_views["enhanced"], clamp=True, channels="GRAY", width="stretch"
                )

            st.markdown(f"""
            <div class='qm-row'>
                <div class='qm-card'>
                    <div class='qm-label'>Contrast · Before → After</div>
                    <div class='qm-value'>{m_filt['contrast']:.1f} → {m_enh['contrast']:.1f}</div>
                    {contrast_gain_str}
                </div>
                <div class='qm-card'>
                    <div class='qm-label'>Dynamic Range · Before → After</div>
                    <div class='qm-value'>{m_filt['dynamic_range']} → {m_enh['dynamic_range']}</div>
                    {dynamic_range_str}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("<div class='section-label'>📈 Pixel Intensity Distribution</div>",
                        unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(12, 3.2))
            fig.patch.set_facecolor("#0b0f19")
            ax.set_facecolor("#0e1520")

            ax.hist(test_stages["grayscale"].ravel(), bins=256, range=(0,255),
                    color="#5fa8ff", alpha=0.55, label="Original Grayscale", density=True)
            ax.hist(test_stages["enhanced"].ravel(), bins=256, range=(0,255),
                    color="#22c55e", alpha=0.65, label="Final CLAHE Enhanced", density=True)

            ax.set_xlabel("Pixel Intensity (0 = black · 255 = white)", color="#4a6890", fontsize=9)
            ax.set_ylabel("Normalised Frequency", color="#4a6890", fontsize=9)
            ax.tick_params(colors="#2a4060", labelsize=8)
            ax.spines[["top","right","left","bottom"]].set_color("#1a2840")
            ax.set_xlim(0, 255)
            ax.set_title(
                "Before Enhancement vs After Enhancement",
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
            st.image(buf, width="stretch")
            plt.close(fig)
            st.caption(
                "The histogram shows how CLAHE redistributes local pixel intensities, "
                "supporting improved contrast in the preprocessed PCB image."
            )

            st.markdown("---")
            st.markdown("<div class='section-label'>🖼️ Original vs Preprocessed Output</div>",
                        unsafe_allow_html=True)

            fc1, fc2 = st.columns(2, gap="medium")
            with fc1:
                st.markdown(
                    "<p style='color:#3a5070;font-size:0.75rem;margin-bottom:5px;'>INPUT — ORIGINAL PCB</p>",
                    unsafe_allow_html=True)
                st.image(test_rgb, width="stretch")

            with fc2:
                st.markdown(
                    "<p style='color:#3a86ff;font-size:0.75rem;margin-bottom:5px;'>PREPROCESSED OUTPUT — CLAHE ENHANCED</p>",
                    unsafe_allow_html=True)
                st.image(processed_test, clamp=True, channels="GRAY", width="stretch")

            st.markdown("---")
            st.markdown("<div class='section-label'>✅ Pre-processing Summary</div>",
                        unsafe_allow_html=True)
            st.markdown(f"""
            <div class='explain-card'>
                <div class='ec-title'>Image ready for defect segmentation</div>
                <div class='ec-desc'>
                    ✓ Grayscale conversion completed<br>
                    ✓ 5×5 median filtering applied to suppress potential impulse noise and small local variations<br>
                    ✓ Local contrast enhanced using CLAHE<br>
                    ✓ Final single-channel image prepared for defect segmentation
                </div>
                <div class='ec-result'>
                    Processing time: {preprocessing_time_summary} &nbsp;|&nbsp;
                    Output: {processed_test.shape[1]}×{processed_test.shape[0]} px &nbsp;|&nbsp;
                    Range: {int(processed_test.min())}–{int(processed_test.max())}
                </div>
            </div>
            """, unsafe_allow_html=True)

            _, dl_col, _ = st.columns([2, 1, 2])
            with dl_col:
                ok, enc = cv2.imencode(".png", processed_test)
                if ok:
                    st.download_button(
                        label="⬇ Download Preprocessed Image",
                        data=enc.tobytes(),
                        file_name=f"preprocessed_{os.path.splitext(result['test_name'])[0]}.png",
                        mime="image/png",
                        width="stretch",
                    )

        # ============================================================
        # DEFECT SEGMENTATION
        # ============================================================
        with tab2:

            st.markdown("<div class='section-label'>🔬 Defect Segmentation Stages</div>",
                        unsafe_allow_html=True)

            # 7 stages, shown in 4 + 3 layout
            stage_order = [
                ("test_image", "Stage 1", "Preprocessed Test", "Pre-processing output"),
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
                        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width="stretch")
                    else:
                        st.image(img, clamp=True, channels="GRAY", width="stretch")

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
                        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width="stretch")
                    else:
                        st.image(img, clamp=True, channels="GRAY", width="stretch")

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
                        width="stretch",
                    )

        # ============================================================
        # FEATURE ANALYSIS
        # ============================================================
        with tab3:

            st.markdown("<div class='section-label'>🔬 Feature Analysis Results</div>",
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
                    width="stretch",
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
                st.markdown("<div class='section-label'>📖 Feature Analysis Overview</div>",
                            unsafe_allow_html=True)

                st.markdown("""
                <div class='explain-card'>
                    <div class='ec-step'>Connected Component Analysis</div>
                    <div class='ec-title'>✓ Defect Region Identification</div>
                    <div class='ec-desc'>
                        Feature analysis applies connected component analysis to the refined binary
                        mask from defect segmentation. Each isolated defect region is identified and
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
        # SEVERITY AND SPATIAL ANALYSIS
        # ============================================================
        with tab4:

            st.markdown("<div class='section-label'>🔬 Defect Severity & Spatial Analysis</div>",
                        unsafe_allow_html=True)

            status = evaluation["status_label"]
            status_color = "#22c55e" if evaluation["inspection_status"] == "NORMAL" else "#ef4444"
            highest_severity = evaluation["highest_severity_level"]
            severity_color = {
                "LOW": "#22c55e",
                "MEDIUM": "#f59e0b",
                "HIGH": "#ef4444",
            }.get(highest_severity, "#c5d4e8")
            priority_id = evaluation["highest_priority_defect_id"]
            priority_display = "N/A" if priority_id == "N/A" else f"Defect #{priority_id}"
            concentrated_display = evaluation["most_concentrated_region"].replace("_", " ").title()
            priority_region_display = evaluation["highest_priority_defect_region"].replace("_", " ").title()
            largest_region_display = evaluation["largest_defect_region"].replace("_", " ").title()

            st.markdown(f"""
            <div class='qm-row'>
                <div class='qm-card'>
                    <div class='qm-label'>Inspection Status</div>
                    <div class='qm-value' style='color:{status_color};'>{status}</div>
                    <div class='qm-delta neu'>Overall quality decision</div>
                </div>
                <div class='qm-card'>
                    <div class='qm-label'>Highest Severity</div>
                    <div class='qm-value' style='color:{severity_color};'>{highest_severity}</div>
                    <div class='qm-delta neu'>Highest per-defect relative score</div>
                </div>
                <div class='qm-card'>
                    <div class='qm-label'>Highest Priority</div>
                    <div class='qm-value'>{priority_display}</div>
                    <div class='qm-delta neu'>Priority rank 1</div>
                </div>
                <div class='qm-card'>
                    <div class='qm-label'>Total Defects</div>
                    <div class='qm-value'>{evaluation['total_defect_count']}</div>
                    <div class='qm-delta neu'>Valid assessed regions</div>
                </div>
            </div>
            <div class='qm-row'>
                <div class='qm-card'>
                    <div class='qm-label'>Most Concentrated Region</div>
                    <div class='qm-value'>{concentrated_display}</div>
                    <div class='qm-delta neu'>Highest defect count by quadrant</div>
                </div>
                <div class='qm-card'>
                    <div class='qm-label'>Priority Defect Region</div>
                    <div class='qm-value'>{priority_region_display}</div>
                    <div class='qm-delta neu'>Location of priority rank 1</div>
                </div>
                <div class='qm-card'>
                    <div class='qm-label'>Largest Defect Region</div>
                    <div class='qm-value'>{largest_region_display}</div>
                    <div class='qm-delta neu'>Quadrant of largest area</div>
                </div>
                <div class='qm-card'>
                    <div class='qm-label'>Processing Time</div>
                    <div class='qm-value'>{proc_time:.2f}s</div>
                    <div class='qm-delta neu'>Full inspection pipeline</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            assessed_defects = evaluation["defects"]
            if assessed_defects:
                st.markdown("---")
                st.markdown("<div class='section-label'>📋 Per-Defect Severity and Priority</div>",
                            unsafe_allow_html=True)
                severity_rows = [
                    {
                        "ID": defect["id"],
                        "Area (px²)": defect["area"],
                        "Area Ratio (%)": f"{defect['area_ratio'] * 100:.4f}",
                        "Severity Score": f"{defect['severity_score']:.6f}",
                        "Severity": defect["severity_level"],
                        "Priority": defect["priority_rank"],
                        "Region": defect["spatial_region"].replace("_", " ").title(),
                    }
                    for defect in assessed_defects
                ]
                st.dataframe(severity_rows, width="stretch", hide_index=True)
            else:
                st.info("No valid defects were detected; severity, priority, and spatial summaries are N/A.")

            st.markdown("---")
            st.markdown("<div class='section-label'>🧭 Spatial Distribution Analysis</div>",
                        unsafe_allow_html=True)
            distribution = evaluation["spatial_distribution"]
            st.markdown(f"""
            <div class='qm-row'>
                <div class='qm-card'><div class='qm-label'>Top Left</div><div class='qm-value'>{distribution['TOP_LEFT']}</div></div>
                <div class='qm-card'><div class='qm-label'>Top Right</div><div class='qm-value'>{distribution['TOP_RIGHT']}</div></div>
                <div class='qm-card'><div class='qm-label'>Bottom Left</div><div class='qm-value'>{distribution['BOTTOM_LEFT']}</div></div>
                <div class='qm-card'><div class='qm-label'>Bottom Right</div><div class='qm-value'>{distribution['BOTTOM_RIGHT']}</div></div>
            </div>
            """, unsafe_allow_html=True)

            thresholds = evaluation["severity_thresholds"]
            weights = evaluation["severity_weights"]
            st.markdown(f"""
            <div class='explain-card'>
                <div class='ec-title'>Configurable Relative Severity Formula</div>
                <div class='ec-desc'>
                    Score = {weights['area']:.2f}(area ratio) + {weights['width']:.2f}(width ratio)
                    + {weights['height']:.2f}(height ratio).<br>
                    LOW: score ≤ {thresholds['low_max_score']:.4f} &nbsp;|&nbsp;
                    MEDIUM: {thresholds['low_max_score']:.4f} &lt; score ≤ {thresholds['medium_max_score']:.4f} &nbsp;|&nbsp;
                    HIGH: above {thresholds['medium_max_score']:.4f}.<br>
                    These are configurable prototype parameters, not scientifically validated
                    manufacturing limits.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ============================================================
        # INSPECTION REPORT
        # ============================================================
        with tab5:

            st.markdown("<div class='section-label'>🔬 Inspection Report Summary</div>",
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
                analysis_metrics=analysis_metrics,
                evaluation_result=evaluation,
            )
            report_defects = inspection_report.get("defects", [])

            st.markdown("---")
            st.markdown("<div class='section-label'>📊 Inspection Summary Metrics</div>",
                        unsafe_allow_html=True)

            # Display high-level metrics
            total_defects = inspection_report["total_defects"]
            total_area = inspection_report["total_defect_area"]
            avg_area = inspection_report["average_defect_area"]
            largest_area = inspection_report["largest_defect_area"]
            status = inspection_report["inspection_status"]

            status_color = "#22c55e" if status.startswith("NORMAL") else "#ef4444"
            status_icon = "✓" if status.startswith("NORMAL") else "⚠"

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
                    st.image(overlay_rgb, width="stretch",
                            caption="Final inspection result — detected defect regions highlighted by bounding boxes.")
                else:
                    st.image(overlay, clamp=True, channels="GRAY", width="stretch",
                            caption="Final inspection result — detected defect regions highlighted by bounding boxes.")
            else:
                st.info("Overlay image is not available from segmentation.py.")

            st.markdown("---")
            st.markdown("<div class='section-label'>📋 Defect Detail Summary</div>",
                        unsafe_allow_html=True)

            if not report_defects or total_defects == 0:
                st.markdown("""
                <div class='explain-card'>
                    <div class='ec-desc'>
                        No individual defect measurements are available because no valid defect regions were detected.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Build and display defect table
                table_data = format_defect_table_data(report_defects)

                st.dataframe(
                    table_data,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Defect ID": st.column_config.NumberColumn(format="%d"),
                        "Area (px²)": st.column_config.NumberColumn(format="%d"),
                        "Width (px)": st.column_config.NumberColumn(format="%d"),
                        "Height (px)": st.column_config.NumberColumn(format="%d"),
                        "Area Ratio (%)": st.column_config.TextColumn(),
                        "Severity Score": st.column_config.TextColumn(),
                        "Severity": st.column_config.TextColumn(),
                        "Priority": st.column_config.NumberColumn(format="%d"),
                        "Region": st.column_config.TextColumn(),
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
    st.info("👆 Upload a defective/test PCB image to get started; add a template if no dataset reference matches.")
