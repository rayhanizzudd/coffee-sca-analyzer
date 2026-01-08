import streamlit as st
import cv2
import numpy as np
import torch
import os
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from detectron2.structures import Instances
import torch.nn as nn
from PIL import Image

# =========================================================
# CUSTOM CSS - MINIMAL & CLEAN
# =========================================================
def load_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit Branding & Sidebar */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stSidebar"] {display: none;}
    header {visibility: hidden;}
    
    /* Main Container */
    .stApp {
        background: #FAFBFC;
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 0rem;
        padding-bottom: 2rem;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1F2937 !important;
        font-weight: 600 !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: #F59E0B;
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton>button:hover {
        background: #D97706;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1rem;
        background: white;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        color: #1F2937 !important;
        font-weight: 600 !important;
    }
    
    /* Remove metric delta arrow */
    [data-testid="stMetricDelta"] {
        display: none;
    }
    
    /* Info/Success boxes */
    .stAlert {
        border-radius: 6px;
        border: 1px solid #E5E7EB;
        padding: 0.8rem 1rem;
    }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        border: 1px solid #E5E7EB;
        border-radius: 6px;
    }
    </style>
    """, unsafe_allow_html=True)

# =========================================================
# NAVBAR COMPONENT - MINIMAL
# =========================================================
def render_navbar(active_page):
    home_class = "active" if active_page == "home" else ""
    detect_class = "active" if active_page == "detect" else ""
    
    st.markdown(f"""
    <style>
    .navbar {{
        background: white;
        padding: 0.8rem 2rem;
        border-bottom: 1px solid #E5E7EB;
        margin-bottom: 1.5rem;
    }}
    .nav-container {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        max-width: 1200px;
        margin: 0 auto;
    }}
    .nav-brand {{
        font-size: 1.2rem;
        font-weight: 600;
        color: #1F2937;
    }}
    .nav-links {{
        display: flex;
        gap: 0.5rem;
    }}
    .nav-link {{
        padding: 0.5rem 1rem;
        color: #6B7280;
        font-weight: 500;
        font-size: 0.9rem;
        border-radius: 6px;
        transition: all 0.2s;
        text-decoration: none !important;
    }}
    .nav-link:hover {{
        background: #F3F4F6;
        color: #1F2937;
        text-decoration: none !important;
    }}
    .nav-link.active {{
        background: #F59E0B;
        color: white;
    }}
    .nav-link:visited,
    .nav-link:active,
    .nav-link:focus {{
        text-decoration: none !important;
    }}
    </style>
    
    <div class="navbar">
        <div class="nav-container">
            <div class="nav-brand">Baseline Coffee Analyzer</div>
            <div class="nav-links">
                <a href="?page=home" class="nav-link {home_class}">Home</a>
                <a href="?page=detect" class="nav-link {detect_class}">Detection</a>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_home_page():
    # Hero Section
    st.markdown('<div style="text-align: center; padding: 3rem 0;">', unsafe_allow_html=True)
    st.markdown('<h1 style="font-size: 3.5rem; margin-bottom: 1rem; color: #D4A017;">☕ Baseline Coffee Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 1.3rem; color: #8B7355; margin-bottom: 3rem;">Sistem Deteksi Cacat Biji Kopi Baseline dengan Standar SCA</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Feature Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('''
        <div style="background: linear-gradient(135deg, #FFFBF0 0%, #FFF8E7 100%); 
                    border: 2px solid #F4C430; border-radius: 15px; padding: 1.5rem; 
                    text-align: center; height: 280px; box-shadow: 0 4px 15px rgba(244, 196, 48, 0.2);">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🔍</div>
            <h3 style="color: #D4A017;">Mask R-CNN Baseline</h3>
            <p style="color: #5C4B2A; font-size: 0.9rem;">Menggunakan arsitektur ResNet-101 FPN baseline untuk mendeteksi 16 jenis cacat kopi</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col2:
        st.markdown('''
        <div style="background: linear-gradient(135deg, #FFFBF0 0%, #FFF8E7 100%); 
                    border: 2px solid #F4C430; border-radius: 15px; padding: 1.5rem; 
                    text-align: center; height: 280px; box-shadow: 0 4px 15px rgba(244, 196, 48, 0.2);">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
            <h3 style="color: #D4A017;">Standar SCA</h3>
            <p style="color: #5C4B2A; font-size: 0.9rem;">Penilaian sesuai Specialty Coffee Association untuk grading kualitas internasional</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col3:
        st.markdown('''
        <div style="background: linear-gradient(135deg, #FFFBF0 0%, #FFF8E7 100%); 
                    border: 2px solid #F4C430; border-radius: 15px; padding: 1.5rem; 
                    text-align: center; height: 280px; box-shadow: 0 4px 15px rgba(244, 196, 48, 0.2);">
            <div style="font-size: 3rem; margin-bottom: 1rem;">⚡</div>
            <h3 style="color: #D4A017;">Real-time</h3>
            <p style="color: #5C4B2A; font-size: 0.9rem;">Analisis cepat dengan visualisasi hasil deteksi dan laporan lengkap</p>
        </div>
        ''', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Defect Types Section
    st.markdown('''
    <div style="background: linear-gradient(135deg, #FFFBF0 0%, #FFF4D6 100%); 
                border: 2px solid #E8C368; border-radius: 15px; padding: 2rem; margin: 2rem 0;">
        <h2 style="text-align: center; color: #D4A017; margin-bottom: 2rem;">16 Jenis Cacat yang Terdeteksi</h2>
    </div>
    ''', unsafe_allow_html=True)
    
    col_def1, col_def2 = st.columns(2)
    with col_def1:
        st.markdown('''
        <div style="padding: 1.5rem; background: linear-gradient(135deg, #FFF0F0 0%, #FFE5E5 100%); 
                    border-radius: 10px; border: 2px solid #FF9999;">
            <h4 style="color: #D32F2F; text-align: center;">🔴 Primary Defects</h4>
            <p style="color: #5C4B2A; font-size: 0.9rem; text-align: center;">Full Black, Full Sour, Fungus, Foreign Material, Cherry Pod, Severe Insect</p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col_def2:
        st.markdown('''
        <div style="padding: 1.5rem; background: linear-gradient(135deg, #FFFEF7 0%, #FFF9E6 100%); 
                    border-radius: 10px; border: 2px solid #FFD54F;">
            <h4 style="color: #F57C00; text-align: center;">🟡 Secondary Defects</h4>
            <p style="color: #5C4B2A; font-size: 0.9rem; text-align: center;">Partial Black, Partial Sour, Parchment, Floater, Immature, Withered, Shell, Broken, Hull, Slight Insect</p>
        </div>
        ''', unsafe_allow_html=True)
    
    # Grading System
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('<h3 style="text-align: center; color: #D4A017;">Sistem Grading SCA</h3>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.markdown('''
        <div style="padding: 2rem; background: linear-gradient(135deg, #81C784 0%, #66BB6A 100%); 
                    border-radius: 15px; text-align: center; color: white; 
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1); height: 180px;">
            <h3 style="color: white; margin-bottom: 1rem;">✨ Specialty Grade</h3>
            <p style="font-size: 1.1rem; margin: 0.5rem 0;">Kategori 1: <strong>0 cacat</strong></p>
            <p style="font-size: 1.1rem; margin: 0.5rem 0;">Kategori 2: <strong>≤ 5 cacat</strong></p>
        </div>
        ''', unsafe_allow_html=True)
    
    with col_g2:
        st.markdown('''
        <div style="padding: 2rem; background: linear-gradient(135deg, #E57373 0%, #EF5350 100%); 
                    border-radius: 15px; text-align: center; color: white; 
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1); height: 180px;">
            <h3 style="color: white; margin-bottom: 1rem;">⚠️ Below Specialty Grade</h3>
            <p style="font-size: 1.1rem; margin: 0.5rem 0;">Kategori 1: <strong>≥ 1 cacat</strong></p>
            <p style="font-size: 1.1rem; margin: 0.5rem 0;">Kategori 2: <strong>&gt; 5 cacat</strong></p>
        </div>
        ''', unsafe_allow_html=True)
    
    # Call to Action
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown('''
    <div style="text-align: center; padding: 2rem; 
                background: linear-gradient(135deg, #FFF8E7 0%, #FFE9B3 100%); 
                border-radius: 15px; border: 2px solid #F4C430;">
        <p style="font-size: 1.2rem; color: #D4A017; margin-bottom: 1rem;">
            ✨ Siap untuk menganalisis kopi Anda?
        </p>
        <p style="color: #5C4B2A;">
            Pilih menu <strong style="color: #D4A017;">🔍 Detection</strong> di navigation bar untuk memulai analisis
        </p>
    </div>
    ''', unsafe_allow_html=True)

# =========================================================
# LOGIKA SCA GRADE
# =========================================================
SCA_RULES = {
    "full black": {"type": 1, "ratio": 1}, "full sour": {"type": 1, "ratio": 1},
    "fungus": {"type": 1, "ratio": 1}, "foreign material": {"type": 1, "ratio": 1},
    "cherry pod": {"type": 1, "ratio": 1}, "severe insect": {"type": 1, "ratio": 5},
    "partial black": {"type": 2, "ratio": 3}, "partial sour": {"type": 2, "ratio": 3},
    "parchment": {"type": 2, "ratio": 5}, "floater": {"type": 2, "ratio": 5},
    "immature": {"type": 2, "ratio": 5}, "withered": {"type": 2, "ratio": 5},
    "shell": {"type": 2, "ratio": 5}, "broken": {"type": 2, "ratio": 5},
    "hull": {"type": 2, "ratio": 5}, "slight insect": {"type": 2, "ratio": 10}
}

def calculate_sca_grade(instances, metadata):
    classes = instances.pred_classes.tolist()
    scores = instances.scores.tolist()
    class_names = metadata.thing_classes
    
    defect_counts, defect_scores_list, full_defects_calc = {}, {}, {}
    total_primary_score, total_secondary_score = 0, 0

    for cls_idx, score in zip(classes, scores):
        name = class_names[cls_idx]
        defect_counts[name] = defect_counts.get(name, 0) + 1
        if name not in defect_scores_list: 
            defect_scores_list[name] = []
        defect_scores_list[name].append(score)

    avg_confidences = {}
    for name, count in defect_counts.items():
        avg_conf = (sum(defect_scores_list[name]) / len(defect_scores_list[name])) * 100
        avg_confidences[name] = f"{avg_conf:.1f}%"
        
        if name in SCA_RULES:
            rule = SCA_RULES[name]
            score_val = count // rule["ratio"]
            full_defects_calc[name] = score_val
            
            if rule["type"] == 1: 
                total_primary_score += score_val
            elif rule["type"] == 2: 
                total_secondary_score += score_val

    total_score = total_primary_score + total_secondary_score
    global_conf = (sum(scores) / len(scores)) * 100 if scores else 0

    grade = "UNCLASSIFIED"
    reason = ""

    # Logika Grading Berdasarkan SCA
    if total_primary_score == 0 and total_secondary_score <= 5:
        grade = "SPECIALTY GRADE"
        reason = f"✅ Memenuhi Specialty Grade: Kategori 1 = {total_primary_score} cacat (harus 0), Kategori 2 = {total_secondary_score} cacat (≤5)."
    else:
        grade = "BELOW SPECIALTY GRADE"
        if total_primary_score >= 1 and total_secondary_score > 5:
            reason = f"❌ Tidak memenuhi standar: Kategori 1 = {total_primary_score} cacat (≥1) dan Kategori 2 = {total_secondary_score} cacat (>5)."
        elif total_primary_score >= 1:
            reason = f"❌ Tidak memenuhi standar: Kategori 1 = {total_primary_score} cacat (≥1)."
        else:
            reason = f"❌ Tidak memenuhi standar: Kategori 2 = {total_secondary_score} cacat (>5)."

    return {
        "counts": defect_counts, 
        "scores": full_defects_calc, 
        "confidences": avg_confidences, 
        "global_confidence": f"{global_conf:.1f}%", 
        "total_primary": int(total_primary_score), 
        "total_secondary": int(total_secondary_score), 
        "total_final": int(total_score), 
        "grade": grade, 
        "reason": reason
    }

# =========================================================
# LOAD MODEL BASELINE
# =========================================================
@st.cache_resource
def load_predictor():
    cfg = get_cfg()
    cfg.MODEL.RPN.LOSS_TYPE = "focal"
    cfg.MODEL.RPN.FOCAL_LOSS_GAMMA = 2.0
    cfg.MODEL.RPN.FOCAL_LOSS_ALPHA = 0.25
    # Gunakan config baseline Mask R-CNN R-101 FPN
    cfg.merge_from_file("config_final_baseline.yaml")
    
    # ==========================================
    # PARAMETER HASIL TUNING (TERBAIK)
    # ==========================================
    cfg.SOLVER.BASE_LR = 0.0001663745199265111
    cfg.SOLVER.WEIGHT_DECAY = 0.00010471156171728883
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6374698594433093
    
    # --- ANCHOR CONFIGURATION ---
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16], [32], [64], [128], [256]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
    
    # --- RPN LOSS CONFIGURATION ---
    
    
    # --- INPUT RESOLUTION ---
    cfg.INPUT.MIN_SIZE_TEST = 1216
    cfg.INPUT.MAX_SIZE_TEST = 1216
    
    # --- DETECTION SETTINGS ---
    cfg.TEST.DETECTIONS_PER_IMAGE = 3000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 3000
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 16
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
    cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.5
    
    # --- MODEL WEIGHTS & THRESHOLD ---
    cfg.MODEL.WEIGHTS = "model_0003499_baseline.pth"  # Ganti dengan path model hasil training
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.45
    cfg.MODEL.DEVICE = "cpu"
    
    # --- CLASS NAMES ---
    thing_classes = [
        'broken', 'cherry pod', 'floater', 'foreign material', 'full black', 'full sour', 
        'fungus', 'hull', 'immature', 'parchment', 'partial black', 'partial sour', 
        'severe insect', 'shell', 'slight insect', 'withered'
    ]
    MetadataCatalog.get("coffee_baseline").set(thing_classes=thing_classes)
    return DefaultPredictor(cfg), MetadataCatalog.get("coffee_baseline")

# =========================================================
# DETECTION PAGE
# =========================================================
def render_detection_page(predictor, metadata):
    st.markdown('<h2 style="color: #1F2937; margin-bottom: 1rem;">Coffee Defect Detection</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"], label_visibility="collapsed")

    if uploaded_file:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        # Inisialisasi session state
        if 'detection_done' not in st.session_state:
            st.session_state.detection_done = False
            st.session_state.result_image = None
            st.session_state.result_data = None
        
        # Gambar Input dan Output
        col_input, col_output = st.columns(2)
        
        with col_input:
            st.markdown("""
            <div style="background: white; border: 1px solid #E5E7EB; border-radius: 6px; padding: 0.8rem; margin-bottom: 0.5rem;">
                <p style="margin: 0; color: #1F2937; font-weight: 600;">Input Image</p>
            </div>
            """, unsafe_allow_html=True)
            st.image(image[:, :, ::-1])
        
        with col_output:
            st.markdown("""
            <div style="background: white; border: 1px solid #E5E7EB; border-radius: 6px; padding: 0.8rem; margin-bottom: 0.5rem;">
                <p style="margin: 0; color: #1F2937; font-weight: 600;">Detection Result</p>
            </div>
            """, unsafe_allow_html=True)
            if st.session_state.detection_done and st.session_state.result_image is not None:
                st.image(st.session_state.result_image)
            else:
                st.info("Click 'Analyze' button to detect defects")
        
        # Tombol Analisis
        if st.button("Analyze", use_container_width=True, type="primary"):
            loading_placeholder = st.empty()
            
            with loading_placeholder.container():
                st.markdown("""
                <div style="text-align: center; padding: 2rem; background: white; border: 1px solid #E5E7EB; border-radius: 6px; margin: 1rem 0;">
                    <div style="display: inline-block; width: 50px; height: 50px; border: 5px solid #F3F4F6; border-top: 5px solid #F59E0B; border-radius: 50%; animation: spin 1s linear infinite;"></div>
                    <p style="color: #6B7280; margin-top: 1rem; font-weight: 500;">Analyzing image, please wait...</p>
                </div>
                <style>
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                </style>
                """, unsafe_allow_html=True)
            
            # Proses analisis
            outputs = predictor(image)
            instances = outputs["instances"].to("cpu")
            result = calculate_sca_grade(instances, metadata)
            
            v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=0.8)
            v = v.draw_instance_predictions(instances)
            
            st.session_state.detection_done = True
            st.session_state.result_image = v.get_image()
            st.session_state.result_data = result
            
            loading_placeholder.empty()
            st.rerun()
        
        # Tombol Reset
        if st.session_state.detection_done:
            if st.button("🔄 Upload New Image", use_container_width=True):
                st.session_state.detection_done = False
                st.session_state.result_image = None
                st.session_state.result_data = None
                st.rerun()
        
        # Hasil Analisis
        if st.session_state.detection_done and st.session_state.result_data is not None:
            result = st.session_state.result_data
            
            st.markdown("---")
            st.markdown('<h3 style="color: #1F2937; margin: 1rem 0;">Analysis Results</h3>', unsafe_allow_html=True)
            
            # Grade Badge
            if result['grade'] == "SPECIALTY GRADE":
                grade_bg = "#D1FAE5"
                grade_text = "#065F46"
            else:
                grade_bg = "#FEE2E2"
                grade_text = "#991B1B"
            
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem; background: {grade_bg}; 
                        border-radius: 6px; margin: 1rem 0; border: 1px solid {grade_text};">
                <h3 style="color: {grade_text}; margin: 0; font-weight: 600;">{result['grade']}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Info
            st.markdown(f"""
            <div style="background: white; border: 1px solid #E5E7EB; border-radius: 6px; padding: 1rem; margin: 1rem 0;">
                <p style="margin: 0; color: #374151;"><strong>Reason:</strong> {result['reason']}</p>
                <p style="margin: 0.5rem 0 0 0; color: #374151;"><strong>Confidence:</strong> {result['global_confidence']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.markdown(f"""
                <div style="background: #FEE2E2; border: 1px solid #EF4444; border-radius: 6px; padding: 1rem; text-align: center;">
                    <p style="color: #6B7280; font-size: 0.85rem; margin: 0;">Category 1</p>
                    <h2 style="color: #991B1B; margin: 0.5rem 0 0 0; font-weight: 600;">{result['total_primary']}</h2>
                </div>
                """, unsafe_allow_html=True)
            with col_m2:
                st.markdown(f"""
                <div style="background: #FEF3C7; border: 1px solid #F59E0B; border-radius: 6px; padding: 1rem; text-align: center;">
                    <p style="color: #6B7280; font-size: 0.85rem; margin: 0;">Category 2</p>
                    <h2 style="color: #92400E; margin: 0.5rem 0 0 0; font-weight: 600;">{result['total_secondary']}</h2>
                </div>
                """, unsafe_allow_html=True)
            with col_m3:
                st.markdown(f"""
                <div style="background: #E0E7FF; border: 1px solid #6366F1; border-radius: 6px; padding: 1rem; text-align: center;">
                    <p style="color: #6B7280; font-size: 0.85rem; margin: 0;">Total Score</p>
                    <h2 style="color: #3730A3; margin: 0.5rem 0 0 0; font-weight: 600;">{result['total_final']}</h2>
                </div>
                """, unsafe_allow_html=True)
            with col_m4:
                total_defects = sum(result['counts'].values())
                st.markdown(f"""
                <div style="background: #FCE7F3; border: 1px solid #EC4899; border-radius: 6px; padding: 1rem; text-align: center;">
                    <p style="color: #6B7280; font-size: 0.85rem; margin: 0;">Total Defects</p>
                    <h2 style="color: #9F1239; margin: 0.5rem 0 0 0; font-weight: 600;">{total_defects}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            # Detail Table
            st.markdown("**Defect Details**")
            table_data = []
            for k, v in result['counts'].items():
                category = "Category 1" if SCA_RULES.get(k, {}).get("type") == 1 else "Category 2"
                table_data.append({
                    "Category": category,
                    "Defect Type": k.title(),
                    "Count": v,
                    "Confidence": result['confidences'][k],
                    "Score": result['scores'].get(k, 0)
                })
            
            table_height = len(table_data) * 35 + 38
            st.dataframe(table_data, use_container_width=True, hide_index=True, height=table_height)

# =========================================================
# MAIN APP
# =========================================================
def main():
    st.set_page_config(
        page_title="Baseline Coffee Analyzer",
        page_icon="☕",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    load_custom_css()
    
    # Inisialisasi session state untuk navigation
    if 'page' not in st.session_state:
        st.session_state.page = 'home'
    
    # Check query params untuk navigation
    query_params = st.query_params
    if 'page' in query_params:
        st.session_state.page = query_params['page']
    
    # Render navbar
    render_navbar(st.session_state.page)
    
    # Page routing tanpa sidebar
    if st.session_state.page == 'home':
        render_home_page()
    elif st.session_state.page == 'detect':
        predictor, metadata = load_predictor()
        render_detection_page(predictor, metadata)

if __name__ == "__main__":
    main()