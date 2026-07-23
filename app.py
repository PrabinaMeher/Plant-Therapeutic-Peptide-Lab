#!/usr/bin/env python3
"""
Plant Therapeutic Peptide Lab - v1.0
"""
import streamlit as st
import pandas as pd
import numpy as np
import tempfile
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import py3Dmol
import streamlit.components.v1 as components
import subprocess
import shutil
import os
import time
import re
import gzip
import requests
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(page_title="Plant Therapeutic Peptide Lab", layout="wide",
                   page_icon="🌿", initial_sidebar_state="expanded")


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
@import url('https://fonts.googleapis.com/icon?family=Material+Symbols+Outlined');
.material-symbols-outlined{font-variation-settings:'FILL' 0,'wght' 400,'GRAD' 0,'opsz' 22;
vertical-align:-5px;font-size:1.15rem;line-height:1;}
html,body,[class*="css"]{font-family:'Inter',sans-serif;color:#1e293b;font-size:14px;}
.header-panel{padding:0.5rem 1.9rem 0.5rem !important;}
.header-panel h1{font-size:3.0rem !important;margin:0.6rem 0 0 !important;}
.header-panel p{font-size:1.2rem !important;margin:0.8rem 0 0 !important;}
.header-eyebrow{font-size:0.65rem !important;padding:0.2rem 0.6rem !important;}
.header-stat-val{font-size:0.95rem !important;}
.header-stat-lbl{font-size:0.62rem !important;}
.header-stat-badge{font-size:0.72rem !important;padding:0.18rem 0.55rem !important;}
.header-divider{margin:0.8rem 0 0.7rem !important;}
.main-hero h1{font-size:2.7rem !important;}
.main-hero p{font-size:1.2rem !important;}
.met-val{font-size:1.5rem !important;}
.met-card{padding:0.7rem !important;}
.sec-hdr{font-size:0.88rem !important;padding:0.2rem 0 0.45rem !important;margin:1.1rem 0 0.7rem !important;}
.brand-title{font-size:1.15rem !important;}
div[data-testid="stSidebar"] div[data-testid="stButton"]>button{
  font-size:0.76rem !important;padding:0.4rem 0.75rem !important;
}
div[data-testid="stSidebar"] div[data-testid="stButton"]>button p{font-size:0.76rem !important;}
.stButton>button{font-size:0.8rem !important;padding:0.4rem 1.1rem !important;}
div[data-testid="stSidebar"]{width:16rem !important;}
.stApp{background:#f7f8fa;}
div[data-testid="stSidebar"]{background:#0f2e1c;border-right:1px solid #0a2015;}
div[data-testid="stSidebar"] .stMarkdown p,
div[data-testid="stSidebar"] label{color:#c9d8ce !important;font-size:0.8rem;}
div[data-testid="stSidebar"] hr{border-color:#1c3d29 !important;}
div[data-testid="stSidebar"] div[data-testid="stButton"]>button{
  color:#fff !important;border:none;border-radius:8px;font-weight:600;
  font-size:0.84rem;padding:0.55rem 1rem;text-align:left;justify-content:flex-start;
  box-shadow:0 1px 4px rgba(0,0,0,0.08);transition:all 0.2s;
}
div[data-testid="stSidebar"] div[data-testid="stButton"]>button:hover{
  transform:translateY(-1px);box-shadow:0 3px 10px rgba(0,0,0,0.18);
}
div[data-testid="stSidebar"] div[data-testid="element-container"]:has(.nav-mark-predict)
  + div[data-testid="element-container"] div[data-testid="stButton"]>button,
div[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.nav-mark-predict)
  + div[data-testid="stElementContainer"] div[data-testid="stButton"]>button{
  background:linear-gradient(135deg,#22c55e,#16a34a);
}
div[data-testid="stSidebar"] div[data-testid="element-container"]:has(.nav-mark-structure)
  + div[data-testid="element-container"] div[data-testid="stButton"]>button,
div[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.nav-mark-structure)
  + div[data-testid="stElementContainer"] div[data-testid="stButton"]>button{
  background:linear-gradient(135deg,#0ea5e9,#0284c7);
}
div[data-testid="stSidebar"] div[data-testid="element-container"]:has(.nav-mark-docking)
  + div[data-testid="element-container"] div[data-testid="stButton"]>button,
div[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.nav-mark-docking)
  + div[data-testid="stElementContainer"] div[data-testid="stButton"]>button{
  background:linear-gradient(135deg,#a855f7,#9333ea);
}
div[data-testid="stSidebar"] div[data-testid="element-container"]:has(.nav-mark-md)
  + div[data-testid="element-container"] div[data-testid="stButton"]>button,
div[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.nav-mark-md)
  + div[data-testid="stElementContainer"] div[data-testid="stButton"]>button{
  background:linear-gradient(135deg,#f59e0b,#d97706);
}
div[data-testid="stSidebar"] div[data-testid="element-container"]:has(.nav-mark-mmpbsa)
  + div[data-testid="element-container"] div[data-testid="stButton"]>button,
div[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.nav-mark-mmpbsa)
  + div[data-testid="stElementContainer"] div[data-testid="stButton"]>button{
  background:linear-gradient(135deg,#14b8a6,#0d9488);
}
div[data-testid="stSidebar"] div[data-testid="element-container"]:has(.nav-mark-about)
  + div[data-testid="element-container"] div[data-testid="stButton"]>button,
div[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.nav-mark-about)
  + div[data-testid="stElementContainer"] div[data-testid="stButton"]>button{
  background:linear-gradient(135deg,#6366f1,#4f46e5);
}
div[data-testid="stSidebar"] div[data-testid="element-container"]:has(.nav-mark-guide)
  + div[data-testid="element-container"] div[data-testid="stButton"]>button,
div[data-testid="stSidebar"] div[data-testid="stElementContainer"]:has(.nav-mark-guide)
  + div[data-testid="stElementContainer"] div[data-testid="stButton"]>button{
  background:linear-gradient(135deg,#166534,#14532d);
}
div[data-testid="stSidebar"] div[data-testid="stButton"]>button[kind="primary"]{
  background:linear-gradient(135deg,#4ade80,#22c55e) !important;
  color:#052e16 !important;font-weight:700;
  box-shadow:0 2px 10px rgba(34,197,94,0.4) !important;
}
.stButton>button{background:#15803d;color:#fff !important;border:none;
border-radius:8px;font-weight:600;font-size:0.84rem;padding:0.5rem 1.3rem;
transition:background 0.15s,transform 0.15s;box-shadow:none;}
.stButton>button:hover{background:#166534;transform:translateY(-1px);}
.stButton>button:disabled{background:#e5e7eb !important;color:#9ca3af !important;box-shadow:none !important;}
.stTextInput>div>div,.stSelectbox>div>div{background:#fff !important;
border:1.5px solid #e2e8f0 !important;border-radius:8px !important;}
.stTextInput>div>div:focus-within{border-color:#15803d !important;
box-shadow:0 0 0 3px rgba(21,128,61,0.12) !important;}
.card{background:#fff;border:1px solid #e5e7eb;border-radius:12px;
padding:1.4rem;box-shadow:0 1px 3px rgba(15,23,42,0.06);}
.sec-hdr{display:flex;align-items:center;gap:0.55rem;background:transparent;
border-bottom:2px solid #15803d;border-radius:0;
padding:0.3rem 0 0.6rem;margin:1.8rem 0 1rem;font-weight:700;
font-size:1.02rem;letter-spacing:0.2px;color:#0f2e1c;}
.met-card{background:#fff;border:1px solid #e5e7eb;border-radius:10px;
padding:1rem;text-align:center;box-shadow:0 1px 3px rgba(15,23,42,0.05);}
.met-val{font-family:'JetBrains Mono',monospace;font-size:1.5rem;font-weight:700;color:#15803d;}
.met-lbl{font-size:0.68rem;color:#6b7280;text-transform:uppercase;letter-spacing:0.8px;margin-top:2px;font-weight:500;}
.badge{display:inline-flex;align-items:center;gap:0.3rem;padding:0.22rem 0.7rem;
border-radius:6px;font-size:0.72rem;font-weight:600;font-family:'JetBrains Mono',monospace;}
.b-idle{background:#f3f4f6;color:#6b7280;border:1px solid #e5e7eb;}
.b-running{background:#eff6ff;color:#2563eb;border:1px solid #bfdbfe;}
.b-done{background:#f0fdf4;color:#15803d;border:1px solid #bbf7d0;}
.b-error{background:#fef2f2;color:#dc2626;border:1px solid #fecaca;}
.ps{display:flex;align-items:center;gap:0.5rem;padding:0.45rem 0.75rem;
border-radius:6px;margin:0.18rem 0;font-size:0.78rem;font-family:'JetBrains Mono',monospace;}
.ps-done{background:#f0fdf4;border:1px solid #86efac;color:#15803d;}
.ps-active{background:#eff6ff;border:1px solid #93c5fd;color:#2563eb;}
.ps-pending{background:#f9fafb;border:1px solid #e5e7eb;color:#9ca3af;}
.info{background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;
padding:0.75rem 1rem;font-size:0.82rem;color:#166534;line-height:1.6;}
.warn{background:#fffbeb;border:1px solid #fde68a;border-radius:8px;
padding:0.75rem 1rem;font-size:0.82rem;color:#92400e;line-height:1.6;}
.rrow{display:flex;justify-content:space-between;padding:0.38rem 0.7rem;
border-bottom:1px solid #f1f5f9;font-size:0.86rem;}
.rk{color:#6b7280;}.rv{font-family:'JetBrains Mono',monospace;color:#15803d;font-weight:600;}
.drug-card{background:#fff;border:1px solid #e5e7eb;border-radius:8px;
padding:0.8rem 1rem;margin:0.3rem 0;display:flex;justify-content:space-between;align-items:center;}
.drug-pass{border-left:4px solid #15803d;}
.drug-fail{border-left:4px solid #dc2626;}

.stProgress>div>div>div{
  background:#ffffff !important;
  border:1px solid #0f2e1c !important;
  border-radius:8px !important;
  height:16px !important;
}
/* Running/active status - green instead of blue */
.b-running{background:#f0fdf4;color:#15803d;border:1px solid #86efac;}
.ps-active{background:#f0fdf4;border:1px solid #86efac;color:#15803d;}
.hb-running{background:rgba(34,197,94,0.18);color:#86efac;}
.hb-running .hb-dot{background:#4ade80;box-shadow:0 0 6px rgba(74,222,128,0.8);}
.stTabs [data-baseweb="tab-list"]{background:#eef1f0;border-radius:10px;padding:3px;gap:2px;}
.stTabs [data-baseweb="tab"]{background:transparent;border-radius:8px;color:#6b7280;
font-weight:500;font-size:0.83rem;padding:0.38rem 0.9rem;}
.stTabs [aria-selected="true"]{background:#fff !important;color:#15803d !important;
font-weight:600 !important;box-shadow:0 1px 4px rgba(0,0,0,0.08);}
.stFileUploader>div{background:#fafbfa !important;border:1.5px dashed #a7c4ac !important;border-radius:10px !important;}
hr{border-color:#e5e7eb !important;margin:1rem 0 !important;}
.stSlider>div>div>div>div{background:#15803d !important;}
.stDataFrame{border-radius:10px;overflow:hidden;border:1px solid #e5e7eb;}
.streamlit-expanderHeader{background:#fafbfa !important;border:1px solid #e5e7eb !important;border-radius:8px !important;}
.guide-step{background:#fff;border:1px solid #e5e7eb;border-radius:10px;padding:1rem 1.2rem;margin:0.6rem 0;}
.guide-num{display:inline-flex;align-items:center;justify-content:center;width:28px;height:28px;
background:#0f2e1c;color:#fff;border-radius:50%;font-weight:700;font-size:0.82rem;margin-right:0.6rem;}
.entry-card{background:#fff;border:1.5px solid #e5e7eb;border-radius:12px;padding:1.2rem 1.4rem;
margin:0.5rem 0;transition:border-color 0.2s;}
.entry-card:hover{border-color:#15803d;}
.entry-card-active{border-color:#15803d !important;background:#f0fdf4 !important;}
.brand-title{color:#ffffff !important;font-weight:900;font-size:1.4rem;line-height:1.2;
text-align:left;letter-spacing:-0.3px;}
.brand-sub{font-size:0.72rem;color:#8fb89c;margin-top:2px;text-align:left;
letter-spacing:0.3px;font-weight:500;}
.nav-label{font-size:0.68rem;font-weight:700;color:#5f8a6c;letter-spacing:1.2px;padding:0 0.3rem 0.4rem;}
.main-hero{text-align:center;padding:1.6rem 0 0.6rem;}
.main-hero h1{margin:0;font-size:2.6rem;font-weight:800;color:#0f2e1c;letter-spacing:-0.5px;}
.main-hero p{color:#5b6b63;font-size:0.95rem;margin:0.6rem auto 0;max-width:720px;line-height:1.6;}
.team-card{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:1.4rem 1.2rem;
text-align:center;box-shadow:0 1px 3px rgba(15,23,42,0.05);}
.team-avatar{width:64px;height:64px;border-radius:50%;background:#e7f3ea;color:#15803d;
display:flex;align-items:center;justify-content:center;margin:0 auto 0.7rem;font-size:1.6rem;}
.diagram-box{background:#fff;border:1.5px dashed #a7c4ac;border-radius:14px;
padding:3rem 1.5rem;text-align:center;color:#6b7280;}
.filter-panel-hdr{display:flex;align-items:center;justify-content:space-between;margin-bottom:0.9rem;
border-bottom:1px solid #f1f5f9;padding-bottom:0.7rem;}
.filter-title{display:flex;align-items:center;gap:0.5rem;font-weight:700;font-size:0.92rem;color:#0f2e1c;}
.header-panel{background:linear-gradient(135deg,#0f2e1c 0%,#1c4a2c 55%,#15803d 100%);
border-radius:16px;padding:1.6rem 2rem 1.5rem;margin:0.4rem 0 1.2rem;
box-shadow:0 8px 24px rgba(15,46,28,0.18);position:relative;overflow:hidden;}
.header-panel::before{content:'';position:absolute;top:-40%;right:-8%;width:280px;height:280px;
border-radius:50%;background:radial-gradient(circle,rgba(255,255,255,0.06) 0%,transparent 70%);}
.header-eyebrow{display:inline-flex;align-items:center;gap:0.4rem;background:rgba(255,255,255,0.1);
border:1px solid rgba(255,255,255,0.18);color:#c9e8d3;font-size:0.78rem;font-weight:700;
letter-spacing:1.3px;text-transform:uppercase;padding:0.3rem 0.85rem;border-radius:20px;}
.header-panel h1{margin:0.7rem 0 0;font-size:3.0rem;font-weight:1000;color:#fff;
letter-spacing:-0.5px;line-height:1.1;}
.header-panel p{color:#cfe7d6;font-size:1.02rem;margin:0.55rem 0 0;max-width:720px;line-height:1.6;}
.header-divider{height:1px;background:linear-gradient(90deg,rgba(255,255,255,0.22),rgba(255,255,255,0.02));
margin:1.2rem 0 1rem;}
.header-stats{display:flex;gap:0;align-items:stretch;}
.header-stat{flex:1;padding:0 1.4rem;border-right:1px solid rgba(255,255,255,0.14);}
.header-stat:first-child{padding-left:0;}
.header-stat:last-child{border-right:none;}
.header-stat-lbl{font-size:0.76rem;color:#9fc7ab;text-transform:uppercase;letter-spacing:0.9px;font-weight:600;}
.header-stat-val{font-family:'JetBrains Mono',monospace;font-size:1.3rem;font-weight:700;color:#fff;margin-top:0.25rem;}
.header-stat-badge{display:inline-flex;align-items:center;gap:0.4rem;font-size:0.82rem;font-weight:600;
padding:0.24rem 0.7rem;border-radius:7px;margin-top:0.25rem;}
.hb-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0;}
.hb-done{background:rgba(34,197,94,0.18);color:#86efac;}
.hb-done .hb-dot{background:#4ade80;}

.hb-idle{background:rgba(255,255,255,0.07);color:#c2cfc7;}
.hb-idle .hb-dot{background:rgba(255,255,255,0.35);}
</style>
""", unsafe_allow_html=True)

#  SESSION STATE
_def = {
    "page": "predict",
    "prediction_df": None, "selected_seq": "", "eval_result": None,
    "tox_result": None, "dl_result": None,
    "pdb_path": None, "pdb_source": "esmfold",
    "receptor_path": None, "valid_seq": "",
    "docking_status": "idle", "docking_result": None,
    "job_dir": None, "haddock_score": None, "docking_pid": None,
    "md_status": "idle", "md_job_dir": None, "md_pid": None,
    "mmpbsa_result": None,
    "md_input_mode": "pipeline",
    "mmpbsa_input_mode": "pipeline",
    "manual_md_dir": None,
    "_f_strict": False,
    "_min_len": 3, "_max_len": 200,
    "_ch_range": (-10, 10), "_pi_range": (3.0, 12.0),
    "_ar_range": (0.0, 1.0), "_hy_range": (-5.0, 5.0),
    "_sampling": 200, "_ncores": 16, "_top_models": 4,
    "_md_ff": "amber99sb-ildn", "_md_water": "tip3p",
    "_md_len": "1 ns", "_md_dt": "0.002 ps",
}
for k, v in _def.items():
    st.session_state.setdefault(k, v)

#  FAQ DB
FAQ_DB = [
    ("What file format do I upload for peptide prediction?",
     "Upload a FASTA file (.fasta, .fa, or .txt). It must start with a '>' "
     "header line followed by the amino acid sequence."),
    ("What does the Bioactivity Task dropdown mean?",
     "It selects which activity class to screen for: ABP=Antibacterial, "
     "ACP=Anticancer, AFP=Antifungal, AHP=Anti-HIV, AIP=Anti-inflammatory, "
     "APP=Antiparasitic, AVP=Antiviral."),
    ("What does the Confidence Threshold slider do?",
     "It filters out peptides whose predicted bioactivity score is below "
     "this value (0-1). Higher = fewer, more confident results."),
    ("What is drug-likeness score?",
     "A percentage showing how many of 7 physicochemical rules a peptide "
     "passes (molecular weight, net charge, H-bond donors/acceptors, "
     "aromatic fraction, hydrophobicity, length). 80%+ is favorable."),
    ("Why is the 3D PCA map empty or not showing?",
     "The PCA map needs at least 3 peptides with a valid score column. If "
     "you changed the Bioactivity Task dropdown after running prediction, "
     "re-run prediction so the score column matches the selected task."),
    ("What is ESMFold?",
     "ESMFold is a deep learning model that predicts a peptide's 3D "
     "structure directly from its sequence, without needing a similar "
     "template structure to already exist."),
    ("How do I get a receptor structure?",
     "Either download one from RCSB PDB by its 4-character ID (e.g. 1YCR "
     "for MDM2), or upload your own receptor PDB file. Ligands, water, and "
     "DNA are automatically removed."),
    ("What does chain selection do in Structure & Receptor?",
     "If your receptor has multiple chains, you choose which one(s) to "
     "keep for docking, so an irrelevant chain doesn't interfere."),
    ("What is HADDOCK3?",
     "A data-driven molecular docking engine used to predict how the "
     "peptide and receptor bind to each other in 3D space."),
    ("What does Sampling Poses control in docking?",
     "The number of random starting orientations tried during docking. "
     "Higher values explore more possibilities but take longer."),
    ("What does Top Models for Refinement mean?",
     "The number of best-scoring poses that get carried into the more "
     "expensive flexible-refinement stage of docking."),
    ("How do I read the HADDOCK score?",
     "Lower (more negative) is better. Below -50 is a strong predicted "
     "interaction, -30 to -50 is moderate, above -30 is weak."),
    ("Can I skip docking?",
     "Yes - use the 'Upload Docked Complex Directly' tab on the Molecular "
     "Docking page if you already have a docked complex PDB."),
    ("What force field should I use for MD?",
     "amber99sb-ildn is the default and works well for most protein/"
     "peptide systems. amber03, charmm27, and oplsaa are alternatives."),
    ("What water model should I use?",
     "tip3p is the default and computationally efficient. tip4p and spce "
     "are more accurate but slightly more expensive."),
    ("What does MD Length mean?",
     "The total simulated time of the production run, in nanoseconds. "
     "Longer runs sample more conformations but take more compute time."),
    ("What software does MD Simulation require?",
     "Both GROMACS (gmx) and gmx_MMPBSA must be installed inside a conda "
     "environment named mmpbsa_env."),
    ("How do I read RMSD results?",
     "RMSD measures how much the structure deviates from its starting "
     "shape over time. Mean RMSD below about 0.2 nm indicates a stable "
     "complex."),
    ("What is MM-PBSA?",
     "A method that estimates the binding free energy (deltaG) of the "
     "docked complex from the MD trajectory, giving a physical energy "
     "value in kcal/mol."),
    ("How do I interpret the MM-PBSA binding energy?",
     "deltaG <= -10 kcal/mol is a very strong binder, -7 to -10 is strong, "
     "-5 to -7 is moderate, and above -5 kcal/mol is a weak binder."),
    ("Can I skip running MD myself?",
     "Yes - on the MM-PBSA Results page, use the 'Upload MD Output Files "
     "Directly' tab to upload FINAL_MMPBSA.dat/.csv and .xvg files from an "
     "external GROMACS/gmx_MMPBSA run."),
    ("What environment does the app itself run in?",
     "haddock_env - that's the environment with streamlit, scikit-learn, "
     "plotly, and py3Dmol installed, and it's what you run "
     "`streamlit run app.py` from."),
    ("What environment does MD simulation run in?",
     "mmpbsa_env - a separate conda environment used only by the "
     "background GROMACS/gmx_MMPBSA script, not the Streamlit app itself."),
]


@st.cache_resource(show_spinner=False)
def _build_faq_index():
    questions = [q for q, _ in FAQ_DB]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(questions)
    return vectorizer, matrix


def answer_faq(user_question: str, threshold: float = 0.20):
    vectorizer, matrix = _build_faq_index()
    query_vec = vectorizer.transform([user_question])
    sims = cosine_similarity(query_vec, matrix)[0]
    best_idx = sims.argmax()
    best_score = sims[best_idx]
    if best_score < threshold:
        return ("I couldn't find a confident match for that in the FAQ. "
                "Try rephrasing your question, or check the full **User "
                "Guide** page in the sidebar for more detail."), None
    return FAQ_DB[best_idx][1], best_score

def _img_to_base64(path):
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None
def sec(icon, title):
    ic = f"<span class='material-symbols-outlined'>{icon}</span>" if icon else ""
    st.markdown(f"<div class='sec-hdr'>{ic}<span>{title}</span></div>",
                unsafe_allow_html=True)

def micon(name):
    return f"<span class='material-symbols-outlined'>{name}</span>"

def met(label, value, unit="", color="#15803d"):
    st.markdown(f"<div class='met-card'><div class='met-val' style='color:{color};'>{value}</div>"
                f"<div class='met-lbl'>{label} {unit}</div></div>", unsafe_allow_html=True)

def pcfg():
    return dict(paper_bgcolor="#ffffff", plot_bgcolor="#f8fafc",
                font_color="#374151", font_family="Inter",
                margin=dict(t=50, b=40, l=50, r=30))

def show_3d(pdb_path, height=420):
    try:
        with open(pdb_path) as f:
            pdb = f.read()
        viewer = py3Dmol.view(width="100%", height=height)
        viewer.addModel(pdb, "pdb")
        viewer.setStyle({"model": -1}, {"cartoon": {"color": "spectrum"}})
        viewer.zoomTo()
        viewer.setBackgroundColor("white")
        viewer.render()
        components.html(viewer._make_html(), height=height)
    except Exception as e:
        st.error(f"3D Viewer Error: {e}")
        try:
            lines = open(pdb_path).readlines()
            atom_count = sum(1 for l in lines if l.startswith("ATOM"))
            chain_ids = set(l[21] for l in lines if l.startswith("ATOM"))
            models = sum(1 for l in lines if l.startswith("MODEL"))
            st.caption(f"PDB debug - {atom_count} ATOM lines · chains: {chain_ids} · MODEL records: {models}")
        except Exception:
            pass

#  RECEPTOR CLEANING
STANDARD_AA = {
    'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE',
    'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL'
}

def clean_pdb(inp, out, keep_chains=None):
    DNA_RNA = {'DA','DC','DG','DT','A','C','G','U',
               'DA3','DA5','DC3','DC5','DG3','DG5','DT3','DT5',
               'ADE','CYT','GUA','THY','URA'}
    lines_in = open(inp).readlines()
    if keep_chains is None:
        keep_chains = set()
        for l in lines_in:
            if l.startswith("ATOM"):
                resn = l[17:20].strip()
                if resn in STANDARD_AA:
                    keep_chains.add(l[21])
    keep_chains = set(keep_chains)
    seen = set(); serial = 1; out_lines = []; prev_chain = None
    for l in lines_in:
        if l.startswith("ATOM"):
            chain = l[21]; resn = l[17:20].strip()
            if chain not in keep_chains: continue
            if resn in DNA_RNA or resn not in STANDARD_AA: continue
            key = (chain, l[22:26].strip(), l[12:16].strip())
            if key in seen: continue
            seen.add(key)
            if prev_chain is not None and chain != prev_chain:
                out_lines.append("TER\n")
            prev_chain = chain
            l = f"ATOM  {serial:5d}{l[11:]}"; serial += 1
            out_lines.append(l)
        elif l.startswith("HETATM"):
            continue
        elif l.startswith(("REMARK", "CRYST1", "HEADER", "TITLE")):
            out_lines.append(l)
    if out_lines and not out_lines[-1].startswith("TER"):
        out_lines.append("TER\n")
    out_lines.append("END\n")
    open(out, "w").writelines(out_lines)
    atoms = [l for l in out_lines if l.startswith("ATOM")]
    chains = set(l[21] for l in atoms)
    resnames = set(l[17:20].strip() for l in atoms)
    nonstd = resnames - STANDARD_AA
    return len(atoms), chains, nonstd

def fix_chain(input_pdb, chain_id, output_pdb):
    serial = 1
    with open(input_pdb) as fin, open(output_pdb, "w") as fout:
        for line in fin:
            if line.startswith(("ATOM", "HETATM")):
                line = line[:21] + chain_id + line[22:]
                line = f"{line[:6]}{serial:5d}{line[11:]}"
                serial += 1
                fout.write(line)
            elif line.startswith(("MODEL", "REMARK", "HEADER", "TITLE", "CRYST1")):
                fout.write(line)
        fout.write("TER\n")
        fout.write("END\n")

def clean_rec(inp, out, keep_chain="A"):
    chains = {keep_chain} if isinstance(keep_chain, str) else set(keep_chain)
    clean_pdb(inp, out, keep_chains=chains)

def clean_rec_select_chains(inp, out, keep_chains):
    clean_pdb(inp, out, keep_chains=set(keep_chains))

#  DOCKING HELPERS
def docking_toml(job_dir, sampling, ncores, top_models=4):
    txt = f"""run_dir = "run1"
molecules = ["receptor_clean.pdb","peptide.pdb"]
[topoaa]
[rigidbody]
cmrest = true
sampling = {sampling}
ncores = {ncores}
[caprieval]
[clustfcc]
[seletopclusts]
top_models = {top_models}
[flexref]
cmrest = true
ncores = {ncores}
tolerance = 60
[caprieval]
[emref]
ncores = {ncores}
tolerance = 60
[caprieval]
"""
    open(os.path.join(job_dir, "docking.toml"), "w").write(txt)

@st.cache_data(show_spinner=False)
def load_capri(path):
    return pd.read_csv(path, sep="\t")

@st.cache_data(show_spinner=False)
def get_score(run_dir):
    for d in ["7_caprieval", "8_caprieval"]:
        f = os.path.join(run_dir, d, "capri_ss.tsv")
        if os.path.exists(f):
            try:
                df = pd.read_csv(f, sep="\t")
                row = df[df["caprieval_rank"] == 1]
                if len(row) == 0: row = df.iloc[[0]]
                return float(row["score"].values[0])
            except Exception: pass
    return None

def get_top_pdb(run_dir):
    out = os.path.join(run_dir, "top_model.pdb")
    if os.path.exists(out) and os.path.getsize(out) > 100:
        return out
    cf = None
    for d in ["7_caprieval", "8_caprieval"]:
        p = os.path.join(run_dir, d, "capri_ss.tsv")
        if os.path.exists(p): cf = p; break
    pn = None
    if cf:
        try:
            df = pd.read_csv(cf, sep="\t")
            row = df[df["caprieval_rank"] == 1]
            if len(row) == 0: row = df.iloc[[0]]
            model_path = str(row["model"].values[0])
            pn = os.path.basename(model_path)
            candidates = [
                model_path,
                os.path.join(run_dir, model_path),
                os.path.join(run_dir, pn),
                os.path.normpath(os.path.join(os.path.dirname(cf), model_path)),
                os.path.normpath(os.path.join(os.path.dirname(cf), "..", "6_emref", pn)),
            ]
            for cand in candidates:
                for fp in [cand, cand + ".gz"]:
                    if os.path.exists(fp):
                        if fp.endswith(".gz"):
                            with gzip.open(fp, "rt") as g, open(out, "w") as o: o.write(g.read())
                        else:
                            shutil.copy(fp, out)
                        return out
        except Exception: pass
    ed = os.path.join(run_dir, "6_emref")
    if os.path.isdir(ed):
        search = ([pn, pn + ".gz"] if pn else []) + sorted(os.listdir(ed))
        for fn in search:
            if not fn: continue
            fp = os.path.join(ed, fn)
            if not os.path.exists(fp): continue
            if fn.endswith(".pdb.gz"):
                with gzip.open(fp, "rt") as g, open(out, "w") as o: o.write(g.read())
                return out
            if fn.endswith(".pdb") and os.path.getsize(fp) > 100:
                shutil.copy(fp, out); return out
    for root, _, files in os.walk(run_dir):
        if "emref" not in root: continue
        for fn in sorted(files):
            fp = os.path.join(root, fn)
            if fn.endswith(".pdb") and os.path.getsize(fp) > 100:
                shutil.copy(fp, out); return out
            if fn.endswith(".pdb.gz"):
                with gzip.open(fp, "rt") as g, open(out, "w") as o: o.write(g.read())
                return out
    return None

def find_runs(base="."):
    found = []
    try:
        for e in os.scandir(base):
            if not e.is_dir(): continue
            r1 = os.path.join(e.path, "run1")
            for d in ["7_caprieval", "8_caprieval"]:
                if os.path.exists(os.path.join(r1, d, "capri_ss.tsv")):
                    found.append(r1); break
                if os.path.exists(os.path.join(e.path, d, "capri_ss.tsv")):
                    found.append(e.path); break
    except Exception: pass
    return sorted(set(found))

def haddock_progress(run_dir):
    steps = ["0_topoaa","1_rigidbody","2_caprieval","3_clustfcc",
             "4_seletopclusts","5_flexref","6_emref","7_caprieval"]
    done = [s for s in steps if os.path.isdir(os.path.join(run_dir, s))]
    return done, len(done) / len(steps)

def capri_is_done(run_dir):
    for d in ["7_caprieval", "8_caprieval"]:
        if os.path.exists(os.path.join(run_dir, d, "capri_ss.tsv")):
            return True
    return False

#  MD HELPERS
def write_mdp(d, name, nsteps=50000, dt=0.002):
    t = {
        "em": (
            "integrator  = steep\nemtol       = 10.0\nemstep      = 0.01\n"
            "nsteps      = 500000\ncutoff-scheme = Verlet\nnstlist     = 20\n"
            "coulombtype = PME\nrcoulomb    = 1.0\nrvdw        = 1.0\n"
            "pbc         = xyz\nconstraints = none\n"
        ),
        "nvt": (
            f"integrator=md\nnsteps=50000\ndt=0.001\ncontinuation=no\n"
            "constraint_algorithm=lincs\nconstraints=h-bonds\n"
            "lincs_iter=4\nlincs_order=6\ncutoff-scheme=Verlet\n"
            "rcoulomb=1.0\nrvdw=1.0\ncoulombtype=PME\ntcoupl=V-rescale\n"
            "tc-grps=Protein Non-Protein\ntau_t=0.1 0.1\nref_t=300 300\n"
            "pcoupl=no\npbc=xyz\ngen_vel=yes\ngen_temp=300\ngen_seed=-1\n"
            "define = -DPOSRES\n"
        ),
        "npt": (
            f"integrator=md\nnsteps=50000\ndt={dt}\ncontinuation=yes\n"
            "constraint_algorithm=lincs\nconstraints=h-bonds\n"
            "lincs_iter=4\nlincs_order=6\ncutoff-scheme=Verlet\n"
            "rcoulomb=1.0\nrvdw=1.0\ncoulombtype=PME\ntcoupl=V-rescale\n"
            "tc-grps=Protein Non-Protein\ntau_t=0.1 0.1\nref_t=300 300\n"
            "pcoupl=Parrinello-Rahman\npcoupltype=isotropic\ntau_p=2.0\nref_p=1.0\n"
            "compressibility=4.5e-5\npbc=xyz\ngen_vel=no\ndefine = -DPOSRES\n"
        ),
        "md": (
            f"integrator=md\nnsteps={nsteps}\ndt={dt}\ncontinuation=yes\n"
            "constraint_algorithm=lincs\nconstraints=h-bonds\ncutoff-scheme=Verlet\n"
            "rcoulomb=1.0\nrvdw=1.0\ncoulombtype=PME\ntcoupl=V-rescale\n"
            "tc-grps=Protein Non-Protein\ntau_t=0.1 0.1\nref_t=300 300\n"
            "pcoupl=Parrinello-Rahman\npcoupltype=isotropic\ntau_p=2.0\nref_p=1.0\n"
            "compressibility=4.5e-5\npbc=xyz\ngen_vel=no\n"
            "nstxout-compressed=5000\nnstenergy=5000\n"
        ),
    }
    open(os.path.join(d, f"{name}.mdp"), "w").write(t.get(name, t["md"]))

def make_md_script(job_dir, ff, water, nsteps, dt, ncores=8):
    mdp_d = os.path.join(job_dir, "mdp")
    os.makedirs(mdp_d, exist_ok=True)
    for m in ["em", "nvt", "npt", "md"]:
        write_mdp(mdp_d, m, nsteps=nsteps, dt=dt)
    ns = round(nsteps * dt / 1000, 2)

    open(os.path.join(job_dir, "mmpbsa.in"), "w").write(
        "&general\nstartframe=1,endframe=1000,interval=10,verbose=2,keep_files=0,\n/\n"
        "&gb\nigb=5,saltcon=0.150,\n/\n"
        "&pb\nistrng=0.150,fillratio=4.0,inp=1,\n/\n"
    )

    sh = f"""#!/bin/bash
set -e
cd {job_dir}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mmpbsa_env
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
echo "=== MD + MM-PBSA Pipeline ({ns} ns) ==="
which gmx      || {{ echo "ERROR: gmx not found";      exit 1; }}
which gmx_MMPBSA || {{ echo "ERROR: gmx_MMPBSA not found"; exit 1; }}
echo "--- Step 0: Preprocessing complex.pdb ---"
python3 - << 'PREPEOF'
lines = open("complex.pdb").readlines()
out = []
for l in lines:
    if not l.startswith(("ATOM","HETATM","TER","END")):
        out.append(l); continue
    if l.startswith(("TER","END")):
        out.append(l); continue
    if l.startswith("HETATM"):
        l = "ATOM  " + l[6:]
    resn = l[17:20].strip()
    atom = l[12:16].strip()
    chain = l[21]
    try: resnum = int(l[22:26])
    except: out.append(l); continue
    if resn == "GLY" and atom == "CB":
        print(f"Removed GLY CB: chain {{chain}} res {{resnum}}")
        continue
    out.append(l)
open("complex_prep.pdb","w").writelines(out)
print(f"Preprocessed: {{sum(1 for l in out if l.startswith('ATOM'))}} ATOM lines")
PREPEOF
echo "--- Step 1: Topology ---"
echo "0" | gmx pdb2gmx -f complex_prep.pdb -o processed.gro \\
    -water {water} -ff {ff} -ignh 2>&1 | tee pdb2gmx.log
grep -i "fatal" pdb2gmx.log && {{ echo "pdb2gmx FAILED"; exit 1; }} || true
echo "--- Step 2: Adding POSRES to chain ITPs ---"
python3 - << 'POSRESEOF'
import glob, os
itp_files = sorted(glob.glob("topol_Protein_chain_*.itp"))
print(f"Chain ITP files: {{itp_files}}")
for itp in itp_files:
    content = open(itp).read()
    if "POSRES" not in content:
        content += "\\n#ifdef POSRES\\n#include \\"posre.itp\\"\\n#endif\\n"
        open(itp,"w").write(content)
        print(f"Added POSRES to {{itp}}")
    else:
        print(f"POSRES already in {{itp}}")
POSRESEOF
echo "--- Step 3: Box + Solvation + Ions ---"
gmx editconf -f processed.gro -o boxed.gro -c -d 1.0 -bt dodecahedron
gmx solvate -cp boxed.gro -cs spc216.gro -o solvated.gro -p topol.top
grep "SOL" topol.top || {{ echo "ERROR: SOL not in topol.top after solvate"; exit 1; }}
gmx grompp -f mdp/em.mdp -c solvated.gro -p topol.top -o ions.tpr -maxwarn 2
echo "SOL" | gmx genion -s ions.tpr -o ionized.gro -p topol.top \\
    -pname NA -nname CL -neutral
echo "--- Step 4: Energy Minimization ---"
gmx grompp -f mdp/em.mdp -c ionized.gro -p topol.top -o em.tpr -maxwarn 2
gmx mdrun -v -deffnm em -ntmpi 1 -ntomp {ncores}
grep "converged to Fmax\\|Maximum force" em.log | tail -3
echo "--- Step 5: Generating position restraints ---"
rm -f posre.itp \\#posre.itp*
echo "Protein" | gmx genrestr -f em.gro -o posre.itp -fc 1000 1000 1000
[ -s posre.itp ] || {{ echo "ERROR: posre.itp empty or missing"; exit 1; }}
echo "posre.itp: $(wc -l < posre.itp) lines"
echo "--- Step 6: NVT ---"
gmx grompp -f mdp/nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr -maxwarn 2
gmx mdrun -v -deffnm nvt -ntmpi 1 -ntomp {ncores}
echo "--- Step 7: NPT ---"
gmx grompp -f mdp/npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt \\
    -p topol.top -o npt.tpr -maxwarn 2
gmx mdrun -v -deffnm npt -ntmpi 1 -ntomp {ncores}
echo "--- Step 8: Production MD ---"
gmx grompp -f mdp/md.mdp -c npt.gro -t npt.cpt -p topol.top -o md.tpr -maxwarn 2
gmx mdrun -v -deffnm md -ntmpi 1 -ntomp {ncores}
echo "--- Step 9: Reading residue ranges ---"
python3 - << 'RANGEEOF'
import os, sys
from collections import defaultdict
rec_res, pep_res = [], []
for fname, store in [("complex_prep.pdb","both")]:
    chains = defaultdict(set)
    for line in open(fname):
        if line.startswith("ATOM"):
            ch = line[21]
            try: chains[ch].add(int(line[22:26].strip()))
            except: pass
    if len(chains) >= 2:
        by_size = sorted(chains.items(), key=lambda x: -len(x[1]))
        rec_res = sorted(by_size[0][1])
        pep_res = sorted(by_size[-1][1])
    elif len(chains) == 1:
        all_res = sorted(list(chains.values())[0])
        mid = len(all_res)//2
        rec_res = all_res[:mid]; pep_res = all_res[mid:]
if not rec_res or not pep_res:
    print("ERROR: Cannot determine residue ranges"); sys.exit(1)
print(f"Receptor: {{rec_res[0]}}-{{rec_res[-1]}} ({{len(rec_res)}} res)")
print(f"Peptide:  {{pep_res[0]}}-{{pep_res[-1]}} ({{len(pep_res)}} res)")
open("residue_ranges.txt","w").write(
    f"{{rec_res[0]}} {{rec_res[-1]}} {{pep_res[0]}} {{pep_res[-1]}}\\n")
RANGEEOF
read REC_START REC_END PEP_START PEP_END < residue_ranges.txt
echo "--- Step 10: Building index ---"
printf "r $REC_START-$REC_END\\nname 17 Receptor\\nr $PEP_START-$PEP_END\\nname 18 Ligand\\nq\\n" \\
    | gmx make_ndx -f em.gro -o index.ndx
echo "--- Step 11: Analysis ---"
printf "0\\n" | gmx trjconv -s md.tpr -f md.xtc -o md_nojump.xtc \\
    -pbc nojump -n index.ndx
printf "17\\n0\\n" | gmx trjconv -s md.tpr -f md_nojump.xtc -o md_center.xtc \\
    -pbc mol -center -n index.ndx
printf "17\\n0\\n" | gmx trjconv -s md.tpr -f md_center.xtc -o md_fit.xtc \\
    -fit rot+trans -n index.ndx
printf "4\\n4\\n" | gmx rms    -s md.tpr -f md_fit.xtc -o rmsd.xvg -tu ns 2>/dev/null || \\
printf "1\\n1\\n" | gmx rms    -s md.tpr -f md_fit.xtc -o rmsd.xvg -tu ns
printf "1\\n"    | gmx rmsf   -s md.tpr -f md_fit.xtc -o rmsf.xvg -res
printf "1\\n"    | gmx gyrate -s md.tpr -f md_fit.xtc -o gyrate.xvg
echo "--- Step 13: Cleaning topology ---"
python3 - << 'TOPCLEANEOF'
import glob, re, shutil, os
content = open("topol.top").read()
itp_files = sorted(glob.glob("topol_Protein_chain_*.itp"))
if itp_files:
    lines = content.splitlines(keepends=True)
    ff_inc = water_inc = ions_inc = ""
    for line in lines:
        ls = line.strip()
        if ls.startswith("#include") and "forcefield" in ls.lower() and not ff_inc:
            ff_inc = ls
        if ls.startswith("#include") and any(x in ls.lower() for x in ["tip","spc"]) and not water_inc:
            water_inc = ls
        if ls.startswith("#include") and "ions" in ls.lower() and not ions_inc:
            ions_inc = ls
    mol_section = ""
    in_mols = False
    for line in lines:
        if re.match(r'\\s*\\[\\s*molecules\\s*\\]', line):
            in_mols = True; mol_section += line; continue
        if in_mols:
            if line.strip().startswith('['): break
            mol_section += line
    new_top = ["; PTPLab Clean Topology\\n\\n"]
    if ff_inc: new_top.append(ff_inc + "\\n\\n")
    for itp in itp_files:
        new_top.append(f'#include "{{itp}}"\\n')
    new_top.append("\\n")
    if water_inc: new_top.append(water_inc + "\\n")
    if ions_inc:  new_top.append(ions_inc  + "\\n")
    new_top.append("\\n[ system ]\\nReceptor-Peptide Complex\\n\\n")
    new_top.append(mol_section)
    shutil.copy("topol.top","topol_backup.top")
    open("topol.top","w").writelines(new_top)
    print("topol.top rebuilt")
TOPCLEANEOF
echo "--- Step 14: MM-PBSA ---"
gmx_MMPBSA -O \\
    -i  mmpbsa.in \\
    -cs md.tpr \\
    -ci index.ndx \\
    -cg Receptor Ligand \\
    -ct md_fit.xtc \\
    -cp topol.top \\
    -o  FINAL_MMPBSA.dat \\
    -eo FINAL_MMPBSA.csv \\
    -nogui && echo "MM-PBSA SUCCESS" || echo "MM-PBSA failed - MD results still saved"
if [ ! -s FINAL_MMPBSA.dat ]; then
    echo "MMPBSA_FAILED" > mmpbsa_status.txt
    echo "WARNING: FINAL_MMPBSA.dat was not created - MM-PBSA step failed"
else
    echo "MMPBSA_OK" > mmpbsa_status.txt
fi
echo "=== COMPLETE ==="
touch PIPELINE_DONE
"""
    script_path = os.path.join(job_dir, "run_md.sh")
    open(script_path, "w").write(sh)
    os.chmod(script_path, 0o755)
    return script_path

def md_progress(d):
    cpts = [
        ("processed.gro","Topology",0.08),("boxed.gro","Box",0.16),
        ("solvated.gro","Solvation",0.24),("ionized.gro","Ions",0.32),
        ("em.gro","Energy Min",0.45),("nvt.gro","NVT Equil",0.58),
        ("npt.gro","NPT Equil",0.70),("md.gro","Production MD",0.84),
        ("rmsd.xvg","Analysis",0.93),("FINAL_MMPBSA.dat","MM-PBSA",1.00),
    ]
    s, f = "Initializing", 0.0
    for fn, st_, fr in cpts:
        if os.path.exists(os.path.join(d, fn)): s, f = st_, fr
    return s, f

def parse_xvg(fp):
    x, y = [], []
    if not os.path.exists(fp): return x, y
    for l in open(fp):
        l = l.strip()
        if l.startswith(("#","@")): continue
        p = l.split()
        if len(p) >= 2:
            try: x.append(float(p[0])); y.append(float(p[1]))
            except Exception: pass
    return x, y

def break_chain_discontinuities(x, y):
    """For multi-chain RMSF data: insert a gap (None) in y wherever the residue
    number (x) resets or jumps, so plotly draws separate line segments per chain
    instead of a straight connector line across the chain boundary."""
    if not x:
        return x, y
    new_x, new_y = [x[0]], [y[0]]
    for i in range(1, len(x)):
        if x[i] <= x[i-1] or (x[i] - x[i-1]) > 1:
            new_x.append(x[i-1])
            new_y.append(None)
        new_x.append(x[i])
        new_y.append(y[i])
    return new_x, new_y

def parse_mmpbsa(job_dir):
    """Parse gmx_MMPBSA output. Reads the structured CSV (Delta Energy Terms
    table) directly instead of regex-matching the human-readable .dat report,
    since the .dat report format varies across gmx_MMPBSA versions and using
    real per-frame CSV data lets us compute genuine mean/std ourselves."""
    import io
    r = {}
    dat = os.path.join(job_dir, "FINAL_MMPBSA.dat")
    csv_path = os.path.join(job_dir, "FINAL_MMPBSA.csv")

    if os.path.exists(dat):
        txt = open(dat).read()
        r["raw"] = txt
        r["validity_warning"] = "VALIDITY OF THESE RESULTS ARE HIGHLY QUESTIONABLE" in txt

    if not os.path.exists(csv_path):
        return r

    csv_text = open(csv_path).read()
    r["df"] = pd.read_csv(csv_path, on_bad_lines="skip")

    def extract_delta_table(method_header):
        """Pull the 'Delta Energy Terms' CSV block that follows the given
        method section header (e.g. 'POISSON BOLTZMANN:' or 'GENERALIZED BORN:')."""
        m = re.search(re.escape(method_header), csv_text)
        if not m:
            return None
        section = csv_text[m.end():]
        dm = re.search(r"Delta Energy Terms\n(.*?)(?:\n\n|\Z)", section, re.S)
        if not dm:
            return None
        block_text = dm.group(1)
        try:
            return pd.read_csv(io.StringIO(block_text))
        except Exception:
            return None

    pb = extract_delta_table("POISSON BOLTZMANN:")
    gb = extract_delta_table("GENERALIZED BORN:")

    def stat(df, col):
        if df is None or col not in df.columns:
            return None, None
        vals = df[col].dropna()
        if vals.empty:
            return None, None
        return float(vals.mean()), float(vals.std())

    src = pb if pb is not None else gb
    if src is not None:
        r["dg"], r["dg_std"] = stat(src, "TOTAL")
        r["vdw"], r["vdw_std"] = stat(src, "VDWAALS")
        r["elec"], r["elec_std"] = stat(src, "EEL")
        # PB section uses EPB/ENPOLAR; GB section uses EGB/ESURF for the equivalent terms
        if "EPB" in src.columns:
            r["polar"], r["polar_std"] = stat(src, "EPB")
            r["nopolar"], r["nopolar_std"] = stat(src, "ENPOLAR")
        else:
            r["polar"], r["polar_std"] = stat(src, "EGB")
            r["nopolar"], r["nopolar_std"] = stat(src, "ESURF")
        r["method"] = "PB" if pb is not None else "GB"

    return r

def proc_alive(pid):
    if not pid: return False
    try: os.kill(int(pid), 0); return True
    except Exception: return False

#  DRUG-LIKENESS
def drug_likeness(seq):
    aa_mw = {'A':89,'R':174,'N':132,'D':133,'C':121,'E':147,'Q':146,'G':75,
             'H':155,'I':131,'L':131,'K':146,'M':149,'F':165,'P':115,'S':105,
             'T':119,'W':204,'Y':181,'V':117}
    mw = sum(aa_mw.get(aa,110) for aa in seq.upper()) - (len(seq)-1)*18
    charge = sum({'R':1,'K':1,'H':0.5,'D':-1,'E':-1}.get(aa,0) for aa in seq.upper())
    hbd = seq.upper().count('S')+seq.upper().count('T')+seq.upper().count('N')+seq.upper().count('Q')
    hba = seq.upper().count('D')+seq.upper().count('E')+seq.upper().count('S')+seq.upper().count('T')
    aromatic = sum(1 for aa in seq.upper() if aa in 'FWY')/max(len(seq),1)
    hydro_aa = {'A':1.8,'V':4.2,'L':3.8,'I':4.5,'P':-1.6,'F':2.8,'W':-0.9,
                'M':1.9,'G':-0.4,'S':-0.8,'T':-0.7,'C':2.5,'Y':-1.3,'H':-3.2,
                'D':-3.5,'E':-3.5,'N':-3.5,'Q':-3.5,'K':-3.9,'R':-4.5}
    hydro = sum(hydro_aa.get(aa,0) for aa in seq.upper())/max(len(seq),1)
    rules = [
        ("Mol. Weight",     f"{mw:.0f} Da",    mw<=2000,        "≤2000 Da", mw),
        ("Net Charge",      f"{charge:+.1f}",   -5<=charge<=5,  "-5 to +5", charge),
        ("H-Bond Donors",   f"{hbd}",           hbd<=10,         "≤10",     hbd),
        ("H-Bond Acceptors",f"{hba}",           hba<=20,         "≤20",     hba),
        ("Aromatic Frac.",  f"{aromatic:.2f}",  aromatic<=0.35,  "≤0.35",   aromatic),
        ("Hydrophobicity",  f"{hydro:.2f}",     -2<=hydro<=3,    "-2 to 3", hydro),
        ("Length",          f"{len(seq)} aa",   3<=len(seq)<=50, "3–50 aa", len(seq)),
    ]
    passed = sum(1 for _,_,ok,_,_ in rules if ok)
    return rules, passed/len(rules)*100, mw, charge, hbd, hba, aromatic, hydro

def embedding_3d_plot(df, score_col, task):
    seqs = df["sequence"].tolist()[:100]; features = []
    for s in seqs:
        s = s.upper()
        aa_mw = {'A':89,'R':174,'N':132,'D':133,'C':121,'E':147,'Q':146,'G':75,
                 'H':155,'I':131,'L':131,'K':146,'M':149,'F':165,'P':115,'S':105,
                 'T':119,'W':204,'Y':181,'V':117}
        mw = sum(aa_mw.get(a,110) for a in s)
        charge = sum({'R':1,'K':1,'H':0.5,'D':-1,'E':-1}.get(a,0) for a in s)
        hydro = sum({'A':1.8,'V':4.2,'L':3.8,'I':4.5,'P':-1.6,'F':2.8,'W':-0.9,
                     'M':1.9,'G':-0.4,'S':-0.8,'T':-0.7,'C':2.5,'Y':-1.3,'H':-3.2,
                     'D':-3.5,'E':-3.5,'N':-3.5,'Q':-3.5,'K':-3.9,'R':-4.5}.get(a,0)
                    for a in s)/max(len(s),1)
        aromatic = sum(1 for a in s if a in 'FWY')/max(len(s),1)
        features.append([len(s),mw/1000,charge,hydro,aromatic,
                         s.count('P')/max(len(s),1), s.count('G')/max(len(s),1)])
    X = np.array(features, dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    stds = X.std(0)
    if np.all(stds < 1e-6):
        st.warning("All candidate peptides have nearly identical physicochemical "
                   "properties, so the 3D map cannot meaningfully separate them. "
                   "Try predicting a more diverse set of sequences.")
        return None
    X = (X - X.mean(0)) / (stds + 1e-8)
    pca = PCA(n_components=3, random_state=42); coords = pca.fit_transform(X)
    coords = np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
    if score_col in df.columns:
        scores = pd.to_numeric(df[score_col], errors="coerce").fillna(0.5).tolist()[:100]
    else:
        scores = [0.5] * len(seqs)
    plot_df = pd.DataFrame({
        "PC1":coords[:,0],"PC2":coords[:,1],"PC3":coords[:,2],
        "Score":scores,
        "Sequence":[s[:20]+"..." if len(s)>20 else s for s in seqs],
        "Length":[max(len(s),1) for s in seqs],
    })
    fig = px.scatter_3d(plot_df, x="PC1", y="PC2", z="PC3", color="Score",
                        size="Length", hover_data=["Sequence","Score","Length"],
                        color_continuous_scale="Greens",
                        title=f"Peptide Chemical Space - {task} (PCA)",
                        labels={"Score":f"{task} Score"})
    fig.update_traces(marker=dict(opacity=0.8, line=dict(width=0)))
    fig.update_layout(paper_bgcolor="#ffffff",
                      scene=dict(bgcolor="#f8fafc",
                          xaxis=dict(backgroundcolor="#f8fafc",gridcolor="#e2e8f0",title="PC1 (size/MW)"),
                          yaxis=dict(backgroundcolor="#f8fafc",gridcolor="#e2e8f0",title="PC2 (charge/polar)"),
                          zaxis=dict(backgroundcolor="#f8fafc",gridcolor="#e2e8f0",title="PC3 (hydrophobicity)")),
                      font_color="#374151", height=600,
                      coloraxis_colorbar=dict(title=f"{task}<br>Score", thickness=15))
    return fig

#  OLLAMA ASSISTANT (defined here, BEFORE page routing, so it's always available)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")

OLLAMA_MODEL = "llama3.2:3b"

def _live_run_context():
    """Summarize the user's current session state into plain text for the LLM."""
    parts = []
    df = st.session_state.get("prediction_df")
    if df is not None and len(df) > 0:
        task = st.session_state.get("prediction_task", "?")
        sc = f"{task}_score"
        top_seq = df.nlargest(1, sc)["sequence"].values[0] if sc in df.columns else df["sequence"].iloc[0]
        top_score = df.nlargest(1, sc)[sc].values[0] if sc in df.columns else None
        avg_drug = df["drug_score_%"].mean() if "drug_score_%" in df.columns else None
        parts.append(
            f"Prediction: {len(df)} peptides passed for task '{task}'. "
            f"Top candidate: {top_seq} (score={top_score:.3f})." if top_score is not None
            else f"Prediction: {len(df)} peptides passed for task '{task}'."
        )
        if avg_drug is not None:
            parts.append(f"Average drug-likeness across candidates: {avg_drug:.1f}%.")
    if st.session_state.get("pdb_path"):
        parts.append(f"Peptide structure loaded (source: {st.session_state.get('pdb_source')}).")
    if st.session_state.get("receptor_path"):
        parts.append("Receptor structure loaded and cleaned.")
    d_status = st.session_state.get("docking_status", "idle")
    if d_status == "done":
        score = st.session_state.get("haddock_score")
        if score is not None:
            interp = "strong" if score < -50 else "moderate" if score < -30 else "weak"
            parts.append(f"Docking complete. HADDOCK score = {score:.2f} ({interp} interaction).")
        else:
            parts.append("Docking complete, but HADDOCK score not yet parsed.")
    elif d_status == "running":
        parts.append("Docking is currently running.")
    elif "error" in str(d_status):
        parts.append(f"Docking ended in an error state: {d_status}")
    m_status = st.session_state.get("md_status", "idle")
    if m_status == "done":
        mdjd = st.session_state.get("md_job_dir", "")
        x, y = parse_xvg(os.path.join(mdjd, "rmsd.xvg")) if mdjd else ([], [])
        if y:
            mean_rmsd = float(np.mean(y))
            stability = "stable" if mean_rmsd < 0.2 else "flexible"
            parts.append(f"MD complete. Mean RMSD = {mean_rmsd:.3f} nm ({stability} complex).")
        else:
            parts.append("MD complete, RMSD data not yet available.")
    elif m_status == "running":
        mdjd = st.session_state.get("md_job_dir", "")
        stage, frac = md_progress(mdjd) if mdjd else ("Initializing", 0.0)
        parts.append(f"MD is running: stage='{stage}', {int(frac*100)}% complete.")
    elif "error" in str(m_status):
        parts.append(f"MD ended in an error state: {m_status}")
    res = st.session_state.get("mmpbsa_result")
    if res:
        dg = res.get("dg")
        if dg is not None:
            if dg <= -10: interp = "very strong binder"
            elif dg <= -7: interp = "strong binder"
            elif dg <= -5: interp = "moderate binder"
            else: interp = "weak binder"
            parts.append(f"MM-PBSA complete. ΔG binding = {dg:.2f} kcal/mol ({interp}).")
    if not parts:
        return "The user hasn't run any pipeline steps yet in this session."
    return "\n".join(f"- {p}" for p in parts)


def _project_context():
    faq_text = "\n".join(f"Q: {q}\nA: {a}" for q, a in FAQ_DB)
    live_text = _live_run_context()
    return f"""You are the PTPLab Assistant, an expert on the Plant Therapeutic
Peptide Lab app (peptide bioactivity prediction, ESMFold structure prediction,
HADDOCK3 docking, GROMACS MD, and gmx_MMPBSA binding energy analysis).

Answer questions about the app, its workflow, parameters, how to interpret
results, AND about the user's specific current session below. If they ask
"why is my score X" or "is my result good", use the live session data to
give a specific, grounded answer rather than a generic one. Be concise.

--- USER'S CURRENT SESSION STATE ---
{live_text}

--- REFERENCE FAQ ---
{faq_text}
"""


def ask_ollama(user_question, history):
    messages = [{"role": "system", "content": _project_context()}]
    for m in history[-6:]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_question})
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except requests.exceptions.ConnectionError:
        return ("⚠️ Ollama isn't running. Start it with `ollama serve`, "
                "or falling back to offline FAQ match below.\n\n"
                + answer_faq(user_question)[0])
    except Exception as e:
        return f"⚠️ Ollama error: {e}\n\n" + answer_faq(user_question)[0]


#  AUTO-RECOVER
_jd = st.session_state.get("job_dir", "")
if _jd and st.session_state.get("docking_status") == "running":
    _rd = os.path.join(_jd, "run1")
    if capri_is_done(_rd):
        _top = get_top_pdb(_rd)
        st.session_state.update({"docking_status":"done","docking_result":_top,"haddock_score":get_score(_rd)})
        if _top and os.path.exists(_top):
            _perm = os.path.join(_jd, "docked_complex.pdb")
            if not os.path.exists(_perm): shutil.copy(_top, _perm)

_md = st.session_state.get("md_job_dir", "")
if _md and st.session_state.get("md_status") == "running":
    if os.path.exists(os.path.join(_md, "PIPELINE_DONE")):
        st.session_state.update({"md_status":"done","mmpbsa_result":parse_mmpbsa(_md)})

#  ON-PAGE FILTER / SETTINGS PANELS
def render_peptide_filters():
    with st.container(border=True):
        st.markdown(f"<div class='filter-panel-hdr'><div class='filter-title'>"
                    f"{micon('filter_alt')}<span>Peptide Filters</span></div></div>",
                    unsafe_allow_html=True)
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1, st.container(border=True):
            min_len = st.slider("Min Length", 1, 500, st.session_state["_min_len"], key="pf_min_len")
        with fc2, st.container(border=True):
            max_len = st.slider("Max Length", 1, 500, st.session_state["_max_len"], key="pf_max_len")
        with fc3, st.container(border=True):
            ch_range = st.slider("Net Charge", -20, 20, st.session_state["_ch_range"], key="pf_ch")
        with fc4, st.container(border=True):
            pi_range = st.slider("pI Range", 0.0, 14.0, st.session_state["_pi_range"], key="pf_pi")
        fc5, fc6, fc7, fc8 = st.columns(4)
        with fc5, st.container(border=True):
            ar_range = st.slider("Aromatic", 0.0, 1.0, st.session_state["_ar_range"], key="pf_ar")
        with fc6, st.container(border=True):
            hy_range = st.slider("Hydrophobicity", -5.0, 5.0, st.session_state["_hy_range"], key="pf_hy")
        with fc7, st.container(border=True):
            f_strict = st.toggle("Strict mode", value=st.session_state["_f_strict"], key="pf_strict")
            st.caption("On: peptides outside any filter range are marked failed. "
                       "Off: out-of-range values are just an advisory warning.")
        with fc8, st.container(border=True):
            st.caption("Restore all filters below to their default values.")
            if st.button("Reset Filters", key="pf_reset", icon=":material/restart_alt:", use_container_width=True):
                for k, v in {"_f_strict": False, "_min_len": 3, "_max_len": 200,
                             "_ch_range": (-10, 10), "_pi_range": (3.0, 12.0),
                             "_ar_range": (0.0, 1.0), "_hy_range": (-5.0, 5.0)}.items():
                    st.session_state[k] = v
                st.rerun()
    st.session_state.update({
        "_f_strict": f_strict, "_min_len": min_len, "_max_len": max_len,
        "_ch_range": ch_range, "_pi_range": pi_range,
        "_ar_range": ar_range, "_hy_range": hy_range,
    })
    return f_strict, min_len, max_len, ch_range, pi_range, ar_range, hy_range

def render_docking_settings():
    with st.container(border=True):
        st.markdown(f"<div class='filter-panel-hdr'><div class='filter-title'>"
                    f"{micon('tune')}<span>Docking Settings</span></div></div>",
                    unsafe_allow_html=True)
        dc1, dc2, dc3, dc4 = st.columns(4)
        with dc1, st.container(border=True):
            sampling = st.slider("Sampling poses", 50, 1000, st.session_state["_sampling"], 50, key="ds_sampling")
        with dc2, st.container(border=True):
            ncores = st.slider("CPU Cores", 1, 64, st.session_state["_ncores"], key="ds_ncores")
        with dc3, st.container(border=True):
            top_models = st.slider("Top models for refinement", 1, 20, st.session_state["_top_models"], key="ds_top")
        with dc4, st.container(border=True):
            if st.button("Reset Settings", key="ds_reset", icon=":material/restart_alt:", use_container_width=True):
                st.session_state.update({"_sampling": 200, "_ncores": 16, "_top_models": 4})
                st.rerun()
    st.session_state.update({"_sampling": sampling, "_ncores": ncores, "_top_models": top_models})
    return sampling, ncores, top_models

def render_md_settings():
    with st.container(border=True):
        st.markdown(f"<div class='filter-panel-hdr'><div class='filter-title'>"
                    f"{micon('tune')}<span>MD Settings</span></div></div>",
                    unsafe_allow_html=True)
        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1, st.container(border=True):
            md_ff = st.selectbox("Force Field", ["amber99sb-ildn","amber03","charmm27","oplsaa"],
                                  index=["amber99sb-ildn","amber03","charmm27","oplsaa"].index(st.session_state["_md_ff"]),
                                  key="ms_ff")
        with mc2, st.container(border=True):
            md_water = st.selectbox("Water Model", ["tip3p","tip4p","spc","spce"],
                                     index=["tip3p","tip4p","spc","spce"].index(st.session_state["_md_water"]),
                                     key="ms_water")
        with mc3, st.container(border=True):
            md_len = st.selectbox("MD Length", ["1 ns","5 ns","10 ns","50 ns","100 ns"],
                                   index=["1 ns","5 ns","10 ns","50 ns","100 ns"].index(st.session_state["_md_len"]),
                                   key="ms_len")
        with mc4, st.container(border=True):
            md_dt = st.selectbox("Time Step", ["0.002 ps","0.001 ps"],
                                  index=["0.002 ps","0.001 ps"].index(st.session_state["_md_dt"]),
                                  key="ms_dt")
    st.session_state.update({"_md_ff":md_ff,"_md_water":md_water,"_md_len":md_len,"_md_dt":md_dt})
    return md_ff, md_water, md_len, md_dt

#  SIDEBAR
with st.sidebar:
    st.markdown("""
    <div style='display:flex;align-items:center;justify-content:center;gap:0.7rem;padding:1.6rem 0 1.3rem;'>
        <div style='font-size:2.6rem;line-height:1;'>🌿</div>
        <div>
            <div style='color:#013220;font-weight:1000;font-size:2.5rem;
                        line-height:1.1;letter-spacing:-0.6px;text-align:left;
                        font-family:"Inter",sans-serif;'>
                PTPLab
            </div>
            <div style='color:#006400;font-weight:1000;font-size:1.0rem;
                        margin-top:4px;text-align:left;letter-spacing:0.5px;'>
                Plant Therapeutic Peptide Lab
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("<div class='nav-label'>NAVIGATION</div>", unsafe_allow_html=True)
    pages = [("predict","science","Peptide Prediction"),("structure","hub","Structure & Receptor"),
             ("docking","link","Molecular Docking"),("md","monitoring","MD Simulation"),
             ("mmpbsa","bolt","MM-PBSA Results"),("about","account_tree","About"),
             ("assistant","smart_toy","Ask Assistant"),
             ("guide","menu_book","User Guide")]
    done_map = {
        "predict":   st.session_state.get("prediction_df") is not None,
        "structure": st.session_state.get("pdb_path") is not None,
        "docking":   st.session_state.get("docking_status") in ("running","done"),
        "md":        st.session_state.get("md_status") in ("running","done"),
        "mmpbsa":    st.session_state.get("md_status") == "done",
        "about":     False,
        "guide":     False,
    }
    for pid, icon, label in pages:
        tick = "  ·  done" if done_map.get(pid,False) and st.session_state.page!=pid else ""
        is_active = st.session_state.page == pid
        active_mark = " ▸" if is_active else ""
        st.markdown(f"<div class='nav-mark-{pid}'></div>", unsafe_allow_html=True)
        if st.button(f"{label}{tick}{active_mark}", key=f"nav_{pid}",
                 icon=f":material/{icon}:", use_container_width=True,
                 type="primary" if is_active else "secondary"):
            st.session_state.page = pid; st.rerun()
    st.divider()
    d_st = st.session_state.get("docking_status","idle")
    m_st = st.session_state.get("md_status","idle")
    db = {"idle":"b-idle","running":"b-running","done":"b-done"}.get(
        "error" if "error" in str(d_st) else d_st, "b-error")
    mb = {"idle":"b-idle","running":"b-running","done":"b-done"}.get(
        "error" if "error" in str(m_st) else m_st, "b-error")
    st.divider()

page = st.session_state.page

#  HEADER
def _hb(status):
    if status == "done": return "hb-done", "Done"
    if status == "running": return "hb-running", "Running"
    return "hb-idle", "Not started"

dock_cls, dock_lbl = _hb(d_st)
md_cls, md_lbl = _hb(m_st)
steps_done = sum(done_map.values())

st.markdown(f"""
<div class='header-panel'>
  <span class='header-eyebrow'>{micon('eco')} AI-Driven Discovery Platform</span>
  <h1>🌿 Plant Therapeutic Peptide Lab</h1>
  <p>Accelerate peptide-based drug discovery with AI <br> From identifying the best therapeutic
  peptides across seven therapeutic classes to structure prediction, docking, molecular dynamics, and binding affinity analysis.</p>
  <div class='header-divider'></div>
  <div class='header-stats'>
    <div class='header-stat'>
      <div class='header-stat-lbl'>Pipeline Progress</div>
      <div class='header-stat-val'>{steps_done}/5 steps</div>
    </div>
    <div class='header-stat'>
      <div class='header-stat-lbl'>Version</div>
      <div class='header-stat-val'>v1.0</div>
    </div>
    <div class='header-stat'>
      <div class='header-stat-lbl'>Docking</div>
      <div class='header-stat-badge {dock_cls}'><span class='hb-dot'></span>{dock_lbl}</div>
    </div>
    <div class='header-stat'>
      <div class='header-stat-lbl'>MD Simulation</div>
      <div class='header-stat-badge {md_cls}'><span class='hb-dot'></span>{md_lbl}</div>
    </div>
    <div class='header-stat'>
      <div class='header-stat-lbl'>Stack</div>
      <div class='header-stat-val' style='font-size:0.82rem;line-height:1.5;'>PLM · ESMFold<br>HADDOCK3 · GROMACS · MM-PBSA</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
st.divider()

#  SHARED FUNCTIONS

def _show_docking_results():
    st.success("Docking complete.")
    jd = st.session_state.get("job_dir",""); rd = os.path.join(jd,"run1")
    rp = st.session_state.get("docking_result")
    if not rp or not os.path.exists(str(rp or "")):
        with st.spinner("Locating top model PDB..."):
            rp = get_top_pdb(rd); st.session_state["docking_result"] = rp
    if not st.session_state.get("haddock_score"):
        st.session_state["haddock_score"] = get_score(rd)
    if rp and os.path.exists(str(rp)):
        perm = os.path.join(jd,"docked_complex.pdb")
        if not os.path.exists(perm): shutil.copy(rp, perm)
    qs1, qs2 = st.columns([3,2])
    with qs1:
        if rp and os.path.exists(str(rp)):
            st.markdown("**Top Docked Complex**")
            show_3d(rp, height=420)
            with open(rp,"rb") as f:
                st.download_button("Download Complex PDB", f,
                                   "docked_complex.pdb", mime="chemical/x-pdb", key="dl_dock_top")
        else:
            st.warning("Could not locate PDB.")
    with qs2:
        cf = None
        for d in ["7_caprieval","8_caprieval"]:
            p = os.path.join(rd,d,"capri_ss.tsv")
            if os.path.exists(p): cf=p; break
        if cf:
            df_c = load_capri(cf)
            top  = df_c[df_c["caprieval_rank"]==1].iloc[0]
            st.markdown("**Rank 1 Properties**")
            for k,lbl in [("score","HADDOCK Score"),("elec","Electrostatics"),
                           ("vdw","Van der Waals"),("desolv","Desolvation"),
                           ("bsa","BSA (Ų)"),("dockq","DOCKQ")]:
                if k in top.index and pd.notna(top[k]):
                    st.markdown(f"<div class='rrow'><span class='rk'>{lbl}</span>"
                                f"<span class='rv'>{top[k]:.3f}</span></div>", unsafe_allow_html=True)
    st.divider()
    bc1,bc2 = st.columns(2)
    with bc1:
        if st.button("Proceed to MD", key="btn_to_md", icon=":material/arrow_forward:"): st.session_state.page="md"; st.rerun()
    with bc2:
        if st.button("New Docking", key="btn_redock", icon=":material/refresh:"):
            st.session_state.update({"docking_status":"idle","docking_result":None,"haddock_score":None}); st.rerun()


def _run_md_ui(complex_pdb, nsteps_val, dt_val, ncores_val, key_suffix=""):
    mm1,mm2,mm3,mm4 = st.columns(4)
    with mm1: met("Force Field",st.session_state.get("_md_ff","amber99sb-ildn"),"")
    with mm2: met("Water Model",st.session_state.get("_md_water","tip3p"),"")
    with mm3: met("Length",st.session_state.get("_md_len","1 ns"),"")
    with mm4: met("CPU Cores",str(ncores_val),"")
    st.markdown("<div class='warn'>GROMACS + gmx_MMPBSA both required in <code>mmpbsa_env</code></div>", unsafe_allow_html=True)
    st.markdown("<div class='info'>📋 pdb2gmx → box(1.0nm) → solvation → ions → EM → NVT → NPT → MD → RMSD/RMSF/Rg → gmx_MMPBSA (auto-detected groups)</div>", unsafe_allow_html=True)
    md_status = st.session_state.get("md_status","idle")
    if md_status == "idle":
        if st.button("Launch MD + MM-PBSA", key=f"btn_md{key_suffix}", icon=":material/rocket_launch:"):
            mdjd = os.path.abspath(f"md_job_{int(time.time())}")
            os.makedirs(mdjd, exist_ok=True)
            shutil.copy(complex_pdb, os.path.join(mdjd,"complex.pdb"))
            sp_full = make_md_script(mdjd,
                                     st.session_state.get("_md_ff","amber99sb-ildn"),
                                     st.session_state.get("_md_water","tip3p"),
                                     nsteps_val, dt_val, ncores=ncores_val)
            lp = os.path.join(mdjd,"md_pipeline.log")
            proc = subprocess.Popen(["bash", sp_full], cwd=mdjd,
                                    stdout=open(lp,"w"), stderr=subprocess.STDOUT)
            open(os.path.join(mdjd,"md.pid"),"w").write(str(proc.pid))
            st.session_state.update({"md_status":"running","md_job_dir":mdjd,
                                      "md_pid":proc.pid,"mmpbsa_result":None})
            st.toast("MD launched", icon=":material/rocket_launch:"); st.rerun()
    elif md_status == "running":
        mdjd = st.session_state.get("md_job_dir","")
        if os.path.exists(os.path.join(mdjd,"PIPELINE_DONE")):
            st.session_state.update({"md_status":"done","mmpbsa_result":parse_mmpbsa(mdjd)})
            st.toast("MD complete", icon=":material/check_circle:"); st.rerun()
        elif not proc_alive(st.session_state.get("md_pid")):
            st.session_state["md_status"]="error: MD process ended unexpectedly"; st.rerun()
        else:
            stage,frac = md_progress(mdjd)
            st.markdown(f"**⟳ MD - {stage} ({int(frac*100)}%)**")
            st.progress(frac)
            st.caption("30 min – several hours depending on length and hardware")
            all_s=["Topology","Box","Solvation","Ions","Energy Min","NVT Equil","NPT Equil","Production MD","Analysis","MM-PBSA"]
            mc = st.columns(5)
            for i,s in enumerate(all_s):
                done_s=frac>=(i+1)/len(all_s)
                with mc[i%5]:
                    css="ps-done" if done_s else ("ps-active" if s==stage else "ps-pending")
                    ic="✓" if done_s else ("⟳" if s==stage else "○")
                    st.markdown(f"<div class='ps {css}'>{ic} {s}</div>", unsafe_allow_html=True)
            lp = os.path.join(mdjd,"md_pipeline.log")
            if os.path.exists(lp):
                with st.expander("📋 Live log"):
                    try: st.code("".join(open(lp).readlines()[-15:]), language="bash")
                    except Exception: pass
            time.sleep(5); st.rerun()
    elif md_status == "done":
        st.success("MD + MM-PBSA complete.")
        c1,c2 = st.columns(2)
        with c1:
            if st.button("View MM-PBSA Results", key=f"btn_to_mmpbsa{key_suffix}", icon=":material/arrow_forward:"):
                st.session_state.page="mmpbsa"; st.rerun()
        with c2:
            if st.button("New MD", key=f"btn_remd{key_suffix}", icon=":material/refresh:"):
                st.session_state.update({"md_status":"idle","mmpbsa_result":None}); st.rerun()
        mdjd = st.session_state.get("md_job_dir","")
        if mdjd:
            st.markdown("### MD Analysis")
            found_any = False
            for xvg, title, xlabel, ylabel in [
                ("rmsd.xvg","RMSD vs Time","Time (ns)","RMSD (nm)"),
                ("rmsf.xvg","RMSF per Residue","Residue","Fluctuation (nm)"),
                ("gyrate.xvg","Radius of Gyration","Time (ns)","Rg (nm)"),
            ]:
                x,y = parse_xvg(os.path.join(mdjd,xvg))
                if x:
                    if xvg == "rmsf.xvg":
                        x, y = break_chain_discontinuities(x, y)
                    found_any = True
                    fig = px.line(x=x,y=y,title=title,
                                  color_discrete_sequence=["#16a34a"])
                    fig.update_layout(xaxis_title=xlabel,yaxis_title=ylabel,**pcfg(),height=350)
                    st.plotly_chart(fig, use_container_width=True)
            if not found_any:
                st.warning("No analysis files found yet.")
    elif "error" in str(md_status):
        st.error(md_status)
        mdjd = st.session_state.get("md_job_dir","")
        lp = os.path.join(mdjd,"md_pipeline.log")
        if os.path.exists(lp):
            with st.expander("Error log"):
                try: st.code("".join(open(lp).readlines()[-25:]), language="bash")
                except Exception: pass
        if st.button("Reset MD", key=f"btn_rst_md{key_suffix}", icon=":material/refresh:"):
            st.session_state.update({"md_status":"idle","mmpbsa_result":None}); st.rerun()


def _show_mmpbsa_results(res, mdjd):
    cfg = pcfg()
    dg=res.get("dg"); dg_std=res.get("dg_std"); vdw=res.get("vdw")
    elec=res.get("elec"); polar=res.get("polar"); nopol=res.get("nopolar")
    if dg is not None:
        if dg<=-10: interp,clr="Very Strong Binder","#16a34a"
        elif dg<=-7: interp,clr="Strong Binder","#ca8a04"
        elif dg<=-5: interp,clr="Moderate Binder","#d97706"
        else: interp,clr="Weak Binder","#dc2626"
    else: interp,clr="N/A","#6b7280"
    r1,r2,r3,r4,r5=st.columns(5)
    with r1: met("ΔG Binding",f"{dg:.2f}{'±'+str(round(dg_std,2)) if dg_std else ''}" if dg else "–","kcal/mol",clr)
    with r2: met("VdW Energy",f"{vdw:.2f}" if vdw else "–","kcal/mol","#2563eb")
    with r3: met("Electrostatics",f"{elec:.2f}" if elec else "–","kcal/mol","#7c3aed")
    with r4: met("Polar Solv.",f"{polar:.2f}" if polar else "–","kcal/mol","#0891b2")
    with r5: met("Affinity",interp,"",clr)
    st.divider()
    if dg is not None:
        fig_g=go.Figure(go.Indicator(mode="gauge+number",value=abs(dg),
            title={"text":"|ΔG| kcal/mol","font":{"color":"#374151","size":13}},
            gauge={"axis":{"range":[0,20]},"bar":{"color":clr},
                   "bgcolor":"#f0fdf4",
                   "steps":[{"range":[0,5],"color":"#fef2f2"},{"range":[5,7],"color":"#fff7ed"},
                             {"range":[7,10],"color":"#f0fdf4"},{"range":[10,20],"color":"#dcfce7"}],
                   "threshold":{"line":{"color":clr,"width":3},"thickness":0.75,"value":abs(dg)}},
            number={"font":{"color":clr,"family":"JetBrains Mono","size":28}}))
        fig_g.update_layout(paper_bgcolor="#ffffff",font_color="#374151",
                            height=280,margin=dict(t=50,b=10,l=20,r=20))
        e_items={k:v for k,v in [("VdW",vdw),("Electrostatics",elec),("Polar Solv.",polar),("Non-Polar",nopol)] if v is not None}
        gc1,gc2=st.columns([2,3])
        with gc1: st.plotly_chart(fig_g, width="stretch", key="gauge_mmpbsa")
        with gc2:
            if e_items:
                fig_e=go.Figure(go.Bar(x=list(e_items.keys()),y=list(e_items.values()),
                                       marker_color=["#16a34a" if v<0 else "#ef4444" for v in e_items.values()],
                                       text=[f"{v:.2f}" for v in e_items.values()],textposition="outside"))
                fig_e.add_hline(y=0,line_color="#94a3b8",line_width=1)
                fig_e.update_layout(**cfg,title="Energy Decomposition (kcal/mol)",
                                    height=280,showlegend=False)
                fig_e.update_layout(margin=dict(t=50,b=30,l=40,r=20))
                st.plotly_chart(fig_e, width="stretch", key="energy_decomp")
    st.divider()
    sec("timeline","MD Trajectory Analysis")
    tt1,tt2,tt3,tt4=st.tabs(["RMSD","RMSF","Radius of Gyration","MM-PBSA Data"])
    with tt1:
        x,y=parse_xvg(os.path.join(mdjd,"rmsd.xvg"))
        if x:
            fig=px.line(x=x,y=y,labels={"x":"Time (ns)","y":"RMSD (nm)"},
                        title="Backbone RMSD vs Time",color_discrete_sequence=["#16a34a"])
            fig.add_hline(y=float(np.mean(y)),line_dash="dot",line_color="#86efac",
                          annotation_text=f"Mean: {np.mean(y):.3f} nm",annotation_font_color="#16a34a")
            fig.update_layout(**cfg,height=420); st.plotly_chart(fig, width="stretch", key="rmsd_plot")
            sc1,sc2,sc3=st.columns(3)
            with sc1: met("Mean RMSD",f"{np.mean(y):.3f}","nm","#16a34a" if np.mean(y)<0.2 else "#d97706")
            with sc2: met("Max RMSD",f"{max(y):.3f}","nm")
            with sc3: met("Stability","Stable ✓" if np.mean(y)<0.2 else "Flexible","",
                          "#16a34a" if np.mean(y)<0.2 else "#d97706")
        else: st.info("RMSD data not available.")
    with tt2:
        x,y=parse_xvg(os.path.join(mdjd,"rmsf.xvg"))
        if x:
            fig=px.bar(x=x,y=y,labels={"x":"Residue","y":"RMSF (nm)"},
                       title="Per-Residue RMSF",color=y,color_continuous_scale="RdYlGn_r")
            fig.update_layout(**cfg,height=420); st.plotly_chart(fig, width="stretch", key="rmsf_plot")
            if y:
                mx_i=y.index(max(y))
                st.markdown(f"<div class='info'>Most flexible residue: <b>{int(x[mx_i])}</b> "
                            f"(RMSF={max(y):.3f} nm) · Mean: <b>{np.mean(y):.3f} nm</b></div>",
                            unsafe_allow_html=True)
        else: st.info("RMSF data not available.")
    with tt3:
        x,y=parse_xvg(os.path.join(mdjd,"gyrate.xvg"))
        if x:
            fig=px.line(x=x,y=y,labels={"x":"Time (ns)","y":"Rg (nm)"},
                        title="Radius of Gyration",color_discrete_sequence=["#2563eb"])
            fig.update_layout(**cfg,height=420); st.plotly_chart(fig, width="stretch", key="rg_plot")
            st.markdown(f"<div class='info'>Mean Rg: <b>{np.mean(y):.3f} nm</b></div>", unsafe_allow_html=True)
        else: st.info("Rg data not available.")
    with tt4:
        if res.get("df") is not None: st.dataframe(res["df"], width="stretch")
        elif res.get("raw"):
            with st.expander("Raw MM-PBSA output"): st.code(res["raw"][:4000], language="text")
        else: st.info("MM-PBSA data not available.")
    st.divider()
    sec("download","Download All Outputs")
    dl_files=[
        ("docked_complex.pdb",st.session_state.get("docking_result",""),"Docked Complex","chemical/x-pdb"),
        ("FINAL_MMPBSA.dat",os.path.join(mdjd,"FINAL_MMPBSA.dat"),"MM-PBSA Results","text/plain"),
        ("FINAL_MMPBSA.csv",os.path.join(mdjd,"FINAL_MMPBSA.csv"),"MM-PBSA CSV","text/csv"),
        ("rmsd.xvg",os.path.join(mdjd,"rmsd.xvg"),"RMSD Data","text/plain"),
        ("rmsf.xvg",os.path.join(mdjd,"rmsf.xvg"),"RMSF Data","text/plain"),
        ("gyrate.xvg",os.path.join(mdjd,"gyrate.xvg"),"Gyration Data","text/plain"),
        ("md.gro",os.path.join(mdjd,"md.gro"),"Final Structure","text/plain"),
        ("topol.top",os.path.join(mdjd,"topol.top"),"Topology","text/plain"),
    ]
    dc=st.columns(4)
    for i,(fname,fpath,label,mime) in enumerate(dl_files):
        with dc[i%4]:
            if fpath and os.path.exists(str(fpath)):
                with open(fpath,"rb") as f:
                    st.download_button(f"{label}",f.read(),fname,mime,key=f"dl_{fname}")
            else: st.button(f"{label}",disabled=True,key=f"dl_na_{fname}")
    jd=st.session_state.get("job_dir","")
    for d in ["7_caprieval","8_caprieval"]:
        cf=os.path.join(jd,"run1",d,"capri_ss.tsv")
        if os.path.exists(cf):
            with open(cf,"rb") as f:
                st.download_button("CAPRI Table (TSV)",f.read(),"capri_results.tsv",
                                   "text/tab-separated-values",key="dl_capri")
            break
    if st.session_state.get("prediction_df") is not None:
        st.download_button("Prediction Table (CSV)",
                           st.session_state.prediction_df.to_csv(index=False),
                           "predictions.csv","text/csv",key="dl_pred_final")
    st.divider()
    st.markdown("<div class='info'><b>MM-PBSA Reference:</b> ΔG ≤ -10 → Very strong (nM) · "
                "-7 to -10 → Strong · -5 to -7 → Moderate (µM) · > -5 → Weak</div>", unsafe_allow_html=True)


#  PAGE ROUTING

if page == "predict":
    sec("science","Peptide Bioactivity Prediction")
    f_strict, min_len, max_len, ch_range, pi_range, ar_range, hy_range = render_peptide_filters()

    @st.cache_resource(show_spinner="Loading prediction model...")
    def _pipeline():
        from pipeline import predict_peptides, evaluate_candidate
        return predict_peptides, evaluate_candidate
    try:
        predict_peptides, evaluate_candidate = _pipeline()
    except Exception as e:
        st.error(f"Pipeline load failed: {e}"); st.stop()

    pc1,pc2,pc3 = st.columns([2,2,2])
    with pc1:
        task = st.selectbox("Bioactivity Task",
            ["ABP","ACP","AFP","AHP","AIP","APP","AVP"],
            help="ABP=Antibacterial · ACP=Anticancer · AFP=Antifungal · "
                 "AHP=AntiHIV · AIP=Anti-inflammatory · APP=Antiparasitic · AVP=Antiviral")
    with pc2:
        threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    with pc3:
        uploaded = st.file_uploader("Upload FASTA", type=["fasta","fa","txt"])

    if st.button("Run Prediction", key="btn_predict", icon=":material/play_arrow:"):
        if not uploaded:
            st.warning("Upload a FASTA file first.")
        else:
            content = uploaded.read().decode("utf-8").strip()
            if not content.startswith(">"):
                st.error("Invalid FASTA - must start with '>'")
            else:
                with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".fasta") as tmp:
                    tmp.write(content); fp = tmp.name
                with st.spinner("Running PLM prediction..."):
                    try:
                        df = predict_peptides(fp, task)
                        sc = f"{task}_score"
                        if sc in df.columns: df = df[df[sc] >= threshold]
                        st.session_state.prediction_df = df
                        st.session_state.prediction_task = task
                        st.toast(f"{len(df)} peptides passed", icon=":material/check_circle:")
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

    if st.session_state.prediction_df is not None:
        df = st.session_state.prediction_df
        task = st.session_state.get("prediction_task", task)
        sc = f"{task}_score"
        st.success(f"**{len(df)} peptides** passed threshold {threshold}")

        sec("table_view","Full Prediction Table")
        st.markdown("<div class='info'>Sort by clicking headers · Search below · Download CSV</div>", unsafe_allow_html=True)
        if "drug_score_%" not in df.columns and len(df) > 0:
            drug_scores = []
            for seq in df.get("sequence", []):
                try:
                    _, score, *_ = drug_likeness(str(seq)); drug_scores.append(round(score,1))
                except Exception: drug_scores.append(0.0)
            df = df.copy(); df["drug_score_%"] = drug_scores
            st.session_state.prediction_df = df

        search = st.text_input("Search sequences", placeholder="Filter by sequence...")
        df_show = df[df["sequence"].str.contains(search,case=False,na=False)] if search else df
        st.dataframe(df_show, width="stretch", height=min(400, max(200, len(df_show)*35+50)))
        st.caption(f"Showing {len(df_show)} of {len(df)} peptides")

        
        def _on_peptide_select():
            new_seq = st.session_state["peptide_select_box"]
            st.session_state.selected_seq = new_seq
            st.session_state["eval_seq_input"] = new_seq
            st.session_state.eval_result = None
            st.session_state.valid_seq = ""

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button("Download CSV", df.to_csv(index=False),
                               "predictions.csv", "text/csv", key="dl_pred_csv")
        with dl2:
            if len(df) > 0:
                seq_options = df["sequence"].tolist()
                default_idx = (seq_options.index(st.session_state.selected_seq)
                               if st.session_state.selected_seq in seq_options else 0)
                seq = st.selectbox("Select peptide for analysis", seq_options,
                                    index=default_idx,
                                    key="peptide_select_box",
                                    on_change=_on_peptide_select)
                st.session_state.selected_seq = seq
        st.divider()
        

        
        
        sec("bar_chart","Confidence Score Chart")
        if sc in df.columns and len(df) > 0:
            n_show = st.slider("Top N", 1, max(1,min(100,len(df))), min(20,max(1,len(df))))
            df_plot = df.nlargest(n_show,sc) if sc in df.columns else df.head(n_show)
            df_plot = df_plot.copy()
            df_plot["label"] = [f"#{i+1}" for i in range(len(df_plot))]
            fig = px.bar(df_plot, x="label", y=sc, color=sc,
                         color_continuous_scale="Greens",
                         title=f"{task} Confidence Scores - Top {n_show}",
                         labels={"label":"Peptide", sc:f"{task} Score"}, text=sc,
                         custom_data=["sequence"])
            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside', marker_line_width=0,
                               hovertemplate="Sequence: %{customdata[0]}<br>"+f"{task} Score: "+"%{y:.4f}<extra></extra>")
            fig.update_layout(**pcfg(), height=500, xaxis_tickangle=0,
                              showlegend=False, xaxis=dict(tickfont=dict(size=11)))
            st.plotly_chart(fig, width="stretch", key="bar_scores")
        st.divider()

        sec("medication","Drug-likeness Evaluation (All Peptides)")
        if "drug_score_%" in df.columns:
            df_drug = df[["sequence","drug_score_%",sc] if sc in df.columns
                         else ["sequence","drug_score_%"]].copy()
            df_drug = df_drug.sort_values("drug_score_%", ascending=False)
            def color_drug(val):
                if val >= 80: return "background-color:#f0fdf4;color:#15803d;font-weight:600"
                elif val >= 60: return "background-color:#fffbeb;color:#92400e;"
                else: return "background-color:#fef2f2;color:#dc2626;"
            st.dataframe(
                df_drug.style.applymap(color_drug, subset=["drug_score_%"]).format(
                    {"drug_score_%":"{:.1f}%", sc:"{:.4f}" if sc in df_drug.columns else None}),
                width="stretch", height=min(350, len(df_drug)*35+50))
            fig_d = px.histogram(df_drug, x="drug_score_%", nbins=20,
                                 color_discrete_sequence=["#16a34a"],
                                 title="Drug-likeness Score Distribution",
                                 labels={"drug_score_%":"Drug-likeness Score (%)","count":"Count"})
            fig_d.add_vline(x=70, line_dash="dot", line_color="#ef4444",
                            annotation_text="70% threshold", annotation_font_color="#ef4444")
            fig_d.update_layout(**pcfg(), height=350, bargap=0.1)
            st.plotly_chart(fig_d, width="stretch", key="drug_dist")
        st.divider()

        sec("scatter_plot","Peptide Chemical Space - 3D PCA Map")
        st.markdown("<div class='info'>Color=score · Size=length · Drag to rotate · Scroll to zoom</div>", unsafe_allow_html=True)
        if sc in df.columns and len(df) >= 3:
            _fig = embedding_3d_plot(df, sc, task)
            if _fig is not None:
                st.plotly_chart(_fig, width="stretch", key="pca_3d")
        else:
            st.info("Need at least 3 peptides for 3D map." f"(Currently viewing task: {task} - re-run prediction if you changed " f"the Bioactivity Task dropdown after your last run.)")
        st.divider()

        if st.session_state.selected_seq:
            sec("biotech",f"Detailed Evaluation - {st.session_state.selected_seq[:30]}"
                     f"{'...' if len(st.session_state.selected_seq)>30 else ''}")
            seq_eval = st.text_input("Sequence", value=st.session_state.selected_seq, key="eval_seq_input")
            if st.button("Run Full Evaluation", key="btn_eval", icon=":material/biotech:"):
                with st.spinner("Evaluating..."):
                    try:
                        r = evaluate_candidate(seq_eval.strip())
                        st.session_state.eval_result = r
                        st.session_state.valid_seq = seq_eval.strip()
                        st.toast("Evaluation complete", icon=":material/biotech:")
                    except Exception as e:
                        st.error(f"Evaluation failed: {e}")

            if st.session_state.eval_result:
                r = st.session_state.eval_result
                ec1,ec2,ec3,ec4,ec5 = st.columns(5)
                for col,lbl,val,unit,clr in [
                    (ec1,"Length",r.get("length","–"),"aa","#15803d"),
                    (ec2,"Net Charge",r.get("charge","–"),"","#2563eb"),
                    (ec3,"pI",r.get("pI","–"),"","#7c3aed"),
                    (ec4,"Hydrophobicity",round(r.get("hydrophobicity",0),3),"","#db2777"),
                    (ec5,"Aromatic",round(r.get("aromatic",0),3),"","#d97706"),
                ]:
                    with col: met(lbl,val,unit,clr)
                st.write("")
                rules,drug_score,mw,charge,hbd,hba,aromatic,hydro = drug_likeness(seq_eval.strip())
                dl_col, radar_col = st.columns([2,3])
                with dl_col:
                    clr = "#16a34a" if drug_score>=80 else "#f59e0b" if drug_score>=60 else "#ef4444"
                    st.markdown(f"**Drug-likeness: {drug_score:.1f}%**")
                    st.markdown(f"<div style='font-size:2rem;font-weight:800;color:{clr};"
                                f"font-family:JetBrains Mono;'>{drug_score:.1f}%</div>",
                                unsafe_allow_html=True)
                    for name,val_str,ok,rule,_ in rules:
                        css="drug-pass" if ok else "drug-fail"
                        ic="✓" if ok else "✗"; ic_clr="#16a34a" if ok else "#ef4444"
                        st.markdown(f"<div class='drug-card {css}'>"
                                    f"<span style='color:#374151;font-size:0.83rem;'>{name}</span>"
                                    f"<span style='display:flex;align-items:center;gap:0.5rem;'>"
                                    f"<span style='font-family:JetBrains Mono;font-size:0.83rem;color:#374151;'>{val_str}</span>"
                                    f"<span style='color:#9ca3af;font-size:0.75rem;'>({rule})</span>"
                                    f"<span style='font-weight:700;color:{ic_clr};'>{ic}</span>"
                                    f"</span></div>", unsafe_allow_html=True)
                with radar_col:
                    fv={"Length":(r.get("length",0),min_len,max_len),
                        "Net Charge":(r.get("charge",0),ch_range[0],ch_range[1]),
                        "pI":(r.get("pI",7),pi_range[0],pi_range[1]),
                        "Aromatic":(r.get("aromatic",0),ar_range[0],ar_range[1]),
                        "Hydrophobicity":(r.get("hydrophobicity",0),hy_range[0],hy_range[1])}
                    checks={k:lo<=v<=hi for k,(v,lo,hi) in fv.items()}
                    cats=list(fv.keys())
                    rvals=[min(max((fv[k][0]-fv[k][1])/(fv[k][2]-fv[k][1]+1e-8),0),1) for k in cats]
                    fig_r=go.Figure(go.Scatterpolar(r=rvals+[rvals[0]],theta=cats+[cats[0]],
                                                    fill="toself",line_color="#16a34a",
                                                    fillcolor="rgba(22,163,74,0.15)",name="Candidate"))
                    fig_r.add_trace(go.Scatterpolar(r=[0.5]*len(cats)+[0.5],theta=cats+[cats[0]],
                                                    line=dict(color="#94a3b8",dash="dot"),name="Reference",fill=None))
                    fig_r.update_layout(polar=dict(radialaxis=dict(visible=True,range=[0,1],gridcolor="#e2e8f0"),
                                                   angularaxis=dict(gridcolor="#e2e8f0"),bgcolor="#f8fafc"),
                                        paper_bgcolor="#ffffff",font_color="#374151",showlegend=True,
                                        height=350,margin=dict(t=40,b=40,l=60,r=60),title="Physicochemical Profile")
                    st.plotly_chart(fig_r, width="stretch", key="radar_eval")
                failed_f=[k for k,ok in checks.items() if not ok]
                if not failed_f: st.success("All physicochemical filters passed.")
                elif st.session_state.get("_f_strict"): st.error(f"Strict mode: failed - {', '.join(failed_f)}")
                else: st.warning(f"Advisory: {', '.join(failed_f)} outside range - can proceed")
                with st.expander("Full evaluation JSON"): st.json(r)

            if st.button("Go to Structure Prediction", key="btn_to_struct", icon=":material/arrow_forward:"):
                st.session_state.page="structure"; st.rerun()


elif page == "structure":
    sec("hub","Structure Prediction & Receptor Preparation")
    st.markdown("### Peptide Structure")
    pt1, pt2 = st.tabs(["ESMFold (Predict From Sequence)","Upload Your Own PDB"])
    with pt1:
        seq_s = st.text_input("Sequence to fold",
                              value=st.session_state.valid_seq or st.session_state.selected_seq,
                              placeholder="Sequence auto-populated from prediction...")
        if st.button("Run ESMFold", key="btn_fold", icon=":material/play_arrow:"):
            if not seq_s.strip(): st.warning("Enter a sequence.")
            else:
                with st.status("Running ESMFold...", expanded=True) as fs:
                    try:
                        from pipeline import fold_peptide as _fp
                        pp = _fp(seq_s.strip())
                        st.session_state.pdb_path = pp; st.session_state.pdb_source = "esmfold"
                        fs.update(label="ESMFold complete!", state="complete", expanded=False)
                        st.toast("Structure ready", icon=":material/check_circle:")
                    except Exception as e:
                        fs.update(label="ESMFold failed", state="error")
                        st.error(f"ESMFold error: {e}")
    with pt2:
        st.markdown("<div class='info'>Upload peptide PDB directly (skip ESMFold)</div>", unsafe_allow_html=True)
        up_pep = st.file_uploader("Upload Peptide PDB", type=["pdb"], key="up_pep")
        if up_pep:
            sd = tempfile.mkdtemp(); sp = os.path.join(sd,"custom_peptide.pdb")
            open(sp,"wb").write(up_pep.read())
            st.session_state.pdb_path = sp; st.session_state.pdb_source = "upload"
            st.success(f"**{up_pep.name}** loaded ({os.path.getsize(sp)//1024 or 1} KB)")

    if st.session_state.pdb_path and os.path.exists(st.session_state.pdb_path):
        src = "ESMFold" if st.session_state.pdb_source=="esmfold" else "Custom Upload"
        st.markdown(f"**Peptide 3D Structure** *(source: {src})*")
        cv1, cv2 = st.columns([4,1])
        with cv1: show_3d(st.session_state.pdb_path, height=380)
        with cv2:
            st.write("")
            with open(st.session_state.pdb_path,"rb") as f:
                st.download_button("Download PDB", f, "peptide.pdb", mime="chemical/x-pdb")

    st.divider()
    st.markdown("### Receptor Preparation")
    rt1, rt2 = st.tabs(["Download From RCSB PDB","Upload Your Own Receptor PDB"])
    with rt1:
        rc1, rc2 = st.columns([3,1])
        with rc1:
            pdb_id = st.text_input("PDB ID",
                placeholder="e.g. 1YCR (MDM2), 1G5M (BCL-2), 2GS7 (EGFR apo), 1E31 (Survivin)...")
        with rc2:
            st.write("")
            if st.button("Download", key="btn_rec", icon=":material/download:"):
                if pdb_id.strip():
                    with st.spinner(f"Downloading {pdb_id.upper()}..."):
                        try:
                            from pipeline import download_receptor as _dr
                            rp = _dr(pdb_id.strip().upper())
                            cleaned = rp.replace(".pdb","_clean.pdb")
                            n_atoms, chains_kept, nonstd = clean_pdb(rp, cleaned)
                            st.session_state.receptor_path = cleaned
                            st.toast(f"{pdb_id.upper()} downloaded and cleaned", icon=":material/check_circle:")
                            st.success(f"Auto-cleaned: {n_atoms} atoms · chains {chains_kept}"
                                       + (f" · non-std removed: {nonstd}" if nonstd else ""))
                        except Exception as e:
                            st.error(f"Download failed: {e}")
    with rt2:
        st.markdown("<div class='info'>Ligands, DNA, and water are auto-removed on upload.</div>", unsafe_allow_html=True)
        up_rec = st.file_uploader("Upload Receptor PDB", type=["pdb"], key="up_rec")
        if up_rec:
            sd2 = tempfile.mkdtemp(); rp2 = os.path.join(sd2,"uploaded_receptor.pdb")
            open(rp2,"wb").write(up_rec.read())
            cleaned2 = os.path.join(sd2,"receptor_clean.pdb")
            n_atoms2, chains2, nonstd2 = clean_pdb(rp2, cleaned2)
            st.session_state.receptor_path = cleaned2
            st.toast("Receptor loaded and cleaned", icon=":material/check_circle:")
            st.success(f"**{up_rec.name}** cleaned: {n_atoms2} atoms · chains {chains2}"
                       + (f" · non-std removed: {nonstd2}" if nonstd2 else ""))

    if st.session_state.receptor_path and os.path.exists(st.session_state.receptor_path):
        chains = set(); nonprot = set(); dna_res={'DA','DC','DG','DT'}
        for l in open(st.session_state.receptor_path):
            if l.startswith(("ATOM","HETATM")):
                chains.add(l[21])
                resn = l[17:20].strip()
                if resn in dna_res: nonprot.add(resn)
        chains_list = sorted(chains)
        st.info(f"Chains in receptor: **{', '.join(chains_list)}**")
        if len(chains_list) > 1:
            st.markdown("### Select Chains to Keep for Docking")
            default_chain = ["A"] if "A" in chains_list else ([chains_list[0]] if chains_list else [])
            selected_chains = st.multiselect("Choose receptor chains", options=chains_list, default=default_chain)
            if st.button("Apply Chain Selection", icon=":material/filter_alt:"):
                if not selected_chains:
                    st.warning("Select at least one chain!")
                else:
                    cleaned_path = os.path.join(tempfile.mkdtemp(),"receptor_clean.pdb")
                    n, c, ns = clean_pdb(st.session_state.receptor_path, cleaned_path,
                                         keep_chains=set(selected_chains))
                    st.session_state.receptor_path = cleaned_path
                    st.success(f"Kept chains: {', '.join(selected_chains)} · {n} atoms")
                    st.rerun()
        if nonprot:
            st.warning(f"DNA residues still present: {nonprot}")
        rv1, rv2 = st.columns([4,1])
        with rv1: show_3d(st.session_state.receptor_path, height=320)
        with rv2:
            st.write("")
            with open(st.session_state.receptor_path,"rb") as f:
                st.download_button("Download PDB", f, "receptor.pdb", mime="chemical/x-pdb")

    st.divider()
    if st.button("Proceed to Docking", icon=":material/arrow_forward:"):
        st.session_state.page="docking"; st.rerun()


elif page == "assistant":
    sec("smart_toy", "Ask the PTPLab Assistant (Local LLM via Ollama)")
    st.markdown("<div class='info'>Powered by a local Ollama model - free, offline, "
                "no API key. Falls back to the built-in FAQ if Ollama isn't running.</div>",
                unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    SUGGESTED_QUESTIONS = [
        "What does the HADDOCK score mean?",
        "Is my docking result good?",
        "What's my best candidate peptide?",
        "How do I interpret MM-PBSA ΔG?",
        "What force field should I use for MD?",
        "Should I trust my MD run?",
    ]

    clicked_q = None
    if not st.session_state.chat_history:
        st.markdown("**Try asking:**")
        sq_cols = st.columns(3)
        for i, q in enumerate(SUGGESTED_QUESTIONS):
            with sq_cols[i % 3]:
                if st.button(q, key=f"sugg_{i}", use_container_width=True):
                    clicked_q = q

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    typed_prompt = st.chat_input("Ask anything about the pipeline, settings, or results...")
    prompt = clicked_q or typed_prompt

    if prompt:
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.spinner("Thinking..."):
            answer = ask_ollama(prompt, st.session_state.chat_history)
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("Clear conversation", icon=":material/delete:"):
            st.session_state.chat_history = []
            st.rerun()


elif page == "docking":
    sec("link","Molecular Docking (HADDOCK3)")
    render_docking_settings()
    dock_tab1, dock_tab2, dock_tab3 = st.tabs([
        "Use Pipeline (ESMFold + Receptor)",
        "Upload Docked Complex Directly",
        "Load Existing HADDOCK3 Run",
    ])
    with dock_tab1:
        ready = (st.session_state.get("pdb_path") and
                 os.path.exists(st.session_state.get("pdb_path","")) and
                 st.session_state.get("receptor_path") and
                 os.path.exists(st.session_state.get("receptor_path","")))
        if not ready:
            st.markdown("<div class='warn'>Complete Structure & Receptor (step 2) first, or upload a docked complex in Tab 2.</div>", unsafe_allow_html=True)
            if st.button("Go to Structure", icon=":material/arrow_back:"): st.session_state.page="structure"; st.rerun()
        else:
            dm1,dm2,dm3,dm4 = st.columns(4)
            with dm1: met("Peptide","ESMFold" if st.session_state.pdb_source=="esmfold" else "Custom","")
            with dm2: met("Sampling",str(st.session_state.get("_sampling",200)),"poses")
            with dm3: met("Top Models",str(st.session_state.get("_top_models",4)),"for refinement")
            with dm4: met("CPU Cores",str(st.session_state.get("_ncores",16)),"")
            status = st.session_state.get("docking_status","idle")
            if status == "idle":
                st.markdown("<div class='info'>Ready. top_models=4 ≈ 7 min for 1 peptide + receptor.</div>", unsafe_allow_html=True)
                if st.button("Run HADDOCK3 Docking", key="btn_dock", icon=":material/rocket_launch:"):
                    jd = os.path.abspath(f"haddock_job_{int(time.time())}")
                    os.makedirs(jd, exist_ok=True)
                    fix_chain(st.session_state.receptor_path, "A", os.path.join(jd,"receptor.pdb"))
                    fix_chain(st.session_state.pdb_path, "B", os.path.join(jd,"peptide.pdb"))
                    clean_pdb(os.path.join(jd,"receptor.pdb"), os.path.join(jd,"receptor_clean.pdb"), keep_chains={"A"})
                    docking_toml(jd, st.session_state.get("_sampling",200),
                                 st.session_state.get("_ncores",16), st.session_state.get("_top_models",4))
                    lp = os.path.join(jd,"haddock3.log")
                    proc = subprocess.Popen(
                        ["conda","run","-n","haddock_env","haddock3","docking.toml"],
                        cwd=jd, stdout=open(lp,"w"), stderr=subprocess.STDOUT)
                    open(os.path.join(jd,"haddock3.pid"),"w").write(str(proc.pid))
                    st.session_state.update({"job_dir":jd,"docking_status":"running",
                                              "docking_result":None,"haddock_score":None,"docking_pid":proc.pid})
                    st.toast("Docking launched", icon=":material/rocket_launch:"); st.rerun()
            elif status == "running":
                jd = st.session_state.get("job_dir",""); rd = os.path.join(jd,"run1")
                done_flag = capri_is_done(rd)
                fail_flag = any(os.path.exists(os.path.join(rd,s,"FAILED"))
                                for s in ["1_rigidbody","5_flexref","6_emref"])
                alive = proc_alive(st.session_state.get("docking_pid"))
                if done_flag:
                    _top = get_top_pdb(rd); _perm = os.path.join(jd,"docked_complex.pdb")
                    if _top and os.path.exists(_top) and not os.path.exists(_perm): shutil.copy(_top, _perm)
                    st.session_state.update({"docking_result":_top,"haddock_score":get_score(rd),"docking_status":"done"})
                    st.toast("Docking complete", icon=":material/check_circle:"); st.rerun()
                elif fail_flag:
                    st.session_state["docking_status"]="error: CNS job failed"; st.rerun()
                elif not alive and os.path.isdir(rd):
                    st.session_state["docking_status"]="error: process ended unexpectedly"; st.rerun()
                else:
                    done_steps, frac = haddock_progress(rd) if os.path.isdir(rd) else ([],0.0)
                    slabels={"0_topoaa":"Topology","1_rigidbody":"Rigid Body","2_caprieval":"CAPRI Eval",
                             "3_clustfcc":"Clustering","4_seletopclusts":"Top Clusters",
                             "5_flexref":"Flex Refinement","6_emref":"EM Refinement","7_caprieval":"Final Eval"}
                    st.markdown(f"**⟳ Docking - {int(frac*100)}% complete**")
                    st.progress(frac)
                    sc_cols = st.columns(4)
                    for i,s in enumerate(slabels.keys()):
                        with sc_cols[i%4]:
                            if s in done_steps: css,ic="ps-done","✓"
                            elif i==len(done_steps): css,ic="ps-active","⟳"
                            else: css,ic="ps-pending","○"
                            st.markdown(f"<div class='ps {css}'>{ic} {slabels[s]}</div>", unsafe_allow_html=True)
                    lp = os.path.join(jd,"haddock3.log")
                    if os.path.exists(lp):
                        with st.expander("📋 Live log"):
                            try: st.code("".join(open(lp).readlines()[-12:]), language="text")
                            except Exception: pass
                    time.sleep(4); st.rerun()
            elif status == "done":
                _show_docking_results()
            elif "error" in str(status):
                st.error(status)
                jd = st.session_state.get("job_dir",""); rd = os.path.join(jd,"run1")
                for s in ["1_rigidbody","5_flexref","6_emref"]:
                    sd_ = os.path.join(rd,s)
                    if os.path.isdir(sd_):
                        errs=[f for f in os.listdir(sd_) if f.endswith(".cnserr.gz")]
                        if errs:
                            with st.expander(f"CNS error - {errs[0]}"):
                                try: st.code(gzip.open(os.path.join(sd_,errs[0]),"rt").read()[-3000:])
                                except Exception: pass
                            break
                if st.button("Reset", key="btn_rst_d", icon=":material/refresh:"):
                    st.session_state.update({"docking_status":"idle","docking_result":None,"haddock_score":None}); st.rerun()

    with dock_tab2:
        st.markdown("<div class='info'>Already have a docked complex PDB? Upload it here and skip straight to MD.</div>", unsafe_allow_html=True)
        up_complex = st.file_uploader("Upload Docked Complex PDB", type=["pdb"], key="up_docked_complex")
        if up_complex:
            sd3 = tempfile.mkdtemp()
            cp3 = os.path.join(sd3, "docked_complex.pdb")
            open(cp3,"wb").write(up_complex.read())
            st.session_state.update({
                "docking_result": cp3, "docking_status": "done",
                "job_dir": sd3, "haddock_score": None,
            })
            st.success(f"**{up_complex.name}** loaded as docked complex ({os.path.getsize(cp3)//1024 or 1} KB)")
            st.markdown("**Preview:**")
            show_3d(cp3, height=350)
            if st.button("Proceed to MD", key="btn_complex_to_md", icon=":material/arrow_forward:"):
                st.session_state.page = "md"; st.rerun()

    with dock_tab3:
        auto = find_runs(".")
        if auto:
            opts = auto + ["Enter path manually..."]
            sel = st.selectbox("Auto-detected runs", opts)
            er = st.text_input("Manual path", placeholder="/path/to/run1") if sel=="Enter path manually..." else sel
        else:
            er = st.text_input("run1 path", placeholder="haddock_job_xxx/run1")
        if st.button("Load Run", key="btn_load", icon=":material/folder_open:"):
            rp_abs = os.path.abspath(er) if er else ""
            if os.path.isdir(rp_abs):
                res = get_top_pdb(rp_abs); sc_ = get_score(rp_abs)
                if res:
                    jd_load = os.path.dirname(rp_abs)
                    st.session_state.update({"docking_result":res,"haddock_score":sc_,
                                              "docking_status":"done","job_dir":jd_load})
                    perm = os.path.join(jd_load,"docked_complex.pdb")
                    if not os.path.exists(perm): shutil.copy(res, perm)
                    st.toast("Loaded", icon=":material/check_circle:"); st.rerun()
                else: st.error("No top model found in that run directory.")
            else: st.error("Directory not found.")


elif page == "md":
    sec("monitoring","Molecular Dynamics Simulation (GROMACS)")
    render_md_settings()
    ns_map={"1 ns":500000,"5 ns":2500000,"10 ns":5000000,"50 ns":25000000,"100 ns":50000000}
    nsteps_val = ns_map.get(st.session_state.get("_md_len","1 ns"),500000)
    dt_val = float(st.session_state.get("_md_dt","0.002 ps").split()[0])
    ncores_val = st.session_state.get("_ncores", 8)
    md_tab1, md_tab2 = st.tabs([
        "Use Docked Complex (Pipeline or Upload)",
        "Upload Complex PDB Directly (Skip Docking)",
    ])
    def _resolve_complex():
        rp = st.session_state.get("docking_result")
        if rp and os.path.exists(str(rp)): return rp
        jd_fb = st.session_state.get("job_dir") or ""
        perm = os.path.join(jd_fb,"docked_complex.pdb") if jd_fb else ""
        if perm and os.path.exists(perm):
            st.session_state["docking_result"] = perm
            return perm
        return None
    with md_tab1:
        rp = _resolve_complex()
        if not rp:
            st.markdown("<div class='warn'>No docked complex found. Complete docking (step 3) or upload a complex in Tab 2.</div>", unsafe_allow_html=True)
            if st.button("Go to Docking", icon=":material/arrow_back:"): st.session_state.page="docking"; st.rerun()
        else:
            st.success(f" Complex ready: `{os.path.basename(rp)}`")
            _run_md_ui(rp, nsteps_val, dt_val, ncores_val)
    with md_tab2:
        st.markdown("<div class='info'>Already have a docked complex PDB? Upload it here to run MD directly without going through HADDOCK3.</div>", unsafe_allow_html=True)
        up_md_complex = st.file_uploader("Upload Complex PDB for MD", type=["pdb"], key="up_md_complex")
        if up_md_complex:
            sd4 = tempfile.mkdtemp()
            cp4 = os.path.join(sd4,"complex_for_md.pdb")
            open(cp4,"wb").write(up_md_complex.read())
            st.success(f"**{up_md_complex.name}** loaded ({os.path.getsize(cp4)//1024 or 1} KB)")
            show_3d(cp4, height=300)
            st.session_state.update({
                "docking_result": cp4, "docking_status": "done", "job_dir": sd4,
            })
            _run_md_ui(cp4, nsteps_val, dt_val, ncores_val, key_suffix="_upload")


elif page == "mmpbsa":
    sec("bolt","MM-PBSA Binding Free Energy Results")
    pbsa_tab1, pbsa_tab2 = st.tabs([
        "Use Pipeline MD Output",
        "Upload MD Output Files Directly",
    ])
    with pbsa_tab1:
        if st.session_state.get("md_status") != "done":
            st.markdown("<div class='warn'>MD must complete first (step 4), or upload your MD output files in Tab 2.</div>", unsafe_allow_html=True)
            if st.button("Go to MD", icon=":material/arrow_back:"): st.session_state.page="md"; st.rerun()
        else:
            mdjd = st.session_state.get("md_job_dir","")
            res  = st.session_state.get("mmpbsa_result") or parse_mmpbsa(mdjd)
            _show_mmpbsa_results(res, mdjd)
    with pbsa_tab2:
        st.markdown("<div class='info'>Already ran GROMACS + gmx_MMPBSA externally? Upload your output files here to visualize results directly.</div>", unsafe_allow_html=True)
        col_u1, col_u2 = st.columns(2)
        with col_u1:
            up_dat  = st.file_uploader("FINAL_MMPBSA.dat",  type=["dat","txt"], key="up_mmpbsa_dat")
            up_csv  = st.file_uploader("FINAL_MMPBSA.csv",  type=["csv"],       key="up_mmpbsa_csv")
        with col_u2:
            up_rmsd = st.file_uploader("rmsd.xvg",          type=["xvg","txt"], key="up_rmsd")
            up_rmsf = st.file_uploader("rmsf.xvg",          type=["xvg","txt"], key="up_rmsf")
            up_gyr  = st.file_uploader("gyrate.xvg",        type=["xvg","txt"], key="up_gyrate")
        if any([up_dat, up_csv, up_rmsd, up_rmsf, up_gyr]):
            manual_dir = st.session_state.get("manual_md_dir") or tempfile.mkdtemp()
            st.session_state["manual_md_dir"] = manual_dir
            def _save(upfile, fname):
                if upfile:
                    p = os.path.join(manual_dir, fname)
                    open(p,"wb").write(upfile.read())
            _save(up_dat,  "FINAL_MMPBSA.dat")
            _save(up_csv,  "FINAL_MMPBSA.csv")
            _save(up_rmsd, "rmsd.xvg")
            _save(up_rmsf, "rmsf.xvg")
            _save(up_gyr,  "gyrate.xvg")
            res_manual = parse_mmpbsa(manual_dir)
            if st.button("Analyze Uploaded Files", key="btn_analyze_upload", icon=":material/insights:"):
                st.session_state.update({
                    "mmpbsa_result": res_manual, "md_status": "done", "md_job_dir": manual_dir,
                })
                st.toast("Files loaded", icon=":material/check_circle:"); st.rerun()




elif page == "about":
    sec("account_tree","Pipeline Overview")

    st.markdown("""
1. **Data mining** - plant-based peptide sequences are extracted from public repositories:
   PlantPepDB, PTPAMP, DRAMP, and MFPPDB.
2. **Feature extraction** - Protein Language Models (PLMs) such as ProtT5 and ProtAlbert generate
   meaningful sequence embeddings.
3. **Classification** - Machine learning and deep learning models classify peptides across
   **7 bioactive classes**: Antibacterial, Antifungal, Anticancer, Antiparasitic,
   Anti-inflammatory, Anti-HIV, and Antiviral - surfacing the most promising bioactive candidates.
4. **Structure prediction** - ESMFold predicts the 3D structure (PDB) of the top candidate peptide.
5. **Receptor selection & docking** - the user selects a suitable receptor, downloads it, and runs
   protein-peptide docking via **HADDOCK3**, producing a HADDOCK score.
6. **Molecular Dynamics** - the docked complex is simulated in **GROMACS** (solvation, ions, EM, NVT,
   NPT, production MD) to assess structural stability (RMSD, RMSF, Rg).
7. **Binding free energy** - **MM-PBSA** (via gmx_MMPBSA) computes the final binding free energy,
   completing a full end-to-end bioactive peptide discovery-to-validation pipeline.
""")

    st.write("")
    wf1,wf2,wf3,wf4,wf5 = st.columns(5)
    for col,ic,lbl in zip([wf1,wf2,wf3,wf4,wf5],
        ["science","hub","link","monitoring","bolt"],
        ["Prediction","Structure","Docking","MD","MM-PBSA"]):
        with col:
            st.markdown(f"""<div class='met-card'>
<span class='material-symbols-outlined' style='font-size:1.6rem;color:#15803d;'>{ic}</span>
<div class='met-lbl' style='margin-top:0.4rem;'>{lbl}</div></div>""", unsafe_allow_html=True)

    st.divider()
    sec("groups","Team")

    team = [
        {
            "name": "Dr. Prabina Kumar Meher",
            "role": "Senior Scientist",
            "affil": "ICAR-IASRI, New Delhi, India",
            "affil_url": "https://scholar.google.com/citations?user=iTBeomoAAAAJ&hl=en",
            "email": "meherprabin@yahoo.com",
            "photo": "Photo/prabin_meher.jpg",
        },
        {
            "name": "Dr. Upendra Kumar Pradhan",
            "role": "Senior Scientist",
            "affil": "ICAR-IASRI, New Delhi, India",
            "affil_url": "https://scholar.google.com/citations?user=lXPvl1MAAAAJ&hl=en",
            "email": "upen4851@gmail.com",
            "photo": "Photo/upendra_pradhan.jpg",
            
        },
        {
            "name": "Shubham Kumar",
            "role": "Young Professional II",
            "affil": "ICAR-IASRI, New Delhi, India",
            "affil_url": "https://scholar.google.com/citations?user=ahGLJ84AAAAJ&hl=en",
            "email": "shubham.kum3105@gmail.com",
            "photo": "Photo/Shubham_Kumar.jpeg",
        },
        {
            "name": "Aanchal Gupta",
            "role": "Project Associate I",
            "affil": "ICAR-IASRI, New Delhi, India",
            "affil_url": "https://scholar.google.com/citations?user=L-qa0PgAAAAJ&hl=en",
            "email": "aanchal.gupta2699@gmail.com",
            "photo": "Photo/Aanchal_gupta.png",
        },
    ]

    t1,t2,t3,t4 = st.columns(4)
    for col, member in zip((t1,t2,t3,t4), team):
        with col:
            affil_html = (
                f"<a href='{member['affil_url']}' target='_blank' style='color:#6b7280;text-decoration:underline;'>{member['affil']}</a>"
                if member["affil_url"] else member["affil"]
            )

            photo_path = member.get("photo", "")
            photo_b64 = None
            if photo_path and os.path.exists(photo_path):
                photo_b64 = _img_to_base64(photo_path)

            ext = os.path.splitext(photo_path)[1].lstrip(".").lower() if photo_path else "jpeg"
            mime = "png" if ext == "png" else "jpeg"

            if photo_b64:
                avatar_html = (
                    f"<img src='data:image/{mime};base64,{photo_b64}' "
                    f"style='width:72px;height:72px;border-radius:60%;object-fit:cover;"
                    f"display:block;margin:0 auto;border:2px solid #15803d;' />"
                )
            else:
                avatar_html = (
                    "<div class='team-avatar'>"
                    "<span class='material-symbols-outlined'>person</span></div>"
                )

            st.markdown(f"""<div class='team-card'>
{avatar_html}
<div style='font-weight:700;font-size:0.9rem;color:#0f2e1c;margin-top:0.5rem;'>{member['name']}</div>
<div style='font-size:0.76rem;color:#374151;margin-top:2px;'>{member['role']}</div>
<div style='font-size:0.72rem;color:#6b7280;margin-top:2px;'>{affil_html}</div>
<div style='font-size:0.72rem;margin-top:4px;'><a href='mailto:{member["email"]}' style='color:#15803d;'>{member['email']}</a></div>
</div>""", unsafe_allow_html=True)


elif page == "guide":
    sec("menu_book","User Guide")

    PDF_PATH = "PTPLab_User_Guide.pdf"

    if os.path.exists(PDF_PATH):
        with open(PDF_PATH, "rb") as f:
            pdf_bytes = f.read()

        st.download_button(
            "Download User Guide (PDF)",
            data=pdf_bytes,
            file_name="PTPLab_User_Guide.pdf",
            mime="application/pdf",
            icon=":material/download:",
        )

        import base64
        b64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
        pdf_display = f"""
        <iframe src="data:application/pdf;base64,{b64_pdf}"
                width="100%" height="900px"
                style="border:1px solid #e5e7eb;border-radius:10px;">
        </iframe>
        """
        st.markdown(pdf_display, unsafe_allow_html=True)
    else:
        st.error(f"Could not find '{PDF_PATH}'. Make sure it's in the same "
                 f"folder as app.py, or update PDF_PATH to the correct location.")

    st.divider()
    sec("link","References")
    st.markdown("""
| Tool | Link |
|---|---|
| HADDOCK3 | [github.com/haddocking/haddock3](https://github.com/haddocking/haddock3) |
| ESMFold | [esmatlas.com](https://esmatlas.com) |
| GROMACS | [gromacs.org](https://www.gromacs.org) |
| gmx_MMPBSA | [github.com/Valdes-Tresanco-MS](https://github.com/Valdes-Tresanco-MS/gmx_MMPBSA) |
""")


#  FOOTER
st.divider()
st.markdown("""<div style='text-align:center;color:#94a3b8;font-size:0.74rem;padding:0.5rem 0 1rem;'>
🌿 <b>Plant Therapeutic Peptide Lab v1.0</b> &nbsp;·&nbsp;
PLM · ESMFold · HADDOCK3 · GROMACS · MM-PBSA &nbsp;·&nbsp;
<a href='https://github.com/haddocking/haddock3' style='color:#16a34a;'>HADDOCK3</a> &nbsp;·&nbsp;
<a href='https://github.com/Valdes-Tresanco-MS/gmx_MMPBSA' style='color:#16a34a;'>gmx_MMPBSA</a>
</div>""", unsafe_allow_html=True)
