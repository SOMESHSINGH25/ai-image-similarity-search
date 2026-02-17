import os
import numpy as np
from PIL import Image
from collections import Counter
import streamlit as st
import tensorflow as tf
import pandas as pd

# ─────────────────────────────────────────────
#  PATHS & CONSTANTS
# ─────────────────────────────────────────────
PROCESSED_DIR = "../data/processed"
MODEL_PATH    = "../models/embedding_model.keras"
CURVES_PATH   = "../results/training_curves.png"
LOG_PATH      = "../logs/training_log.csv"
IMG_SIZE      = 64

CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

CLASS_EMOJI = {
    'airplane':    '✈️',  'automobile': '🚗',
    'bird':        '🐦',  'cat':        '🐱',
    'deer':        '🦌',  'dog':        '🐶',
    'frog':        '🐸',  'horse':      '🐴',
    'ship':        '🚢',  'truck':      '🚛'
}

CLASS_DESC = {
    'airplane':   'Fixed-wing aircraft',
    'automobile': 'Four-wheeled vehicle',
    'bird':       'Feathered vertebrate',
    'cat':        'Domestic feline',
    'deer':       'Hoofed mammal',
    'dog':        'Domestic canine',
    'frog':       'Amphibian species',
    'horse':      'Large equine mammal',
    'ship':       'Large watercraft',
    'truck':      'Heavy goods vehicle'
}

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SimiliAI — Neural Image Search",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
#  GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Syne:wght@300;400;600;700&display=swap');

:root {
  --bg-deep:    #020408;
  --bg-panel:   #060d14;
  --bg-card:    #0a1520;
  --bg-hover:   #0d1e2e;
  --cyan:       #00d4ff;
  --cyan-dim:   #00a8cc;
  --cyan-glow:  rgba(0,212,255,0.15);
  --cyan-trace: rgba(0,212,255,0.06);
  --teal:       #00ffcc;
  --gold:       #ffd700;
  --red:        #ff4466;
  --green:      #00ff99;
  --text-pri:   #e8f4f8;
  --text-sec:   #7aacbe;
  --text-dim:   #3a6070;
  --border:     rgba(0,212,255,0.12);
  --border-hi:  rgba(0,212,255,0.35);
}

html, body, [class*="css"] {
  font-family: 'Syne', sans-serif;
  background-color: var(--bg-deep);
  color: var(--text-pri);
}

/* Grid background */
.stApp {
  background-color: var(--bg-deep);
  background-image:
    linear-gradient(rgba(0,212,255,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,212,255,0.03) 1px, transparent 1px);
  background-size: 40px 40px;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }

/* ── LOGO HEADER ── */
.simili-logo {
  font-family: 'Orbitron', monospace;
  font-size: 3.2rem;
  font-weight: 900;
  letter-spacing: 0.08em;
  background: linear-gradient(135deg, #00d4ff 0%, #00ffcc 50%, #00d4ff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  text-shadow: none;
  filter: drop-shadow(0 0 20px rgba(0,212,255,0.4));
  line-height: 1;
  margin-bottom: 0.1rem;
}

.simili-tagline {
  font-family: 'Syne', sans-serif;
  font-size: 0.82rem;
  font-weight: 400;
  color: var(--text-dim);
  letter-spacing: 0.35em;
  text-transform: uppercase;
  margin-bottom: 0.5rem;
}

.simili-divider {
  height: 1px;
  background: linear-gradient(90deg,
    transparent, var(--cyan), var(--teal), var(--cyan), transparent);
  margin: 0.8rem 0 1.5rem 0;
  opacity: 0.5;
}

/* ── SECTION LABELS ── */
.section-label {
  font-family: 'Orbitron', monospace;
  font-size: 0.6rem;
  font-weight: 700;
  color: var(--cyan);
  letter-spacing: 0.25em;
  text-transform: uppercase;
  margin-bottom: 0.6rem;
  display: flex;
  align-items: center;
  gap: 8px;
}
.section-label::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border);
}

/* ── PANELS ── */
.panel {
  background: var(--bg-panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 20px;
  margin-bottom: 16px;
  position: relative;
  overflow: hidden;
}
.panel::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--cyan), transparent);
  opacity: 0.5;
}

/* ── STAT BOXES ── */
.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 12px; }
.stat-box {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 12px;
  text-align: center;
  position: relative;
  overflow: hidden;
}
.stat-box::after {
  content: '';
  position: absolute;
  bottom: 0; left: 0; right: 0;
  height: 1px;
  background: var(--cyan);
  opacity: 0.3;
}
.stat-num {
  font-family: 'Orbitron', monospace;
  font-size: 1.6rem;
  font-weight: 700;
  color: var(--cyan);
  line-height: 1;
}
.stat-label {
  font-size: 0.65rem;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 0.12em;
  margin-top: 4px;
}

/* ── QUERY PANEL ── */
.query-panel {
  background: linear-gradient(135deg, #021520, #040d18);
  border: 1px solid var(--border-hi);
  border-radius: 10px;
  padding: 20px;
  margin-bottom: 20px;
  box-shadow: 0 0 40px var(--cyan-glow), inset 0 0 60px rgba(0,212,255,0.02);
}

/* ── RESULT CARDS ── */
.result-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px;
  margin-top: 8px;
  transition: border-color 0.2s, box-shadow 0.2s;
}
.result-card.correct {
  border-color: rgba(0,255,153,0.4);
  box-shadow: 0 0 16px rgba(0,255,153,0.08);
}
.result-card.wrong {
  border-color: rgba(255,68,102,0.3);
  box-shadow: 0 0 16px rgba(255,68,102,0.05);
}

/* ── RANK PILL ── */
.rank-pill {
  display: inline-block;
  font-family: 'Orbitron', monospace;
  font-size: 0.55rem;
  font-weight: 700;
  color: var(--cyan);
  border: 1px solid var(--cyan-dim);
  border-radius: 3px;
  padding: 2px 7px;
  letter-spacing: 0.1em;
  margin-bottom: 5px;
}

/* ── ACCURACY DISPLAY ── */
.acc-display {
  font-family: 'Orbitron', monospace;
  font-size: 2.8rem;
  font-weight: 900;
  text-align: center;
  padding: 1.2rem 1.5rem;
  border-radius: 10px;
  margin: 1rem 0;
  letter-spacing: 0.05em;
  line-height: 1;
}
.acc-high {
  color: var(--green);
  background: rgba(0,255,153,0.06);
  border: 1px solid rgba(0,255,153,0.25);
  box-shadow: 0 0 30px rgba(0,255,153,0.08);
}
.acc-mid {
  color: var(--gold);
  background: rgba(255,215,0,0.06);
  border: 1px solid rgba(255,215,0,0.25);
  box-shadow: 0 0 30px rgba(255,215,0,0.08);
}
.acc-low {
  color: var(--red);
  background: rgba(255,68,102,0.06);
  border: 1px solid rgba(255,68,102,0.25);
  box-shadow: 0 0 30px rgba(255,68,102,0.08);
}

/* ── SIM BAR ── */
.sim-track {
  background: rgba(0,212,255,0.08);
  border-radius: 2px;
  height: 4px;
  margin-top: 5px;
  overflow: hidden;
}
.sim-fill {
  height: 4px;
  border-radius: 2px;
  background: linear-gradient(90deg, var(--cyan-dim), var(--teal));
  transition: width 0.4s ease;
}

/* ── CLASS CHIP ── */
.class-chip {
  display: inline-block;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 3px 9px;
  font-size: 0.72rem;
  color: var(--text-sec);
  margin: 2px;
  font-family: 'Syne', sans-serif;
}

/* ── TECH TAG ── */
.tech-tag {
  display: inline-block;
  background: rgba(0,212,255,0.06);
  border: 1px solid rgba(0,212,255,0.2);
  border-radius: 3px;
  padding: 2px 8px;
  font-size: 0.65rem;
  color: var(--cyan-dim);
  margin: 2px;
  font-family: 'Orbitron', monospace;
  letter-spacing: 0.05em;
}

/* ── INFO ROW ── */
.info-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 7px 0;
  border-bottom: 1px solid var(--border);
  font-size: 0.8rem;
}
.info-row:last-child { border-bottom: none; }
.info-key { color: var(--text-dim); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; }
.info-val { color: var(--cyan); font-family: 'Orbitron', monospace; font-size: 0.75rem; }

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--bg-panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 4px;
  gap: 4px;
}
.stTabs [data-baseweb="tab"] {
  font-family: 'Orbitron', monospace;
  font-size: 0.6rem;
  letter-spacing: 0.1em;
  color: var(--text-dim);
  border-radius: 5px;
  padding: 8px 16px;
  background: transparent;
  border: none;
}
.stTabs [aria-selected="true"] {
  background: var(--bg-card) !important;
  color: var(--cyan) !important;
  box-shadow: inset 0 0 20px var(--cyan-trace);
}

/* ── BUTTONS ── */
.stButton > button {
  font-family: 'Orbitron', monospace;
  font-size: 0.65rem;
  letter-spacing: 0.15em;
  font-weight: 700;
  background: linear-gradient(135deg, #003344, #005566);
  color: var(--cyan);
  border: 1px solid var(--cyan-dim);
  border-radius: 6px;
  padding: 0.55rem 1.4rem;
  text-transform: uppercase;
  transition: all 0.2s;
  box-shadow: 0 0 20px rgba(0,212,255,0.1);
}
.stButton > button:hover {
  background: linear-gradient(135deg, #004455, #006677);
  box-shadow: 0 0 30px rgba(0,212,255,0.25);
  border-color: var(--cyan);
}

/* ── SLIDER ── */
.stSlider > div > div > div > div { background: var(--cyan); }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
  background: var(--bg-panel);
  border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .stMarkdown p { color: var(--text-sec); font-size: 0.85rem; }

/* ── SELECTBOX ── */
.stSelectbox > div > div {
  background: var(--bg-card);
  border-color: var(--border);
  color: var(--text-pri);
}

/* ── UPLOAD ── */
[data-testid="stFileUploadDropzone"] {
  background: var(--bg-card);
  border: 1px dashed var(--border-hi);
  border-radius: 8px;
}

/* ── DATAFRAME ── */
.stDataFrame { border: 1px solid var(--border); border-radius: 6px; }

/* ── PREDICTION CARD ── */
.prediction-card {
  background: linear-gradient(135deg, #021a10, #030d08);
  border: 1px solid rgba(0,255,153,0.3);
  border-radius: 10px;
  padding: 20px 24px;
  margin: 16px 0;
  box-shadow: 0 0 30px rgba(0,255,153,0.06);
}

/* ── QUERY IMAGE FRAME ── */
.query-frame {
  border: 2px solid var(--cyan-dim);
  border-radius: 8px;
  padding: 4px;
  background: var(--bg-card);
  box-shadow: 0 0 20px var(--cyan-glow);
  display: inline-block;
}

/* ── PIPELINE STEP ── */
.pipeline-step {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-left: 3px solid var(--cyan);
  border-radius: 0 6px 6px 0;
  padding: 10px 14px;
  margin-bottom: 8px;
  font-size: 0.82rem;
}
.pipeline-num {
  font-family: 'Orbitron', monospace;
  color: var(--cyan);
  font-size: 0.65rem;
  margin-bottom: 3px;
}

/* ── HOVER GLOW on images ── */
img { border-radius: 6px; }

/* ── SUCCESS / INFO ── */
.stSuccess, .stInfo { border-radius: 6px; font-size: 0.85rem; }

/* ── METRIC OVERRIDE ── */
[data-testid="metric-container"] {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 12px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD DATA & MODEL
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️  Initialising SimiliAI neural engine...")
def load_all():
    train_emb   = np.load(os.path.join(PROCESSED_DIR, "train_embeddings.npy"))
    train_lab   = np.load(os.path.join(PROCESSED_DIR, "train_labels.npy"))
    train_paths = np.load(os.path.join(PROCESSED_DIR, "train_image_paths.npy"))
    test_emb    = np.load(os.path.join(PROCESSED_DIR, "test_embeddings.npy"))
    test_lab    = np.load(os.path.join(PROCESSED_DIR, "test_labels.npy"))
    test_paths  = np.load(os.path.join(PROCESSED_DIR, "test_image_paths.npy"))
    model = None
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return train_emb, train_lab, train_paths, test_emb, test_lab, test_paths, model


# ─────────────────────────────────────────────
#  SEARCH  (Euclidean — matches triplet loss)
# ─────────────────────────────────────────────
def find_similar(query_emb, db_emb, top_k=5):
    diffs     = db_emb - query_emb
    distances = np.sqrt(np.sum(diffs ** 2, axis=1))
    top_idx   = distances.argsort()[:top_k]
    sims      = 1.0 / (1.0 + distances[top_idx])
    return top_idx, sims, distances[top_idx]


# ─────────────────────────────────────────────
#  EMBED UPLOADED IMAGE
# ─────────────────────────────────────────────
def embed_image(pil_img, model):
    img = pil_img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return model.predict(arr, verbose=0)[0]


# ─────────────────────────────────────────────
#  HELPER: similarity bar HTML
# ─────────────────────────────────────────────
def sim_bar(score, color=None):
    pct = int(score * 100)
    c   = color or "linear-gradient(90deg, #00a8cc, #00ffcc)"
    return f'<div class="sim-track"><div class="sim-fill" style="width:{pct}%;background:{c}"></div></div>'


# ─────────────────────────────────────────────
#  HELPER: accuracy style class
# ─────────────────────────────────────────────
def acc_cls(pct):
    if pct >= 60: return "acc-high"
    if pct >= 30: return "acc-mid"
    return "acc-low"


# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
try:
    (train_emb, train_lab, train_paths,
     test_emb,  test_lab,  test_paths, emb_model) = load_all()
    data_ok = True
except Exception as e:
    data_ok   = False
    data_err  = str(e)


# ═══════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════
with st.sidebar:
    # Logo
    st.markdown("""
    <div style="text-align:center;padding:16px 0 8px 0;">
      <div style="font-family:'Orbitron',monospace;font-size:1.6rem;font-weight:900;
                  background:linear-gradient(135deg,#00d4ff,#00ffcc);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  filter:drop-shadow(0 0 12px rgba(0,212,255,0.5));">
        SimiliAI
      </div>
      <div style="font-size:0.6rem;color:#3a6070;letter-spacing:0.3em;
                  text-transform:uppercase;margin-top:2px;">
        Neural Image Search
      </div>
    </div>
    <div style="height:1px;background:linear-gradient(90deg,transparent,#00d4ff,transparent);
                opacity:0.4;margin-bottom:16px;"></div>
    """, unsafe_allow_html=True)

    if data_ok:
        # Dataset stats
        st.markdown('<div class="section-label">Dataset</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stat-grid">
          <div class="stat-box">
            <div class="stat-num">{len(train_emb)}</div>
            <div class="stat-label">Train</div>
          </div>
          <div class="stat-box">
            <div class="stat-num">{len(test_emb)}</div>
            <div class="stat-label">Test</div>
          </div>
          <div class="stat-box">
            <div class="stat-num">128</div>
            <div class="stat-label">Embed Dim</div>
          </div>
          <div class="stat-box">
            <div class="stat-num">10</div>
            <div class="stat-label">Classes</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Model info
        st.markdown('<div class="section-label">Architecture</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="panel" style="padding:14px;">
          <div class="info-row">
            <span class="info-key">Type</span>
            <span class="info-val">Triplet CNN</span>
          </div>
          <div class="info-row">
            <span class="info-key">Mining</span>
            <span class="info-val">Semi-Hard</span>
          </div>
          <div class="info-row">
            <span class="info-key">Metric</span>
            <span class="info-val">Euclidean</span>
          </div>
          <div class="info-row">
            <span class="info-key">Pooling</span>
            <span class="info-val">Global Avg</span>
          </div>
          <div class="info-row">
            <span class="info-key">Output</span>
            <span class="info-val">tanh · 128D</span>
          </div>
          <div class="info-row">
            <span class="info-key">Dataset</span>
            <span class="info-val">CIFAR-10</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Classes
        st.markdown('<div class="section-label">Classes</div>', unsafe_allow_html=True)
        chips = "".join([
            f'<span class="class-chip">{CLASS_EMOJI[c]} {c}</span>'
            for c in CIFAR10_CLASSES
        ])
        st.markdown(f'<div style="line-height:2;">{chips}</div>', unsafe_allow_html=True)

        # Pipeline
        st.markdown('<div class="section-label" style="margin-top:14px;">Pipeline</div>',
                    unsafe_allow_html=True)
        steps = [
            ("01", "Download CIFAR-10", "2000 images saved"),
            ("02", "Split Dataset",     "80% train / 20% test"),
            ("03", "Train Triplet Net", "Semi-hard mining"),
            ("04", "Extract Embeddings","128-dim vectors"),
            ("05", "Similarity Search", "Euclidean k-NN"),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div class="pipeline-step">
              <div class="pipeline-num">STEP {num}</div>
              <div style="font-weight:600;font-size:0.78rem;">{title}</div>
              <div style="color:#3a6070;font-size:0.7rem;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:20px;text-align:center;color:#1e3a4a;font-size:0.65rem;
                letter-spacing:0.1em;text-transform:uppercase;">
      Deep Learning Project<br>AI Image Similarity Search
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  MAIN — HEADER
# ═══════════════════════════════════════════════════════
st.markdown("""
<div class="simili-logo">SimiliAI</div>
<div class="simili-tagline">Triplet Network · CIFAR-10 · Euclidean Embedding Space · Trained from Scratch</div>
<div class="simili-divider"></div>
""", unsafe_allow_html=True)

if not data_ok:
    st.error(f"❌ Failed to load data: {data_err}")
    st.info("Run `prepare_dataset.py` first to generate embeddings.")
    st.stop()

# ── Controls row ──
c1, c2, c3 = st.columns([2, 1.2, 1])
with c1:
    top_k = st.slider("**K — Number of similar images to retrieve**", 1, 10, 5)
with c2:
    search_db = st.selectbox("**Search database**", ["Train set (1596)", "Test set (404)"])
with c3:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:var(--bg-card);border:1px solid var(--border);
                border-radius:6px;padding:8px 12px;text-align:center;">
      <div style="font-family:'Orbitron',monospace;font-size:0.55rem;
                  color:var(--text-dim);letter-spacing:0.1em;">QUERY POOL</div>
      <div style="font-family:'Orbitron',monospace;font-size:1rem;
                  color:var(--cyan);">{len(test_emb)} imgs</div>
    </div>
    """, unsafe_allow_html=True)

use_train = "Train" in search_db
db_emb    = train_emb   if use_train else test_emb
db_lab    = train_lab   if use_train else test_lab
db_paths  = train_paths if use_train else test_paths

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──
tab1, tab2, tab3, tab4 = st.tabs([
    "⬡  RANDOM QUERY",
    "⬡  UPLOAD IMAGE",
    "⬡  TRAINING CURVES",
    "⬡  HOW IT WORKS",
])


# ═══════════════════════════════════════════════════════
#  TAB 1 — RANDOM QUERY
# ═══════════════════════════════════════════════════════
with tab1:
    st.markdown("<br>", unsafe_allow_html=True)

    btn_col, info_col = st.columns([1, 3])
    with btn_col:
        run = st.button("⬡  EXECUTE QUERY", key="run_random", use_container_width=True)
    with info_col:
        st.markdown("""
        <div style="background:var(--bg-card);border:1px solid var(--border);
                    border-radius:6px;padding:10px 16px;font-size:0.8rem;color:var(--text-sec);">
          Selects a <strong style="color:var(--cyan)">random test image</strong>,
          encodes it into a 128-dimensional embedding vector, then performs
          <strong style="color:var(--cyan)">Euclidean k-NN search</strong>
          across the database to retrieve the most visually similar images.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if run:
        query_idx     = np.random.randint(0, len(test_emb))
        query_vec     = test_emb[query_idx]
        query_label   = int(test_lab[query_idx])
        query_path    = test_paths[query_idx]
        query_class   = CIFAR10_CLASSES[query_label]

        top_idx, sims, dists = find_similar(query_vec, db_emb, top_k + 1)

        # Remove self if searching test set
        if not use_train:
            top_idx = np.array([i for i in top_idx if i != query_idx][:top_k])
            diffs   = db_emb[top_idx] - query_vec
            dists   = np.sqrt(np.sum(diffs ** 2, axis=1))
            sims    = 1.0 / (1.0 + dists)
        else:
            top_idx = top_idx[:top_k]
            sims    = sims[:top_k]
            dists   = dists[:top_k]

        retrieved_labels = db_lab[top_idx]
        correct_count    = int(np.sum(retrieved_labels == query_label))
        accuracy_pct     = (correct_count / top_k) * 100

        # ── Layout: Query | Accuracy | Stats ──
        left, mid, right = st.columns([1.4, 1.8, 1.4])

        with left:
            st.markdown('<div class="section-label">Query Image</div>',
                        unsafe_allow_html=True)
            try:
                st.image(Image.open(query_path), width=180)
            except:
                st.warning("Image not found")
            st.markdown(f"""
            <div class="panel" style="margin-top:10px;">
              <div class="info-row">
                <span class="info-key">Class</span>
                <span style="color:var(--teal);font-weight:700;">
                  {CLASS_EMOJI[query_class]} {query_class}
                </span>
              </div>
              <div class="info-row">
                <span class="info-key">Index</span>
                <span class="info-val">#{query_idx}</span>
              </div>
              <div class="info-row">
                <span class="info-key">Description</span>
                <span style="color:var(--text-sec);font-size:0.72rem;">
                  {CLASS_DESC[query_class]}
                </span>
              </div>
              <div class="info-row">
                <span class="info-key">Embed Dim</span>
                <span class="info-val">128D</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with mid:
            st.markdown('<div class="section-label">Retrieval Accuracy</div>',
                        unsafe_allow_html=True)
            ac = acc_cls(accuracy_pct)
            st.markdown(f"""
            <div class="acc-display {ac}">
              {correct_count}/{top_k}
              <div style="font-size:0.9rem;margin-top:6px;letter-spacing:0.05em;">
                {accuracy_pct:.0f}% TOP-{top_k}
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Per-result breakdown
            st.markdown('<div class="section-label">Result Breakdown</div>',
                        unsafe_allow_html=True)
            for i, (idx, sim, dist) in enumerate(zip(top_idx, sims, dists)):
                lbl   = int(db_lab[idx])
                cname = CIFAR10_CLASSES[lbl]
                ok    = lbl == query_label
                color = "#00ff99" if ok else "#ff4466"
                mark  = "✓" if ok else "✗"
                st.markdown(f"""
                <div style="display:flex;align-items:center;justify-content:space-between;
                            padding:5px 0;border-bottom:1px solid var(--border);
                            font-size:0.78rem;">
                  <span style="color:var(--text-dim);font-family:'Orbitron',monospace;
                               font-size:0.6rem;">#{i+1}</span>
                  <span>{CLASS_EMOJI[cname]} {cname}</span>
                  <span style="font-family:'Orbitron',monospace;font-size:0.68rem;
                               color:var(--cyan);">{sim:.3f}</span>
                  <span style="color:{color};font-weight:700;">{mark}</span>
                </div>
                """, unsafe_allow_html=True)

        with right:
            st.markdown('<div class="section-label">Embedding Stats</div>',
                        unsafe_allow_html=True)
            st.markdown(f"""
            <div class="panel">
              <div class="info-row">
                <span class="info-key">Min distance</span>
                <span class="info-val">{dists.min():.4f}</span>
              </div>
              <div class="info-row">
                <span class="info-key">Max distance</span>
                <span class="info-val">{dists.max():.4f}</span>
              </div>
              <div class="info-row">
                <span class="info-key">Avg similarity</span>
                <span class="info-val">{sims.mean():.4f}</span>
              </div>
              <div class="info-row">
                <span class="info-key">Correct matches</span>
                <span style="color:var(--green);font-family:'Orbitron',monospace;
                             font-size:0.75rem;">{correct_count}/{top_k}</span>
              </div>
              <div class="info-row">
                <span class="info-key">Search space</span>
                <span class="info-val">{len(db_emb):,}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Class distribution of results
            st.markdown('<div class="section-label" style="margin-top:10px;">Retrieved Classes</div>',
                        unsafe_allow_html=True)
            class_counts = Counter(CIFAR10_CLASSES[int(db_lab[i])] for i in top_idx)
            for cls_name, cnt in class_counts.most_common():
                bar_w = int((cnt / top_k) * 100)
                color = "#00ff99" if cls_name == query_class else "#3a6070"
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;
                            font-size:0.75rem;">
                  <span style="width:70px;color:var(--text-sec);">
                    {CLASS_EMOJI[cls_name]} {cls_name}
                  </span>
                  <div style="flex:1;background:rgba(0,212,255,0.06);
                              border-radius:2px;height:6px;overflow:hidden;">
                    <div style="width:{bar_w}%;height:6px;border-radius:2px;
                                background:{color};"></div>
                  </div>
                  <span style="color:var(--text-dim);font-size:0.68rem;">{cnt}</span>
                </div>
                """, unsafe_allow_html=True)

        # ── Results Grid ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Retrieved Images</div>',
                    unsafe_allow_html=True)

        n_cols = min(top_k, 5)
        cols   = st.columns(n_cols)

        for i, (idx, sim, dist) in enumerate(zip(top_idx[:5], sims[:5], dists[:5])):
            lbl    = int(db_lab[idx])
            cname  = CIFAR10_CLASSES[lbl]
            ok     = lbl == query_label
            card_c = "correct" if ok else "wrong"
            status = "✅ MATCH" if ok else "❌ MISS"
            s_col  = "#00ff99" if ok else "#ff4466"

            with cols[i]:
                try:
                    st.image(Image.open(db_paths[idx]), use_container_width=True)
                except:
                    st.error("img err")

                st.markdown(f"""
                <div class="result-card {card_c}">
                  <div class="rank-pill">RANK #{i+1}</div><br>
                  <span style="font-size:1.1rem;">{CLASS_EMOJI[cname]}</span>
                  <strong style="font-size:0.85rem;"> {cname}</strong><br>
                  <span style="color:{s_col};font-size:0.7rem;font-family:'Orbitron',monospace;">
                    {status}
                  </span><br>
                  <span style="color:var(--text-dim);font-size:0.7rem;">
                    sim&nbsp;{sim:.4f} &nbsp;|&nbsp; d={dist:.3f}
                  </span>
                  {sim_bar(sim)}
                </div>
                """, unsafe_allow_html=True)

        # Second row if top_k > 5
        if top_k > 5:
            st.markdown("<br>", unsafe_allow_html=True)
            cols2 = st.columns(top_k - 5)
            for i, (idx, sim, dist) in enumerate(
                    zip(top_idx[5:], sims[5:], dists[5:])):
                lbl    = int(db_lab[idx])
                cname  = CIFAR10_CLASSES[lbl]
                ok     = lbl == query_label
                card_c = "correct" if ok else "wrong"
                status = "✅ MATCH" if ok else "❌ MISS"
                s_col  = "#00ff99" if ok else "#ff4466"
                with cols2[i]:
                    try:
                        st.image(Image.open(db_paths[idx]), use_container_width=True)
                    except:
                        st.error("img err")
                    st.markdown(f"""
                    <div class="result-card {card_c}">
                      <div class="rank-pill">RANK #{i+6}</div><br>
                      {CLASS_EMOJI[cname]} <strong>{cname}</strong>
                      <span style="color:{s_col};font-size:0.7rem;
                                   font-family:'Orbitron',monospace;"> {status}</span><br>
                      <span style="color:var(--text-dim);font-size:0.7rem;">
                        sim {sim:.4f} | d={dist:.3f}
                      </span>
                      {sim_bar(sim)}
                    </div>
                    """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center;padding:5rem 2rem;">
          <div style="font-family:'Orbitron',monospace;font-size:3rem;
                      color:rgba(0,212,255,0.15);letter-spacing:0.2em;">
            AWAITING QUERY
          </div>
          <div style="color:#1e3a4a;margin-top:1rem;font-size:0.85rem;">
            Press <strong style="color:#00a8cc;">EXECUTE QUERY</strong>
            to begin similarity search
          </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  TAB 2 — UPLOAD IMAGE
# ═══════════════════════════════════════════════════════
with tab2:
    st.markdown("<br>", unsafe_allow_html=True)

    if emb_model is None:
        st.error("❌ Embedding model not found. Run `train_triplet.py` first.")
        st.stop()

    up_left, up_right = st.columns([1.2, 2])

    with up_left:
        st.markdown('<div class="section-label">Upload Image</div>',
                    unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drop any image file",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed"
        )
        st.markdown("""
        <div class="panel" style="margin-top:10px;">
          <div style="font-size:0.75rem;color:var(--text-sec);line-height:1.7;">
            Upload <strong style="color:var(--cyan)">any photo</strong> and SimiliAI
            will:<br><br>
            <span style="color:var(--text-dim);">①</span>
            Resize to 64×64 pixels<br>
            <span style="color:var(--text-dim);">②</span>
            Pass through the trained CNN<br>
            <span style="color:var(--text-dim);">③</span>
            Output a 128-dim tanh embedding<br>
            <span style="color:var(--text-dim);">④</span>
            Run Euclidean k-NN search<br>
            <span style="color:var(--text-dim);">⑤</span>
            Return the most similar images<br><br>
            <strong style="color:var(--cyan)">Best results</strong> with images
            containing: cats, dogs, cars, birds, airplanes, ships, trucks,
            deer, frogs, or horses.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with up_right:
        if uploaded:
            pil_img = Image.open(uploaded)

            col_img, col_info = st.columns([1, 1.5])
            with col_img:
                st.markdown('<div class="section-label">Your Image</div>',
                            unsafe_allow_html=True)
                st.image(pil_img, width=180)
                st.markdown(f"""
                <div style="font-size:0.72rem;color:var(--text-dim);margin-top:4px;">
                  {pil_img.width}×{pil_img.height}px &nbsp;|&nbsp; {pil_img.mode}
                </div>
                """, unsafe_allow_html=True)

            with col_info:
                st.markdown('<div class="section-label">Processing</div>',
                            unsafe_allow_html=True)
                with st.spinner("Running neural embedding..."):
                    query_vec     = embed_image(pil_img, emb_model)
                    top_idx, sims, dists = find_similar(query_vec, db_emb, top_k)

                retrieved_labels = db_lab[top_idx]
                pred_label = Counter(retrieved_labels.tolist()).most_common(1)[0][0]
                pred_class = CIFAR10_CLASSES[pred_label]

                st.markdown(f"""
                <div class="prediction-card">
                  <div style="font-size:0.6rem;color:#00cc77;letter-spacing:0.2em;
                              text-transform:uppercase;margin-bottom:6px;">
                    Predicted Class
                  </div>
                  <div style="font-size:2.5rem;line-height:1;">
                    {CLASS_EMOJI[pred_class]}
                  </div>
                  <div style="font-family:'Orbitron',monospace;font-size:1.1rem;
                              color:#00ff99;font-weight:700;margin-top:4px;">
                    {pred_class.upper()}
                  </div>
                  <div style="font-size:0.72rem;color:#3a6070;margin-top:4px;">
                    {CLASS_DESC[pred_class]}<br>
                    Based on top-{top_k} nearest neighbours
                  </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="panel">
                  <div class="info-row">
                    <span class="info-key">Embedding norm</span>
                    <span class="info-val">{np.linalg.norm(query_vec):.4f}</span>
                  </div>
                  <div class="info-row">
                    <span class="info-key">Nearest distance</span>
                    <span class="info-val">{dists.min():.4f}</span>
                  </div>
                  <div class="info-row">
                    <span class="info-key">Avg similarity</span>
                    <span class="info-val">{sims.mean():.4f}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # Results grid
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Nearest Neighbours</div>',
                        unsafe_allow_html=True)
            cols = st.columns(min(top_k, 5))
            for i, (idx, sim, dist) in enumerate(zip(top_idx[:5], sims[:5], dists[:5])):
                lbl   = int(db_lab[idx])
                cname = CIFAR10_CLASSES[lbl]
                with cols[i]:
                    try:
                        st.image(Image.open(db_paths[idx]), use_container_width=True)
                    except:
                        st.error("err")
                    st.markdown(f"""
                    <div class="result-card">
                      <div class="rank-pill">#{i+1}</div><br>
                      {CLASS_EMOJI[cname]} <strong>{cname}</strong><br>
                      <span style="color:var(--text-dim);font-size:0.7rem;">
                        {sim:.4f} | d={dist:.3f}
                      </span>
                      {sim_bar(sim)}
                    </div>
                    """, unsafe_allow_html=True)

            if top_k > 5:
                cols2 = st.columns(top_k - 5)
                for i, (idx, sim, dist) in enumerate(zip(top_idx[5:], sims[5:], dists[5:])):
                    lbl   = int(db_lab[idx])
                    cname = CIFAR10_CLASSES[lbl]
                    with cols2[i]:
                        try:
                            st.image(Image.open(db_paths[idx]), use_container_width=True)
                        except:
                            st.error("err")
                        st.markdown(f"""
                        <div class="result-card">
                          <div class="rank-pill">#{i+6}</div><br>
                          {CLASS_EMOJI[cname]} <strong>{cname}</strong><br>
                          <span style="color:var(--text-dim);font-size:0.7rem;">
                            {sim:.4f}
                          </span>
                          {sim_bar(sim)}
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:5rem 2rem;">
              <div style="font-family:'Orbitron',monospace;font-size:3rem;
                          color:rgba(0,212,255,0.12);letter-spacing:0.2em;">
                UPLOAD FILE
              </div>
              <div style="color:#1e3a4a;margin-top:1rem;font-size:0.85rem;">
                Drag and drop or click to upload an image
              </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  TAB 3 — TRAINING CURVES
# ═══════════════════════════════════════════════════════
with tab3:
    st.markdown("<br>", unsafe_allow_html=True)

    if os.path.exists(CURVES_PATH):
        st.markdown('<div class="section-label">Loss Curves — Triplet Network Training</div>',
                    unsafe_allow_html=True)
        st.image(CURVES_PATH, use_container_width=True)

        if os.path.exists(LOG_PATH):
            st.markdown("<br>", unsafe_allow_html=True)
            df = pd.read_csv(LOG_PATH)
            best_row = df.loc[df["val_loss"].idxmin()]

            st.markdown('<div class="section-label">Training Summary</div>',
                        unsafe_allow_html=True)

            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.markdown(f"""
                <div class="stat-box">
                  <div class="stat-num">{len(df)}</div>
                  <div class="stat-label">Epochs Run</div>
                </div>""", unsafe_allow_html=True)
            with s2:
                st.markdown(f"""
                <div class="stat-box">
                  <div class="stat-num">{best_row['val_loss']:.4f}</div>
                  <div class="stat-label">Best Val Loss</div>
                </div>""", unsafe_allow_html=True)
            with s3:
                st.markdown(f"""
                <div class="stat-box">
                  <div class="stat-num">{int(best_row['epoch'])}</div>
                  <div class="stat-label">Best Epoch</div>
                </div>""", unsafe_allow_html=True)
            with s4:
                improvement = df["train_loss"].iloc[0] - df["train_loss"].iloc[-1]
                st.markdown(f"""
                <div class="stat-box">
                  <div class="stat-num">{improvement:.2f}</div>
                  <div class="stat-label">Loss Drop</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Epoch Log</div>',
                        unsafe_allow_html=True)
            st.dataframe(
                df.style.highlight_min(subset=["val_loss"], color="#082010")
                        .highlight_min(subset=["train_loss"], color="#082010")
                        .format({"train_loss": "{:.5f}", "val_loss": "{:.5f}"}),
                use_container_width=True,
                hide_index=True
            )
    else:
        st.markdown("""
        <div style="text-align:center;padding:5rem 2rem;">
          <div style="font-family:'Orbitron',monospace;font-size:2rem;
                      color:rgba(0,212,255,0.12);letter-spacing:0.15em;">
            NO TRAINING DATA
          </div>
          <div style="color:#1e3a4a;margin-top:1rem;font-size:0.85rem;">
            Run <code style="color:#00a8cc;">train_triplet.py</code>
            to generate training curves
          </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
#  TAB 4 — HOW IT WORKS
# ═══════════════════════════════════════════════════════
with tab4:
    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-label">What is a Triplet Network?</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="panel">
          <div style="font-size:0.85rem;color:var(--text-sec);line-height:1.8;">
            A <strong style="color:var(--cyan)">Triplet Network</strong> is a deep
            learning architecture that learns to map images into an embedding space
            where <strong style="color:var(--teal)">similar images are close</strong>
            and <strong style="color:var(--red)">dissimilar images are far apart</strong>.<br><br>
            It trains on <em>triplets</em> of images:
            <ul style="margin:8px 0 8px 16px;color:var(--text-sec);">
              <li><strong style="color:var(--teal)">Anchor</strong> — the reference image</li>
              <li><strong style="color:var(--green)">Positive</strong> — same class as anchor</li>
              <li><strong style="color:var(--red)">Negative</strong> — different class from anchor</li>
            </ul>
            The loss function pushes the anchor closer to the positive
            and further from the negative in 128-dimensional space.
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:14px;">Triplet Loss Formula</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="panel" style="font-family:'Orbitron',monospace;font-size:0.72rem;
                                  color:var(--cyan);text-align:center;padding:20px;">
          L = max( d(A,P) − d(A,N) + α, 0 )
          <div style="margin-top:12px;font-size:0.58rem;color:var(--text-dim);
                      font-family:'Syne',sans-serif;letter-spacing:0.05em;">
            d = Euclidean distance &nbsp;|&nbsp; α = margin (0.3) &nbsp;|&nbsp;
            A = Anchor &nbsp;|&nbsp; P = Positive &nbsp;|&nbsp; N = Negative
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:14px;">Semi-Hard Mining</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="panel">
          <div style="font-size:0.82rem;color:var(--text-sec);line-height:1.8;">
            <strong style="color:var(--cyan)">Semi-hard mining</strong> selects negatives
            that satisfy:<br><br>
            <div style="font-family:'Orbitron',monospace;font-size:0.68rem;
                        color:var(--teal);text-align:center;padding:8px;
                        background:var(--bg-card);border-radius:4px;margin:8px 0;">
              d(A,P) &lt; d(A,N) &lt; d(A,P) + margin
            </div>
            This means the negative is <em>harder than the positive</em>
            but <em>not so hard</em> that gradients vanish — the sweet spot
            for stable, meaningful learning from scratch.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-label">CNN Architecture</div>',
                    unsafe_allow_html=True)

        layers_info = [
            ("INPUT",        "64×64×3 RGB image",            "var(--text-dim)"),
            ("CONV2D × 3",   "32→64→128 filters, ReLU, Same","var(--cyan)"),
            ("BATCHNORM × 3","Normalise activations",         "var(--text-sec)"),
            ("MAXPOOL × 3",  "Downsample spatial dims",       "var(--text-sec)"),
            ("GLOBAL AVG",   "Aggregate spatial features",    "var(--cyan)"),
            ("DENSE 256",    "ReLU activation",               "var(--cyan)"),
            ("DROPOUT 0.3",  "Regularisation",                "var(--text-sec)"),
            ("DENSE 128",    "tanh → embedding vector",       "var(--teal)"),
            ("OUTPUT",       "128-dim embedding ∈ [-1, 1]",   "var(--teal)"),
        ]

        for name, desc, color in layers_info:
            st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;
                        padding:8px 12px;margin-bottom:4px;
                        background:var(--bg-card);border:1px solid var(--border);
                        border-left:3px solid {color};border-radius:0 6px 6px 0;">
              <div style="font-family:'Orbitron',monospace;font-size:0.58rem;
                          color:{color};width:90px;flex-shrink:0;">{name}</div>
              <div style="font-size:0.75rem;color:var(--text-sec);">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:16px;">Search Strategy</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="panel">
          <div style="font-size:0.82rem;color:var(--text-sec);line-height:1.8;">
            At query time, SimiliAI:<br><br>
            <div class="pipeline-step">
              <div class="pipeline-num">STEP 01</div>
              Encodes the query image into a 128-dim vector
            </div>
            <div class="pipeline-step">
              <div class="pipeline-num">STEP 02</div>
              Computes Euclidean distance to all 1596 train embeddings
            </div>
            <div class="pipeline-step">
              <div class="pipeline-num">STEP 03</div>
              Sorts by distance — smallest = most visually similar
            </div>
            <div class="pipeline-step">
              <div class="pipeline-num">STEP 04</div>
              Returns top-K images with similarity scores
            </div>
            <br>
            <strong style="color:var(--cyan)">Why Euclidean?</strong><br>
            The triplet loss is defined using squared Euclidean distance,
            so the embedding space is shaped by Euclidean geometry.
            Cosine similarity would ignore the magnitude of displacement
            between embeddings and produce incorrect rankings.
          </div>
        </div>
        """, unsafe_allow_html=True)