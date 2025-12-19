import os
import random
import time
from datetime import datetime, timedelta
import textwrap 
import torch
import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import chromadb
from sentence_transformers import SentenceTransformer
import json
import re

# ================== IMPORT FROM MAPPINGS ==================
from mappings import (
    PREDEFINED_QUESTIONS,
    SERVICE_MAPPING,
    PROTOCOL_MAPPING,
    ATTACK_CLASS_MAP,
    FEATURE_LABELS,
)

def strip_html(text):
    if not isinstance(text, str):
        return text
    return re.sub(r"<.*?>", "", text)

# ================== CONSTANTS & PATHS ==================
from pathlib import Path

# Dossier où se trouve app.py  → .../Dashboard/dashboard
BASE_DIR = Path(__file__).resolve().parent

# Racine du projet → .../CYBERMENTOR-DL-NLP
PROJECT_ROOT = BASE_DIR.parent.parent

# ============ PATHS pour projet ============

# Modèle NLP (il sera téléchargé automatiquement depuis HuggingFace)
MODEL_DIR = "Naaima/nlp_multiclass_model"   

# Fichier de test 
TEST_CSV = BASE_DIR / "data" / "nlp_test.csv"

# RAG
RAG_BASE = PROJECT_ROOT / "rag"
KB_PATH = PROJECT_ROOT / "Dashboard" / "chroma_db"  
DOCS_PATH = RAG_BASE / "docs.json"

EMBED_MODEL_NAME = "all-mpnet-base-v2"

# LLM Qwen téléchargé depuis HuggingFace
LLM_PATH = "Qwen/Qwen2.5-0.5B-Instruct"

# ================== HELPER FUNCTIONS ==================

def render_stat_card(title, value, subtitle, border_color):
    """Render a consistent statistics card"""
    st.markdown(
        f"""
        <div style='
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            border-radius: 16px;
            padding: 1.5rem;
            color: white;
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
            border: 1px solid #475569;
            border-left: 4px solid {border_color};
        '>
            <div style='font-size: 0.875rem; opacity: 0.9; margin-bottom: 0.5rem; color: #94a3b8;'>{title}</div>
            <div style='font-size: 2rem; font-weight: 700; color: #f1f5f9;'>{value}</div>
            <div style='font-size: 0.75rem; opacity: 0.8; margin-top: 0.5rem; color: #cbd5e1;'>{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_status_banner(running: bool):
    """Render simulation status banner"""
    status_color = "🟢" if running else "🔴"
    status_text = "RUNNING" if running else "STOPPED"
    status_bg = (
        "linear-gradient(135deg, #10b981, #059669)"
        if running else
        "linear-gradient(135deg, #ef4444, #dc2626)"
    )
    st.markdown(
        f"""
        <div style='
            {status_bg};
            border-radius: 12px;
            padding: 1rem 1.5rem;
            color: white;
            margin: 1rem 0;
            text-align: center;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        '>
            <div style='font-size: 1.1rem;'>
                {status_color} Simulation Status: {status_text}
            </div>
            <div style='font-size: 0.875rem; opacity: 0.9; margin-top: 0.25rem;'>
                Last update: {datetime.now().strftime("%H:%M:%S")}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_autorefresh_banner(text: str):
    """Render auto-refresh banner"""
    st.markdown(
        f"""
        <div style='
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            padding: 1rem 1.5rem;
            color: white;
            margin-bottom: 1.5rem;
            text-align: center;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        '>
            {text}
        </div>
        """,
        unsafe_allow_html=True
    )

def style_plot(fig, theme="dark"):
    """Style plotly figures with dark or light theme"""
    if theme == "dark":
        paper = plot = "#1e293b"
        font_color = "#f1f5f9"
        grid = "#334155"
        legend_bg = "rgba(30, 41, 59, 0.9)"
        legend_border = "#334155"
    else:
        paper = plot = "#ffffff"
        font_color = "#0f172a"
        grid = "#e5e7eb"
        legend_bg = "rgba(255,255,255,0.85)"
        legend_border = "#e5e7eb"

    fig.update_layout(
        paper_bgcolor=paper,
        plot_bgcolor=plot,
        font=dict(family="Inter", size=12, color=font_color),
        margin=dict(l=40, r=25, t=50, b=40),
        legend=dict(
            bgcolor=legend_bg,
            bordercolor=legend_border,
            borderwidth=1,
        ),
        xaxis=dict(gridcolor=grid, zerolinecolor=grid),
        yaxis=dict(gridcolor=grid, zerolinecolor=grid),
    )
    return fig

def show_event_details(row_dict):
    """Show event details in a formatted dataframe"""
    details_data = []
    for col, val in row_dict.items():
        if col in ("text_features", "attack_context"):
            continue
        label = FEATURE_LABELS.get(col, col)
        if isinstance(val, (int, float)) and val > 1000:
            val = f"{val:,}"
        elif isinstance(val, str) and len(val) > 50:
            val = val[:50] + "..."
        details_data.append({"Feature": label, "Value": str(val)})

    df_details = pd.DataFrame(details_data)
    st.dataframe(
        df_details,
        use_container_width=True,
        height=300,
        hide_index=True,
    )

def show_model_performance(ev):
    """Show model performance details"""
    true_label = ev["true_label"]
    pred_binary = ev["nlp_pred_binary"]
    probs = ev["nlp_probs"]
    prob_normal = probs[9]
    prob_attack = 1.0 - prob_normal

    st.write(f"- **True type:** {['Normal', 'Attack'][true_label]}")
    st.write(f"- **Predicted:** {['Normal', 'Attack'][pred_binary]}")
    if pred_binary == 1:
        st.write(f"- **Predicted attack type:** {ev['predicted_attack_type']}")
    st.write(f"- **Prob Normal:** {prob_normal:.3f}")
    st.write(f"- **Prob Attack:** {prob_attack:.3f}")

def get_detection_badge(true_label, pred_binary, pred_attack_type, true_attack_cat):
    """Return appropriate badge based on prediction correctness"""
    if true_label == pred_binary:
        if true_label == 1 and pred_attack_type == true_attack_cat:
            return " Perfect detection", "perfect-badge"
        else:
            return " Correct detection", "correct-badge"
    else:
        return " Bad detection", "bad-badge"

# ================== RAG FUNCTIONS ==================

def search_rag_documents(question: str, attack_type: str, collection, embed_model, n_results: int = 3):
    """Search for relevant documents in ChromaDB based on question and attack type"""
    try:
        query_text = f"{question} {attack_type} attack"
        query_embedding = embed_model.encode([query_text]).tolist()[0]
        
        where_filter = None
        if attack_type != "Generic" and attack_type in ["Fuzzers", "Analysis", "Backdoors", "DoS", "Exploits", "Reconnaissance", "Shellcode", "Worms"]:
            where_filter = {"attack_type": attack_type}
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
        
        return results
    except Exception as e:
        st.error(f"Error searching RAG database: {e}")
        return None

def generate_rag_answer(question: str, attack_type: str, rag_results, tokenizer_llm, model_llm):
    """Generate a short, human and professional answer using ALL RAG documents found."""
    try:
        # ===== 1) Construire le contexte à partir de TOUS les documents RAG =====
        context = ""
        if rag_results and "documents" in rag_results and rag_results["documents"]:
            all_docs = rag_results["documents"][0]
            for i, doc in enumerate(all_docs):
                context += f"[Doc {i+1}]\n{doc}\n\n"

        # ===== 2) Prompts =====
        if not context.strip():
            # Pas de contexte RAG → réponse générale
            prompt = f"""
You are a cybersecurity assistant.

No specific knowledge base context is available for this attack type: {attack_type}.

The user asked:
\"{question}\"

Give a concise, human explanation in 3–6 sentences.
Be practical and operational (logs to check, controls, protections).
Do NOT use bullet points or numbered lists.
Answer in English in a single short paragraph.

Answer:
"""
        else:
            # Contexte RAG disponible
            prompt = f"""
You are a cybersecurity assistant.

Use ONLY the context below about {attack_type} attacks
to answer the user's question in a short and practical way.

Context:
{context}

Question:
\"{question}\"

Give a concise, human explanation in 3–6 sentences.
Be practical and operational (logs to check, controls, protections).
Do NOT use bullet points or numbered lists.
Answer in English in a single short paragraph.

Answer:
"""

        # ===== 3) Appel au LLM =====
        inputs = tokenizer_llm(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(model_llm.device)

        with torch.no_grad():
            output_ids = model_llm.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                pad_token_id=tokenizer_llm.eos_token_id,
            )

        raw_text = tokenizer_llm.decode(output_ids[0], skip_special_tokens=True).strip()

        # ===== 4) Nettoyage : enlever le prompt éventuel =====
        if "Answer:" in raw_text:
            raw_text = raw_text.split("Answer:", 1)[1].strip()

        return raw_text

    except Exception as e:
        st.error(f"Error generating RAG answer: {e}")
        return "I encountered an error while generating the answer. Please try again."

# ================== ANALYTICS FUNCTIONS ==================

def calculate_ip_reputation(events, ip_type='srcip'):
    """Calculate IP reputation scores"""
    ip_stats = {}
    for event in events:
        if event['label'] == 'Attack':
            ip = event[ip_type]
            probs = event['nlp_probs']
            prob_normal = float(probs[9])
            prob_attack = float(1.0 - prob_normal)
            
            if ip not in ip_stats:
                ip_stats[ip] = {'count': 0, 'total_severity': 0}
            
            ip_stats[ip]['count'] += 1
            ip_stats[ip]['total_severity'] += prob_attack
    
    reputation_scores = {}
    for ip, stats in ip_stats.items():
        avg_severity = stats['total_severity'] / stats['count']
        frequency_score = min(stats['count'] / 10, 1.0)
        reputation_scores[ip] = (avg_severity * 0.7 + frequency_score * 0.3) * 100
    
    return reputation_scores

# ================== STATE INIT ==================

if "events" not in st.session_state:
    st.session_state["events"] = []

if "simulation_running" not in st.session_state:
    st.session_state["simulation_running"] = False

if "last_sim_time" not in st.session_state:
    st.session_state["last_sim_time"] = None

if "assistant_visible" not in st.session_state:
    st.session_state["assistant_visible"] = {}

if "assistant_questions" not in st.session_state:
    st.session_state["assistant_questions"] = {}

if "assistant_answers" not in st.session_state:
    st.session_state["assistant_answers"] = {}

# AJOUT DES NOUVEAUX STATES PERSONNALISÉS
if "custom_questions" not in st.session_state:
    st.session_state["custom_questions"] = {}

# ================== MODEL LOADERS ==================

@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()
    return tokenizer, model, device

@st.cache_data
def load_demo_data():
    return pd.read_csv(TEST_CSV, dtype={'dsport': str, 'sport': str}, low_memory=False)

def predict(text, tokenizer, model, device):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        pred_label = torch.argmax(probs).item()

    return pred_label, probs.cpu().numpy()

@st.cache_resource
def load_rag():
    """
    Charge le modèle d'embedding + la collection ChromaDB.
    Si la collection 'security_kb' n'existe pas, on la reconstruit à partir de docs.json.
    """
    # 1) Charger le modèle d'embedding
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    # 2) Client ChromaDB local
    client = chromadb.PersistentClient(path=KB_PATH)
    collection_name = "security_kb"

    # 3) Essayer de récupérer la collection
    try:
        collection = client.get_collection(collection_name)
    except Exception as e:    
        try:
            with open(DOCS_PATH, "r", encoding="utf-8") as f:
                attacks = json.load(f)

            texts, ids, metas = [], [], []

            for attack in attacks:
                mitigation_texts = []
                for m in attack.get("mitigation_recommendations", []):
                    mitigation_texts.append(
                        f"Recommendation: {m['recommendation']}. "
                        f"Implementation: {m['implementation']}. "
                        f"Priority: {m['priority']}. "
                        f"Source: {m['source']} ({m['source_detail']})"
                    )

                text = (
                    f"Attack ID: {attack['id']}\n"
                    f"Type: {attack['attack_type']}\n"
                    f"Description: {attack['description']}\n"
                    f"Indicators: {', '.join(attack.get('indicators', []))}\n"
                    f"Mitigation Recommendations:\n" + "\n".join(mitigation_texts) + "\n"
                    f"Defense Layers: {', '.join(attack.get('defense_layers', []))}\n"
                    f"Source: {attack.get('source', '')}"
                )

                texts.append(text)
                ids.append(attack["id"])
                metas.append({
                    "source": attack.get("source", ""),
                    "attack_type": attack.get("attack_type", "")
                })

            # Embeddings
            embeddings = embed_model.encode(
                texts,
                show_progress_bar=False,
                batch_size=16
            ).tolist()

            # Créer la collection et ajouter les docs
            collection = client.get_or_create_collection(name=collection_name)
            collection.add(
                ids=ids,
                documents=texts,
                metadatas=metas,
                embeddings=embeddings
            )
        except Exception as e2:
            st.error(f"❌ Error rebuilding ChromaDB: {e2}")
            # on ne fait PAS st.stop(), on laisse l'app continuer mais sans RAG
            return embed_model, None, {}

    try:
        with open(DOCS_PATH, "r", encoding="utf-8") as f:
            attacks = json.load(f)

        attacks_by_type = {}
        for a in attacks:
            attacks_by_type.setdefault(a["attack_type"], []).append(a)

    except Exception as e:
        st.warning(f"⚠️ Could not load attacks_by_type from {DOCS_PATH}: {e}")
        attacks_by_type = {}

    return embed_model, collection, attacks_by_type

@st.cache_resource
def load_llm():
    tokenizer_llm = AutoTokenizer.from_pretrained(LLM_PATH, trust_remote_code=True)
    model_llm = AutoModelForCausalLM.from_pretrained(
        LLM_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model_llm.eval()
    return tokenizer_llm, model_llm

# ================== MAIN APP ==================

st.set_page_config(page_title="CyberMentor Dashboard", layout="wide")

# ---------- MODERN WHITE THEME CSS (FUSIONNÉ ET AMÉLIORÉ) ----------
st.markdown(
    """
<style>
/* Global styles */
[data-testid="stAppViewContainer"] {
    background-color: #ffffff;
}

h1, h2, h3, h4, h5, .stTitle {
    font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #0f172a;
    font-weight: 600;
}

/* Header styles */
.dashboard-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-radius: 20px;
    padding: 2rem;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    border: 1px solid rgba(255,255,255,0.1);
}

.dashboard-header h1 {
    font-size: 2.5rem;
    margin-bottom: 1rem;
    font-weight: 700;
    color: white;
}

/* Modern tabs */
[data-testid="stTabs"] [data-baseweb="tab"] {
    padding: 0.75rem 1.5rem;
    font-weight: 500;
}

/* Cards and metrics */
[data-testid="metric-container"] {
    background: #1e293b;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    border: 1px solid #334155;
    color: #f1f5f9;
}

.stats-card {
    background: #1e293b;
    border-radius: 12px;
    padding: 1.5rem;
    border: 1px solid #334155;
    margin-bottom: 1rem;
    color: #f1f5f9;
}

/* Buttons */
.stButton>button {
    border-radius: 8px;
    border: 1px solid #e5e7eb;
    background: #0f172a;
    color: #ffffff;
    padding: 0.5rem 1rem;
    font-weight: 500;
}

/* Event cards */
.event-card {
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.5rem;
    background: #ffffff;
    border: 1px solid #e5e7eb;
}
.event-card.attack {
    border-left: 4px solid #ef4444;
}
.event-card.normal {
    border-left: 4px solid #22c55e;
}

.event-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
    flex-wrap: wrap;
    gap: 0.5rem;
}

.event-badge {
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.25rem 0.75rem;
    border-radius: 6px;
    background-color: #64748b;
    color: #ffffff;
}

/* Badges de détection */
.correct-badge { background: #10b981 !important; }
.perfect-badge { background: #8b5cf6 !important; }
.bad-badge { background: #ef4444 !important; }

/* Assistant section améliorée */
.assistant-section {
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #3b82f6;
}

/* Styles pour les questions personnalisées */
.stTextArea textarea {
    border-radius: 8px;
    border: 1px solid #d1d5db;
    font-family: 'Inter', sans-serif;
}

.stTextArea textarea:focus {
    border-color: #3b82f6;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1);
}

/* Style pour les boutons radio */
.stRadio > div {
    flex-direction: row !important;
    gap: 1rem;
}

.stRadio > div[role="radiogroup"] > label {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    margin-right: 0.5rem;
}

.stRadio > div[role="radiogroup"] > label[data-testid="stRadio"] {
    background: #f1f5f9;
}

/* Button rows */
.button-row {
    display: flex;
    gap: 0.5rem;
    margin: 0.5rem 0;
    flex-wrap: wrap;
}

/* Styles pour les tables améliorées */
.stDataFrame {
    border: 1px solid #e2e8f0;
    border-radius: 12px;
}

/* Style pour les boutons download améliorés */
.stDownloadButton button {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
    border: none !important;
    font-weight: 600 !important;
}

.stDownloadButton button:hover {
    background: linear-gradient(135deg, #1e293b 0%, #374151 100%) !important;
}

/* Style pour les dataframes améliorés */
.dataframe {
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

.dataframe thead tr {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
}

.dataframe thead th {
    color: white !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    padding: 1rem !important;
    border: none !important;
}

.dataframe tbody tr {
    transition: background-color 0.2s ease;
}

.dataframe tbody tr:nth-child(even) {
    background-color: #f8fafc !important;
}

.dataframe tbody tr:hover {
    background-color: #f1f5f9 !important;
}

.dataframe tbody td {
    padding: 0.75rem 1rem !important;
    font-size: 0.875rem !important;
    border-color: #e2e8f0 !important;
    color: #475569 !important;
}
</style>

<!-- HEADER HTML INTÉGRÉ DANS LE MÊME BLOC -->
<div class="dashboard-header">
    <h1>CyberMentor Dashboard</h1>
</div>
""",
    unsafe_allow_html=True,
)

# Load models
with st.spinner("Loading NLP model..."):
    tokenizer, model, device = load_model()

with st.spinner("Loading demo data..."):
    df_demo = load_demo_data()

with st.spinner("Loading RAG system..."):
    embed_model, rag_collection, attacks_by_type = load_rag()

with st.spinner("Loading LLM..."):
    tokenizer_llm, model_llm = load_llm()

# Prepare data
df_attacks = df_demo[df_demo["Label"] == 1].reset_index(drop=True)
df_normals = df_demo[df_demo["Label"] == 0].reset_index(drop=True)

# ================== TABS ==================

# Custom CSS for modern tabs
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f8fafc;
        padding: 8px;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid #e2e8f0;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background: white;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        color: #64748b;
        font-weight: 600;
        padding: 0 24px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #f1f5f9;
        color: #475569;
        border-color: #cbd5e1;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%) !important;
        color: white !important;
        border-color: #0f172a !important;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.2) !important;
        transform: translateY(-2px);
    }
    
    .stTabs [data-baseweb="tab-highlight"] {
        background: transparent !important;
    }
    
    /* Tab content styling */
    .tab-content {
        padding: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Create tabs with modern design
tab_sim, tab_history, tab_stats = st.tabs([
    " **Real-time Simulation**", 
    " **Incident History**", 
    " **Analytics Dashboard**"
])

# Add some spacing after tabs
st.markdown('<div class="tab-content"></div>', unsafe_allow_html=True)

# ================== TAB 1: REAL-TIME SIMULATION ==================

with tab_sim:
    st.subheader("Network Traffic Simulation")
    
    # DARKER Stats Cards
    col1, col2, col3, col4 = st.columns(4)
    
    total_events = len(st.session_state["events"])
    total_attacks = sum(1 for e in st.session_state["events"] if e["label"] == "Attack")
    total_normals = total_events - total_attacks
    attack_ratio = (total_attacks / total_events * 100) if total_events > 0 else 0
    
    with col1:
        render_stat_card("TOTAL EVENTS", total_events, "📊 All activities", "#3b82f6")
    
    with col2:
        render_stat_card("TOTAL ATTACKS", total_attacks, "🚨 Threats detected", "#ef4444")
    
    with col3:
        render_stat_card("NORMAL EVENTS", total_normals, "✅ Legitimate traffic", "#22c55e")
    
    with col4:
        render_stat_card("ATTACK RATIO", f"{attack_ratio:.1f}%", "📈 Threat percentage", "#8b5cf6")

    # Configuration Section
    st.markdown("---")
    st.subheader("⚙️ Simulation Configuration")
    
    config_col1, config_col2, config_col3 = st.columns(3)
    
    with config_col1:
        events_per_sec = st.number_input(
            " Events per second",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.5,
            help="Control the simulation speed"
        )
    
    with config_col2:
        attack_percent = st.slider(
            " Attack percentage (%)",
            min_value=0,
            max_value=100,
            value=50,
            step=5,
            help="Percentage of attack events in simulation"
        )
    
    with config_col3:
        max_events_display = st.number_input(
            " Max events displayed",
            min_value=10,
            max_value=500,
            value=50,
            step=10,
            help="Maximum number of events to show in real-time"
        )

    # Controls Section
    st.markdown("---")
    st.subheader("Simulation Controls")
    
    col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)
    
    with col_btn1:
        if st.button("▶️ Start Simulation", use_container_width=True, type="primary"):
            st.session_state["simulation_running"] = True
            st.session_state["last_sim_time"] = time.time()
            st.rerun()
    
    with col_btn2:
        if st.button("⏹️ Stop Simulation", use_container_width=True):
            st.session_state["simulation_running"] = False
            st.rerun()
    
    with col_btn3:
        if st.button("🔄 Reset Stats", use_container_width=True):
            st.session_state["events"] = []
            st.session_state["assistant_visible"] = {}
            st.session_state["assistant_questions"] = {}
            st.session_state["assistant_answers"] = {}
            st.session_state["custom_questions"] = {}
            st.rerun()
    
    with col_btn4:
        if st.button("📊 Export Data", use_container_width=True):
            if st.session_state["events"]:
                df_export = pd.DataFrame(st.session_state["events"])
                csv = df_export.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"cybermentor_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    # Simulation Status
    render_status_banner(st.session_state["simulation_running"])

    # Simulation logic
    if st.session_state["simulation_running"]:
        now = time.time()
        last = st.session_state["last_sim_time"] or now
        elapsed = now - last
        
        expected_events = int(elapsed * events_per_sec)
        
        if expected_events > 0:
            events_to_generate = min(expected_events, 10)
            
            for _ in range(events_to_generate):
                is_attack = random.random() < (attack_percent / 100.0)

                if is_attack and len(df_attacks) > 0:
                    row = df_attacks.sample(n=1).iloc[0]
                else:
                    row = df_normals.sample(n=1).iloc[0]

                text = f"{row['text_features']} {row.get('attack_context', '')}"
                true_label = int(row["Label"])
                true_multi = int(row.get("multi_class_label", 9))
                attack_cat = row.get("attack_cat", "unknown")

                pred_label, probs = predict(text, tokenizer, model, device)

                if pred_label == 9:
                    pred_binary = 0
                    predicted_label_text = "Normal"
                    predicted_attack_type = "Normal"
                else:
                    pred_binary = 1
                    predicted_label_text = "Attack"
                    predicted_attack_type = ATTACK_CLASS_MAP.get(pred_label, "Unknown")
                
                true_attack_cat = row.get("attack_cat", "unknown")

                event = {
                    "timestamp": datetime.now().isoformat(),
                    "label": "Attack" if true_label == 1 else "Normal",
                    "true_label": true_label,
                    "true_attack_cat": true_attack_cat,
                    "predicted_label": predicted_label_text,
                    "predicted_attack_type": predicted_attack_type,
                    "nlp_pred": int(pred_label),
                    "nlp_pred_binary": int(pred_binary),
                    "nlp_probs": probs.tolist(),
                    "attack_cat": attack_cat,
                    "srcip": row.get("srcip", "-"),
                    "dstip": row.get("dstip", "-"),
                    "dsport": str(row.get("dsport", "-")),
                    "proto": row.get("proto", "-"),
                    "service": row.get("service", "-"),
                    "row": row.to_dict(),
                    "true_multi_label": true_multi,
                    "text_features": strip_html(row["text_features"]),
                    "attack_context": strip_html(row.get("attack_context", "")),
                }
                st.session_state["events"].append(event)
            
            st.session_state["last_sim_time"] = now

    # Display events
    st.markdown("---")
    st.subheader("Live Events Stream")

    events = st.session_state.get("events", [])
    if not events:
        st.info("🎯 No events recorded yet. Start the simulation to see live events.")
    else:
        displayed_events = list(reversed(events[-max_events_display:]))

        for idx, ev in enumerate(displayed_events):
            event_key = f"{ev['timestamp']}_{ev['srcip']}_{ev['dstip']}"
            ts = datetime.fromisoformat(ev["timestamp"])
            time_str = ts.strftime("%H:%M:%S")
            is_attack = ev["nlp_pred_binary"] == 1
            attack_cat = ev["predicted_attack_type"]

            # Get detection badge
            badge_text, badge_class = get_detection_badge(
                ev["true_label"], 
                ev["nlp_pred_binary"], 
                ev["predicted_attack_type"],
                ev["true_attack_cat"]
            )

            card_type = "attack" if is_attack else "normal"
            type_badge = "ATTACK" if is_attack else "NORMAL"
            
            attack_cat_html = (
                f"<div class='event-attack-cat'>Attack category: <strong>{attack_cat}</strong></div>"
                if is_attack and attack_cat != "unknown"
                else ""
            )

            # Badge de détection
            detection_badge_html = (
                f"<span class='event-badge {badge_class}' style='margin-left: 0.5rem;'>{badge_text}</span>"
                if badge_text
                else ""
            )

            card_html = f"""
<div class="event-card {card_type}">
  <div class="event-header">
    <div style="display:flex;align-items:center;">
      <span class="event-badge">{type_badge}</span>
      {detection_badge_html}
    </div>
    <span class="event-time">{time_str}</span>
  </div>
  <div class="event-meta">
    <span>src: <strong>{ev['srcip']}</strong> → dst: <strong>{ev['dstip']}</strong></span>
    <span>port: <strong>{ev['dsport']}</strong></span>
    <span>proto: <strong>{ev.get('proto', '-')}</strong></span>
  </div>
  {attack_cat_html}
</div>
"""
            st.markdown(card_html, unsafe_allow_html=True)

            # Action buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if is_attack:
                    btn_text = " Details"
                else:
                    btn_text = "📋 Details"
                    
                if st.button(btn_text, key=f"det_{event_key}"):
                    with st.expander("🔍 Event Details", expanded=True):
                        show_event_details(ev["row"])
            
            with col2:
                if is_attack:
                    btn_text = " Model Performance"
                else:
                    btn_text = "📈 Model Performance"
                    
                if st.button(btn_text, key=f"perf_{event_key}"):
                    with st.expander("📊 Model Performance", expanded=True):
                        show_model_performance(ev)
            
            with col3:
                if is_attack:
                    if st.button(" Ask Assistant", key=f"ask_btn_{event_key}"):
                        st.session_state["assistant_visible"][event_key] = not st.session_state["assistant_visible"].get(event_key, False)
                        st.rerun()

            # NOUVELLE SECTION ASSISTANT AMÉLIORÉE
            if is_attack and st.session_state["assistant_visible"].get(event_key, False):
                st.markdown('<div class="assistant-section">', unsafe_allow_html=True)
                st.markdown("####  Cybersecurity Assistant")
                
                # Initialize session state for this event
                if event_key not in st.session_state["assistant_questions"]:
                    st.session_state["assistant_questions"][event_key] = ""
                if event_key not in st.session_state["custom_questions"]:
                    st.session_state["custom_questions"][event_key] = ""
                
                # Two options: predefined questions OR custom input
                option = st.radio(
                    "Choose question type:",
                    [" Use predefined question", " Ask custom question"],
                    key=f"option_{event_key}",
                    horizontal=True
                )
                
                current_question = ""
                
                if option == " Use predefined question":
                    # Predefined questions dropdown
                    predefined_questions = PREDEFINED_QUESTIONS.get(attack_cat, PREDEFINED_QUESTIONS["Generic"])
                    selected_question = st.selectbox(
                        "Select a question:",
                        options=["Choose a question..."] + predefined_questions,
                        key=f"predef_{event_key}",
                        label_visibility="visible"
                    )
                    
                    if selected_question and selected_question != "Choose a question...":
                        st.session_state["assistant_questions"][event_key] = selected_question
                        current_question = selected_question
                
                else:  # Custom question
                    custom_question = st.text_area(
                        "Ask your own question:",
                        placeholder=f"Ask anything about {attack_cat} attacks, mitigation strategies, detection methods...",
                        key=f"custom_{event_key}",
                        height=80
                    )
                    
                    if custom_question:
                        st.session_state["custom_questions"][event_key] = custom_question
                        current_question = custom_question
                
                # Ask button - only enabled if there's a question
                col_ask1, col_ask2 = st.columns([1, 4])
                with col_ask1:
                    ask_button = st.button(
                        " Get Answer", 
                        key=f"ask_{event_key}", 
                        use_container_width=True,
                        disabled=not current_question.strip(),  # Désactivé si pas de question
                        type="primary" if current_question.strip() else "secondary"
                    )
                
                # Question display
                if current_question:
                    st.markdown(f"**Question:** {current_question}")
                
                # Process the question when button is clicked
                if ask_button and current_question:
                    with st.spinner(" Searching knowledge base..."):
                        rag_results = search_rag_documents(
                            current_question, 
                            attack_cat, 
                            rag_collection, 
                            embed_model
                        )
                        
                        answer = generate_rag_answer(
                            current_question,
                            attack_cat,
                            rag_results,
                            tokenizer_llm,
                            model_llm
                        )
                        
                        # Store both question and answer
                        st.session_state["assistant_answers"][event_key] = {
                            "question": current_question,
                            "answer": answer,
                            "timestamp": datetime.now().isoformat()
                        }
                
                # Display answer
                if st.session_state["assistant_answers"].get(event_key):
                    answer_data = st.session_state["assistant_answers"][event_key]
                    st.markdown("---")
                    st.markdown("####  Assistant Answer")
                    
                    # Afficher la question posée
                    st.markdown(f"**Q:** {answer_data['question']}")
                    
                    # Afficher la réponse formatée
                    st.success(answer_data['answer'])
                    
                    # Option pour poser une nouvelle question
                    if st.button("🔄 Ask New Question", key=f"new_question_{event_key}"):
                        # Reset pour permettre une nouvelle question
                        st.session_state["assistant_questions"][event_key] = ""
                        st.session_state["custom_questions"][event_key] = ""
                        st.session_state["assistant_answers"][event_key] = None
                        st.rerun()
                
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("---")

    # Auto-refresh
    if st.session_state["simulation_running"]:
        time.sleep(0.5)
        st.rerun()

# ================== TAB 2: INCIDENT HISTORY ==================

with tab_history:
    st.subheader("Incident History & Analytics")
    
    if st.session_state["simulation_running"]:
        render_autorefresh_banner("🔄 Auto-refresh enabled - updates every 2 seconds")
    
    events = st.session_state.get("events", [])
    if not events:
        st.info("📭 No events recorded yet. Start the simulation to see incident history.")
    else:
        df_events = pd.DataFrame(events)
        df_events["timestamp"] = pd.to_datetime(df_events["timestamp"])

        # Modern Filter Card
        st.markdown(
            """
            <div style='
                background: #ffffff;
                border-radius: 16px;
                padding: 1.5rem;
                border: 1px solid #e2e8f0;
                margin-bottom: 2rem;
                box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            '>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown(
            """
            <div style='
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                border-radius: 12px;
                padding: 1rem 1.5rem;
                color: white;
                margin-bottom: 1.5rem;
                text-align: center;
                font-weight: 600;
            '>
                 Advanced Event Filtering
            </div>
            """,
            unsafe_allow_html=True
        )
        
        col_f1, col_f2, col_f3 = st.columns(3)

        with col_f1:
            label_filter = st.multiselect(
                "Filter by Event Type",
                options=sorted(df_events["label"].unique().tolist()),
                default=sorted(df_events["label"].unique().tolist()),
                help="Select event types to display"
            )

        with col_f2:
            attack_cats = sorted(
                [c for c in df_events["attack_cat"].dropna().unique().tolist() if c != "unknown"]
            )
            attack_cat_filter = st.multiselect(
                "Filter by Attack Category",
                options=attack_cats,
                default=attack_cats,
                help="Select specific attack categories"
            )

        with col_f3:
            period = st.selectbox(
                "Time Period Filter",
                ["All", "Last hour", "Last 24h", "Last 7 days"],
                index=0,
                help="Filter events by time range"
            )
        
        st.markdown("</div>", unsafe_allow_html=True)

        # Apply filters
        now = datetime.now()
        df_filtered = df_events.copy()

        if period == "Last hour":
            df_filtered = df_filtered[df_filtered["timestamp"] >= now - timedelta(hours=1)]
        elif period == "Last 24h":
            df_filtered = df_filtered[df_filtered["timestamp"] >= now - timedelta(hours=24)]
        elif period == "Last 7 days":
            df_filtered = df_filtered[df_filtered["timestamp"] >= now - timedelta(days=7)]

        if label_filter:
            df_filtered = df_filtered[df_filtered["label"].isin(label_filter)]

        if attack_cat_filter:
            df_filtered = df_filtered[df_filtered["attack_cat"].isin(attack_cat_filter)]

        # Results Summary with DARK Cards
        st.markdown("---")
        st.subheader(" Filter Results Summary")
        
        col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
        
        total_filtered = len(df_filtered)
        total_attacks_filtered = len(df_filtered[df_filtered["label"] == "Attack"])
        total_normals_filtered = len(df_filtered[df_filtered["label"] == "Normal"])
        coverage_percent = (total_filtered / len(df_events)) * 100 if len(df_events) > 0 else 0
        
        with col_sum1:
            render_stat_card("FILTERED EVENTS", total_filtered, "📊 Matching criteria", "#3b82f6")
        
        with col_sum2:
            render_stat_card("ATTACKS FILTERED", total_attacks_filtered, "🚨 Threats found", "#ef4444")
        
        with col_sum3:
            render_stat_card("NORMAL EVENTS", total_normals_filtered, "✅ Legitimate traffic", "#22c55e")
        
        with col_sum4:
            render_stat_card("DATA COVERAGE", f"{coverage_percent:.1f}%", "📈 Of total events", "#8b5cf6")

        # Modern Table Display AMÉLIORÉE
        st.markdown("---")
        st.subheader(" Incident Details Table")
        
        df_display = df_filtered[[
            "timestamp", "label", "attack_cat", "srcip", "dstip", "dsport", "proto"
        ]].sort_values("timestamp", ascending=False)
        
        # Format timestamp for better readability
        df_display["timestamp"] = df_display["timestamp"].dt.strftime("%m/%d %H:%M:%S")

        # Afficher le tableau avec configuration améliorée
        st.dataframe(
            df_display,
            use_container_width=True,
            height=400,
            column_config={
                "timestamp": "Timestamp",
                "label": "Type",
                "attack_cat": "Attack Category", 
                "srcip": "Source IP",
                "dstip": "Destination IP",
                "dsport": "Port",
                "proto": "Protocol"
            }
        )

        # Export Section AMÉLIORÉE
        st.markdown("---")
        st.subheader("📥 Data Export")

        col_exp1, col_exp2 = st.columns([1, 2])

        with col_exp1:
            # Bouton download avec meilleur design
            csv_data = df_filtered.to_csv(index=False).encode("utf-8")
            
            st.download_button(
                label="** Download CSV**",
                data=csv_data,
                file_name=f"cybermentor_incidents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary",
                help="Download filtered data as CSV file"
            )

        with col_exp2:
            st.markdown(f"""
            <div style='
                background: #f8fafc;
                border-radius: 8px;
                padding: 1rem;
                border-left: 4px solid #3b82f6;
            '>
                <div style='font-weight: 600; color: #0f172a; margin-bottom: 0.5rem;'>Export Information</div>
                <div style='font-size: 0.875rem; color: #64748b;'>
                • Contains <strong>{len(df_filtered)}</strong> filtered events<br>
                • File includes all event details and timestamps<br>
                • Compatible with Excel and data analysis tools
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Auto-refresh for history tab
    if st.session_state["simulation_running"]:
        time.sleep(2)
        st.rerun()

# ================== TAB 3: ANALYTICS DASHBOARD ==================

with tab_stats:
    st.subheader(" Analytics Dashboard")
    
    if st.session_state["simulation_running"]:
        render_autorefresh_banner("🔄 Auto-refresh enabled - updates every 3 seconds")
    
    events = st.session_state.get("events", [])
    
    if not events:
        st.info("📭 No events recorded yet. Start the simulation to see analytics.")
    else:
        df = pd.DataFrame(events)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Summary Statistics with DARK Cards
        st.markdown("### Summary Statistics")
        
        total_events = len(events)
        total_attacks = sum(1 for e in events if e['label'] == 'Attack')
        total_normals = total_events - total_attacks

        correct_predictions = 0
        for e in events:
            pred_bin = e['nlp_pred_binary']
            if e['true_label'] == pred_bin:
                correct_predictions += 1
        accuracy = correct_predictions / total_events if total_events > 0 else 0
        attack_ratio = (total_attacks / total_events * 100) if total_events > 0 else 0
        
        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
        
        with col_stat1:
            render_stat_card("TOTAL EVENTS", total_events, "📊 All activities", "#3b82f6")
        
        with col_stat2:
            render_stat_card("TOTAL ATTACKS", total_attacks, "🚨 Threats detected", "#ef4444")
        
        with col_stat3:
            render_stat_card("NORMAL EVENTS", total_normals, "✅ Legitimate traffic", "#22c55e")
        
        with col_stat4:
            render_stat_card("MODEL ACCURACY", f"{accuracy*100:.1f}%", "🤖 AI performance", "#8b5cf6")
        
        st.markdown("---")
        
        # ================== TRAFFIC OVERVIEW SECTION ==================
        with st.expander("📈 **Traffic Overview**", expanded=True):
            st.markdown("""
            <div style='
                background: #f8fafc;
                border-radius: 8px;
                padding: 1rem;
                margin-bottom: 1rem;
                border-left: 4px solid #3b82f6;
            '>
                <strong>📖 Readme:</strong> These charts show the overall distribution of network traffic between normal and attack events, 
                and break down the different types of attacks detected in your network.
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🚦 Traffic Distribution")
                
                label_counts = df['label'].value_counts()
                fig_traffic = px.pie(
                    values=label_counts.values,
                    names=label_counts.index,
                    color=label_counts.index,
                    color_discrete_map={'Normal': '#10b981', 'Attack': '#ef4444'},
                    hole=0.4
                )
                fig_traffic.update_traces(
                    textinfo='percent+label',
                    marker=dict(line=dict(color='#ffffff', width=2))
                )
                fig_traffic.update_layout(
                    showlegend=False, 
                    height=350,
                    margin=dict(t=50, b=50, l=50, r=50)
                )
                fig_traffic = style_plot(fig_traffic, theme="light")
                st.plotly_chart(fig_traffic, use_container_width=True)
                st.caption(" Shows the ratio of normal vs attack events across ALL network traffic")
            
            with col2:
                st.markdown("####  Attack Types Breakdown")
                
                attacks_df = df[df['label'] == 'Attack']
                if not attacks_df.empty:
                    attack_type_counts = attacks_df['attack_cat'].value_counts()
                    fig_attack_types = px.bar(
                        x=attack_type_counts.values,
                        y=attack_type_counts.index,
                        orientation='h',
                        color=attack_type_counts.values,
                        color_continuous_scale='reds'
                    )
                    fig_attack_types.update_layout(
                        xaxis_title="Number of Attacks",
                        yaxis_title="Attack Type",
                        height=350,
                        showlegend=False,
                    )
                    fig_attack_types.update_traces(
                        marker=dict(line=dict(color='rgba(0,0,0,0.1)', width=1))
                    )
                    fig_attack_types = style_plot(fig_attack_types, theme="light")
                    st.plotly_chart(fig_attack_types, use_container_width=True)
                    st.caption(" Displays different attack categories detected (ONLY attack events)")
                else:
                    st.info("📭 No attacks recorded yet")
        
        st.markdown("---")
        
        # ================== NETWORK ANALYSIS SECTION ==================
        with st.expander(" **Network Analysis**", expanded=False):
            st.markdown("""
            <div style='
                background: #f8fafc;
                border-radius: 8px;
                padding: 1rem;
                margin-bottom: 1rem;
                border-left: 4px solid #10b981;
            '>
                <strong>📖 Readme:</strong> These charts analyze network patterns including source/destination IPs, protocols, 
                and services. They show data from <strong>ALL events</strong> (both normal and attack traffic) to give you 
                a complete view of your network activity.
            </div>
            """, unsafe_allow_html=True)
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown("####  Top Source IPs")
                
                # ALL events (normal + attacks) for source IPs
                top_src_ips = df['srcip'].value_counts().head(10)
                
                if not top_src_ips.empty:
                    fig_src_ips = px.bar(
                        x=top_src_ips.values,
                        y=top_src_ips.index,
                        orientation='h',
                        color=top_src_ips.values,
                        color_continuous_scale='blues',
                        title=""
                    )
                    fig_src_ips.update_layout(
                        xaxis_title="Total Events",
                        yaxis_title="Source IP",
                        height=350,
                        showlegend=False,
                    )
                    fig_src_ips.update_traces(
                        marker=dict(line=dict(color='rgba(0,0,0,0.1)', width=1))
                    )
                    fig_src_ips = style_plot(fig_src_ips, theme="light")
                    st.plotly_chart(fig_src_ips, use_container_width=True)
                    st.caption(" Most active source IPs (ALL events - normal + attacks)")
                else:
                    st.info(" No source IP data available")
            
            with col4:
                st.markdown("####  Top Destination IPs")
                
                # ALL events (normal + attacks) for destination IPs
                top_dst_ips = df['dstip'].value_counts().head(10)
                
                if not top_dst_ips.empty:
                    fig_dst_ips = px.bar(
                        x=top_dst_ips.values,
                        y=top_dst_ips.index,
                        orientation='h',
                        color=top_dst_ips.values,
                        color_continuous_scale='greens',
                        title=""
                    )
                    fig_dst_ips.update_layout(
                        xaxis_title="Total Events",
                        yaxis_title="Destination IP",
                        height=350,
                        showlegend=False,
                    )
                    fig_dst_ips.update_traces(
                        marker=dict(line=dict(color='rgba(0,0,0,0.1)', width=1))
                    )
                    fig_dst_ips = style_plot(fig_dst_ips, theme="light")
                    st.plotly_chart(fig_dst_ips, use_container_width=True)
                    st.caption(" Most targeted destination IPs (ALL events - normal + attacks)")
                else:
                    st.info(" No destination IP data available")
            
            # Second row in Network Analysis
            col5, col6 = st.columns(2)
            
            with col5:
                st.markdown("####  Protocol Distribution")
                
                # ALL events for protocol distribution
                protocol_counts = df['proto'].value_counts()
                
                if not protocol_counts.empty:
                    # Map protocol codes to names
                    protocol_names = [PROTOCOL_MAPPING.get(str(proto), f"Proto {proto}") for proto in protocol_counts.index]
                    
                    fig_protocol = px.bar(
                        x=protocol_counts.values,
                        y=protocol_names,
                        orientation='h',
                        color=protocol_counts.values,
                        color_continuous_scale='purples',
                        title=""
                    )
                    fig_protocol.update_layout(
                        xaxis_title="Total Events",
                        yaxis_title="Protocol",
                        height=350,
                        showlegend=False,
                    )
                    fig_protocol.update_traces(
                        marker=dict(line=dict(color='rgba(0,0,0,0.1)', width=1))
                    )
                    fig_protocol = style_plot(fig_protocol, theme="light")
                    st.plotly_chart(fig_protocol, use_container_width=True)
                    st.caption(" Network protocols used (ALL events - normal + attacks)")
                else:
                    st.info(" No protocol data available")
            
            with col6:
                st.markdown("####  Top Targeted Services")
                
                # ALL events for services
                service_counts = df['service'].value_counts().head(10)
                
                if not service_counts.empty:
                    # Map service codes to names
                    service_names = [SERVICE_MAPPING.get(str(svc), f"Service {svc}") for svc in service_counts.index]
                    
                    fig_services = px.bar(
                        x=service_counts.values,
                        y=service_names,
                        orientation='h',
                        color=service_counts.values,
                        color_continuous_scale='oranges',
                        title=""
                    )
                    fig_services.update_layout(
                        height=350,
                        xaxis_title="Total Events",
                        yaxis_title="Service",
                    )
                    fig_services.update_traces(
                        marker=dict(line=dict(color='rgba(0,0,0,0.1)', width=1))
                    )
                    fig_services = style_plot(fig_services, theme="light")
                    st.plotly_chart(fig_services, use_container_width=True)
                    st.caption(" Network services accessed (ALL events - normal + attacks)")
                else:
                    st.info(" No service data available")
        
        st.markdown("---")
        
        # ================== SECURITY ANALYSIS SECTION ==================
        with st.expander(" **Security Analysis**", expanded=False):
            st.markdown("""
            <div style='
                background: #fef2f2;
                border-radius: 8px;
                padding: 1rem;
                margin-bottom: 1rem;
                border-left: 4px solid #ef4444;
            '>
                <strong>📖 Readme:</strong> These security-focused charts analyze <strong>ONLY attack events</strong> to help you 
                identify threats, vulnerable ports, and suspicious IP addresses. Use this information to strengthen 
                your security posture.
            </div>
            """, unsafe_allow_html=True)
            
            attacks_df = df[df['label'] == 'Attack']
            
            if not attacks_df.empty:
                col7, col8 = st.columns(2)
                
                with col7:
                    st.markdown("#### 🔌 Top Targeted Ports in Attacks")
                    
                    top_ports = attacks_df['dsport'].value_counts().head(10)
                    if not top_ports.empty:
                        fig_ports = px.bar(
                            x=top_ports.index,
                            y=top_ports.values,
                            color=top_ports.values,
                            color_continuous_scale='reds'
                        )
                        fig_ports.update_layout(
                            xaxis_title="Port Number",
                            yaxis_title="Number of Attacks",
                            height=350,
                        )
                        fig_ports.update_traces(
                            marker=dict(line=dict(color='rgba(0,0,0,0.1)', width=1))
                        )
                        fig_ports = style_plot(fig_ports, theme="light")
                        st.plotly_chart(fig_ports, use_container_width=True)
                        st.caption(" Ports most frequently targeted by attackers (ONLY attack events)")
                    else:
                        st.info(" No port attack data available")
                
                with col8:
                    st.markdown("####  Top Attack Sources")
                    
                    top_attack_src = attacks_df['srcip'].value_counts().head(10)
                    if not top_attack_src.empty:
                        fig_attack_src = px.bar(
                            x=top_attack_src.values,
                            y=top_attack_src.index,
                            orientation='h',
                            color=top_attack_src.values,
                            color_continuous_scale='reds',
                            title=""
                        )
                        fig_attack_src.update_layout(
                            xaxis_title="Number of Attacks",
                            yaxis_title="Source IP",
                            height=350,
                            showlegend=False,
                        )
                        fig_attack_src.update_traces(
                            marker=dict(line=dict(color='rgba(0,0,0,0.1)', width=1))
                        )
                        fig_attack_src = style_plot(fig_attack_src, theme="light")
                        st.plotly_chart(fig_attack_src, use_container_width=True)
                        st.caption(" IP addresses generating the most attacks")
                    else:
                        st.info(" No attack source data available")
                
                # TROISIÈME GRAPHIQUE SEUL (sans columns)
                st.markdown("####  Most Targeted Services in Attacks")
                
                attack_services = attacks_df['service'].value_counts().head(10)
                if not attack_services.empty:
                    # Map service codes to names
                    attack_service_names = [SERVICE_MAPPING.get(str(svc), f"Service {svc}") for svc in attack_services.index]
                    
                    fig_attack_services = px.bar(
                        x=attack_services.values,
                        y=attack_service_names,
                        orientation='h',
                        color=attack_services.values,
                        color_continuous_scale='oranges',
                        title=""
                    )
                    fig_attack_services.update_layout(
                        height=350,
                        xaxis_title="Number of Attacks",
                        yaxis_title="Service",
                    )
                    fig_attack_services.update_traces(
                        marker=dict(line=dict(color='rgba(0,0,0,0.1)', width=1))
                    )
                    fig_attack_services = style_plot(fig_attack_services, theme="light")
                    st.plotly_chart(fig_attack_services, use_container_width=True)
                    st.caption(" Services most frequently attacked")
                else:
                    st.info(" No service attack data available")
                    
            else:
                st.info(" No attacks recorded yet - Security analysis will appear here when attacks are detected")
        
        st.markdown("---")
        
        # ================== MODEL PERFORMANCE SECTION ==================
        with st.expander(" **Model Performance**", expanded=False):
            st.markdown("""
            <div style='
                background: #f0f9ff;
                border-radius: 8px;
                padding: 1rem;
                margin-bottom: 1rem;
                border-left: 4px solid #8b5cf6;
            '>
                <strong>📖 Readme:</strong> These metrics evaluate the performance of your AI detection model. 
                Monitor accuracy, precision, and recall to ensure your threat detection system is working effectively.
            </div>
            """, unsafe_allow_html=True)
            
            col_bin1, col_bin2 = st.columns(2)

            # Recompute confusion matrix using binary prediction
            true_positives = true_negatives = false_positives = false_negatives = 0

            for e in events:
                true_label = e['true_label']
                pred_bin = e['nlp_pred_binary']

                if true_label == 1 and pred_bin == 1:
                    true_positives += 1
                elif true_label == 0 and pred_bin == 0:
                    true_negatives += 1
                elif true_label == 0 and pred_bin == 1:
                    false_positives += 1
                elif true_label == 1 and pred_bin == 0:
                    false_negatives += 1

            total_events = len(events)
            correct_predictions = true_positives + true_negatives
            accuracy = correct_predictions / total_events if total_events > 0 else 0

            with col_bin1:
                st.markdown("####  Detection Accuracy")
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=accuracy * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Detection Accuracy (%)", 'font': {'size': 16}},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "#0f172a"},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 50], 'color': '#fee2e2'},
                            {'range': [50, 80], 'color': '#fef3c7'},
                            {'range': [80, 100], 'color': '#dcfce7'},
                        ],
                    }
                ))
                fig_gauge.update_layout(
                    height=350,
                    paper_bgcolor='rgba(0,0,0,0)',
                    font={'color': "#0f172a", 'family': "Arial"}
                )
                st.plotly_chart(fig_gauge, use_container_width=True)
                st.caption(" Overall accuracy of threat detection model")

            with col_bin2:
                st.markdown("####  Confusion Matrix")
                matrix_data = [[true_negatives, false_positives],
                               [false_negatives, true_positives]]

                fig_matrix = px.imshow(
                    matrix_data,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Normal', 'Attack'],
                    y=['Normal', 'Attack'],
                    color_continuous_scale='Blues',
                    aspect="auto",
                    text_auto=True,
                )
                fig_matrix.update_layout(
                    height=350,
                    xaxis_side="top",
                )
                fig_matrix = style_plot(fig_matrix, theme="light")
                st.plotly_chart(fig_matrix, use_container_width=True)
                st.caption(" Model performance breakdown: True/False Positives/Negatives")

                # Performance metrics
                col_met1, col_met2, col_met3 = st.columns(3)
                with col_met1:
                    st.metric(
                        "Correct Detections", 
                        f"{correct_predictions}/{total_events}",
                        delta=None
                    )
                with col_met2:
                    precision = (true_positives / (true_positives + false_positives)
                                 if (true_positives + false_positives) > 0 else 0)
                    st.metric(
                        "Precision", 
                        f"{precision*100:.1f}%" if precision > 0 else "N/A",
                        delta=None
                    )
                with col_met3:
                    recall = (true_positives / (true_positives + false_negatives)
                              if (true_positives + false_negatives) > 0 else 0)
                    st.metric(
                        "Recall", 
                        f"{recall*100:.1f}%" if recall > 0 else "N/A",
                        delta=None
                    )
        
    # Auto-refresh for statistics tab
    if st.session_state["simulation_running"]:
        time.sleep(3)
        st.rerun()