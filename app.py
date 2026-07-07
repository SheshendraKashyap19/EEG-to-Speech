import streamlit as st
import pickle
import torch
import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
from collections import defaultdict
from eegcnn import EEGcnn, PositionalEncoding
 
# ── Model class ──────────────────────────────────────────────
class EEGclassification(torch.nn.Module):
    def __init__(self, chans=122, timestamp=165, cls=3, dropout1=0.1,
                 dropout2=0.1, layer=0, pooling=None, size1=8, size2=8,
                 feel1=125, feel2=25):
        super().__init__()
        self.eegcnn = EEGcnn(Chans=chans, kernLength1=feel1, kernLength2=feel2,
                             F1=size1, D=size2, F2=size1*size2,
                             P1=2, P2=5, dropoutRate=dropout1)
        self.linear = torch.nn.Linear(
            timestamp*size1*size2 if pooling is None else size1*size2, cls)
        self.layer = layer
        self.pooling = pooling
        if self.layer > 0:
            self.poscode = PositionalEncoding(size1*size2, dropout=dropout2,
                                              max_len=timestamp)
            self.encoder = torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(
                    d_model=size1*size2, nhead=size1*size2//8,
                    dim_feedforward=4*size1*size2,
                    batch_first=True, dropout=dropout2),
                num_layers=self.layer)
 
    def forward(self, inputs, mask):
        hidden = self.eegcnn(inputs).permute(0, 2, 1)
        if self.layer > 0:
            hidden = self.poscode(hidden)
            hidden = self.encoder(hidden, src_key_padding_mask=(mask.bool()==False))
        if self.pooling is None:
            hidden = torch.flatten(hidden, start_dim=1)
        if self.pooling == "mean":
            hidden = torch.sum(hidden*mask.unsqueeze(2), dim=1) / \
                     torch.sum(mask, dim=1).unsqueeze(1)
        output = self.linear(hidden)
        return output
 
 
# ── Page config ───────────────────────────────────────────────
st.set_page_config(page_title="Chisco EEG Decoder", page_icon="🧠", layout="wide")
 
 
# ── Load textmaps ─────────────────────────────────────────────
@st.cache_resource
def load_textmaps():
    # Chinese textmaps
    with open("Chisco/json/textmaps.json", "r", encoding="utf-8") as f:
        textmaps_data = json.load(f)
    textmaps = defaultdict(lambda: -1, textmaps_data)
    reverse_map = defaultdict(list)
    for phrase, cat in textmaps_data.items():
        reverse_map[cat].append(phrase)
 
    # English textmaps (category -> english phrases)
    english_map = defaultdict(list)
    try:
        with open("Chisco/json/textmaps_english.json", "r", encoding="utf-8") as f:
            english_data = json.load(f)
        for phrase, cat in english_data.items():
            english_map[cat].append(phrase)
    except Exception:
        pass
 
    # Chinese -> English direct lookup
    zh_to_en = {}
    try:
        with open("Chisco/json/zh_to_en.json", "r", encoding="utf-8") as f:
            zh_to_en = json.load(f)
    except Exception:
        pass
 
    return textmaps, reverse_map, english_map, zh_to_en
 
 
# ── Find latest checkpoint ────────────────────────────────────
def get_latest_checkpoint():
    ckpt_dir = "checkpoint"
    if not os.path.exists(ckpt_dir):
        return None, None
    files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    if not files:
        return None, None
    files.sort(key=lambda x: int(x.replace("checkpoint-", "").replace(".pt", "")))
    latest = files[-1]
    step = latest.replace("checkpoint-", "").replace(".pt", "")
    return os.path.join(ckpt_dir, latest), step
 
 
# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    ckpt_path, step = get_latest_checkpoint()
    if ckpt_path is None:
        return None, None
    try:
        model = EEGclassification(
            chans=122, timestamp=165, cls=39,
            dropout1=0.5, dropout2=0.5,
            layer=0, pooling='mean',
            size1=8, size2=8,
            feel1=20, feel2=10
        )
        model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        model.eval()
        return model, step
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None, None
 
 
# ── SIDEBAR ───────────────────────────────────────────────────
st.sidebar.title("🧠 Chisco EEG Decoder")
st.sidebar.markdown("**Chinese Imagined Speech Decoding**")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["🏠 Home", "📂 Predict", "📊 Dashboard", "ℹ️ About"])
 
 
# ── HOME ──────────────────────────────────────────────────────
if page == "🏠 Home":
    st.title("🧠 Chisco EEG Imagined Speech Decoder")
    st.markdown("### Decode imagined Chinese speech from EEG brain signals")
    st.markdown("---")
 
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Subjects", "5")
    with col2:
        st.metric("Categories", "39")
    with col3:
        st.metric("EEG Channels", "122")
    with col4:
        ckpt_path, step = get_latest_checkpoint()
        st.metric("Latest Checkpoint", f"Step {step}" if step else "Not trained")
 
    st.markdown("---")
    st.markdown("""
    ### How it works
    1. **EEG signals** recorded while person imagines speaking Chinese phrases
    2. Signals preprocessed into **122 channels × 1651 timepoints**
    3. **CNN model** classifies signal into one of **39 semantic categories**
    4. Predicted category reveals what the person was **thinking about**
 
    ### Navigate using the sidebar
    - 📂 **Predict** — Upload EEG pkl file and get predictions
    - 📊 **Dashboard** — View training accuracy graphs
    - ℹ️ **About** — Project details and improvements
    """)
 
 
# ── PREDICT ───────────────────────────────────────────────────
elif page == "📂 Predict":
    st.title("📂 Upload EEG & Predict")
    st.markdown("---")
 
    textmaps, reverse_map, english_map, zh_to_en = load_textmaps()
    model, step = load_model()
 
    if model is None:
        st.warning("⚠️ No trained model checkpoint found in `checkpoint/` folder.")
        st.info("Please finish training first using EEGclassify.py")
    else:
        st.success(f"✅ Model loaded from checkpoint step {step}")
 
    uploaded_file = st.file_uploader("Upload EEG pkl file", type=["pkl"])
 
    if uploaded_file and model is not None:
        try:
            data = pickle.load(uploaded_file)
            st.success(f"✅ File loaded — **{len(data)} trials** found")
 
            trial_idx = st.slider("Select trial to predict", 0, len(data)-1, 0)
            trial = data[trial_idx]
 
            # EEG signal preview
            st.markdown("### 📈 EEG Signal Preview (first 10 channels)")
            eeg_preview = trial['input_features'][0, :10, :300] * 1000000
            fig = px.line(
                np.array(eeg_preview).T,
                title="EEG Signal (10 channels × 300 timepoints)",
                labels={"index": "Time", "value": "Amplitude (µV)", "variable": "Channel"}
            )
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
 
            # Actual label
            actual_text = trial['text'].strip()
            actual_category = textmaps[actual_text]
            actual_english = zh_to_en.get(actual_text, "No translation available")
 
            st.info(f"🇨🇳 Actual phrase: **{actual_text}**  |  🇬🇧 English: **{actual_english}**  |  Category: **{actual_category}**")
 
            # Predict button
            if st.button("🔮 Predict Category", type="primary"):
                input_features = trial['input_features'][0, :122, :] * 1000000
                input_features = np.float32(input_features)
                tensor = torch.tensor(input_features).unsqueeze(0)
                mask = torch.ones(1, 165)
 
                with torch.no_grad():
                    output = model(tensor, mask)
                    probs = torch.softmax(output, dim=1)[0]
                    predicted_cat = torch.argmax(probs).item()
                    confidence = probs[predicted_cat].item()
 
                st.markdown("### 🎯 Prediction Result")
                col1, col2, col3 = st.columns(3)
                with col1:
                    if predicted_cat == actual_category:
                        st.success(f"✅ Predicted: **Category {predicted_cat}**")
                    else:
                        st.error(f"❌ Predicted: **Category {predicted_cat}**")
                with col2:
                    st.metric("Actual Category", actual_category)
                with col3:
                    st.metric("Confidence", f"{confidence*100:.1f}%")
 
                # Top 5 bar chart
                top5 = torch.topk(probs, 5)
                fig2 = go.Figure(go.Bar(
                    x=[f"Cat {i.item()}" for i in top5.indices],
                    y=[v.item()*100 for v in top5.values],
                    marker_color=['green' if i.item()==actual_category
                                  else 'steelblue' for i in top5.indices],
                    text=[f"{v.item()*100:.1f}%" for v in top5.values],
                    textposition='auto'
                ))
                fig2.update_layout(
                    title="Top 5 Predicted Categories",
                    xaxis_title="Category",
                    yaxis_title="Confidence (%)",
                    height=350
                )
                st.plotly_chart(fig2, use_container_width=True)
 
                # Show Chinese and English phrases side by side
                st.markdown(f"### 💬 Phrases in Predicted Category {predicted_cat}")
                chinese_phrases = reverse_map.get(predicted_cat, [])[:5]
                english_phrases = english_map.get(predicted_cat, [])[:5]
 
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("🇨🇳 **Chinese**")
                    for p in chinese_phrases:
                        st.write(f"• {p}")
                with col2:
                    st.markdown("🇬🇧 **English**")
                    for p in english_phrases:
                        st.write(f"• {p}")
 
                # Show actual translation
                st.markdown("### 🔄 Translation Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"🇨🇳 **Actual:** {actual_text}\n\n🇬🇧 **English:** {actual_english}")
                with col2:
                    predicted_chinese = reverse_map.get(predicted_cat, ["Unknown"])[0]
                    predicted_english = zh_to_en.get(predicted_chinese, english_map.get(predicted_cat, ["Unknown"])[0] if english_map.get(predicted_cat) else "Unknown")
                    st.info(f"🎯 **Predicted phrase:** {predicted_chinese}\n\n🇬🇧 **English:** {predicted_english}")
 
                # Text to Speech
                st.markdown("### 🔊 Speech Output")
                try:
                    from gtts import gTTS
                    import tempfile
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Actual phrase:**")
                        tts1 = gTTS(text=actual_english, lang='en')
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
                            tts1.save(f.name)
                            st.audio(f.name, format='audio/mp3')
                    with col2:
                        st.markdown("**Predicted phrase:**")
                        tts2 = gTTS(text=predicted_english, lang='en')
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
                            tts2.save(f.name)
                            st.audio(f.name, format='audio/mp3')
                except Exception:
                    st.caption("Install gtts for audio: pip install gtts")
 
        except Exception as e:
            st.error(f"Error: {e}")
 
 
# ── DASHBOARD ─────────────────────────────────────────────────
elif page == "📊 Dashboard":
    st.title("📊 Training Results Dashboard")
    st.markdown("---")
 
    ckpt_dir = "checkpoint"
    if os.path.exists(ckpt_dir):
        files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
        if files:
            steps_available = sorted([int(f.replace("checkpoint-", "").replace(".pt", "")) for f in files])
            st.success(f"✅ Found {len(files)} checkpoints — latest step: {steps_available[-1]}")
 
    st.markdown("### Paste your training log here")
    st.caption("Copy accuracy lines from Anaconda Prompt and paste below")
 
    log_input = st.text_area("Training log", height=250,
        placeholder="step:100(epoch2 1/49) valid_loss:3.73 accuracy:0.005 max_accuracy:0.005 f1:0.001 max_f1:0.001")
 
    if log_input:
        import re
        steps, accs, max_accs, f1s = [], [], [], []
        for line in log_input.strip().split("\n"):
            s = re.search(r'step:(\d+)', line)
            a = re.search(r'accuracy:([\d.]+)', line)
            m = re.search(r'max_accuracy:([\d.]+)', line)
            f = re.search(r' f1:([\d.]+)', line)
            if s and a and m:
                steps.append(int(s.group(1)))
                accs.append(float(a.group(1))*100)
                max_accs.append(float(m.group(1))*100)
                if f:
                    f1s.append(float(f.group(1))*100)
 
        if steps:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Best Accuracy", f"{max(max_accs):.2f}%")
            with col2:
                st.metric("Latest Accuracy", f"{accs[-1]:.2f}%")
            with col3:
                st.metric("Steps Trained", steps[-1])
 
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=steps, y=accs, mode='lines+markers',
                name='Accuracy', line=dict(color='royalblue', width=2)))
            fig.add_trace(go.Scatter(x=steps, y=max_accs, mode='lines',
                name='Max Accuracy', line=dict(color='green', width=2, dash='dash')))
            if f1s:
                fig.add_trace(go.Scatter(x=steps, y=f1s, mode='lines+markers',
                    name='F1 Score', line=dict(color='orange', width=2)))
            fig.add_hline(y=14, line_dash="dot", line_color="red",
                         annotation_text="Paper baseline 14%")
            fig.update_layout(
                title="Accuracy & F1 vs Training Steps",
                xaxis_title="Step", yaxis_title="Score (%)", height=400)
            st.plotly_chart(fig, use_container_width=True)
 
    st.markdown("### 📊 Model Comparison Table")
    st.table({
        "Method": ["layer=1 (original README)", "layer=0 (fixed)", "Our result"],
        "Accuracy (%)": ["~5%", "~14%", "~11.7%"],
        "Notes": ["Wrong hyperparameter", "Correct setting", "7/45 files used"]
    })
 
 
# ── ABOUT ─────────────────────────────────────────────────────
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    ### Chisco: Chinese Imagined Speech EEG Decoding
 
    This project decodes **imagined speech from EEG brain signals** using deep learning.
 
    **Dataset:** OpenNeuro ds005170 — 5 subjects, 45 runs each, 39 semantic categories
 
    **Model:** CNN (EEGNet-style) + optional Transformer
 
    ---
    ### Bugs Fixed & Improvements Made
    | Issue | Fix |
    |---|---|
    | Missing `textmaps.json` | Downloaded from OpenNeuro `json/` folder |
    | Corrupted pkl files crash | Added try/except in `data_imagine.py` |
    | Windows UTF-8 encoding error | Added `encoding="utf-8"` |
    | Wrong hyperparameter in README | Found `--layer 0` fix from GitHub issues |
    | Chinese to English translation | Added NLP translation using Google Translate |
    | No interface | Built this Streamlit web app |
    | Speech output | Added gTTS text-to-speech audio |
 
    ---
    ### Tech Stack
    - Python 3.10 + PyTorch
    - Streamlit + Plotly
    - HuggingFace Transformers
    - Deep Translator (NLP)
    - gTTS (Text-to-Speech)
    """)