"""
STREAMLIT APP - Image Captioning with Seq2Seq
Neural Storyteller: Generate captions for images using deep learning
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models, transforms

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Neural Storyteller - Image Captioning",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📸 Neural Storyteller: Image Captioning with Seq2Seq")
st.markdown("Generate natural language descriptions for images using deep learning")

# ============================================================================
# MODELS
# ============================================================================

class Encoder(nn.Module):
    def __init__(self, feature_size=2048, hidden_size=512, dropout=0.5):
        super(Encoder, self).__init__()
        self.fc = nn.Linear(feature_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, features):
        x = self.fc(features)
        x = self.layer_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(BahdanauAttention, self).__init__()
        self.W_encoder = nn.Linear(encoder_dim, attention_dim)
        self.W_decoder = nn.Linear(decoder_dim, attention_dim)
        self.V = nn.Linear(attention_dim, 1)
    
    def forward(self, encoder_out, decoder_hidden, encoder_proj=None):
        if encoder_proj is None:
            encoder_proj = self.W_encoder(encoder_out)
        decoder_proj = self.W_decoder(decoder_hidden).unsqueeze(1)
        scores = self.V(torch.tanh(encoder_proj + decoder_proj)).squeeze(2)
        alpha = F.softmax(scores, dim=1)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512, num_layers=2, dropout=0.5):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.attention = BahdanauAttention(hidden_size, hidden_size, hidden_size)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size + hidden_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size + hidden_size, vocab_size)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.init_h = nn.Linear(hidden_size, hidden_size)
        self.init_c = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
    
    def _init_hidden(self, encoder_out):
        h0 = torch.tanh(self.init_h(encoder_out)).unsqueeze(0).repeat(self.num_layers, 1, 1)
        c0 = torch.tanh(self.init_c(encoder_out)).unsqueeze(0).repeat(self.num_layers, 1, 1)
        return h0, c0
    
    def sample(self, encoder_out, max_len=20):
        """Greedy decoding"""
        batch_size = encoder_out.size(0)
        sampled_ids = []
        
        h, c = self._init_hidden(encoder_out)
        encoder_out_expanded = encoder_out.unsqueeze(1)
        encoder_proj = self.attention.W_encoder(encoder_out_expanded)
        
        current_input = torch.ones(batch_size, dtype=torch.long).to(encoder_out.device)
        
        for t in range(max_len):
            embeddings = self.embedding(current_input)
            context, alpha = self.attention(encoder_out_expanded, h[-1], encoder_proj)
            gate = torch.sigmoid(self.gate(h[-1]))
            gated_context = gate * context
            lstm_input = torch.cat([embeddings, gated_context], dim=1).unsqueeze(1)
            lstm_out, (h, c) = self.lstm(lstm_input, (h, c))
            output = self.fc(self.dropout(torch.cat([lstm_out.squeeze(1), context], dim=1)))
            predicted = output.argmax(1)
            sampled_ids.append(predicted.cpu().numpy())
            current_input = predicted
        
        return np.column_stack(sampled_ids)


# ============================================================================
# CACHING & INITIALIZATION
# ============================================================================

@st.cache_resource
def load_models():
    """Load encoder, decoder, and vocabulary"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load vocab
    with open('vocab.pkl', 'rb') as f:
        vocab_data = pickle.load(f)
    
    vocab_size = len(vocab_data['word2idx'])
    
    # Initialize models
    encoder = Encoder(feature_size=2048, hidden_size=512, dropout=0.5)
    decoder = Decoder(vocab_size=vocab_size, embed_size=256, hidden_size=512, num_layers=2, dropout=0.5)
    
    # Load checkpoint
    checkpoint = torch.load('best_model.pth', map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    encoder.to(device).eval()
    decoder.to(device).eval()
    
    return encoder, decoder, vocab_data, device


@st.cache_resource
def load_feature_extractor():
    """Load pre-trained ResNet50 for feature extraction"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model = nn.Sequential(*list(model.children())[:-1])
    model.to(device).eval()
    
    return model, device


@st.cache_resource
def load_training_history():
    """Load training history"""
    try:
        with open('training_history.json', 'r') as f:
            return json.load(f)
    except:
        return None


@st.cache_resource
def load_evaluation_metrics():
    """Load evaluation metrics"""
    try:
        import pandas as pd
        return pd.read_csv('evaluation_metrics.csv')
    except:
        return None


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_features(image, feature_extractor, device):
    """Extract features from image using ResNet50"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = feature_extractor(img_tensor).view(1, -1)
    
    return features


# ============================================================================
# CAPTION GENERATION
# ============================================================================

def generate_caption(encoder, decoder, features, vocab_data, device, max_len=20):
    """Generate caption from image features"""
    features = features.to(device)
    
    with torch.no_grad():
        encoded = encoder(features)
        sampled_ids = decoder.sample(encoded, max_len=max_len)
        sampled_ids = sampled_ids[0]
    
    idx2word = vocab_data['idx2word']
    caption = []
    
    for word_id in sampled_ids:
        word = idx2word.get(int(word_id), '<unk>')
        if word == '<end>':
            break
        if word not in ['<start>', '<pad>']:
            caption.append(word)
    
    return ' '.join(caption)


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.header("⚙️ Settings")
    
    max_caption_len = st.slider(
        "Max Caption Length",
        min_value=5,
        max_value=30,
        value=20,
        step=1
    )
    
    st.markdown("---")
    st.header("📊 Model Info")
    st.info("""
    **Architecture:**
    - Encoder: ResNet50 + Linear Layer
    - Decoder: LSTM with Bahdanau Attention
    - Loss: CrossEntropy with Label Smoothing
    - Optimizer: Adam
    
    **Training Details:**
    - Epochs: 20
    - Batch Size: 128
    - Learning Rate: 3e-4
    """)


# ============================================================================
# MAIN APP
# ============================================================================

# Load models
encoder, decoder, vocab_data, device = load_models()
feature_extractor, fe_device = load_feature_extractor()

# Main layout
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Generate Caption", "📈 Training Metrics", "🔍 Model Info", "ℹ️ About"])

# ============================================================================
# TAB 1: CAPTION GENERATION
# ============================================================================

with tab1:
    st.header("Generate Caption from Image")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Upload an image to generate a caption"
        )
    
    with col2:
        st.subheader("Or Use Example")
        use_example = st.checkbox("Use example image")
    
    if uploaded_file or use_example:
        # Load image
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
        else:
            # Create a sample image
            image = Image.new('RGB', (224, 224), color='blue')
        
        # Display image
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.markdown("### 📝 Generated Caption")
            
            # Extract features
            with st.spinner("🔄 Extracting features..."):
                features = extract_features(image, feature_extractor, fe_device)
            
            # Generate caption
            with st.spinner("🤖 Generating caption..."):
                caption = generate_caption(
                    encoder, decoder, features, vocab_data, device, max_len=max_caption_len
                )
            
            # Display caption
            st.success(f"✨ {caption}")
            
            # Copy button
            st.text_area(
                "Copy caption:",
                value=caption,
                height=100,
                disabled=False
            )


# ============================================================================
# TAB 2: TRAINING METRICS
# ============================================================================

with tab2:
    st.header("📊 Training Metrics")
    
    history = load_training_history()
    metrics_df = load_evaluation_metrics()
    
    if history:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Loss Curves")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(history['train_loss'], 'b-o', label='Training Loss', linewidth=2)
            ax.plot(history['val_loss'], 'r-s', label='Validation Loss', linewidth=2)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.subheader("Accuracy Curves")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(history['train_acc'], 'b-o', label='Training Accuracy', linewidth=2)
            ax.plot(history['val_acc'], 'r-s', label='Validation Accuracy', linewidth=2)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Accuracy (%)', fontsize=12)
            ax.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.warning("Training history not found")


# ============================================================================
# TAB 3: EVALUATION METRICS
# ============================================================================

with tab3:
    st.header("🔍 Model Evaluation")
    
    metrics_df = load_evaluation_metrics()
    
    if metrics_df is not None:
        st.subheader("Quantitative Evaluation Results")
        
        col1, col2, col3 = st.columns(3)
        
        # BLEU scores
        with col1:
            st.metric("BLEU-1", f"{metrics_df.loc[0, 'Score']:.4f}")
            st.metric("BLEU-2", f"{metrics_df.loc[1, 'Score']:.4f}")
            st.metric("BLEU-3", f"{metrics_df.loc[2, 'Score']:.4f}")
            st.metric("BLEU-4", f"{metrics_df.loc[3, 'Score']:.4f}")
        
        # Token-level metrics
        with col2:
            st.metric("Precision", f"{metrics_df.loc[4, 'Score']:.4f}")
            st.metric("Recall", f"{metrics_df.loc[5, 'Score']:.4f}")
            st.metric("F1-Score", f"{metrics_df.loc[6, 'Score']:.4f}")
        
        with col3:
            st.markdown("### Interpretation")
            st.info("""
            **BLEU Scores**: Measure n-gram overlap between generated and reference captions
            - BLEU-1: Unigram overlap
            - BLEU-4: 4-gram overlap (stricter)
            
            **Token Metrics**: Evaluate token-level accuracy
            - Precision: % of predicted tokens in reference
            - Recall: % of reference tokens predicted
            - F1: Harmonic mean of precision & recall
            """)
        
        # Full table
        st.subheader("Full Metrics Table")
        st.dataframe(metrics_df, use_container_width=True)
    else:
        st.warning("Evaluation metrics not found")


# ============================================================================
# TAB 4: ABOUT
# ============================================================================

with tab4:
    st.header("ℹ️ About Neural Storyteller")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🎯 Objective")
        st.markdown("""
        Build a multimodal deep learning model that generates natural language 
        descriptions for images using a Sequence-to-Sequence (Seq2Seq) architecture.
        """)
        
        st.subheader("🏗️ Architecture")
        st.markdown("""
        **Encoder:**
        - ResNet50 (pre-trained on ImageNet)
        - Extracts 2048-dim feature vectors
        - Linear projection to hidden size (512)
        
        **Decoder:**
        - LSTM (2 layers, 512 hidden units)
        - Bahdanau Attention mechanism
        - Word embeddings (256 dim)
        - Vocabulary: 7,727 tokens
        """)
    
    with col2:
        st.subheader("📚 Training Details")
        st.markdown("""
        **Dataset:** Flickr30k
        - 31,783 images
        - 158,915 captions
        - 80/10/10 train/val/test split
        
        **Training:**
        - Optimizer: Adam (lr=3e-4)
        - Loss: CrossEntropy with label smoothing (0.1)
        - Batch size: 128
        - Epochs: 20
        - Early stopping: patience=5
        
        **Hardware:**
        - GPU: NVIDIA T4 x2
        - Framework: PyTorch
        """)
    
    st.markdown("---")
    
    st.subheader("🚀 Key Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ✨ **Real-time Generation**
        - Instant caption generation
        - Greedy search decoding
        """)
    
    with col2:
        st.markdown("""
        📊 **Evaluation Metrics**
        - BLEU-1, 2, 3, 4
        - Precision, Recall, F1
        """)
    
    with col3:
        st.markdown("""
        🎨 **User-Friendly**
        - Upload any image
        - Instant results
        - Copy captions
        """)
    
    st.markdown("---")
    
    st.subheader("📖 How It Works")
    st.markdown("""
    1. **Feature Extraction**: Image is passed through pre-trained ResNet50 to extract 2048-dim vectors
    2. **Encoding**: Features are projected to hidden dimension (512) via encoder
    3. **Decoding**: LSTM with attention generates caption word-by-word
    4. **Attention**: Dynamically focuses on different image regions while generating
    5. **Output**: Natural language caption describing the image
    """)


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>💡 Neural Storyteller - Image Captioning with Seq2Seq</p>
        <p>Made with ❤️ using PyTorch & Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)
