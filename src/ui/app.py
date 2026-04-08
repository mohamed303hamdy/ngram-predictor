import streamlit as st
import os
from dotenv import load_dotenv
import sys 
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from src.model.ngram_model import NGramModel
from src.data_prep.normalizer import Normalizer
from src.inference.predictor import Predictor

# Load environment
load_dotenv("config/.env")

class PredictorUI:
    """Streamlit web interface for the next-word predictor."""
    
    def __init__(self):
        st.set_page_config(page_title="Sherlock Next-Word Predictor", page_icon="️")
        self.setup_styles()

    def setup_styles(self):
        st.markdown("""
            <style>
            .stButton>button { width: 100%; border-radius: 20px; }
            .prediction-box { padding: 10px; border: 1px solid #ddd; border-radius: 10px; }
            </style>
        """, unsafe_allow_html=True)

    @st.cache_resource
    def load_resources(_self):
        """Cache model and normalizer so they don't reload on every keystroke."""
        norm = Normalizer()
        n_order = int(os.getenv("NGRAM_ORDER", 4))
        model = NGramModel(n_order=n_order)
        model.load(os.getenv("MODEL"), os.getenv("VOCAB"))
        predictor = Predictor(model, norm, n_order)
        return predictor

    def run(self):
        st.title("️ Sherlock Holmes Predictor")
        st.write("Type a phrase below to see what Sherlock might say next.")
        
        predictor = self.load_resources()
        top_k = int(os.getenv("TOP_K", 3))
        
        user_text = st.text_input("Enter text:", placeholder="Elementary, my dear...")
        
        if user_text:
            predictions = predictor.predict_next(user_text, top_k)
            
            if predictions:
                st.subheader("Top Predictions:")
                cols = st.columns(len(predictions))
                for i, word in enumerate(predictions):
                    with cols[i]:
                        st.button(word, key=f"pred_{i}")
            else:
                st.info("No predictions found for this context.")

if __name__ == "__main__":
    ui = PredictorUI()
    ui.run()