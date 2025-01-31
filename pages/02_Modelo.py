import streamlit as st
import torch
from model import SentimentLSTM
from transformers import AutoTokenizer
from deep_translator import GoogleTranslator
import numpy as np
from eda import SENTIMENT_MAP, cargar_datos
import pandas as pd

st.set_page_config(page_title="Modelo de Sentimientos", layout="wide")



@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = SentimentLSTM(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=768,
        hidden_dim=512,
        n_layers=3,
        n_classes=3,
        dropout=0.3
    )
    if torch.cuda.is_available():
        model.load_state_dict(torch.load('best_model.pt'))
    else:
        model.load_state_dict(torch.load('best_model.pt', map_location=torch.device('cpu')))
    model.eval()
    return model, tokenizer

def translate_to_english(text):
    translator = GoogleTranslator(source='es', target='en')
    return translator.translate(text)

def predict_sentiment(text, model, tokenizer):
    english_text = translate_to_english(text)
    encoding = tokenizer(
        english_text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    with torch.no_grad():
        outputs = model(encoding['input_ids'], encoding['attention_mask'])
        probabilities = outputs.numpy()
        prediction = np.argmax(probabilities)
        confidence = float(probabilities[0][prediction]) * 100
    
    return SENTIMENT_MAP[prediction], confidence, english_text

st.title("Análisis de Sentimientos Financieros")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Información del Modelo")
    
    # Métricas de entrenamiento
    st.markdown("### Métricas de Validación")
    metrics_col1, metrics_col2 = st.columns(2)
    with metrics_col1:
        st.metric("Precisión de Validación", "89.2%")  # Valor del último epoch en train.py
        st.metric("Loss de Validación", "0.32")        # Valor del último epoch en train.py
    with metrics_col2:
        st.metric("Precisión de Entrenamiento", "91.5%")  # Valor del último epoch en train.py
        st.metric("Loss de Entrenamiento", "0.28")        # Valor del último epoch en train.py
    
    # Características del modelo
    st.markdown("### Arquitectura del Modelo")
    st.markdown("""
    - **Base**: LSTM Bidireccional con FinBERT
    - **Embeddings**: 768 dimensiones
    - **Capas Ocultas**: 512 unidades
    - **Capas LSTM**: 3
    - **Dropout**: 0.3
    
    ### Técnicas de Mejora
    - **Balanceo**: SMOTE para equilibrar clases
    - **Label Smoothing**: 0.1
    - **Early Stopping**: Paciencia de 5 épocas
    - **Optimizador**: AdamW con learning rate 2e-5
    - **Traducción**: Utiliza Google Translator para traducir textos en español al inglés
    
    ### Procesamiento de Texto
    - El modelo está optimizado para textos financieros en inglés
    - Los textos en español son automáticamente traducidos usando Google Translator
    - Se utiliza el tokenizador de FinBERT con longitud máxima de 128 tokens
    """)

with col2:
    st.subheader("Análisis de Texto")
    st.write("""
    Este modelo está entrenado con textos financieros en inglés del dataset Financial PhraseBank. 
    Para permitir el análisis de textos en español, se utiliza Google Translator para traducir 
    automáticamente el texto antes del análisis. Puedes ver la traducción utilizada expandiendo 
    la sección 'Ver traducción' después del análisis.
    """)
    st.write("### Ejemplos:")
    st.info('"En el tercer trimestre de 2010, las ventas netas aumentó un 5,2% a 205,5 millones EUR y el beneficio operativo un 34,9% a 23,5 millones EUR" (Positivo)')
    st.error('"La empresa reportó una pérdida neta de 150 millones debido a la caída en ventas" (Negativo)')
    st.warning('"La compañía no tiene planes de trasladar toda la producción a Rusia, aunque ahí es donde la empresa está creciendo" (Neutro)')
    
    text_input = st.text_area("Texto a analizar:", height=100)
    model, tokenizer = load_model()
    
    if st.button("Analizar Sentimiento"):
        if text_input:
            with st.spinner("Analizando sentimiento..."):
                sentiment, confidence, english_text = predict_sentiment(text_input, model, tokenizer)
                
                sentiment_colors = {
                    "Positivo": "green",
                    "Negativo": "red",
                    "Neutro": "orange"
                }
                
                st.markdown(f"### Sentimiento: <span style='color:{sentiment_colors[sentiment]}'>{sentiment}</span>", unsafe_allow_html=True)
                st.progress(confidence/100)
                st.markdown(f"**Confianza**: {confidence:.1f}%")
                
                with st.expander("Ver traducción"):
                    st.info(english_text)
        else:
            st.warning("Por favor, ingresa un texto para analizar.")

# Información del autor
st.markdown("""
<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px'>
    <h4>Desarrollado por: Ana Lorena Jiménez Preciado</h4>
</div>
""", unsafe_allow_html=True)
