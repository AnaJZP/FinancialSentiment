import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datasets import load_dataset
from collections import Counter
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


#  NLTK
nltk.download("punkt_tab")

# Configuraciones globales
SENTIMENT_MAP = {
    0: "Negativo",
    1: "Neutro",
    2: "Positivo"
}

COLOR_MAP = {
    "Negativo": "#440154",  # Violeta oscuro
    "Neutro": "#21908C",    # Verde azulado
    "Positivo": "#FDE725"   # Amarillo
}

def descargar_datos(configuracion="sentences_allagree"):
    """Descarga los datos del dataset Financial PhraseBank."""
    datos = load_dataset("takala/financial_phrasebank", configuracion, split="train")
    return datos.to_pandas()

def cargar_datos():
    """Carga y prepara los datos iniciales."""
    df = descargar_datos()
    df.columns = ["texto", "etiqueta"]
    df["sentimiento"] = df["etiqueta"].map(SENTIMENT_MAP)
    return df

def obtener_distribucion_sentimientos(df):
    """Genera el gráfico de distribución de sentimientos."""
    conteo = df["sentimiento"].value_counts().reset_index()
    conteo.columns = ["Sentimiento", "Frecuencia"]
    
    fig = px.bar(
        conteo, 
        x="Sentimiento", 
        y="Frecuencia",
        color="Sentimiento",
        color_discrete_map=COLOR_MAP,
        text="Frecuencia",
        title="Distribución de Sentimientos en Textos Financieros"
    )
    
    fig.update_traces(
        textposition="outside",
        textfont=dict(size=14)
    )
    fig.update_layout(
        title_x=0.5,
        title_font_size=20,
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        margin=dict(t=60),
        showlegend=False
    )
    
    return fig

def limpiar_texto(texto):
    """Limpia y preprocesa el texto."""
    texto = texto.lower()
    texto = re.sub(r"[^a-zA-Záéíóúüñ\s]", "", texto)
    tokens = word_tokenize(texto)
    # Agregar palabras específicas a excluir
    stop_words = set(stopwords.words("english"))
    stop_words.update(["mn", "eur"])
    palabras_filtradas = [palabra for palabra in tokens if palabra not in stop_words]
    return " ".join(palabras_filtradas)

def obtener_palabras_frecuentes(df, sentimiento=None, n_palabras=15):
    """Obtiene las palabras más frecuentes para un sentimiento específico o todos."""
    if sentimiento and sentimiento != "Todos":
        textos = df[df["sentimiento"] == sentimiento]["texto"]
    else:
        textos = df["texto"]
    
    # Unir todos los textos y limpiarlos
    texto_completo = " ".join(textos.apply(limpiar_texto))
    palabras = texto_completo.split()
    
    # Contar frecuencias
    frecuencias = Counter(palabras)
    palabras_frecuentes = pd.DataFrame(
        frecuencias.most_common(n_palabras),
        columns=["Palabra", "Frecuencia"]
    )
    
    # Crear gráfico
    fig = px.bar(
        palabras_frecuentes,
        x="Palabra",
        y="Frecuencia",
        title=f"Palabras más frecuentes - {sentimiento if sentimiento else ''}",
        color="Frecuencia",
        color_continuous_scale="viridis"
    )
    
    fig.update_layout(
        xaxis_title="Palabra",
        yaxis_title="Frecuencia",
        xaxis_tickangle=-45,
        title_x=0.5,
        showlegend=False
    )
    
    return fig

def generar_nube_palabras(df, sentimiento=None):
    """Genera la nube de palabras para un sentimiento específico o todos."""
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    
    if sentimiento and sentimiento != "Todos":
        texto_completo = " ".join(df[df["sentimiento"] == sentimiento]["texto"].apply(limpiar_texto))
        titulo = f"Nube de Palabras - {sentimiento}"
    else:
        texto_completo = " ".join(df["texto"].apply(limpiar_texto))
        titulo = "Nube de Palabras"
    
    nube = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="viridis"
    ).generate(texto_completo)
    
    fig = go.Figure(go.Image(z=nube.to_array()))
    fig.update_layout(
        title=titulo,
        title_x=0.5,
        xaxis_visible=False,
        yaxis_visible=False
    )
    
    return fig

def obtener_ejemplos_texto(df, sentimiento):
    """Obtiene ejemplos de texto para un sentimiento específico."""
    return df[df["sentimiento"] == sentimiento]["texto"].head(3)

def obtener_estadisticas(df):
    """Obtiene estadísticas básicas del dataset."""
    stats = {
        "total_textos": len(df),
        "distribucion": df["sentimiento"].value_counts().to_dict(),
        "longitud_promedio": df["texto"].str.len().mean()
    }
    return stats