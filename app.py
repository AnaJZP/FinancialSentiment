import streamlit as st
from eda import (
    cargar_datos, 
    obtener_distribucion_sentimientos,
    generar_nube_palabras,
    obtener_ejemplos_texto,
    obtener_estadisticas,
    obtener_palabras_frecuentes,
    SENTIMENT_MAP
)

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Sentimientos Financieros",
    layout="wide",
    page_icon="📈"
)

# Título y descripción
st.title("Análisis Exploratorio de Datos - Financial Phrasebank")
st.write("Este dashboard muestra el análisis de sentimientos de textos financieros.")

# Información del autor
st.markdown("""
<div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px'>
    <h4>Desarrollado por: Ana Lorena Jiménez Preciado</h4>
    <p>Usa la barra lateral para navegar entre las diferentes páginas:</p>
    <ul>
        <li>📊 Página Principal - Análisis Exploratorio</li>
        <li>🤖 Modelo - Análisis de Sentimientos en Tiempo Real</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Cargar datos
with st.spinner("Cargando datos..."):
    df = cargar_datos()

# Estadísticas en tres columnas
col1, col2, col3 = st.columns(3)
stats = obtener_estadisticas(df)

with col1:
    st.metric("Total de Textos", f"{stats['total_textos']:,}")
with col2:
    st.metric("Longitud Promedio", f"{stats['longitud_promedio']:.1f} caracteres")
with col3:
    sentimiento_comun = max(stats['distribucion'].items(), key=lambda x: x[1])[0]
    st.metric("Sentimiento más común", sentimiento_comun)

# Distribución de sentimientos
st.plotly_chart(
    obtener_distribucion_sentimientos(df),
    use_container_width=True
)

# Análisis de palabras
st.subheader("Análisis de Palabras")
sentimiento_seleccionado = st.selectbox(
    "Seleccione un sentimiento para la nube de palabras y obtener la frecuencia:",
    ["Todos"] + list(SENTIMENT_MAP.values())
)

col_nube, col_freq = st.columns(2)

with col_nube:
    with st.spinner("Generando nube de palabras..."):
        fig_nube = generar_nube_palabras(df, sentimiento_seleccionado)
        st.plotly_chart(fig_nube, use_container_width=True)

with col_freq:
    with st.spinner("Generando gráfico de frecuencias..."):
        fig_freq = obtener_palabras_frecuentes(df, sentimiento_seleccionado)
        st.plotly_chart(fig_freq, use_container_width=True)

# Ejemplos de textos
st.subheader("Ejemplos de Textos por Sentimiento")
for sentimiento in SENTIMENT_MAP.values():
    with st.expander(f"Ver ejemplos de textos {sentimiento.lower()}s"):
        ejemplos = obtener_ejemplos_texto(df, sentimiento)
        for i, ejemplo in enumerate(ejemplos, 1):
            st.write(f"{i}. {ejemplo}")