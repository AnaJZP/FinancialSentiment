# Análisis de Sentimientos en Textos Financieros

Este proyecto implementa un modelo de análisis de sentimientos para textos financieros, utilizando un LSTM bidireccional con FinBERT. El modelo está especialmente diseñado para analizar textos financieros en español, realizando una traducción automática al inglés para su procesamiento.

## 🚀 Características

- Modelo LSTM Bidireccional con FinBERT
- Traducción automática español-inglés
- Dashboard interactivo con Streamlit
- Análisis exploratorio de datos
- Visualización de sentimientos y métricas

## 📊 Estructura del Proyecto

```
├── app.py                 # Dashboard principal de EDA
├── eda.py                 # Funciones de análisis exploratorio
├── model.py              # Definición del modelo LSTM
├── train.py              # Script de entrenamiento
├── pages/
│   └── 02_Modelo.py      # Página de predicción de sentimientos
├── requirements.txt      # Dependencias del proyecto
└── README.md            # Este archivo
```

## 🛠️ Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/AnaJZP/FinancialSentiment.git
cd Bourbaki_Analisis_Sentimiento
```

2. Crear y activar entorno virtual:
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## Nota sobre el modelo `best_model.pt`

El archivo `best_model.pt` no se encuentra en el repositorio porque excede el límite de tamaño permitido por GitHub. Para generar este archivo, sigue los siguientes pasos:

1. Corre el script de entrenamiento:
   ```bash
   python train.py


## 🖥️ Uso

1. Ejecutar la aplicación localmente:
```bash
streamlit run app.py
```

2. Acceder a la aplicación en el navegador:
```
http://localhost:8501
```

## 📝 Características del Modelo

- **Arquitectura**: LSTM Bidireccional con FinBERT
- **Embeddings**: 768 dimensiones
- **Capas Ocultas**: 512 unidades
- **Capas LSTM**: 3
- **Dropout**: 0.3

### Técnicas de Mejora
- Balanceo de clases con SMOTE
- Label Smoothing (0.1)
- Early Stopping (paciencia de 5 épocas)
- Optimizador AdamW (learning rate 2e-5)


## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, abre primero un issue para discutir los cambios que te gustaría hacer.

## ✍️ Autor

Ana Lorena Jiménez Preciado
