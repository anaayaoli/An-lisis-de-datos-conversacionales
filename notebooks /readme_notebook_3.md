# Chatbot Intents Analysis

Este proyecto muestra un flujo completo de **análisis de intenciones en conversaciones de un chatbot**, desde la carga de datos hasta la detección de temas. Fue realizado durante mis prácticas como parte de un proyecto para un cliente real, por este motivo no se muestran resultados ni datos reales.  


---

## Objetivos
- Procesar y limpiar los textos de entrada de los usuarios.
- Generar estadísticas descriptivas (longitud de mensajes, palabras más frecuentes).
- Visualizar patrones lingüísticos con nubes de palabras y gráficos.
- Detectar temas recurrentes mediante **LDA (Latent Dirichlet Allocation)**.
- Documentar un flujo reproducible de análisis para bots conversacionales.

---

## Tecnologías utilizadas
- **Python 3.9+**
- **pandas**: manipulación de datos
- **spaCy**: preprocesamiento y tokenización en español
- **matplotlib** & **wordcloud**: visualización
- **gensim**: modelado de tópicos con LDA
- **pyLDAvis**: visualización interactiva de temas

---

## 📂 Estructura del proyecto

**1. Carga y preprocesamiento de datos**
- Se carga un archivo IA.xlsx con frases de usuarios.
- Se filtran únicamente los mensajes asociados al intent: "0.0. Enviar mensaje a LLM Default".
- Se eliminan entradas vacías, frases cortas, saludos y cortesías.

**2. Modelado de temas con BERTopic**
- Se usa HDBSCAN como clusterizador para reducir fragmentación.
- Se entrena un modelo BERTopic multilingüe.
- Se reducen los temas a un número manejable (nr_topics=20).
- Se exportan resultados a temas_bot_limpios.xlsx.
- Se asignan etiquetas manuales a cada tema detectado (ej. “Problemas con retiros”, “Promociones de giros”, etc.).

**3. Análisis de sentimientos con PySentimiento**
- Se utiliza el modelo RoBERTa en español para clasificar los mensajes en:

     - POS → Positivo
      
     - NEG → Negativo
      
     - NEU → Neutral
       
- Los resultados se exportan a analisis_sentimientos.xlsx.

**4. Integración de resultados**
- Se unen los temas y sentimientos en un único DataFrame.
- Se calculan porcentajes de distribución de sentimientos por tema.
- Se ordenan los temas por mayor porcentaje de negatividad.
- Se genera una visualización comparativa de los 10 temas con más feedback negativo.
