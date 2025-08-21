"""
Chatbot Intents Topic Modeling with BERTopic
--------------------------------------------
Este script carga frases de un chatbot desde un Excel,
las limpia y aplica BERTopic para descubrir los temas principales
en las conversaciones.
"""

# ==============================
# 📌 1. Importación de librerías
# ==============================
import pandas as pd
import re

from bertopic import BERTopic
from hdbscan import HDBSCAN


# ==============================
# 📌 2. Cargar y filtrar datos
# ==============================
# Cargar el dataset desde Excel
df = pd.read_excel("IA.xlsx")

# Filtrar solo las filas con el intent de interés
df_filtrado = df[df["Nombre de Intent"] == "0.0. Enviar mensaje a LLM Default"]

print("Número de filas filtradas:", df_filtrado.shape)
print(df_filtrado.head())

# Extraer la columna con los textos de entrada
textos_entrada = df_filtrado["Texto de Entrada"].dropna().tolist()
print("Ejemplo de frases originales:", textos_entrada[:10])


# =======================================
# 📌 3. Definir frases irrelevantes a filtrar
# =======================================
frases_irrelevantes = {
    "sí", "ok", "vale", "gracias", "muchas gracias", "por favor", "hola",
    "buenos días", "buenas", "buenas tardes", "buenas noches",
    "de acuerdo", "está bien", "dale", "quiero ayuda"
}

def es_relevante(frase: str) -> bool:
    """
    Determina si una frase es relevante para el análisis.
    Filtra frases vacías, muy cortas o que sean saludos/cortesías.
    """
    frase_limpia = frase.strip().lower()
    if len(frase_limpia.split()) <= 2:
        return False
    if frase_limpia in frases_irrelevantes:
        return False
    return True


# =======================================
# 📌 4. Limpieza de textos
# =======================================
# 1. Eliminar frases con "event detection"
textos_filtrados = [
    texto for texto in textos_entrada
    if "event detection" not in texto.lower()
]

# 2. Función para normalización mínima
def limpiar_frase(texto: str) -> str:
    """
    Convierte a minúsculas y elimina signos de puntuación,
    manteniendo tildes y la letra ñ.
    """
    texto = texto.lower()
    texto = re.sub(r"[^\w\sáéíóúüñ]", "", texto)
    return texto

# 3. Aplicar limpieza
frases_limpias = [limpiar_frase(texto) for texto in textos_filtrados]


# =======================================
# 📌 5. Modelado de temas con BERTopic
# =======================================
# Crear un clusterizador HDBSCAN para reducir fragmentación
hdbscan_model = HDBSCAN(
    min_cluster_size=200,  # tamaño mínimo de cluster
    min_samples=1,
    metric="euclidean",
    prediction_data=True
)

# Crear modelo BERTopic multilingüe (soporta español)
modelo_topic = BERTopic(
    hdbscan_model=hdbscan_model,
    language="multilingual",
    verbose=True
)

# Entrenar modelo
temas, probabilidades = modelo_topic.fit_transform(frases_limpias)


# =======================================
# 📌 6. Resultados iniciales
# =======================================
# Información de los temas
df_temas = modelo_topic.get_topic_info()
print("Top 10 temas detectados:")
print(df_temas.head(10))

# Ejemplo de textos asignados al tema 1
indices_tema_1 = [i for i, t in enumerate(temas) if t == 1]
print("\nEjemplos de frases del tema 1:")
for i in indices_tema_1[:5]:
    print(f"- {frases_limpias[i]}")


# =======================================
# 📌 7. Visualización
# =======================================
# Visualización de todos los temas (mapa interactivo)
modelo_topic.visualize_topics()

# Distribución de frecuencias de los temas
modelo_topic.visualize_barchart(top_n_topics=42)


# =======================================
# 📌 8. Reducción del número de temas
# =======================================
modelo_topic_reducido = modelo_topic.reduce_topics(
    frases_limpias,
    nr_topics=20  # reducir a 20 temas principales
)

# Reasignar documentos con el modelo reducido
nuevo_temas, probabilidades = modelo_topic_reducido.transform(frases_limpias)

# Visualización de la reducción
modelo_topic_reducido.visualize_barchart(top_n_topics=20)

print("\n✅ Análisis de temas completado.")
