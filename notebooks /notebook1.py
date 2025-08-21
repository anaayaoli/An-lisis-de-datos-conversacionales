# 📌 Sección 1: Importar librerías y cargar datos
import pandas as pd

# Cargar archivo Excel con los datos de intenciones
df = pd.read_excel("../data/IA.xlsx")

# Vista previa de las primeras filas
df.head()

# ---
# 📌 Sección 2: Filtrar datos
# Nos quedamos solo con las filas donde el intent es el especificado
df_filtrado = df[df["Nombre de Intent"] == "0.0. Enviar mensaje a LLM Default"]

print("Número de filas filtradas:", df_filtrado.shape)
df_filtrado.head()

# Asumimos que la columna de texto se llama "Texto" (ajustar si cambia)
textos_entrada = df_filtrado["Texto"].dropna().tolist()

# ---
# 📌 Sección 3: Preprocesamiento con spaCy
!pip install -q spacy
!python -m spacy download es_core_news_sm -q

import spacy
import string

# Cargar modelo en español
nlp = spacy.load("es_core_news_sm")

# Función de limpieza y lematización
def limpiar_y_lemmatizar(texto):
    doc = nlp(texto.lower())  # pasar a minúsculas y procesar con spaCy
    tokens_limpios = [
        token.lemma_ for token in doc
        if token.is_alpha and not token.is_stop
    ]
    return tokens_limpios

# Aplicamos la función a todos los textos
textos_procesados = [limpiar_y_lemmatizar(texto) for texto in textos_entrada]

# ---
# 📌 Sección 4: Filtrar palabras irrelevantes
palabras_excluir = {
    "event", "detection", "eh", "querer", "salir", "llegar", "q", "ok", "favor",
    "bienvenida", "ayudar", "necesitar", "gracias", "pasar", "ayer",
    "esperar", "dejar", "hola", "dar", "bastante", "hacer"
}

textos_filtrados = [
    [token for token in texto if token not in palabras_excluir]
    for texto in textos_procesados
]

print("Texto original:", textos_entrada[0])
print("Texto procesado:", textos_filtrados[0])

# ---
# 📌 Sección 5: Frecuencia de palabras
from itertools import chain
from collections import Counter

# Aplanar todos los tokens
tokens_totales = list(chain.from_iterable(textos_filtrados))

# Contar palabras
frecuencia_palabras = Counter(tokens_totales)

# Mostrar top 20 palabras
frecuencia_palabras.most_common(20)

# ---
# 📌 Sección 6: Visualización de frecuencias
import matplotlib.pyplot as plt

palabras, frecuencias = zip(*frecuencia_palabras.most_common(25))
plt.figure(figsize=(10,5))
plt.bar(palabras, frecuencias)
plt.xticks(rotation=45)
plt.title("Top 25 palabras más frecuentes")
plt.show()

# ---
# 📌 Sección 7: Topic Modeling con LDA
from sklearn.feature_extraction.text import CountVectorizer

# Unimos los tokens por documento
documentos = [" ".join(texto) for texto in textos_filtrados]

vectorizer = CountVectorizer(max_df=0.95, min_df=2)
X = vectorizer.fit_transform(documentos)

from sklearn.decomposition import LatentDirichletAllocation

lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(X)

# Función para mostrar palabras top por tema
def mostrar_top_palabras_por_tema(modelo, vectorizer, n_palabras=10):
    palabras = vectorizer.get_feature_names_out()
    for idx, tema in enumerate(modelo.components_):
        print(f"\n🔹 Tema {idx + 1}:")
        top_palabras = [palabras[i] for i in tema.argsort()[:-n_palabras - 1:-1]]
        print(", ".join(top_palabras))

mostrar_top_palabras_por_tema(lda, vectorizer)

# ---
# 📌 Sección 8: Visualización interactiva de temas
import pyLDAvis
import pyLDAvis.sklearn

pyLDAvis.enable_notebook()
pyLDAvis.sklearn.prepare(lda, X, vectorizer)
