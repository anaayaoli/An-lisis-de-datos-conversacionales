

import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from pysentimiento import create_analyzer
import matplotlib.pyplot as plt

# =========================
# 1. Cargar y limpiar datos
# =========================
df = pd.read_excel("IA.xlsx")

# Filtrar por el intent del modelo LLM
df_filtrado = df[df["Nombre de Intent"] == "0.0. Enviar mensaje a LLM Default"]

# Eliminar entradas vacías y las que contengan "from_zendesk_full"
textos = df_filtrado["Texto de Entrada"].dropna()
textos = textos[~textos.str.contains("from_zendesk_full", case=False)]

# Convertir a lista
frases = textos.tolist()

# Filtrar frases muy cortas o irrelevantes
frases_limpias = [
    f for f in frases
    if isinstance(f, str)
    and len(f.split()) > 2
    and not any(p in f.lower() for p in ["gracias", "hola", "sí", "ok", "por favor", "event"])
]

# ===============================
# 2. Modelado de tópicos (BERTopic)
# ===============================
hdbscan_model = HDBSCAN(
    min_cluster_size=200,
    min_samples=10,
    metric="euclidean",
    prediction_data=True
)

modelo_topic = BERTopic(
    hdbscan_model=hdbscan_model,
    language="multilingual",
    verbose=True
)

# Entrenar modelo
temas, probs = modelo_topic.fit_transform(frases_limpias)

# Reducir número de temas
modelo_topic_reducido = modelo_topic.reduce_topics(frases_limpias, nr_topics=20)

# Obtener info de temas
df_temas = modelo_topic_reducido.get_topic_info()

# Eliminar temas irrelevantes
temas_a_eliminar = []
palabras_irrelevantes = ["gracias", "hola", "ok", "por", "favor", "sí"]

for topic in df_temas["Topic"]:
    palabras_topico = [w[0] for w in modelo_topic_reducido.get_topic(topic)]
    if any(pal in palabras_topico for pal in palabras_irrelevantes):
        temas_a_eliminar.append(topic)

df_temas_limpios = df_temas[~df_temas.Topic.isin(temas_a_eliminar)]
df_temas_ordenados = df_temas_limpios.sort_values(by="Count", ascending=False)

# Exportar resultados
df_temas_ordenados.to_excel("temas_bot_limpios.xlsx", index=False)

# ===============================
# 3. Etiquetado manual de temas
# ===============================
etiquetas_manual = {
    -1: "Problemas generales con la cuenta",
     0: "Promociones de giros gratuitos",
     1: "Cambio de correo electrónico",
     2: "Peticiones de ayuda genéricas",
     3: "Reclamación por giros no entregados",
     4: "Problemas con bono de registro",
     5: "Dudas sobre apuestas deportivas o casino",
     6: "Preguntas sobre juegos y beneficios",
     7: "Problemas para retirar dinero",
     8: "Verificación de cuenta",
     9: "Dónde están mis giros / promociones",
    10: "Solicitud para hablar con un agente",
    11: "Consultas sobre promociones del día",
    12: "Bono de bienvenida / cómo usarlo",
    13: "Promoción de pronóstico sin riesgo",
    14: "Primer depósito / recarga inicial",
    15: "Reclamaciones por espera de 24-72h",
    16: "Promoción de cumpleaños",
    17: "Promoción Día del Padre",
    18: "Frases poco informativas"
}

df_temas["Etiqueta Manual"] = df_temas["Topic"].map(etiquetas_manual)
df_temas_ordenado = df_temas.sort_values(by="Count", ascending=False)
df_temas_ordenado.to_excel("temas_etiquetados.xlsx", index=False)

# ===============================
# 4. Análisis de Sentimientos
# ===============================
analyzer = create_analyzer(task="sentiment", lang="es")

resultados = [analyzer.predict(frase) for frase in frases_limpias]

sentimientos = [r.output for r in resultados]
probabilidades = [r.probas for r in resultados]

df_sentimientos = pd.DataFrame({
    "Frase": frases_limpias,
    "Sentimiento": sentimientos,
    "Probabilidad": [proba[max(proba, key=proba.get)] for proba in probabilidades]
})

# Exportar sentimientos
df_sentimientos.to_excel("analisis_sentimientos.xlsx", index=False)

# ===============================
# 5. Unión de resultados
# ===============================
df_temas_completo = modelo_topic_reducido.get_document_info(frases_limpias)
df_temas_completo = df_temas_completo[["Document", "Topic"]].rename(
    columns={"Document": "Frase", "Topic": "TemaReducido"}
)
df_temas_completo.to_excel("temas_bot_limpios_completo.xlsx", index=False)

df_temas = pd.read_excel("temas_bot_limpios_completo.xlsx")
df_sentimientos = pd.read_excel("analisis_sentimientos.xlsx")

# Merge
df_completo = pd.merge(df_temas, df_sentimientos, on="Frase", how="inner")

# Resumen por tema y sentimiento
resumen = df_completo.groupby(["TemaReducido", "Sentimiento"]).size().unstack(fill_value=0)
resumen_pct = resumen.div(resumen.sum(axis=1), axis=0) * 100
resumen_ordenado = resumen_pct.sort_values(by="NEG", ascending=False)

print("\n📊 Resumen de sentimientos por tema:\n")
print(resumen_ordenado.round(2))

# ===============================
# 6. Visualización
# ===============================
resumen_ordenado.head(10).plot(
    kind="barh",
    stacked=True,
    color=["red", "gray", "green"],
    figsize=(10, 6)
)
plt.xlabel("Porcentaje")
plt.ylabel("Tema Reducido")
plt.title("Distribución de Sentimientos por Tema Reducido")
plt.legend(title="Sentimiento", loc="lower right")
plt.tight_layout()
plt.show()
