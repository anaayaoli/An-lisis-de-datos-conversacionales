# Chatbot Intents Topic Modeling with BERTopic

This project applies **topic modeling** techniques to chatbot user inputs extracted from an Excel file.  
The goal is to explore the main conversational themes and group semantically similar utterances.

---

## 🚀 Project Overview

1. **Data ingestion**  
   - Load an Excel file (`IA.xlsx`) containing chatbot interactions.  
   - Filter only the rows corresponding to a specific intent (`"0.0. Enviar mensaje a LLM Default"`).  

2. **Preprocessing**  
   - Remove irrelevant phrases (e.g., greetings, short answers like *sí*, *ok*, *hola*).  
   - Exclude technical logs (e.g., `"event detection"`).  
   - Normalize text: lowercasing and removing punctuation (while keeping accents and ñ).  

3. **Topic Modeling with BERTopic**  
   - Train a **BERTopic model** (multilingual embeddings for Spanish).  
   - Apply **HDBSCAN** as the clustering method to reduce fragmentation and create more coherent themes.  
   - Explore the most frequent topics and their associated keywords.  

4. **Visualization**  
   - Interactive topic map (`visualize_topics`).  
   - Bar chart of topic frequencies.  
   - Reduced topic set (e.g., top 20 main themes).  

---

## 🛠️ Tech Stack
- **Python** (pandas, re)
- **NLP**: BERTopic (multilingual embeddings), spaCy (optional for preprocessing)
- **Clustering**: HDBSCAN
- **Visualization**: BERTopic built-in visualizations (Plotly)

---

## 📊 Example Workflow
- Load Excel file with intents.  
- Clean and preprocess input phrases.  
- Apply BERTopic to discover latent themes.  
- Visualize results and interpret main topics.  

Sample outputs include:  
- Top 10 most frequent topics with representative keywords.  
- Example user inputs assigned to each topic.  
- Interactive topic visualizations.

---

## ⚙️ Installation

```bash
pip install pandas bertopic hdbscan
