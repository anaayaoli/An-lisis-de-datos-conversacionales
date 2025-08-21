# Chatbot Intents NLP Analysis

This project explores text data coming from chatbot conversations in order to 
extract **frequent words** and **main topics** using Natural Language Processing (NLP).

It is part of my portfolio to showcase skills in **linguistic preprocessing, topic modeling and conversational AI**.

---

## 🚀 Project Overview
1. **Data ingestion**: Load an Excel file with chatbot intents.
2. **Text preprocessing**:
   - Lowercasing
   - Tokenization
   - Lemmatization (spaCy, Spanish model)
   - Stopwords & irrelevant words removal
3. **Exploratory analysis**:
   - Word frequency counts
   - Top 25 most common words
4. **Topic modeling**:
   - Bag-of-Words representation
   - Latent Dirichlet Allocation (LDA) with 5 topics
   - Top words per topic
5. **Visualization**:
   - Word frequencies
   - Topic visualization with `pyLDAvis`

---

## 🛠️ Tech Stack
- **Python** (pandas, scikit-learn, spaCy)
- **NLP**: tokenization, lemmatization, stopword removal
- **Machine Learning**: Latent Dirichlet Allocation (topic modeling)
- **Visualization**: matplotlib, pyLDAvis

---

## ✨ What I learned
- How to clean and lemmatize Spanish texts with spaCy
- How to compute word frequency distributions
- How to apply LDA topic modeling to chatbot datasets
- How to combine linguistics + ML for practical use cases

