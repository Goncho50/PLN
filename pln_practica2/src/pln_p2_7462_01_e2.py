# -----------------------------------------------------------
# Ejercicio 2 - Representaciones vectoriales: TF-IDF + features linguisticas
# PLN P2 - BoardGameGeek Reviews
# -----------------------------------------------------------

import json
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem import WordNetLemmatizer
import pickle

# Descarga de recursos necesarios
nltk.download("wordnet", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("omw-1.4", quiet=True)

# Configuracion de entradas y salidas
INPUT_FILE = "corpus_features_balanced.json"

OUTPUT_TFIDF = "X_tfidf.npz"
OUTPUT_FEATS = "X_feats.npz"
OUTPUT_TEXTS = "texts.json"
OUTPUT_VECTORIZER = "tfidf_vectorizer.pkl"

lemmatizer = WordNetLemmatizer()

# -----------------------------------------------------------
# 1. Carga de datos
# -----------------------------------------------------------

print("Cargando corpus balanceado...")
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

texts = []
ling_feats = []

for d in data:
    text = (d.get("text_clean") or d.get("text_raw") or "").strip()
    if not text:
        continue

    # Lematizacion ligera para reducir variabilidad
    tokens = nltk.word_tokenize(text.lower())
    lemmas = [lemmatizer.lemmatize(t) for t in tokens]
    text_lemmatized = " ".join(lemmas)
    texts.append(text_lemmatized)

    # Cargamos todas las caracteristicas linguisticas extraidas en el ejercicio 1
    ling_feats.append([
        d.get("sent_pos", 0.0),
        d.get("sent_neg", 0.0),
        d.get("sent_neu", 0.0),
        d.get("compound", 0.0),
        d.get("num_tokens", 0),
        d.get("num_sentences", 0),
        d.get("num_adj", 0),
        d.get("num_adv", 0),
        d.get("num_verbs", 0),
        d.get("num_pos_words", 0),
        d.get("num_neg_words", 0),
        d.get("sum_lex_polarity", 0.0),
        d.get("num_negations", 0),
        d.get("num_intensifiers", 0),
        d.get("num_domain_terms", 0),
    ])

print("Resenas procesadas:", len(texts))

# -----------------------------------------------------------
# 2. Generar matriz TF-IDF
# -----------------------------------------------------------

print("Generando matriz TF-IDF optimizada...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",    # elimina ruido de palabras vacias
    ngram_range=(1, 2),      # unigramas y bigramas
    min_df=3,                # elimina terminos muy raros
    max_df=0.80,             # elimina terminos demasiado frecuentes
    max_features=12000       # limite de vocabulario
)

X_tfidf = vectorizer.fit_transform(texts)
print("TF-IDF generado con dimensiones:", X_tfidf.shape)

# -----------------------------------------------------------
# 3. Convertir features linguisticos en matriz dispersa
# -----------------------------------------------------------

X_feats = np.array(ling_feats, dtype=float)
X_feats_sparse = sparse.csr_matrix(X_feats)

print("Matriz de features linguisticos:", X_feats_sparse.shape)

# -----------------------------------------------------------
# 4. Guardar resultados
# -----------------------------------------------------------

print("Guardando representaciones en disco...")

sparse.save_npz(OUTPUT_TFIDF, X_tfidf)
sparse.save_npz(OUTPUT_FEATS, X_feats_sparse)

with open(OUTPUT_TEXTS, "w", encoding="utf-8") as f:
    json.dump(texts, f, ensure_ascii=False, indent=2)

with open(OUTPUT_VECTORIZER, "wb") as f:
    pickle.dump(vectorizer, f)

print("Guardado en:", OUTPUT_TFIDF)
print("Guardado en:", OUTPUT_FEATS)
print("Guardado en:", OUTPUT_TEXTS)
print("Guardado en:", OUTPUT_VECTORIZER)

print("Ejercicio 2 completado correctamente.")
