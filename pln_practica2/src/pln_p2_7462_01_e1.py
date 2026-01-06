# -----------------------------------------------------------
# Ejercicio 1 - Extraccion de caracteristicas linguisticas
# PLN P2 - Analisis de polaridad en resenas de juegos de mesa
# -----------------------------------------------------------

import re
import json
import nltk
from collections import Counter
from nltk.sentiment import SentimentIntensityAnalyzer

# Descarga de recursos necesarios para NLTK
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# -----------------------------------------------------------
# Patrones linguisticos basados en expresiones regulares
# -----------------------------------------------------------

NEG_RE = re.compile(r"\b(?:not|no|never|none|hardly|scarcely|barely|without|lacks?|fail to)\b", re.I)

INT_RE = re.compile(
    r"\b(?:very|extremely|really|so|too|quite|fairly|highly|super|pretty|"
    r"sort of|kind of|barely|slightly|somewhat)\b",
    re.I
)

DOM_RE = re.compile(
    r"\b(?:rulebook|mechanic|mechanics|dice|component|components|miniatures?|"
    r"theme|replayability|downtime|setup|strategy|board|cards?|player|expansion)\b",
    re.I
)

# Inicializo VADER y obtengo su diccionario lexicon
sid = SentimentIntensityAnalyzer()
VADER_LEX = sid.lexicon


# -----------------------------------------------------------
# Funcion auxiliar para clasificar una resena segun rating
# -----------------------------------------------------------

def classify_rating(r):
    if r is None:
        return None
    if r >= 7:
        return "positive"
    elif r >= 4:
        return "neutral"
    else:
        return "negative"


# -----------------------------------------------------------
# Extraccion de caracteristicas de una resena
# -----------------------------------------------------------

def extract_features(text: str) -> dict:
    if not text or not text.strip():
        return None

    # Analisis global con VADER
    scores = sid.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        vader_label = "positive"
    elif compound <= -0.05:
        vader_label = "negative"
    else:
        vader_label = "neutral"

    # Tokenizacion y etiquetado PoS
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)

    num_tokens = len(tokens)
    num_sentences = len(nltk.sent_tokenize(text))

    # Conteo de categorias PoS
    num_adj = sum(1 for _, p in pos_tags if p.startswith("JJ"))
    num_adv = sum(1 for _, p in pos_tags if p.startswith("RB"))
    num_verbs = sum(1 for _, p in pos_tags if p.startswith("VB"))

    # Analisis de palabras con polaridad en el lexicon de VADER
    num_pos_words = 0
    num_neg_words = 0
    sum_lex_polarity = 0

    for t in tokens:
        w = t.lower()
        if w in VADER_LEX:
            v = VADER_LEX[w]
            sum_lex_polarity += v
            if v > 0.3:
                num_pos_words += 1
            elif v < -0.3:
                num_neg_words += 1

    # Patrones especificos: negaciones, intensificadores y terminos de dominio
    num_negations = len(NEG_RE.findall(text))
    num_intensifiers = len(INT_RE.findall(text))
    num_domain_terms = len(DOM_RE.findall(text))

    return {
        "sent_pos": scores["pos"],
        "sent_neg": scores["neg"],
        "sent_neu": scores["neu"],
        "compound": compound,
        "sent_label_vader": vader_label,
        "num_tokens": num_tokens,
        "num_sentences": num_sentences,
        "num_adj": num_adj,
        "num_adv": num_adv,
        "num_verbs": num_verbs,
        "num_pos_words": num_pos_words,
        "num_neg_words": num_neg_words,
        "sum_lex_polarity": sum_lex_polarity,
        "num_negations": num_negations,
        "num_intensifiers": num_intensifiers,
        "num_domain_terms": num_domain_terms,
    }


# -----------------------------------------------------------
# Fase 1: Preprocesar y guardar resenas validas
# -----------------------------------------------------------

def preprocess():
    INPUT_FILE = "corpus_total.json"
    OUTPUT_FILE = "corpus_features_only_text.json"

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned = []
    kept_reviews = 0

    for review in data:
        text = (review.get("text_raw") or review.get("text_clean") or "").strip()
        if not text:
            continue

        feats = extract_features(text)
        if feats:
            review.update(feats)
            cleaned.append(review)
            kept_reviews += 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)

    print("Preprocesado completado")
    print("Resenas con texto valido:", kept_reviews)
    print("Guardado en:", OUTPUT_FILE)


# -----------------------------------------------------------
# Fase 2: Balancear corpus
# -----------------------------------------------------------

def balance():
    INPUT_FILE = "corpus_features_only_text.json"
    OUTPUT_FILE = "corpus_features_balanced.json"

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    rating_counts = Counter()

    for review in data:
        r = review.get("rating")
        label = classify_rating(r)
        if label:
            rating_counts[label] += 1

    print("Distribucion original del corpus:")
    for k, v in rating_counts.items():
        print("-", k, ":", v)

    min_size = min(rating_counts.values())
    print("Se guardaran", min_size, "resenas por clase")

    balanced_data = []
    class_bucket = {"positive": 0, "neutral": 0, "negative": 0}

    for review in data:
        r = review.get("rating")
        label = classify_rating(r)

        if label and class_bucket[label] < min_size:
            balanced_data.append(review)
            class_bucket[label] += 1

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(balanced_data, f, ensure_ascii=False, indent=2)

    print("Distribucion balanceada final:")
    for k, v in class_bucket.items():
        print("-", k, ":", v)

    print("Resenas finales:", len(balanced_data))
    print("Guardado en:", OUTPUT_FILE)


# -----------------------------------------------------------
# Ejecucion del script
# -----------------------------------------------------------

if __name__ == "__main__":
    preprocess()
    balance()
