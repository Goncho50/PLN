# -----------------------------------------------------------
# Ejercicio 3 - Creacion de ficheros de entrenamiento y test
# PLN P2 - Particionado de matrices vectoriales
# -----------------------------------------------------------

import json
import numpy as np
import pandas as pd
from scipy import sparse


# Configuracion de archivos de entrada y salida
INPUT_CORPUS = "corpus_features_balanced.json"
INPUT_TFIDF = "X_tfidf.npz"
INPUT_FEATS = "X_feats.npz"

OUTPUT_X_TRAIN = "X_train.npz"
OUTPUT_X_TEST = "X_test.npz"
OUTPUT_Y_TRAIN = "y_train.npy"
OUTPUT_Y_TEST = "y_test.npy"

RANDOM_STATE = 42

# Funcion auxiliar que traduce ratings a etiquetas
def label_from_rating(r):
    if r >= 7:
        return "positive"
    elif r < 4:
        return "negative"
    else:
        return "neutral"


def main():

    print("Cargando corpus para generar etiquetas...")
    with open(INPUT_CORPUS, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    # Extraccion de etiquetas desde el rating
    y = np.array([label_from_rating(d["rating"]) for d in corpus])

    print("Distribucion de clases:")
    print(pd.Series(y).value_counts())

    # Cargar matrices vectoriales generadas en el ejercicio 2
    print("Cargando matrices TF-IDF y features linguisticos...")

    X_tfidf = sparse.load_npz(INPUT_TFIDF)
    X_feats = sparse.load_npz(INPUT_FEATS)

    print("Dimensiones X_tfidf:", X_tfidf.shape)
    print("Dimensiones X_feats:", X_feats.shape)

    # Concatenacion de TF-IDF y features
    print("Concatenando matrices...")
    X = sparse.hstack([X_tfidf, X_feats], format="csr")

    print("Matriz final X:", X.shape)

    # Division en train/test estratificada
    print("Dividiendo en train y test (80/20 estratificado)...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)

    # Guardado de los ficheros finales
    print("Guardando ficheros de entrenamiento y test...")

    sparse.save_npz(OUTPUT_X_TRAIN, X_train)
    sparse.save_npz(OUTPUT_X_TEST, X_test)

    np.save(OUTPUT_Y_TRAIN, y_train)
    np.save(OUTPUT_Y_TEST, y_test)

    print("Guardado en:", OUTPUT_X_TRAIN)
    print("Guardado en:", OUTPUT_X_TEST)
    print("Guardado en:", OUTPUT_Y_TRAIN)
    print("Guardado en:", OUTPUT_Y_TEST)

    print("Ejercicio 3 completado correctamente.")


if __name__ == "__main__":
    main()
