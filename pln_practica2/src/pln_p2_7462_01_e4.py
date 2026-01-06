# -----------------------------------------------------------
# Ejercicio 4 - Construccion y evaluacion de modelos de clasificacion
# PLN P2 - Clasificacion de polaridad en resenas
# -----------------------------------------------------------

import numpy as np
from scipy import sparse

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.preprocessing import MinMaxScaler

# Archivos generados en el ejercicio 3
X_TRAIN_FILE = "X_train.npz"
X_TEST_FILE = "X_test.npz"
Y_TRAIN_FILE = "y_train.npy"
Y_TEST_FILE = "y_test.npy"

# Archivos del ejercicio 2 para saber separacion TF-IDF / features
TFIDF_FILE = "X_tfidf.npz"
FEATS_FILE = "X_feats.npz"

# Funcion auxiliar para evaluar modelos
def evaluar_modelo(nombre, modelo, X_train, y_train, X_test, y_test):
    print("\nEntrenando modelo:", nombre)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print("Accuracy: {:.3f} | F1 weighted: {:.3f}".format(acc, f1))
    print("Matriz de confusion:")
    print(confusion_matrix(y_test, y_pred))
    print("\nInforme de clasificacion:")
    print(classification_report(y_test, y_pred, digits=3))


def main():

    # -------------------------------------------------------
    # 1. Cargar X_train / X_test / y_train / y_test
    # -------------------------------------------------------
    print("Cargando matrices de entrenamiento y test...")

    X_train = sparse.load_npz(X_TRAIN_FILE)
    X_test = sparse.load_npz(X_TEST_FILE)

    y_train = np.load(Y_TRAIN_FILE, allow_pickle=True)
    y_test = np.load(Y_TEST_FILE, allow_pickle=True)

    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    print("y_train:", y_train.shape)
    print("y_test:", y_test.shape)

    # -------------------------------------------------------
    # 2. Recuperar dimensiones originales TF-IDF y features
    # -------------------------------------------------------
    X_tfidf_full = sparse.load_npz(TFIDF_FILE)
    X_feats_full = sparse.load_npz(FEATS_FILE)

    tfidf_dim = X_tfidf_full.shape[1]
    feats_dim = X_feats_full.shape[1]

    print("\nDimensiones originales:")
    print("TF-IDF:", tfidf_dim)
    print("Features linguisticas:", feats_dim)

    # Separar cada parte segun las columnas
    X_tfidf_train = X_train[:, :tfidf_dim]
    X_feats_train_raw = X_train[:, tfidf_dim:tfidf_dim + feats_dim]

    X_tfidf_test = X_test[:, :tfidf_dim]
    X_feats_test_raw = X_test[:, tfidf_dim:tfidf_dim + feats_dim]

    print("\nSeparacion de representaciones:")
    print("X_tfidf_train:", X_tfidf_train.shape)
    print("X_feats_train_raw:", X_feats_train_raw.shape)
    print("X_tfidf_test:", X_tfidf_test.shape)
    print("X_feats_test_raw:", X_feats_test_raw.shape)

    # -------------------------------------------------------
    # 3. Escalado de features linguisticas (0-1)
    # -------------------------------------------------------
    scaler = MinMaxScaler()
    X_feats_train_scaled = scaler.fit_transform(X_feats_train_raw.toarray())
    X_feats_test_scaled = scaler.transform(X_feats_test_raw.toarray())

    # Se devuelven a formato sparse para combinar con TF-IDF
    X_feats_train_scaled_sparse = sparse.csr_matrix(X_feats_train_scaled)
    X_feats_test_scaled_sparse = sparse.csr_matrix(X_feats_test_scaled)

    # Matrices combinadas
    X_comb_train = sparse.hstack([X_tfidf_train, X_feats_train_scaled_sparse], format="csr")
    X_comb_test = sparse.hstack([X_tfidf_test, X_feats_test_scaled_sparse], format="csr")

    print("\nMatrices combinadas:")
    print("X_comb_train:", X_comb_train.shape)
    print("X_comb_test:", X_comb_test.shape)

    # -------------------------------------------------------
    # 4. Definir modelos a evaluar
    # -------------------------------------------------------
    modelos = {
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "SVM LinearSVC": LinearSVC(C=1.0, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
        ),
    }

    # -------------------------------------------------------
    # 5. Experimento 1 - Solo TF-IDF
    # -------------------------------------------------------
    print("\n===============================")
    print("EXPERIMENTO 1: Solo TF-IDF")
    print("===============================")

    for nombre, modelo in modelos.items():
        evaluar_modelo(nombre, modelo, X_tfidf_train, y_train, X_tfidf_test, y_test)

    # -------------------------------------------------------
    # 6. Experimento 2 - Solo features linguisticas
    # -------------------------------------------------------
    print("\n===============================")
    print("EXPERIMENTO 2: Solo features linguisticas")
    print("===============================")

    for nombre, modelo in modelos.items():
        evaluar_modelo(nombre, modelo, X_feats_train_scaled, y_train, X_feats_test_scaled, y_test)

    # -------------------------------------------------------
    # 7. Experimento 3 - TF-IDF + features linguisticas
    # -------------------------------------------------------
    print("\n===============================")
    print("EXPERIMENTO 3: TF-IDF + features linguisticas")
    print("===============================")

    for nombre, modelo in modelos.items():
        evaluar_modelo(nombre, modelo, X_comb_train, y_train, X_comb_test, y_test)

    print("\nEjercicio 4 completado correctamente.")


if __name__ == "__main__":
    main()
