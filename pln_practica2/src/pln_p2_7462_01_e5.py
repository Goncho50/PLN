# -----------------------------------------------------------
# Ejercicio 5 - Evaluacion rigurosa con GridSearchCV
# PLN P2 - Polaridad en resenas BGG
# -----------------------------------------------------------

import json
import numpy as np
import pandas as pd

from scipy import sparse
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# Archivos generados en los ejercicios anteriores
X_TRAIN_FILE = "X_train.npz"
X_TEST_FILE  = "X_test.npz"
Y_TRAIN_FILE = "y_train.npy"
Y_TEST_FILE  = "y_test.npy"

TFIDF_FILE = "X_tfidf.npz"
FEATS_FILE = "X_feats.npz"

RESULTS_CSV = "resultados_e5.csv"
BEST_MODELS_JSON = "mejores_modelos_e5.json"

RANDOM_STATE = 42

# -----------------------------------------------------------
# Carga de matrices y etiquetas
# -----------------------------------------------------------

print("Cargando matrices de entrenamiento y test...")

X_train = sparse.load_npz(X_TRAIN_FILE)
X_test  = sparse.load_npz(X_TEST_FILE)

y_train = np.load(Y_TRAIN_FILE, allow_pickle=True)
y_test  = np.load(Y_TEST_FILE, allow_pickle=True)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# Dimensiones originales TF-IDF y features
X_tfidf_full = sparse.load_npz(TFIDF_FILE)
X_feats_full = sparse.load_npz(FEATS_FILE)

tfidf_dim = X_tfidf_full.shape[1]
feats_dim = X_feats_full.shape[1]

# -----------------------------------------------------------
# Separar TF-IDF y features
# -----------------------------------------------------------

print("Separando representaciones...")

X_tfidf_train = X_train[:, :tfidf_dim]
X_feats_train_raw = X_train[:, tfidf_dim:tfidf_dim + feats_dim]

X_tfidf_test = X_test[:, :tfidf_dim]
X_feats_test_raw = X_test[:, tfidf_dim:tfidf_dim + feats_dim]

# Escalar las features para que esten en rango 0-1
scaler = MinMaxScaler()
X_feats_train_scaled = scaler.fit_transform(X_feats_train_raw.toarray())
X_feats_test_scaled = scaler.transform(X_feats_test_raw.toarray())

X_feats_train_sparse = sparse.csr_matrix(X_feats_train_scaled)
X_feats_test_sparse = sparse.csr_matrix(X_feats_test_scaled)

# Matrices combinadas
X_comb_train = sparse.hstack([X_tfidf_train, X_feats_train_sparse])
X_comb_test  = sparse.hstack([X_tfidf_test,  X_feats_test_sparse])

representaciones = {
    "tfidf":    (X_tfidf_train, X_tfidf_test),
    "feats":    (X_feats_train_scaled, X_feats_test_scaled),
    "combined": (X_comb_train, X_comb_test)
}

# -----------------------------------------------------------
# Definicion de modelos y grids
# -----------------------------------------------------------

def get_model_and_grid(name):
    if name == "MultinomialNB":
        return MultinomialNB(), {
            "alpha": [0.1, 0.5, 1.0]
        }
    if name == "LinearSVC":
        return LinearSVC(random_state=RANDOM_STATE), {
            "C": [0.1, 1.0, 10.0]
        }
    if name == "RandomForest":
        return RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1), {
            "n_estimators": [200, 400],
            "max_depth": [None, 20, 40]
        }
    raise ValueError("Modelo no reconocido")


# -----------------------------------------------------------
# Grid Search completo
# -----------------------------------------------------------

resultados = []
mejores_modelos = []

for rep_name, (X_tr, X_te) in representaciones.items():
    print("\n======================================================")
    print("Representacion:", rep_name)
    print("======================================================")

    for model_name in ["MultinomialNB", "LinearSVC", "RandomForest"]:

        print("\nModelo:", model_name)

        model, grid_params = get_model_and_grid(model_name)

        grid = GridSearchCV(
            estimator=model,
            param_grid=grid_params,
            scoring="f1_weighted",
            cv=3,
            n_jobs=-1,
            verbose=0
        )

        grid.fit(X_tr, y_train)
        best_model = grid.best_estimator_

        print("Mejores parametros:", grid.best_params_)

        y_pred = best_model.predict(X_te)

        acc = accuracy_score(y_test, y_pred)
        f1_w = f1_score(y_test, y_pred, average="weighted")
        f1_m = f1_score(y_test, y_pred, average="macro")

        print("Accuracy: {:.3f} | F1_weighted: {:.3f}".format(acc, f1_w))

        cm = confusion_matrix(y_test, y_pred)
        print("Matriz de confusion:")
        print(cm)

        resultados.append({
            "representacion": rep_name,
            "modelo": model_name,
            "mejores_parametros": grid.best_params_,
            "accuracy": acc,
            "f1_macro": f1_m,
            "f1_weighted": f1_w
        })

        mejores_modelos.append({
            "representacion": rep_name,
            "modelo": model_name,
            "parametros": grid.best_params_,
            "confusion_matrix": cm.tolist()
        })

# -----------------------------------------------------------
# Guardar resultados
# -----------------------------------------------------------

pd.DataFrame(resultados).to_csv(RESULTS_CSV, index=False, encoding="utf-8")

with open(BEST_MODELS_JSON, "w", encoding="utf-8") as f:
    json.dump(mejores_modelos, f, indent=2, ensure_ascii=False)

print("\nEvaluacion completada correctamente.")
