# =====================================================
# Script: Train Model
# Layer: FEATURES -> MODEL
# Author: Jean Bezerra
# =====================================================

import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import joblib

# =====================================================
# Paths
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]

FEATURES_PATH = BASE_DIR / "data" / "features"
MODELS_PATH = BASE_DIR / "models"

MODELS_PATH.mkdir(parents=True, exist_ok=True)

# =====================================================
# Configuração do modelo
# =====================================================

CONFIG = {
    "test_size": 0.2,
    "random_state": 42,

    "logistic_regression": {
        "max_iter": 5000,
        "solver": "lbfgs"
    },

    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "n_jobs": -1
    },

    # flag futura (não ativa agora)
    "use_gpu": False
}

# =====================================================
# Logging
# =====================================================

def log(level: str, step: str, message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] [{step}] {message}")

# =====================================================
# Load Data
# =====================================================

def load_data():
    path = FEATURES_PATH / "flights_features.parquet"

    log("INFO", "READ", f"Carregando dataset de features: {path}")

    df = pd.read_parquet(path)

    log("INFO", "READ", f"Dataset carregado com {len(df)} linhas")

    return df

# =====================================================
# Preprocessing
# =====================================================

def preprocess(df: pd.DataFrame):

    log("INFO", "PROCESS", "Separando target e features")

    y = df["IS_DELAYED"]

    X = df.drop(columns=["IS_DELAYED"])

    # =========================
    # Encoding categórico
    # =========================

    log("INFO", "PROCESS", "Aplicando encoding em variáveis categóricas")

    categorical_cols = X.select_dtypes(include=["string"]).columns

    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    return X, y, encoders

# =====================================================
# Train Models
# =====================================================

def train_models(X, y):

    log("INFO", "TRAIN", "Dividindo dados (train/test)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"],
        stratify=y
    )

    # =========================
    # Logistic Regression
    # =========================

    log("INFO", "TRAIN", "Treinando Logistic Regression")

    lr = LogisticRegression(
        max_iter=CONFIG["logistic_regression"]["max_iter"],
        solver=CONFIG["logistic_regression"]["solver"]
    )

    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    # =========================
    # Random Forest
    # =========================

    log("INFO", "TRAIN", "Treinando Random Forest")

    rf = RandomForestClassifier(
        n_estimators=CONFIG["random_forest"]["n_estimators"],
        max_depth=CONFIG["random_forest"]["max_depth"],
        n_jobs=CONFIG["random_forest"]["n_jobs"],
        random_state=CONFIG["random_state"]
    )

    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    # =========================
    # Avaliação
    # =========================

    def evaluate(y_true, y_pred, model_name):

        log("INFO", "EVAL", f"Avaliando modelo: {model_name}")

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred)
        }

        for k, v in metrics.items():
            log("INFO", "EVAL", f"{model_name} - {k}: {v:.4f}")

        return metrics

    metrics_lr = evaluate(y_test, y_pred_lr, "LogisticRegression")
    metrics_rf = evaluate(y_test, y_pred_rf, "RandomForest")

    return rf, lr, metrics_rf, metrics_lr

# =====================================================
# Save Model
# =====================================================

def save_model(model, encoders):

    model_path = MODELS_PATH / "model_rf.pkl"
    encoder_path = MODELS_PATH / "encoders.pkl"

    log("INFO", "WRITE", f"Salvando modelo em: {model_path}")

    joblib.dump(model, model_path)
    joblib.dump(encoders, encoder_path)

    log("SUCCESS", "WRITE", "Modelo salvo com sucesso")

# =====================================================
# Main
# =====================================================

def main():

    log("INFO", "PIPELINE", "Iniciando pipeline de treinamento")

    df = load_data()

    X, y, encoders = preprocess(df)

    model_rf, model_lr, metrics_rf, metrics_lr = train_models(X, y)

    save_model(model_rf, encoders)

    log("SUCCESS", "PIPELINE", "Treinamento finalizado com sucesso")

# =====================================================

if __name__ == "__main__":
    main()