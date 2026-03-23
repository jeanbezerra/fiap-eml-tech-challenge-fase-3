# =====================================================
# Script: Modeling & Evaluation
# Author: Jean Bezerra (Refined Version - No Icons)
# =====================================================
import pandas as pd
import numpy as np
import os
import sys
import joblib
from datetime import datetime
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- Configuração de Ambiente ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "features", "flights_features.parquet")
LOG_DIR = os.path.join(BASE_DIR, "log")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

CLASSIFICATION_THRESHOLD = 0.30
VALIDATION_SAMPLE_SIZE = 50000

# --- Funções de Utilidade (ISO 8601) ---
def iso8601_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")

def iso8601_filename_timestamp() -> str:
    return iso8601_timestamp().replace(":", "-")

LOG_FILE = os.path.join(LOG_DIR, f"audit_modeling_{iso8601_filename_timestamp()}.log")

def audit_log(message):
    entry = f"[{iso8601_timestamp()}] {message}"
    print(entry)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(entry + "\n")


def positive_rate(series) -> float:
    return float(series.mean())


def diagnose_fit(train_f1: float, test_f1: float, test_recall: float) -> str:
    gap = train_f1 - test_f1

    if train_f1 >= 0.80 and gap > 0.15:
        return "OVERFITTING"
    if train_f1 < 0.60 and test_f1 < 0.60:
        return "UNDERFITTING"
    if test_recall < 0.50 and gap <= 0.15:
        return "UNDERFITTING"
    if gap <= 0.10 and test_f1 >= 0.60:
        return "AJUSTE_EQUILIBRADO"
    return "AJUSTE_INTERMEDIARIO"


def drop_compatibility_aliases(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    alias_pairs = {
        "CITY": "ORIGIN_CITY",
        "STATE": "ORIGIN_STATE",
    }
    dropped_columns = []
    model_frame = df.copy()

    for alias_column, canonical_column in alias_pairs.items():
        if alias_column not in model_frame.columns or canonical_column not in model_frame.columns:
            continue

        alias_values = model_frame[alias_column].astype("string").fillna("UNKNOWN")
        canonical_values = model_frame[canonical_column].astype("string").fillna("UNKNOWN")

        if alias_values.equals(canonical_values):
            model_frame = model_frame.drop(columns=[alias_column])
            dropped_columns.append(alias_column)

    return model_frame, dropped_columns


def fit_frequency_encoder(df: pd.DataFrame, categorical_columns: list[str]) -> dict[str, pd.Series]:
    encoder = {}
    row_count = len(df)

    for column in categorical_columns:
        frequencies = df[column].astype("string").fillna("UNKNOWN").value_counts(dropna=False) / row_count
        encoder[column] = frequencies.astype("float32")

    return encoder


def transform_frequency_encoded(
    df: pd.DataFrame,
    numeric_columns: list[str],
    categorical_columns: list[str],
    encoder: dict[str, pd.Series],
) -> pd.DataFrame:
    transformed = df[numeric_columns].copy()

    for column in categorical_columns:
        mapped = (
            df[column]
            .astype("string")
            .fillna("UNKNOWN")
            .map(encoder[column])
            .fillna(0)
            .astype("float32")
        )
        transformed[f"{column}_FREQ"] = mapped

    return transformed


def format_log_table(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    widths = {
        column: max(len(str(column)), df[column].astype(str).map(len).max() if len(df) else 0)
        for column in columns
    }

    header = " | ".join(f"{column:<{widths[column]}}" for column in columns)
    separator = "-+-".join("-" * widths[column] for column in columns)
    rows = [
        " | ".join(f"{str(row[column]):<{widths[column]}}" for column in columns)
        for _, row in df.iterrows()
    ]

    return "\n".join([header, separator] + rows)


def calculate_vif_table(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = df.replace([np.inf, -np.inf], np.nan).fillna(0).astype("float64")
    variance_mask = clean_df.var() > 0
    clean_df = clean_df.loc[:, variance_mask]

    if clean_df.shape[1] < 2:
        return pd.DataFrame([{"feature": "N/A", "value": "N/A", "status": "INSUFFICIENT_FEATURES"}])

    corr = clean_df.corr().fillna(0)
    corr_values = corr.to_numpy(copy=True)
    corr_values += np.eye(corr_values.shape[0]) * 1e-6
    inv_corr = np.linalg.pinv(corr_values)
    vif_values = np.diag(inv_corr)

    vif_df = pd.DataFrame({
        "feature": clean_df.columns,
        "value": np.round(vif_values, 4),
    }).sort_values("value", ascending=False).head(10)

    vif_df["status"] = np.where(
        vif_df["value"] >= 10, "HIGH",
        np.where(vif_df["value"] >= 5, "MODERATE", "LOW")
    )

    return vif_df


def calculate_ks_drift_table(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for column in train_df.columns:
        stat, pvalue = ks_2samp(train_df[column], test_df[column])
        rows.append({
            "feature": column,
            "value": f"stat={stat:.4f}; p={pvalue:.4g}",
            "status": "DRIFT" if pvalue < 0.05 else "STABLE",
        })

    return pd.DataFrame(rows).sort_values("status", ascending=False).head(10)


def calculate_psi(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    expected = pd.to_numeric(expected, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    actual = pd.to_numeric(actual, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)

    breakpoints = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(breakpoints) < 3:
        return 0.0

    expected_bins = pd.cut(expected, bins=breakpoints, include_lowest=True)
    actual_bins = pd.cut(actual, bins=breakpoints, include_lowest=True)

    expected_pct = expected_bins.value_counts(normalize=True, sort=False).replace(0, 1e-6)
    actual_pct = actual_bins.value_counts(normalize=True, sort=False).replace(0, 1e-6)

    psi = ((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)).sum()
    return float(psi)


def calculate_psi_drift_table(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for column in train_df.columns:
        psi_value = calculate_psi(train_df[column], test_df[column])
        status = "HIGH_DRIFT" if psi_value >= 0.25 else "MODERATE_DRIFT" if psi_value >= 0.10 else "LOW_DRIFT"
        rows.append({
            "feature": column,
            "value": f"{psi_value:.4f}",
            "status": status,
        })

    return pd.DataFrame(rows).sort_values("value", ascending=False).head(10)


def calculate_isolation_forest_table(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    sample_size = min(VALIDATION_SAMPLE_SIZE, len(train_df), len(test_df))
    train_sample = train_df.sample(n=sample_size, random_state=42)
    test_sample = test_df.sample(n=sample_size, random_state=42)

    model = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train_sample)

    train_pred = model.predict(train_sample)
    test_pred = model.predict(test_sample)

    train_anomaly_rate = float((train_pred == -1).mean())
    test_anomaly_rate = float((test_pred == -1).mean())
    delta = test_anomaly_rate - train_anomaly_rate

    rows = [
        {
            "feature": "train_anomaly_rate",
            "value": f"{train_anomaly_rate:.4f}",
            "status": "BASELINE",
        },
        {
            "feature": "test_anomaly_rate",
            "value": f"{test_anomaly_rate:.4f}",
            "status": "HIGH" if test_anomaly_rate >= 0.15 else "MODERATE" if test_anomaly_rate >= 0.05 else "LOW",
        },
        {
            "feature": "delta_test_vs_train",
            "value": f"{delta:.4f}",
            "status": "SHIFT" if delta >= 0.05 else "STABLE",
        },
    ]

    return pd.DataFrame(rows)

# ==============================
# 1. CARREGAR E PREPARAR DADOS
# ==============================
audit_log(f"Iniciando Pipeline. Arquivo: {DATA_PATH}")

if not os.path.exists(DATA_PATH):
    audit_log("ERRO: Dataset nao encontrado.")
    sys.exit(1)

df_raw = pd.read_parquet(DATA_PATH)
audit_log(f"Dados carregados: linhas={len(df_raw)} | colunas={len(df_raw.columns)}")
df = df_raw.sample(min(500000, len(df_raw)), random_state=42).copy()
audit_log(f"Amostra utilizada: linhas={len(df)} | colunas={len(df.columns)} | seed=42")

# Preparacao do Target
df["TARGET_DELAY"] = df["IS_DELAYED"].astype(int)
feature_frame = df.drop(columns=["IS_DELAYED", "TARGET_DELAY"], errors="ignore")
feature_frame, dropped_alias_columns = drop_compatibility_aliases(feature_frame)
numeric_columns = feature_frame.select_dtypes(include=[np.number]).columns.tolist()
categorical_columns = feature_frame.select_dtypes(exclude=[np.number]).columns.tolist()
X = feature_frame.copy()
y = df["TARGET_DELAY"]

# Split com estratificacao para manter proporcao de classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

audit_log(
    f"Setup concluido. Features numericas={len(numeric_columns)} | categoricas={len(categorical_columns)} | Threshold={CLASSIFICATION_THRESHOLD:.2f}"
)
audit_log(
    f"Distribuicao alvo: full={positive_rate(y):.4f} | train={positive_rate(y_train):.4f} | test={positive_rate(y_test):.4f}"
)
if dropped_alias_columns:
    audit_log(
        f"Aliases de compatibilidade removidas do treino por duplicarem colunas canonicas: {dropped_alias_columns}"
    )
if categorical_columns:
    audit_log(
        f"Features categoricas incluidas no treino: {categorical_columns}"
    )
    cardinality_summary = {
        column: int(X_train[column].astype("string").nunique(dropna=True))
        for column in categorical_columns
    }
    audit_log(f"Cardinalidade das categoricas no treino: {cardinality_summary}")

# ==============================
# 2. TREINAMENTO (CLASSIFICACAO)
# ==============================
audit_log("--- Treinamento: Classificacao ---")

preprocessor_lr = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_columns),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical_columns),
    ]
)

frequency_encoder = fit_frequency_encoder(X_train, categorical_columns)
X_train_rf = transform_frequency_encoded(X_train, numeric_columns, categorical_columns, frequency_encoder)
X_test_rf = transform_frequency_encoded(X_test, numeric_columns, categorical_columns, frequency_encoder)

# Parametros com Class Weight para classes desbalanceadas
params_rf = {"n_estimators": 100, "random_state": 42, "class_weight": "balanced", "n_jobs": -1, "oob_score": True}
params_lr = {"max_iter": 2000, "class_weight": "balanced"}

model_lr = Pipeline(
    steps=[
        ("preprocessor", preprocessor_lr),
        ("classifier", LogisticRegression(**params_lr)),
    ]
)
model_rf = RandomForestClassifier(**params_rf)

audit_log(f"Treinando Logistic Regression com parametros: {params_lr}")
model_lr.fit(X_train, y_train)

audit_log(
    f"Treinando Random Forest com parametros: {params_rf} | "
    f"categoricas codificadas por frequencia: {[f'{column}_FREQ' for column in categorical_columns]}"
)
model_rf.fit(X_train_rf, y_train)
audit_log(f"Random Forest OOB score: {model_rf.oob_score_:.4f}")

# ==============================
# 3. AVALIACAO DE PERFORMANCE
# ==============================
def evaluate_model(model, X_val, y_val, name, is_scaled=False):
    probs = model.predict_proba(X_val)[:, 1]
    preds = (probs >= CLASSIFICATION_THRESHOLD).astype(int)
    
    # Metricas de Teste
    acc = accuracy_score(y_val, preds)
    f1 = f1_score(y_val, preds)
    rec = recall_score(y_val, preds)
    pre = precision_score(y_val, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_val, preds).ravel()
    
    # Metricas de Treino (para check de Overfitting)
    train_probs = model.predict_proba(X_train if is_scaled else X_train_rf)[:, 1]
    train_preds = (train_probs >= CLASSIFICATION_THRESHOLD).astype(int)
    train_f1 = f1_score(y_train, train_preds, zero_division=0)
    train_rec = recall_score(y_train, train_preds, zero_division=0)
    train_pre = precision_score(y_train, train_preds, zero_division=0)
    predicted_positive_rate = float(preds.mean())
    fit_status = diagnose_fit(train_f1, f1, rec)
    
    audit_log(f"RESULTADO {name}:")
    audit_log(
        f"  > TESTE: Acc={acc:.4f} | F1={f1:.4f} | Recall={rec:.4f} | Precision={pre:.4f} | Pred_Pos_Rate={predicted_positive_rate:.4f}"
    )
    audit_log(
        f"  > TREINO: F1={train_f1:.4f} | Recall={train_rec:.4f} | Precision={train_pre:.4f}"
    )
    audit_log(f"  > MATRIZ_CONFUSAO: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    
    gap = train_f1 - f1
    audit_log(f"  > DIAGNOSTICO: {fit_status} | Gap_F1={gap:.4f}")

    if gap > 0.15:
        audit_log(f"  ! ALERTA: Gap de overfitting alto ({gap:.4f})")
    if rec < 0.50:
        audit_log(f"  ! ALERTA: Recall baixo ({rec:.4f}); ainda ha muitos falsos negativos.")

    return {
        "name": name,
        "test_f1": f1,
        "test_recall": rec,
        "test_precision": pre,
        "test_accuracy": acc,
        "train_f1": train_f1,
        "fit_status": fit_status,
        "predicted_positive_rate": predicted_positive_rate,
    }

lr_results = evaluate_model(model_lr, X_test, y_test, "Logistic Regression", is_scaled=True)
rf_results = evaluate_model(model_rf, X_test_rf, y_test, "Random Forest", is_scaled=False)

# ==============================
# 4. VALIDACAO ADICIONAL
# ==============================
audit_log("--- Validacao Adicional ---")

validation_train = X_train_rf.copy()
validation_test = X_test_rf.copy()

isolation_forest_table = calculate_isolation_forest_table(validation_train, validation_test)
vif_table = calculate_vif_table(validation_train[numeric_columns])
ks_drift_table = calculate_ks_drift_table(validation_train, validation_test)
psi_drift_table = calculate_psi_drift_table(validation_train, validation_test)

validation_summary = pd.DataFrame([
    {
        "validation": "Isolation Forest",
        "focus": "Anomalias / outliers",
        "scope": "train_vs_test",
        "top_status": isolation_forest_table.iloc[-1]["status"] if len(isolation_forest_table) else "N/A",
    },
    {
        "validation": "VIF",
        "focus": "Multicolinearidade",
        "scope": "train_numeric",
        "top_status": vif_table.iloc[0]["status"] if len(vif_table) else "N/A",
    },
    {
        "validation": "KS-Test",
        "focus": "Drift de distribuicao",
        "scope": "train_vs_test",
        "top_status": ks_drift_table.iloc[0]["status"] if len(ks_drift_table) else "N/A",
    },
    {
        "validation": "PSI",
        "focus": "Shift populacional",
        "scope": "train_vs_test",
        "top_status": psi_drift_table.iloc[0]["status"] if len(psi_drift_table) else "N/A",
    },
])

audit_log("TABELA_RESUMO_VALIDACAO:")
audit_log("\n" + format_log_table(validation_summary))
audit_log("TABELA_ISOLATION_FOREST:")
audit_log("\n" + format_log_table(isolation_forest_table))
audit_log("TABELA_VIF_TOP10:")
audit_log("\n" + format_log_table(vif_table))
audit_log("TABELA_KS_TEST_TOP10:")
audit_log("\n" + format_log_table(ks_drift_table))
audit_log("TABELA_PSI_TOP10:")
audit_log("\n" + format_log_table(psi_drift_table))

# ==============================
# 5. EXPORTACAO E INSIGHTS
# ==============================
audit_log("--- Finalizacao e Artefatos ---")

best_model_name = "Random Forest" if rf_results["test_f1"] >= lr_results["test_f1"] else "Logistic Regression"
best_model = model_rf if best_model_name == "Random Forest" else model_lr
best_model_path = os.path.join(MODEL_DIR, "best_model.pkl")

joblib.dump(best_model, best_model_path)
audit_log(f"Melhor modelo por F1 de teste: {best_model_name}")
audit_log(f"Modelo salvo em: {best_model_path}")

if best_model_name == "Random Forest":
    rf_encoder_path = os.path.join(MODEL_DIR, "best_model_frequency_encoder.pkl")
    joblib.dump(
        {
            "frequency_encoder": frequency_encoder,
            "numeric_columns": numeric_columns,
            "categorical_columns": categorical_columns,
        },
        rf_encoder_path,
    )
    audit_log(f"Encoder de frequencia salvo em: {rf_encoder_path}")

# Importancia das Features
rf_feature_names = numeric_columns + [f"{column}_FREQ" for column in categorical_columns]
importances = pd.Series(model_rf.feature_importances_, index=rf_feature_names).sort_values(ascending=False).head(5)
audit_log(f"Top 5 Predictores: {importances.to_dict()}")

audit_log(
    "RESUMO FINAL: "
    f"LR={lr_results['fit_status']} (F1={lr_results['test_f1']:.4f}, Recall={lr_results['test_recall']:.4f}) | "
    f"RF={rf_results['fit_status']} (F1={rf_results['test_f1']:.4f}, Recall={rf_results['test_recall']:.4f})"
)
audit_log(f"Pipeline finalizado com sucesso. Log gerado em: {LOG_FILE}")
