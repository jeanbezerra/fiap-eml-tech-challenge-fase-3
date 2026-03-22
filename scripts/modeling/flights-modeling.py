# ==============================
# 1. IMPORTS & SETUP
# ==============================
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configuração de caminhos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "features", "flights_features.parquet")
LOG_DIR = os.path.join(BASE_DIR, "log")
os.makedirs(LOG_DIR, exist_ok=True)

# Gerador de log de auditoria
log_filename = os.path.join(LOG_DIR, f"audit_modeling_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

def audit_log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {message}"
    print(entry)
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(entry + "\n")

# ==============================
# 2. CARREGAR E PREPARAR DADOS
# ==============================
audit_log(f"Iniciando execução. Arquivo: {DATA_PATH}")

if not os.path.exists(DATA_PATH):
    audit_log(f"ERRO: Arquivo não encontrado em {DATA_PATH}")
    sys.exit(1)

df_raw = pd.read_parquet(DATA_PATH)
audit_log(f"Dados brutos carregados: {df_raw.shape}")

# Amostragem
sample_size = 500000
df = df_raw.sample(min(sample_size, len(df_raw)), random_state=42).copy()
audit_log(f"Amostra utilizada: {df.shape} (Seed: 42)")

# Criar Target (Classificação)
df['TARGET_DELAY'] = (df['ARRIVAL_DELAY'] > 15).astype(int)

# Filtro de Leakage
cols_leakage = [
    'ARRIVAL_DELAY', 'DEPARTURE_DELAY', 'DELAY_RATIO', 
    'AIR_SYSTEM_DELAY', 'AIRLINE_DELAY', 'LATE_AIRCRAFT_DELAY', 
    'WEATHER_DELAY', 'IS_DELAYED', 'DATE'
]

X = df.drop(columns=cols_leakage + ['TARGET_DELAY'], errors='ignore').select_dtypes(include=[np.number])
y = df['TARGET_DELAY']

audit_log(f"Features selecionadas: {X.shape[1]} colunas.")

# Split único para Classificação
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================
# 3. CLASSIFICAÇÃO
# ==============================
audit_log("--- Início: Classificação ---")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Parâmetros para auditoria
params_rf = {"n_estimators": 100, "max_depth": None, "random_state": 42}
params_lr = {"max_iter": 2000}

model_lr = LogisticRegression(**params_lr)
model_rf = RandomForestClassifier(**params_rf)

audit_log(f"Treinando Logistic Regression (params: {params_lr})...")
model_lr.fit(X_train_scaled, y_train)

audit_log(f"Treinando Random Forest (params: {params_rf})...")
model_rf.fit(X_train, y_train) # RF não exige escala, usamos X_train original

def get_metrics_class(model, X_val, y_val, name):
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    pre = precision_score(y_val, preds, zero_division=0)
    rec = recall_score(y_val, preds, zero_division=0)
    f1 = f1_score(y_val, preds, zero_division=0)
    return f"METRICAS {name}: Acc: {acc:.4f} | Prec: {pre:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}"

audit_log(get_metrics_class(model_lr, X_test_scaled, y_test, "Logistic Regression"))
audit_log(get_metrics_class(model_rf, X_test, y_test, "Random Forest"))

# ==============================
# 4. REGRESSÃO
# ==============================
audit_log("--- Início: Regressão (Apenas voos com atraso > 0) ---")

df_reg = df[df['ARRIVAL_DELAY'] > 0].copy()

if len(df_reg) < 100:
    audit_log("AVISO: Dados insuficientes para regressão.")
else:
    X_reg = df_reg[X.columns] # Usa as mesmas features numéricas já limpas
    y_reg = df_reg['ARRIVAL_DELAY']
    
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    model_lin = LinearRegression()
    model_rfr = RandomForestRegressor(n_estimators=50, random_state=42) # Reduzi n_estimators para velocidade

    model_lin.fit(Xr_train, yr_train)
    model_rfr.fit(Xr_train, yr_train)

    def get_metrics_reg(model, X_val, y_val, name):
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        r2 = r2_score(y_val, preds)
        return f"METRICAS {name}: MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.4f}"

    audit_log(get_metrics_reg(model_lin, Xr_test, yr_test, "Linear Reg"))
    audit_log(get_metrics_reg(model_rfr, Xr_test, yr_test, "RF Regressor"))

# ==============================
# 5. CLUSTERIZAÇÃO E IMPORTÂNCIA
# ==============================
audit_log("--- Início: Clusterização & Insights ---")

kmeans = KMeans(n_clusters=5, n_init=10, random_state=42)
# Usando subset para clusterizar (Ex: Distancia e Tempo)
X_cl = scaler.fit_transform(df[['DISTANCE', 'SCHEDULED_TIME']].fillna(0))
df['CLUSTER'] = kmeans.fit_predict(X_cl)
audit_log(f"Clusters criados. Distribuição: {df['CLUSTER'].value_counts().to_dict()}")

# Feature Importance (Random Forest Classificação)
importances = pd.Series(model_rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(5)
audit_log(f"Top 5 Features: {importances.to_dict()}")

audit_log("Finalizado com sucesso. Log gerado em: " + log_filename)