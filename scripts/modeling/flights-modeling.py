# ==============================
# 1. IMPORTS
# ==============================
import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==============================
# 2. CARREGAR DADOS
# ==============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
file_path = os.path.join(BASE_DIR, "data", "features", "flights_features.parquet")

print("📂 Carregando arquivo:", file_path)

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

df = pd.read_parquet(file_path)

print("✅ Dados carregados:", df.shape)

# ==============================
# 3. AMOSTRAGEM
# ==============================
df = df.sample(min(500000, len(df)), random_state=42)
print("📊 Amostra utilizada:", df.shape)

# ==============================
# 4. TARGET
# ==============================
print("\n🎯 Criando variável target...")
df['TARGET_DELAY'] = (df['ARRIVAL_DELAY'] > 15).astype(int)

# ==============================
# 5. REMOVER LEAKAGE
# ==============================
print("\n🚫 Removendo colunas com vazamento...")

cols_leakage = [
    'ARRIVAL_DELAY',
    'DEPARTURE_DELAY',
    'DELAY_RATIO',
    'AIR_SYSTEM_DELAY',
    'AIRLINE_DELAY',
    'LATE_AIRCRAFT_DELAY',
    'WEATHER_DELAY',
    'IS_DELAYED',
    'DATE'
]

X = df.drop(columns=cols_leakage + ['TARGET_DELAY'], errors='ignore')

# manter apenas numéricas
X = X.select_dtypes(include=[np.number])

y = df['TARGET_DELAY']

print("✔ Features finais:", X.shape)

# ==============================
# SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 6. CLASSIFICAÇÃO
# ==============================
print("\n===== 🤖 CLASSIFICAÇÃO =====")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_lr = LogisticRegression(max_iter=2000)
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)

print("🚀 Treinando Logistic Regression...")
model_lr.fit(X_train_scaled, y_train)

print("🌲 Treinando Random Forest...")
model_rf.fit(X_train, y_train)

def eval_class(y_true, y_pred, name):
    print(f"\n📊 {name}")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, zero_division=0))
    print("F1-score:", f1_score(y_true, y_pred, zero_division=0))

eval_class(y_test, model_lr.predict(X_test_scaled), "Logistic Regression")
eval_class(y_test, model_rf.predict(X_test), "Random Forest")

# ==============================
# 7. REGRESSÃO (SEM LEAKAGE)
# ==============================
print("\n===== 📈 REGRESSÃO =====")

df_reg = df[df['ARRIVAL_DELAY'] > 0]

if len(df_reg) < 100:
    print("⚠ Poucos dados para regressão. Pulando etapa.")
else:
    X_reg = df_reg.drop(columns=cols_leakage + ['TARGET_DELAY'], errors='ignore')
    X_reg = X_reg.select_dtypes(include=[np.number])

    y_reg = df_reg['ARRIVAL_DELAY']

    X_train, X_test, y_train, y_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    model_lr_reg = LinearRegression()
    model_rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)

    print("🚀 Treinando Linear Regression...")
    model_lr_reg.fit(X_train, y_train)

    print("🌲 Treinando Random Forest Regressor...")
    model_rf_reg.fit(X_train, y_train)

    def eval_reg(y_true, y_pred, name):
        print(f"\n📊 {name}")
        print("MAE:", mean_absolute_error(y_true, y_pred))
        print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
        print("R2:", r2_score(y_true, y_pred))

    eval_reg(y_test, model_lr_reg.predict(X_test), "Linear Regression")
    eval_reg(y_test, model_rf_reg.predict(X_test), "Random Forest Regressor")

# ==============================
# 8. CLUSTERIZAÇÃO
# ==============================
print("\n===== 🧩 CLUSTERIZAÇÃO =====")

features_cluster = ['DISTANCE', 'SCHEDULED_TIME']

features_cluster = [col for col in features_cluster if col in df.columns]

df_cluster = df[features_cluster].copy()
df_cluster = df_cluster.select_dtypes(include=[np.number])

scaler_cluster = StandardScaler()
X_scaled = scaler_cluster.fit_transform(df_cluster)

print("🚀 Treinando KMeans...")

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['CLUSTER'] = kmeans.fit_predict(X_scaled)

print("\n📊 Distribuição dos clusters:")
print(df['CLUSTER'].value_counts())

# ==============================
# 9. FEATURE IMPORTANCE
# ==============================
print("\n===== 🔍 FEATURE IMPORTANCE =====")

importances = pd.Series(
    model_rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(importances.head(10))

# ==============================
# FINAL
# ==============================
print("\n✅ Pipeline SEM LEAKAGE executado com sucesso!")