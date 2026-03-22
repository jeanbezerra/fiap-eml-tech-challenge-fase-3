# ==============================
# 1. IMPORTS
# ==============================
import pandas as pd
import numpy as np

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
# 2. CARREGAR DADOS NORMALIZADOS
# ==============================
df = pd.read_parquet("data/features/flights_features.parquet")

print("Dados carregados:", df.shape)

# 🔥 OPCIONAL (RECOMENDADO) → acelerar processamento
df = df.sample(500000, random_state=42)
print("Amostra utilizada:", df.shape)

# ==============================
# 3. TARGET (CLASSIFICAÇÃO)
# ==============================
df['TARGET_DELAY'] = (df['ARRIVAL_DELAY'] > 15).astype(int)

# ==============================
# 4. PREPARAÇÃO DOS DADOS
# ==============================
cols_drop = ['ARRIVAL_DELAY', 'DATE']

X = df.drop(columns=cols_drop + ['TARGET_DELAY'], errors='ignore')
y = df['TARGET_DELAY']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================
# 5. CLASSIFICAÇÃO
# ==============================
print("\n===== CLASSIFICAÇÃO =====")

# 🔹 Escala apenas para Logistic Regression
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🔹 Modelos
model_lr = LogisticRegression(max_iter=2000)
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)

# 🔹 Treinamento
model_lr.fit(X_train_scaled, y_train)
model_rf.fit(X_train, y_train)

# 🔹 Avaliação
def eval_class(y_true, y_pred, name):
    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1-score:", f1_score(y_true, y_pred))

eval_class(y_test, model_lr.predict(X_test_scaled), "Logistic Regression")
eval_class(y_test, model_rf.predict(X_test), "Random Forest")

# ==============================
# 6. REGRESSÃO
# ==============================
print("\n===== REGRESSÃO =====")

# usar apenas voos com atraso
df_reg = df[df['ARRIVAL_DELAY'] > 0]

X_reg = df_reg.drop(columns=['ARRIVAL_DELAY', 'TARGET_DELAY', 'DATE'], errors='ignore')
y_reg = df_reg['ARRIVAL_DELAY']

X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# 🔹 Modelos
model_lr_reg = LinearRegression()
model_rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)

# 🔹 Treinamento
model_lr_reg.fit(X_train, y_train)
model_rf_reg.fit(X_train, y_train)

# 🔹 Avaliação
def eval_reg(y_true, y_pred, name):
    print(f"\n{name}")
    print("MAE:", mean_absolute_error(y_true, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
    print("R2:", r2_score(y_true, y_pred))

eval_reg(y_test, model_lr_reg.predict(X_test), "Linear Regression")
eval_reg(y_test, model_rf_reg.predict(X_test), "Random Forest Regressor")

# ==============================
# 7. CLUSTERIZAÇÃO
# ==============================
print("\n===== CLUSTERIZAÇÃO =====")

features_cluster = ['DISTANCE', 'SCHEDULED_TIME', 'DEPARTURE_DELAY']

df_cluster = df[features_cluster].copy()

scaler_cluster = StandardScaler()
X_scaled = scaler_cluster.fit_transform(df_cluster)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['CLUSTER'] = kmeans.fit_predict(X_scaled)

print("\nDistribuição dos clusters:")
print(df['CLUSTER'].value_counts())

# ==============================
# 8. FEATURE IMPORTANCE
# ==============================
print("\n===== FEATURE IMPORTANCE (Random Forest) =====")

importances = pd.Series(
    model_rf.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(importances.head(10))

# ==============================
# FINAL
# ==============================
print("\nPipeline de modelagem executado com sucesso!")