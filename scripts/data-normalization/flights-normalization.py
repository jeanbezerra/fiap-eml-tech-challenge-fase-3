# ==============================
# 1. IMPORTS
# ==============================
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# ==============================
# 2. CARREGAMENTO DOS DADOS
# ==============================
url = "data/raw/flights.csv"
df = pd.read_csv(url, low_memory=False)

# ==============================
# 3. FEATURE: DATA
# ==============================
df['DATE'] = pd.to_datetime(df[['YEAR','MONTH','DAY']], errors='coerce')

# ==============================
# 4. REMOVER COLUNAS IRRELEVANTES
# ==============================
df.drop(columns=['TAIL_NUMBER'], inplace=True, errors='ignore')

# ==============================
# 5. CONVERSÃO DE HORÁRIOS
# ==============================
def hhmm_to_minutes(x):
    try:
        if pd.isna(x):
            return np.nan
        x = int(float(x))
        hours = x // 100
        minutes = x % 100
        
        if hours >= 24 or minutes >= 60:
            return np.nan
            
        return hours * 60 + minutes
    except:
        return np.nan

cols_time = [
    'SCHEDULED_DEPARTURE', 'DEPARTURE_TIME', 'WHEELS_OFF',
    'WHEELS_ON', 'SCHEDULED_ARRIVAL', 'ARRIVAL_TIME'
]

for col in cols_time:
    if col in df.columns:
        df[col] = df[col].apply(hhmm_to_minutes)

# ==============================
# 6. TRATAMENTO DE NULOS
# ==============================
df['CANCELLATION_REASON'] = df['CANCELLATION_REASON'].fillna('None')

df['DEPARTURE_DELAY'] = pd.to_numeric(df['DEPARTURE_DELAY'], errors='coerce').fillna(0)
df['ARRIVAL_DELAY'] = pd.to_numeric(df['ARRIVAL_DELAY'], errors='coerce').fillna(0)

df.loc[df['CANCELLED'] == 1, 'DEPARTURE_DELAY'] = 0

delay_cols = [
    'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY',
    'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY'
]

for col in delay_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# ==============================
# 7. AJUSTE DE TIPOS
# ==============================
df['CANCELLED'] = pd.to_numeric(df['CANCELLED'], errors='coerce').fillna(0).astype(int)
df['DIVERTED'] = pd.to_numeric(df['DIVERTED'], errors='coerce').fillna(0).astype(int)

# ==============================
# 8. FEATURE ENGINEERING
# ==============================
df['SCHEDULED_TIME'] = pd.to_numeric(df['SCHEDULED_TIME'], errors='coerce')

df['DELAY_RATIO'] = df['DEPARTURE_DELAY'] / df['SCHEDULED_TIME'].replace(0, np.nan)
df['DELAY_RATIO'] = df['DELAY_RATIO'].replace([np.inf, -np.inf], np.nan).fillna(0)

# ==============================
# 9. ENCODING
# ==============================
for col in ['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

low_cardinality_cols = ['AIRLINE', 'CANCELLATION_REASON']
existing_cats = [col for col in low_cardinality_cols if col in df.columns]

df = pd.get_dummies(df, columns=existing_cats, dummy_na=False)

# ==============================
# 10. NORMALIZAÇÃO
# ==============================
if 'DISTANCE' in df.columns:
    df['DISTANCE'] = pd.to_numeric(df['DISTANCE'], errors='coerce')
    df['DISTANCE'] = df['DISTANCE'].fillna(df['DISTANCE'].median())

    scaler = MinMaxScaler()
    df[['DISTANCE']] = scaler.fit_transform(df[['DISTANCE']])

# ==============================
# 11. LIMPEZA FINAL
# ==============================
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
df[num_cols] = df[num_cols].fillna(0)

# ==============================
# 12. SALVAR CSV (AGORA EM data/raw)
# ==============================
output_dir = "data/raw"
output_file = os.path.join(output_dir, "flights_normalized.csv")

# garante que a pasta existe (já deve existir, mas por segurança)
os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(output_file):
    df.to_csv(output_file, index=False)
    print(f"Arquivo criado: {output_file}")

else:
    if os.path.getsize(output_file) == 0:
        df.to_csv(output_file, index=False)
        print(f"Arquivo estava vazio e foi preenchido: {output_file}")
    else:
        print(f"Arquivo já existe e NÃO foi sobrescrito: {output_file}")

# ==============================
# 13. OUTPUT FINAL
# ==============================
print("Dataset tratado com sucesso!")
print("Shape:", df.shape)

print("\nHEAD:")
print(df.head())

print("\nTAIL:")
print(df.tail())