# =====================================================
# Script: Build Features
# Layer: CURATED -> FEATURES
# Author: Jean Bezerra
# =====================================================

import pandas as pd
from pathlib import Path
from datetime import datetime

# =====================================================
# Paths
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]

CURATED_PATH = BASE_DIR / "data" / "curated"
FEATURES_PATH = BASE_DIR / "data" / "features"

FEATURES_PATH.mkdir(parents=True, exist_ok=True)

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
    log("INFO", "READ", "Carregando datasets parquet")

    flights = pd.read_parquet(CURATED_PATH / "flights" / "flights.parquet")
    airports = pd.read_parquet(CURATED_PATH / "airports" / "airports.parquet")

    log("INFO", "READ", f"Flights: {len(flights)} linhas")
    log("INFO", "READ", f"Airports: {len(airports)} linhas")

    return flights, airports

# =====================================================
# Feature Engineering
# =====================================================

def build_features(df: pd.DataFrame, airports: pd.DataFrame) -> pd.DataFrame:

    log("INFO", "FEATURE", "Iniciando criação de features")

    # =====================================================
    # Target
    # =====================================================
    log("INFO", "FEATURE", "Criando target: IS_DELAYED")
    df["IS_DELAYED"] = (df["ARRIVAL_DELAY"] > 15).astype("Int8")

    # =====================================================
    # Features temporais
    # =====================================================
    log("INFO", "FEATURE", "Criando features temporais")

    df["DEPARTURE_HOUR"] = (df["SCHEDULED_DEPARTURE"] // 100).astype("Int8")
    df["IS_WEEKEND"] = df["DAY_OF_WEEK"].isin([6, 7]).astype("Int8")

    def get_period(hour):
        if pd.isna(hour):
            return "UNKNOWN"
        if hour < 6:
            return "NIGHT"
        elif hour < 12:
            return "MORNING"
        elif hour < 18:
            return "AFTERNOON"
        else:
            return "EVENING"

    df["PERIOD_OF_DAY"] = df["DEPARTURE_HOUR"].apply(get_period).astype("string")

    # =====================================================
    # Features de rota
    # =====================================================
    log("INFO", "FEATURE", "Criando features de rota")

    df["ROUTE"] = (
        df["ORIGIN_AIRPORT"].astype("string") + "_" +
        df["DESTINATION_AIRPORT"].astype("string")
    )

    # =====================================================
    # Enriquecimento com airports
    # =====================================================
    log("INFO", "FEATURE", "Enriquecendo com dados de aeroportos")

    airports = airports.rename(columns={"IATA_CODE": "ORIGIN_AIRPORT"})

    df = df.merge(
        airports[["ORIGIN_AIRPORT", "CITY", "STATE"]],
        on="ORIGIN_AIRPORT",
        how="left"
    )

    # =====================================================
    # Limpeza de colunas desnecessárias
    # =====================================================
    log("INFO", "PROCESS", "Removendo colunas irrelevantes")

    drop_cols = [
        "CANCELLED",
        "DIVERTED",
        "CANCELLATION_REASON"
    ]

    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # =====================================================
    # Tratamento de valores nulos (CORRETO)
    # =====================================================
    log("INFO", "PROCESS", "Tratando valores nulos (NaN)")

    # numéricos → mediana
    num_cols = df.select_dtypes(include=["number"]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # categóricos → UNKNOWN
    cat_cols = df.select_dtypes(include=["string"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna("UNKNOWN")

    log("INFO", "PROCESS", "Valores nulos tratados com sucesso")

    # =====================================================
    # Dataset final
    # =====================================================
    log("INFO", "FEATURE", f"Dataset final preparado com {len(df)} linhas")

    return df

# =====================================================
# Save
# =====================================================

def save_features(df: pd.DataFrame):

    output_path = FEATURES_PATH / "flights_features.parquet"

    log("INFO", "WRITE", f"Salvando dataset em: {output_path}")

    df.to_parquet(
        output_path,
        engine="pyarrow",
        index=False
    )

    log("SUCCESS", "WRITE", "Dataset de features salvo com sucesso")

# =====================================================
# Main
# =====================================================

def main():

    log("INFO", "PIPELINE", "Iniciando pipeline de feature engineering")

    flights, airports = load_data()

    df_features = build_features(flights, airports)

    save_features(df_features)

    log("SUCCESS", "PIPELINE", "Pipeline finalizado com sucesso")

# =====================================================

if __name__ == "__main__":
    main()