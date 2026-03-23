# =====================================================
# Script: Create Parquet Structure
# Layer: RAW -> CURATED
# Author: Jean Bezerra
# =====================================================

import pandas as pd
from pathlib import Path
from datetime import datetime

# =====================================================
# Paths
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_PATH = BASE_DIR / "data" / "raw"
CURATED_PATH = BASE_DIR / "data" / "curated"

CURATED_PATH.mkdir(parents=True, exist_ok=True)

# =====================================================
# Config datasets
# =====================================================

DATASETS = {
    "flights": {
        "input": RAW_PATH / "flights_normalized.csv",
        "output": CURATED_PATH / "flights" / "flights.parquet"
    },
    "airports": {
        "input": RAW_PATH / "airports_normalized.csv",
        "output": CURATED_PATH / "airports" / "airports.parquet"
    },
    "airlines": {
        "input": RAW_PATH / "airlines_normalized.csv",
        "output": CURATED_PATH / "airlines" / "airlines.parquet"
    }
}

# =====================================================
# Logging
# =====================================================

def log(level: str, step: str, message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] [{step}] {message}")


# =====================================================
# Helpers
# =====================================================

def ensure_directory(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def read_csv_safe(path: Path) -> pd.DataFrame:
    log("INFO", "READ", f"Lendo arquivo CSV: {path}")

    dtype_map = {
        "YEAR": "Int16",
        "MONTH": "Int8",
        "DAY": "Int8",
        "DAY_OF_WEEK": "Int8",

        "AIRLINE": "string",
        "FLIGHT_NUMBER": "Int32",
        "TAIL_NUMBER": "string",

        "ORIGIN_AIRPORT": "string",
        "DESTINATION_AIRPORT": "string",

        "SCHEDULED_DEPARTURE": "Int32",
        "DEPARTURE_TIME": "Int32",
        "WHEELS_OFF": "Int32",
        "WHEELS_ON": "Int32",
        "SCHEDULED_ARRIVAL": "Int32",
        "ARRIVAL_TIME": "Int32",

        "SCHEDULED_TIME": "float32",
        "ELAPSED_TIME": "float32",
        "AIR_TIME": "float32",
        "TAXI_OUT": "float32",
        "TAXI_IN": "float32",

        "DISTANCE": "float32",

        "DEPARTURE_DELAY": "float32",
        "ARRIVAL_DELAY": "float32",

        "DIVERTED": "Int8",
        "CANCELLED": "Int8",

        "CANCELLATION_REASON": "string",

        "AIR_SYSTEM_DELAY": "float32",
        "SECURITY_DELAY": "float32",
        "AIRLINE_DELAY": "float32",
        "LATE_AIRCRAFT_DELAY": "float32",
        "WEATHER_DELAY": "float32",
    }

    try:
        df = pd.read_csv(path, dtype=dtype_map, low_memory=False)
    except UnicodeDecodeError:
        log("WARN", "READ", "Encoding padrão falhou, tentando latin1")
        df = pd.read_csv(path, dtype=dtype_map, encoding="latin1", low_memory=False)

    log("INFO", "READ", f"Arquivo carregado com sucesso: {len(df)} linhas e {len(df.columns)} colunas")

    return df


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    log("INFO", "PROCESS", "Normalizando nomes das colunas (UPPERCASE)")
    df.columns = [c.strip().upper() for c in df.columns]

    log("INFO", "PROCESS", "Garantindo consistência de tipos (string)")

    # Para colunas de texto, garantir que sejam do tipo string e remover espaços em branco
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == "object":
            df[col] = df[col].astype("string").str.strip()

    return df


def write_parquet(df: pd.DataFrame, path: Path):
    ensure_directory(path)

    log("INFO", "WRITE", f"Gravando arquivo Parquet em: {path}")

    df.to_parquet(
        path,
        engine="pyarrow",
        index=False
    )

    log("SUCCESS", "WRITE", "Arquivo Parquet gerado com sucesso")


# =====================================================
# Core Process
# =====================================================

def process_dataset(name: str, config: dict):
    log("INFO", "INGESTION", f"Iniciando processamento do dataset: {name}")

    input_path = config["input"]
    output_path = config["output"]

    if not input_path.exists():
        log("ERROR", "INGESTION", f"Arquivo não encontrado: {input_path}")
        return

    try:
        df = read_csv_safe(input_path)
        df = normalize_dataframe(df)

        write_parquet(df, output_path)

        log("SUCCESS", "INGESTION", f"Dataset '{name}' processado com sucesso")
        log("INFO", "INGESTION", f"###############################################################")

    except Exception as e:
        log("ERROR", "INGESTION", f"Erro ao processar dataset '{name}': {str(e)}")


# =====================================================
# Main
# =====================================================

def main():
    log("INFO", "PIPELINE", "Iniciando pipeline de conversão CSV → Parquet")

    for name, config in DATASETS.items():
        process_dataset(name, config)

    log("SUCCESS", "PIPELINE", "Pipeline finalizado com sucesso")


if __name__ == "__main__":
    main()
