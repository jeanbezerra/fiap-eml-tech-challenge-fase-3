# =====================================================
# Script: Airports Normalization
# Layer: RAW -> NORMALIZED
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
INPUT_PATH = RAW_PATH / "airports.csv"
OUTPUT_PATH = RAW_PATH / "airports_normalized.csv"


# =====================================================
# Logging
# =====================================================

def log(level: str, step: str, message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] [{step}] {message}")


# =====================================================
# Helpers
# =====================================================

def read_csv_safe(path: Path) -> pd.DataFrame:
    log("INFO", "READ", f"Lendo arquivo CSV: {path}")

    try:
        df = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        log("WARN", "READ", "Encoding padrão falhou, tentando latin1")
        df = pd.read_csv(path, encoding="latin1", low_memory=False)

    log("INFO", "READ", f"Arquivo carregado com sucesso: {len(df)} linhas e {len(df.columns)} colunas")
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    log("INFO", "PROCESS", "Normalizando nomes das colunas")
    df.columns = [column.strip().upper() for column in df.columns]
    return df


def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    log("INFO", "PROCESS", "Padronizando colunas textuais")

    for column in df.columns:
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
            df[column] = df[column].astype("string").str.strip()

    uppercase_code_columns = ["IATA_CODE"]
    for column in uppercase_code_columns:
        if column in df.columns:
            df[column] = df[column].str.upper()

    title_case_columns = ["AIRPORT", "CITY", "STATE", "COUNTRY"]
    for column in title_case_columns:
        if column in df.columns:
            df[column] = df[column].str.title()

    return df


def normalize_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    log("INFO", "PROCESS", "Ajustando tipos numericos")

    numeric_columns = ["LATITUDE", "LONGITUDE"]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def validate_keys(df: pd.DataFrame) -> None:
    if "IATA_CODE" not in df.columns:
        log("WARN", "VALIDATE", "Coluna IATA_CODE não encontrada no dataset de airports")
        return

    null_keys = int(df["IATA_CODE"].isna().sum())
    duplicate_keys = int(df["IATA_CODE"].duplicated().sum())

    log("INFO", "VALIDATE", f"IATA_CODE nulos: {null_keys}")
    log("INFO", "VALIDATE", f"IATA_CODE duplicados: {duplicate_keys}")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    log("INFO", "WRITE", f"Gravando dataset normalizado em: {path}")
    df.to_csv(path, index=False)
    log("SUCCESS", "WRITE", "Arquivo CSV normalizado salvo com sucesso")


# =====================================================
# Core Process
# =====================================================

def normalize_airports() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {INPUT_PATH}")

    df = read_csv_safe(INPUT_PATH)
    df = normalize_columns(df)
    df = normalize_text_columns(df)
    df = normalize_numeric_columns(df)
    validate_keys(df)

    log("INFO", "PROCESS", f"Dataset final preparado com {len(df)} linhas e {len(df.columns)} colunas")
    return df


# =====================================================
# Main
# =====================================================

def main():
    log("INFO", "PIPELINE", "Iniciando pipeline de normalização de airports")

    try:
        df = normalize_airports()
        write_csv(df, OUTPUT_PATH)
        log("SUCCESS", "PIPELINE", "Pipeline finalizado com sucesso")
    except Exception as exc:
        log("ERROR", "PIPELINE", f"Erro na normalização: {str(exc)}")
        raise


if __name__ == "__main__":
    main()
