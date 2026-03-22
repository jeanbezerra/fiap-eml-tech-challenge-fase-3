# =====================================================
# Script: Flights Normalization
# Layer: RAW -> NORMALIZED
# Author: Jean Bezerra
# =====================================================

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# =====================================================
# Paths
# =====================================================

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_PATH = BASE_DIR / "data" / "raw"
INPUT_PATH = RAW_PATH / "flights.csv"
OUTPUT_PATH = RAW_PATH / "flights_normalized.csv"

# =====================================================
# Logging
# =====================================================

def log(level: str, step: str, message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] [{step}] {message}")


# =====================================================
# Helpers
# =====================================================

def hhmm_to_minutes(value):
    if pd.isna(value):
        return np.nan

    try:
        value = int(float(value))
    except (TypeError, ValueError):
        return np.nan

    hours = value // 100
    minutes = value % 100

    if hours >= 24 or minutes >= 60:
        return np.nan

    return hours * 60 + minutes


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


def add_date_column(df: pd.DataFrame) -> pd.DataFrame:
    log("INFO", "PROCESS", "Criando coluna DATE")
    df["DATE"] = pd.to_datetime(df[["YEAR", "MONTH", "DAY"]], errors="coerce")
    return df


def remove_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    log("INFO", "PROCESS", "Removendo colunas irrelevantes para a camada normalizada")
    return df.drop(columns=["TAIL_NUMBER"], errors="ignore")


def normalize_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    log("INFO", "PROCESS", "Convertendo colunas de horario de HHMM para minutos")

    time_columns = [
        "SCHEDULED_DEPARTURE",
        "DEPARTURE_TIME",
        "WHEELS_OFF",
        "WHEELS_ON",
        "SCHEDULED_ARRIVAL",
        "ARRIVAL_TIME",
    ]

    for column in time_columns:
        if column in df.columns:
            df[column] = df[column].apply(hhmm_to_minutes)

    return df


def normalize_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    log("INFO", "PROCESS", "Ajustando tipos numericos e tratando valores invalidos")

    integer_columns = [
        "YEAR",
        "MONTH",
        "DAY",
        "DAY_OF_WEEK",
        "FLIGHT_NUMBER",
        "CANCELLED",
        "DIVERTED",
    ]

    float_columns = [
        "SCHEDULED_DEPARTURE",
        "DEPARTURE_TIME",
        "WHEELS_OFF",
        "WHEELS_ON",
        "SCHEDULED_ARRIVAL",
        "ARRIVAL_TIME",
        "SCHEDULED_TIME",
        "ELAPSED_TIME",
        "AIR_TIME",
        "TAXI_OUT",
        "TAXI_IN",
        "DISTANCE",
        "DEPARTURE_DELAY",
        "ARRIVAL_DELAY",
        "AIR_SYSTEM_DELAY",
        "SECURITY_DELAY",
        "AIRLINE_DELAY",
        "LATE_AIRCRAFT_DELAY",
        "WEATHER_DELAY",
    ]

    for column in integer_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in float_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "CANCELLED" in df.columns:
        df["CANCELLED"] = df["CANCELLED"].fillna(0).astype("Int8")

    if "DIVERTED" in df.columns:
        df["DIVERTED"] = df["DIVERTED"].fillna(0).astype("Int8")

    return df


def normalize_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    log("INFO", "PROCESS", "Padronizando colunas textuais")

    text_columns = [
        "AIRLINE",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT",
        "CANCELLATION_REASON",
    ]

    for column in text_columns:
        if column in df.columns:
            df[column] = df[column].astype("string").str.strip()

    if "CANCELLATION_REASON" in df.columns:
        df["CANCELLATION_REASON"] = df["CANCELLATION_REASON"].fillna("NONE")

    return df


def validate_airport_code_formats(df: pd.DataFrame) -> None:
    log("INFO", "VALIDATE", "Validando formato das chaves de aeroportos em flights")

    airport_columns = ["ORIGIN_AIRPORT", "DESTINATION_AIRPORT"]

    for column in airport_columns:
        if column not in df.columns:
            continue

        series = df[column].astype("string").str.strip().str.upper()
        numeric_mask = series.str.fullmatch(r"\d+").fillna(False)
        iata_mask = series.str.fullmatch(r"[A-Z]{3}").fillna(False)

        numeric_rows = int(numeric_mask.sum())
        iata_rows = int(iata_mask.sum())
        other_rows = int((~numeric_mask & ~iata_mask & series.notna()).sum())

        log("INFO", "VALIDATE", f"{column}: linhas IATA={iata_rows:,} | linhas numéricas={numeric_rows:,} | outros formatos={other_rows:,}")

        if numeric_rows > 0:
            sample_numeric = sorted(series[numeric_mask].dropna().unique())[:10]
            log(
                "WARN",
                "VALIDATE",
                f"{column} contém códigos numéricos sem mapeamento no dataset de airports. Exemplos: {sample_numeric}"
            )


def filter_iata_airport_rows(df: pd.DataFrame) -> pd.DataFrame:
    log("INFO", "PROCESS", "Removendo linhas com aeroportos fora do padrão IATA")

    origin_mask = df["ORIGIN_AIRPORT"].astype("string").str.strip().str.upper().str.fullmatch(r"[A-Z]{3}").fillna(False)
    destination_mask = df["DESTINATION_AIRPORT"].astype("string").str.strip().str.upper().str.fullmatch(r"[A-Z]{3}").fillna(False)
    valid_mask = origin_mask & destination_mask

    removed_rows = int((~valid_mask).sum())

    if removed_rows > 0:
        log("WARN", "PROCESS", f"{removed_rows:,} linhas removidas por ORIGIN_AIRPORT/DESTINATION_AIRPORT fora do padrão IATA")

    return df.loc[valid_mask].copy()


def create_business_features(df: pd.DataFrame) -> pd.DataFrame:
    log("INFO", "PROCESS", "Criando atributos derivados de suporte")

    if "SCHEDULED_TIME" in df.columns:
        df["DELAY_RATIO"] = df["DEPARTURE_DELAY"] / df["SCHEDULED_TIME"].replace(0, np.nan)
        df["DELAY_RATIO"] = df["DELAY_RATIO"].replace([np.inf, -np.inf], np.nan).fillna(0)

    if "CANCELLED" in df.columns and "DEPARTURE_DELAY" in df.columns:
        df.loc[df["CANCELLED"] == 1, "DEPARTURE_DELAY"] = 0

    return df


def fill_numeric_nulls(df: pd.DataFrame) -> pd.DataFrame:
    log("INFO", "PROCESS", "Tratando nulos numericos")

    zero_fill_columns = [
        "DEPARTURE_DELAY",
        "ARRIVAL_DELAY",
        "AIR_SYSTEM_DELAY",
        "SECURITY_DELAY",
        "AIRLINE_DELAY",
        "LATE_AIRCRAFT_DELAY",
        "WEATHER_DELAY",
        "DELAY_RATIO",
    ]

    for column in zero_fill_columns:
        if column in df.columns:
            df[column] = df[column].fillna(0)

    return df


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    log("INFO", "WRITE", f"Gravando dataset normalizado em: {path}")
    df.to_csv(path, index=False)
    log("SUCCESS", "WRITE", "Arquivo CSV normalizado salvo com sucesso")


# =====================================================
# Core Process
# =====================================================

def normalize_flights() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {INPUT_PATH}")

    df = read_csv_safe(INPUT_PATH)
    df = normalize_columns(df)
    df = add_date_column(df)
    df = remove_irrelevant_columns(df)
    df = normalize_time_columns(df)
    df = normalize_numeric_columns(df)
    df = normalize_text_columns(df)
    validate_airport_code_formats(df)
    df = filter_iata_airport_rows(df)
    df = create_business_features(df)
    df = fill_numeric_nulls(df)

    log("INFO", "PROCESS", f"Dataset final preparado com {len(df)} linhas e {len(df.columns)} colunas")
    return df


# =====================================================
# Main
# =====================================================

def main():
    log("INFO", "PIPELINE", "Iniciando pipeline de normalização de flights")

    try:
        df = normalize_flights()
        write_csv(df, OUTPUT_PATH)
        log("SUCCESS", "PIPELINE", "Pipeline finalizado com sucesso")
    except Exception as exc:
        log("ERROR", "PIPELINE", f"Erro na normalização: {str(exc)}")
        raise


if __name__ == "__main__":
    main()
