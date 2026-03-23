# =====================================================
# Script: Build Features
# Layer: CURATED -> FEATURES
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

CURATED_PATH = BASE_DIR / "data" / "curated"
FEATURES_PATH = BASE_DIR / "data" / "features"

FEATURES_PATH.mkdir(parents=True, exist_ok=True)

LEAKAGE_COLUMNS = [
    "DEPARTURE_TIME",
    "ARRIVAL_TIME",
    "WHEELS_OFF",
    "WHEELS_ON",
    "ACTUAL_ELAPSED_TIME",
    "ELAPSED_TIME",
    "AIR_TIME",
    "TAXI_IN",
    "TAXI_OUT",
    "DEPARTURE_DELAY",
    "ARRIVAL_DELAY",
    "AIR_SYSTEM_DELAY",
    "SECURITY_DELAY",
    "AIRLINE_DELAY",
    "LATE_AIRCRAFT_DELAY",
    "WEATHER_DELAY",
    "DELAY_RATIO",
    "CANCELLED",
    "DIVERTED",
    "CANCELLATION_REASON",
    "DATE",
]

FINAL_FEATURE_COLUMNS = [
    "YEAR",
    "MONTH",
    "DAY_OF_MONTH",
    "DAY_OF_WEEK",
    "IS_WEEKEND",
    "SCHEDULED_DEPARTURE",
    "SCHEDULED_ARRIVAL",
    "SCHEDULED_TIME",
    "DISTANCE",
    "DEPARTURE_HOUR",
    "ARRIVAL_HOUR",
    "SCHEDULED_DEPARTURE_SIN",
    "SCHEDULED_DEPARTURE_COS",
    "SCHEDULED_ARRIVAL_SIN",
    "SCHEDULED_ARRIVAL_COS",
    "PERIOD_OF_DAY",
    "AIRLINE",
    "FLIGHT_NUMBER",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "ROUTE",
    "CITY",
    "STATE",
    "ORIGIN_CITY",
    "ORIGIN_STATE",
    "DESTINATION_CITY",
    "DESTINATION_STATE",
    "AIRLINE_HIST_DELAY_RATE",
    "ROUTE_HIST_DELAY_RATE",
    "IS_DELAYED",
]

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
# Helpers
# =====================================================

def normalize_airport_key(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.upper()


def normalize_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def hhmm_to_hour(series: pd.Series) -> pd.Series:
    hours = (pd.to_numeric(series, errors="coerce") // 60) % 24
    return hours.fillna(-1).astype("Int8")


def hour_to_period(series: pd.Series) -> pd.Series:
    labels = pd.Series(pd.array(["UNKNOWN"] * len(series), dtype="string"), index=series.index)
    labels = labels.mask(series.between(0, 5), "NIGHT")
    labels = labels.mask(series.between(6, 11), "MORNING")
    labels = labels.mask(series.between(12, 17), "AFTERNOON")
    labels = labels.mask(series.between(18, 23), "EVENING")
    return labels.astype("string")


def encode_cyclical_minutes(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    minutes = pd.to_numeric(series, errors="coerce")
    angle = 2 * np.pi * (minutes % 1440) / 1440

    sin_values = pd.Series(np.sin(angle), index=series.index).fillna(0).astype("float32")
    cos_values = pd.Series(np.cos(angle), index=series.index).fillna(0).astype("float32")

    return sin_values, cos_values


def log_airport_enrichment_coverage(df: pd.DataFrame, airports: pd.DataFrame) -> None:
    flight_keys = normalize_airport_key(df["ORIGIN_AIRPORT"])
    airport_keys = normalize_airport_key(airports["IATA_CODE"])

    missing_keys = sorted(set(flight_keys.dropna().unique()) - set(airport_keys.dropna().unique()))
    missing_rows = int(flight_keys.isin(missing_keys).sum())
    numeric_missing = [key for key in missing_keys if key.isdigit()]
    alpha_missing = [key for key in missing_keys if not key.isdigit()]

    log(
        "INFO",
        "VALIDATE",
        f"Cobertura airports x flights: origens únicas em flights={flight_keys.nunique(dropna=True)} | chaves em airports={airport_keys.nunique(dropna=True)} | chaves sem match={len(missing_keys)}"
    )

    if missing_keys:
        sample_missing = missing_keys[:10]
        log(
            "WARN",
            "VALIDATE",
            f"Enriquecimento incompleto: {missing_rows:,} linhas de flights ficarão sem dados de aeroporto. Exemplos sem match: {sample_missing}"
        )
        log(
            "WARN",
            "VALIDATE",
            f"Quebra por tipo de chave sem match: numéricas={len(numeric_missing)} | não numéricas={len(alpha_missing)}"
        )


def enrich_airports(df: pd.DataFrame, airports: pd.DataFrame) -> pd.DataFrame:
    log("INFO", "FEATURE", "Enriquecendo com dados de aeroportos")

    log_airport_enrichment_coverage(df, airports)

    airport_dim = airports.copy()
    airport_dim["IATA_CODE"] = normalize_airport_key(airport_dim["IATA_CODE"])

    origin_airports = airport_dim.rename(
        columns={
            "IATA_CODE": "ORIGIN_AIRPORT",
            "CITY": "ORIGIN_CITY",
            "STATE": "ORIGIN_STATE",
        }
    )

    destination_airports = airport_dim.rename(
        columns={
            "IATA_CODE": "DESTINATION_AIRPORT",
            "CITY": "DESTINATION_CITY",
            "STATE": "DESTINATION_STATE",
        }
    )

    df = df.merge(
        origin_airports[["ORIGIN_AIRPORT", "ORIGIN_CITY", "ORIGIN_STATE"]],
        on="ORIGIN_AIRPORT",
        how="left"
    )

    df = df.merge(
        destination_airports[["DESTINATION_AIRPORT", "DESTINATION_CITY", "DESTINATION_STATE"]],
        on="DESTINATION_AIRPORT",
        how="left"
    )

    return df


def add_airport_compatibility_aliases(df: pd.DataFrame) -> pd.DataFrame:
    log("INFO", "FEATURE", "Criando aliases de compatibilidade CITY/STATE a partir da origem")

    if "ORIGIN_CITY" in df.columns:
        df["CITY"] = df["ORIGIN_CITY"]
    if "ORIGIN_STATE" in df.columns:
        df["STATE"] = df["ORIGIN_STATE"]

    return df


def add_schedule_features(df: pd.DataFrame) -> pd.DataFrame:
    log("INFO", "FEATURE", "Criando features de agendamento")

    df["DAY_OF_MONTH"] = pd.to_numeric(df["DAY"], errors="coerce").fillna(-1).astype("Int8")
    df["MONTH"] = pd.to_numeric(df["MONTH"], errors="coerce").fillna(-1).astype("Int8")
    df["DAY_OF_WEEK"] = pd.to_numeric(df["DAY_OF_WEEK"], errors="coerce").fillna(-1).astype("Int8")
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").fillna(-1).astype("Int16")
    df["IS_WEEKEND"] = df["DAY_OF_WEEK"].isin([6, 7]).astype("Int8")

    df["DEPARTURE_HOUR"] = hhmm_to_hour(df["SCHEDULED_DEPARTURE"])
    df["ARRIVAL_HOUR"] = hhmm_to_hour(df["SCHEDULED_ARRIVAL"])
    df["PERIOD_OF_DAY"] = hour_to_period(df["DEPARTURE_HOUR"])

    dep_sin, dep_cos = encode_cyclical_minutes(df["SCHEDULED_DEPARTURE"])
    arr_sin, arr_cos = encode_cyclical_minutes(df["SCHEDULED_ARRIVAL"])

    df["SCHEDULED_DEPARTURE_SIN"] = dep_sin
    df["SCHEDULED_DEPARTURE_COS"] = dep_cos
    df["SCHEDULED_ARRIVAL_SIN"] = arr_sin
    df["SCHEDULED_ARRIVAL_COS"] = arr_cos

    return df


def add_route_features(df: pd.DataFrame) -> pd.DataFrame:
    log("INFO", "FEATURE", "Criando features de rota")

    df["ROUTE"] = (
        df["ORIGIN_AIRPORT"].astype("string") + "_" +
        df["DESTINATION_AIRPORT"].astype("string")
    )

    return df


def add_historical_delay_rates(df: pd.DataFrame) -> pd.DataFrame:
    log("INFO", "FEATURE", "Criando histórico acumulado de atraso por companhia e rota")

    df = df.sort_values(
        by=["YEAR", "MONTH", "DAY", "SCHEDULED_DEPARTURE", "FLIGHT_NUMBER"],
        kind="stable"
    ).copy()

    global_rate = float(df["IS_DELAYED"].mean())
    delay_as_float = df["IS_DELAYED"].astype("float32")

    airline_group = df.groupby("AIRLINE", sort=False)["IS_DELAYED"]
    airline_cumsum = airline_group.cumsum().astype("float32")
    airline_previous_sum = airline_cumsum - delay_as_float
    airline_previous_count = airline_group.cumcount().astype("float32")

    route_group = df.groupby("ROUTE", sort=False)["IS_DELAYED"]
    route_cumsum = route_group.cumsum().astype("float32")
    route_previous_sum = route_cumsum - delay_as_float
    route_previous_count = route_group.cumcount().astype("float32")

    df["AIRLINE_HIST_DELAY_RATE"] = np.where(
        airline_previous_count > 0,
        airline_previous_sum / airline_previous_count,
        global_rate
    ).astype("float32")

    df["ROUTE_HIST_DELAY_RATE"] = np.where(
        route_previous_count > 0,
        route_previous_sum / route_previous_count,
        global_rate
    ).astype("float32")

    df["IS_DELAYED"] = delay_as_float.astype("Int8")

    return df


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    log("INFO", "PROCESS", "Removendo colunas com data leakage e eventos pós-voo")

    drop_cols = [column for column in LEAKAGE_COLUMNS if column in df.columns]
    return df.drop(columns=drop_cols)


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    log("INFO", "PROCESS", "Tratando valores nulos")

    categorical_unknown = [
        "AIRLINE",
        "ORIGIN_AIRPORT",
        "DESTINATION_AIRPORT",
        "ROUTE",
        "CITY",
        "STATE",
        "ORIGIN_CITY",
        "ORIGIN_STATE",
        "DESTINATION_CITY",
        "DESTINATION_STATE",
        "PERIOD_OF_DAY",
    ]

    for column in categorical_unknown:
        if column in df.columns:
            df[column] = normalize_text(df[column]).fillna("UNKNOWN").astype("string")

    numeric_default_zero = [
        "SCHEDULED_TIME",
        "DISTANCE",
        "SCHEDULED_DEPARTURE",
        "SCHEDULED_ARRIVAL",
        "SCHEDULED_DEPARTURE_SIN",
        "SCHEDULED_DEPARTURE_COS",
        "SCHEDULED_ARRIVAL_SIN",
        "SCHEDULED_ARRIVAL_COS",
        "AIRLINE_HIST_DELAY_RATE",
        "ROUTE_HIST_DELAY_RATE",
    ]

    for column in numeric_default_zero:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    for column in ["SCHEDULED_TIME", "DISTANCE"]:
        if column in df.columns:
            df[column] = df[column].astype("float32")

    for column in [
        "SCHEDULED_DEPARTURE_SIN",
        "SCHEDULED_DEPARTURE_COS",
        "SCHEDULED_ARRIVAL_SIN",
        "SCHEDULED_ARRIVAL_COS",
        "AIRLINE_HIST_DELAY_RATE",
        "ROUTE_HIST_DELAY_RATE",
    ]:
        if column in df.columns:
            df[column] = df[column].astype("float32")

    for column, dtype in {
        "FLIGHT_NUMBER": "Int32",
        "SCHEDULED_DEPARTURE": "Int32",
        "SCHEDULED_ARRIVAL": "Int32",
        "YEAR": "Int16",
        "MONTH": "Int8",
        "DAY_OF_MONTH": "Int8",
        "DAY_OF_WEEK": "Int8",
        "IS_WEEKEND": "Int8",
        "DEPARTURE_HOUR": "Int8",
        "ARRIVAL_HOUR": "Int8",
        "IS_DELAYED": "Int8",
    }.items():
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").fillna(-1 if dtype != "Int32" else 0).astype(dtype)

    log("INFO", "PROCESS", "Valores nulos tratados com sucesso")

    return df


def select_final_columns(df: pd.DataFrame) -> pd.DataFrame:
    log("INFO", "PROCESS", "Selecionando schema final limpo para modelagem")

    selected_columns = [column for column in FINAL_FEATURE_COLUMNS if column in df.columns]
    df = df[selected_columns].copy()

    log("INFO", "PROCESS", f"Colunas finais: {selected_columns}")

    return df


# =====================================================
# Feature Engineering
# =====================================================

def build_features(df: pd.DataFrame, airports: pd.DataFrame) -> pd.DataFrame:
    log("INFO", "FEATURE", "Iniciando criação de features")

    df = df.copy()
    airports = airports.copy()

    df["ORIGIN_AIRPORT"] = normalize_airport_key(df["ORIGIN_AIRPORT"])
    df["DESTINATION_AIRPORT"] = normalize_airport_key(df["DESTINATION_AIRPORT"])
    airports["IATA_CODE"] = normalize_airport_key(airports["IATA_CODE"])
    df["AIRLINE"] = normalize_text(df["AIRLINE"]).str.upper()

    log("INFO", "FEATURE", "Criando target: IS_DELAYED")
    df["IS_DELAYED"] = (pd.to_numeric(df["ARRIVAL_DELAY"], errors="coerce") > 15).astype("Int8")

    df = add_schedule_features(df)
    df = add_route_features(df)
    df = enrich_airports(df, airports)
    df = add_airport_compatibility_aliases(df)
    df = add_historical_delay_rates(df)
    df = drop_leakage_columns(df)
    df = fill_missing_values(df)
    df = select_final_columns(df)

    log("INFO", "FEATURE", f"Dataset final preparado com {len(df)} linhas e {len(df.columns)} colunas")

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
