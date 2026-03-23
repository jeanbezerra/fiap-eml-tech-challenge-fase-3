@echo off
SETLOCAL
cls

echo [1/7] Preparando ambiente Python...
:: Verifica se o uv está instalado, senão usa o fluxo padrão silencioso
uv venv .venv >nul 2>&1 || python -m venv .venv >nul

echo [2/7] Sincronizando dependencias...
uv python -m pip install -r requirements.txt >nul 2>&1 || (
    python -m pip install --upgrade pip >nul
    python -m pip install -r requirements.txt >nul
)

echo [3/7] Executando: Download de dados...
python scripts/data-download/download-raw-data.py >nul

echo [4/7] Executando: Normalizacao de voos...
python scripts/data-normalization/flights-normalization.py >nul

echo [5/7] Executando: Normalizacao de aeroportos...
python scripts/data-normalization/airports-normalization.py >nul

echo [6/7] Executando: Normalizacao de companhias...
python scripts/data-normalization/airlines-normalization.py >nul

echo [7/7] Executando: Geracao curated + features...
python scripts/data-download/build-curated-parquet.py >nul
python scripts/feature-engineering/build_features.py >nul

echo.
echo Processo concluido com sucesso!
pause
