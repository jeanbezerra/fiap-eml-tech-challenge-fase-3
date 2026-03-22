@echo off
SETLOCAL
cls

echo [1/5] Preparando ambiente Python...
:: Verifica se o uv está instalado, senão usa o fluxo padrão silencioso
uv venv .venv >nul 2>&1 || python -m venv .venv >nul

echo [2/5] Sincronizando dependencias...
uv pip install -r requirements.txt >nul 2>&1 || (
    python -m pip install --upgrade pip >nul
    pip install -r requirements.txt >nul
)

echo [3/5] Executando: Download de dados...
python scripts/data-download/script-data-download.py >nul

echo [4/5] Executando: Normalizacao de voos...
python scripts/data-normalization/flights-normalization.py >nul

echo [5/5] Executando: Engenharia de atributos (Features)...
python scripts/feature-engineering/build_features.py >nul

echo.
echo Processo concluido com sucesso!
pause