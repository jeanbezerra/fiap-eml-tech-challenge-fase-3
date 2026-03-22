# fiap-eml-tech-challenge-fase-3

## Cloud Providers

### AWS S3

#### Bucket (fiap-eml-tech-challenge-static-files-177862772785-sa-east-1-an)

- https://fiap-eml-tech-challenge-static-files-177862772785-sa-east-1-an.s3.sa-east-1.amazonaws.com/fase-3/airlines.csv  
- https://fiap-eml-tech-challenge-static-files-177862772785-sa-east-1-an.s3.sa-east-1.amazonaws.com/fase-3/airports.csv  
- https://fiap-eml-tech-challenge-static-files-177862772785-sa-east-1-an.s3.sa-east-1.amazonaws.com/fase-3/flights.csv

## Configuração do ambiente de desenvolvimento/estudos/pesquisas

```sh
python -m venv .venv
.venv\Scripts\activate
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
```

## Download dos dados

Antes de rodar notebooks, rotinas de tratamento ou rotinas de machine learning, o engenheiro de ML deve fazer o download dos arquivos base para a pasta `data/raw/`.

```sh
python scripts/data-download/script-data-download.py
```

## Iniciar um novo Jupyter Notebook

```sh
Ctrl + Shift + P
```

```sh
Create: New Jupyter Notebook
```

