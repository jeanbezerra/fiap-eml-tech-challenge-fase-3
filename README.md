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
python -m pip install -r requirements.txt
```

## Download dos dados

Antes de rodar notebooks, rotinas de tratamento ou rotinas de machine learning, o engenheiro de ML deve fazer o download dos arquivos base para a pasta `data/raw/`.

```sh
python -m pip install -r requirements.txt
python scripts/data-download/download-raw-data.py
python scripts/data-normalization/flights-normalization.py
python scripts/data-normalization/airports-normalization.py
python scripts/data-normalization/airlines-normalization.py
python scripts/data-download/build-curated-parquet.py
python scripts/feature-engineering/build_features.py
```

## Iniciar um novo Jupyter Notebook

```sh
Ctrl + Shift + P
```

```sh
Create: New Jupyter Notebook
```

# Requisitos e insights a serem respondidos: 
# MODELAGEM SUPERVISIONADA (mínimo uma abordagem):

Escolha entre:
Classificação: prever se um voo vai atrasar ou não.
OU
Regressão: prever quanto tempo o atraso vai durar.
Além disso: comparar pelo menos dois algoritmos diferentes e avalie com métricas adequadas.

# MODELAGEM NÃO SUPERVISIONADA (mínimo uma abordagem):
Use clusterização (ex.: agrupar rotas, aeroportos ou companhias aéreas)
Redução de dimensionalidade (ex.: PCA)

# EXTRAS:
● Criar variáveis derivadas (ex.: período do dia, feriados, estações do ano).
● Analisar atrasos por aeroporto, companhia ou estado.
● Criar mapas geográficos de rotas e atrasos.
● Identificar padrões sazonais ou horários críticos.
● Quais aeroportos são mais críticos em relação a atrasos?
● Que características aumentam a chance de atraso em um voo?
● Os atrasos são mais comuns em certos dias da semana ou horários?
