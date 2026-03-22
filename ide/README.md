# IDE para uso do Codex em projetos de ML

## Interpretação deste projeto

Este repositório implementa um pipeline de machine learning sobre atrasos de voos com foco em classificação supervisionada e preparação de dados para análises adicionais.

Fluxo principal identificado:

1. Download dos CSVs públicos para `data/raw/`.
2. Normalização de `flights.csv` e geração de `flights_normalized.csv`.
3. Conversão dos datasets para Parquet em `data/curated/`.
4. Criação de features em `data/features/`.
5. Treinamento de modelos em `scripts/modeling/`.

Pastas e responsabilidades:

- `scripts/data-download/`: ingestão inicial e preparação da camada raw/curated.
- `scripts/data-normalization/`: limpeza, tipagem e transformação inicial de voos.
- `scripts/feature-engineering/`: geração de target e atributos derivados.
- `scripts/modeling/`: treino e comparação de modelos.
- `data/raw/`: dados baixados do bucket público, não versionados.
- `data/curated/`: datasets em Parquet organizados por domínio.
- `notebooks/`: exploração e EDA.

## Como o Codex deve trabalhar neste tipo de projeto

Em projetos parecidos, o Codex deve assumir estas regras operacionais:

- Ler primeiro `README.md`, `requirements.txt`, `run.cmd` e a árvore de `scripts/`.
- Tratar o projeto como pipeline por camadas: ingestão, normalização, features, modelagem e artefatos.
- Preservar caminhos relativos ao diretório raiz do projeto.
- Colocar qualquer dependência nova em `requirements.txt` com versão explícita.
- Não versionar datasets baixados, arquivos temporários, modelos grandes e saídas locais.
- Preferir scripts reproduzíveis em vez de lógica solta em notebooks.
- Antes de editar modelagem, verificar o formato real gerado pelas etapas anteriores.
- Ao alterar o pipeline, manter compatibilidade com execução local via terminal Windows/PowerShell.

## Convenções recomendadas para outros projetos

Estrutura mínima recomendada:

```text
project/
  data/
    raw/
    curated/
    features/
  scripts/
    data-download/
    data-normalization/
    feature-engineering/
    modeling/
  notebooks/
  models/
  README.md
  requirements.txt
  .gitignore
```

Sequência padrão que o Codex deve privilegiar:

```sh
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/data-download/script-data-download.py
python scripts/data-normalization/flights-normalization.py
python scripts/data-download/script-create-parquet-structure.py
python scripts/feature-engineering/build_features.py
python scripts/modeling/train.py
```

## Prompt-base reutilizável para Codex

Use este contexto em outros projetos de ML quando quiser acelerar o trabalho do Codex:

```text
Interprete o projeto como um pipeline de dados e machine learning.
Leia primeiro README, requirements, scripts e estrutura de pastas.
Preserve o fluxo raw -> curated -> features -> model.
Qualquer dependência nova deve entrar no requirements.txt com versão explícita.
Evite soluções só em notebook quando a rotina puder virar script reproduzível.
Antes de alterar treino ou features, valide os artefatos produzidos nas etapas anteriores.
Considere execução local em Windows/PowerShell, a menos que o projeto diga o contrário.
```

## Lacunas percebidas neste repositório

Pontos que o Codex deve observar ao continuar este projeto:

- `data/features/` não está documentada no README principal.
- `scripts/modeling/flights-modeling.py` e `scripts/modeling/train.py` parecem coexistir com objetivos sobrepostos.
- `train.py` depende de `joblib`, então essa dependência deve existir em `requirements.txt` se o script for usado.
- Falta um comando único no README para executar treinamento ponta a ponta.
- O pipeline atual mistura CSV normalizado e Parquet; isso deve ser mantido consistente nas próximas evoluções.
