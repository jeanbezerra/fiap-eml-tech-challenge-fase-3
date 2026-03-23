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

## Como interpretar os resultados da modelagem

O script `scripts/modeling/flights-modeling.py` gera um log de auditoria com métricas de classificação, sinais de ajuste do modelo e validações adicionais de estabilidade dos dados. A leitura correta dessas saídas é importante para decidir se o modelo está pronto para apoiar operação, se ainda precisa de melhoria ou se o comportamento observado pode trazer risco para o negócio.

### Métricas principais de classificação

`Accuracy`

Mostra a taxa geral de acerto. É útil como visão ampla, mas não deve ser a principal referência quando o volume de voos sem atraso é maior do que o volume de voos com atraso. Um modelo pode ter `Accuracy` razoável e ainda assim deixar passar muitos atrasos importantes.

`Precision`

Mostra, entre os voos que o modelo marcou como atrasados, quantos realmente atrasaram. Para o negócio, isso indica o custo de alarmes falsos. Se a `Precision` estiver muito baixa, a operação pode reagir a muitos voos que não precisariam de atenção.

`Recall`

Mostra, entre os voos que realmente atrasaram, quantos o modelo conseguiu capturar. Para o negócio, esta é uma das métricas mais importantes quando o objetivo é prevenir impacto operacional. `Recall` baixo significa que o modelo está deixando atrasos relevantes passarem sem alerta.

`F1-Score`

É o equilíbrio entre `Precision` e `Recall`. Ajuda a decidir se o modelo está encontrando um ponto de trabalho saudável entre excesso de alertas e perda de atrasos reais. Em geral, é a melhor métrica para comparar modelos neste problema.

`Pred_Pos_Rate`

Mostra o percentual de voos que o modelo está classificando como atrasados. Essa métrica ajuda a avaliar se o modelo está exagerando nos alertas ou sendo conservador demais. Se esse número estiver muito distante da taxa real de atrasos da base, isso indica desequilíbrio na decisão do modelo ou threshold inadequado.

`Threshold`

É o ponto de corte usado para transformar probabilidade em decisão final. Threshold menor aumenta a chance de capturar atrasos, mas normalmente gera mais falsos positivos. Threshold maior reduz alarmes falsos, mas tende a perder atrasos reais. Essa escolha deve refletir o custo operacional de cada tipo de erro.

### Matriz de confusão

O log também mostra a matriz de confusão em formato de tabela lógica:

- `TP`: atrasos reais que o modelo acertou
- `FP`: voos sem atraso que o modelo marcou como atraso
- `FN`: atrasos reais que o modelo perdeu
- `TN`: voos sem atraso corretamente classificados

Para decisão de negócio:

- `FN` alto significa risco operacional, porque atrasos reais não serão tratados a tempo
- `FP` alto significa custo operacional, porque a equipe pode agir sem necessidade
- o melhor cenário depende da estratégia: evitar perda de atraso ou reduzir ruído operacional

### Sinais de overfitting, underfitting e ajuste equilibrado

O log compara métricas de treino e teste e gera um diagnóstico automático.

`OVERFITTING`

Significa que o modelo aprendeu muito bem o treino, mas perdeu qualidade fora dele. Em negócio, isso representa baixa confiabilidade em produção. O modelo parece forte no laboratório, mas não entrega o mesmo resultado na operação real.

`UNDERFITTING`

Significa que o modelo está simples ou pobre em representação e não consegue capturar bem o padrão nem no treino nem no teste. Em negócio, isso indica que ainda há espaço claro para melhorar a modelagem, usar mais atributos ou rever a forma de representar os dados.

`AJUSTE_EQUILIBRADO`

Significa que treino e teste estão próximos e que o modelo está generalizando melhor. Em negócio, este é o sinal mais saudável, porque indica maior previsibilidade do desempenho em ambiente real.

`Gap_F1`

Mostra a distância entre o `F1` de treino e o `F1` de teste. Gap alto sugere sobreajuste. Gap muito baixo com desempenho fraco sugere subajuste. Gap baixo com bom resultado sugere equilíbrio.

### Validações adicionais de qualidade do modelo e dos dados

Além das métricas de classificação, o projeto passou a gerar validações complementares para ajudar a decidir se o modelo é confiável e se a base usada no teste continua parecida com a base de treino.

`Isolation Forest`

Usado para medir o nível de anomalias e outliers entre treino e teste. Se a taxa de anomalias aumentar muito no teste, isso sugere que os dados avaliados estão menos parecidos com o padrão aprendido. Em negócio, isso é um sinal de risco de degradação de performance em produção.

`VIF - Variance Inflation Factor`

Usado para identificar excesso de redundância entre variáveis numéricas. VIF alto indica que várias variáveis estão carregando praticamente a mesma informação. Em negócio, isso significa complexidade sem ganho real, além de possível instabilidade em modelos lineares.

Leitura prática:

- VIF abaixo de 5: situação mais saudável
- VIF entre 5 e 10: atenção
- VIF acima de 10: forte redundância e necessidade de revisão

`Drift Detection - KS-Test`

Compara a distribuição das variáveis entre treino e teste. Se muitas variáveis aparecerem com status de drift, isso significa que o teste já está se afastando do padrão em que o modelo foi treinado. Em negócio, isso indica risco de queda de performance quando a operação muda de perfil.

`Drift Detection - PSI`

Mede mudança populacional entre treino e teste. É especialmente útil para monitoramento contínuo.

Leitura prática:

- PSI abaixo de 0,10: mudança baixa
- PSI entre 0,10 e 0,25: mudança moderada
- PSI acima de 0,25: mudança alta e necessidade de investigação

### Como usar essas saídas para decisão

Se o objetivo for capturar o máximo de atrasos possível:

- priorize `Recall`
- acompanhe `FN`
- aceite alguma perda de `Precision`, desde que a operação suporte esse volume de alertas

Se o objetivo for reduzir ruído operacional:

- priorize `Precision`
- acompanhe `FP`
- revise o threshold para evitar excesso de alarmes

Se o objetivo for equilíbrio geral:

- use `F1-Score` como métrica principal
- confirme que o diagnóstico não está em `OVERFITTING`
- valide que `KS-Test`, `PSI` e `Isolation Forest` não estão sinalizando mudança relevante entre treino e teste

Se houver muitos sinais de drift ou anomalias:

- não basta trocar algoritmo
- primeiro confirme se o dado atual ainda representa o cenário operacional real

Se houver muito `OVERFITTING`:

- simplifique o modelo, reduza complexidade ou reveja as features

Se houver `UNDERFITTING`:

- inclua mais informação útil no treino, melhore encoding das variáveis ou revise a engenharia de atributos

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