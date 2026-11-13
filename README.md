# Recomendação Musical por Agregação de Vizinhança Aprendida em Grafos

## Integrantes

*Arthur Jones Bicalho - 10420317 - 10420317@mackenzista.com.br*
*Alex Cruz de Santana - 10420357 - 10420357@mackenzista.com.br*
*Luis Felipe Santos do Nascimento - 10420572 - luisfelipesdn12@gmail.com*

## Dataset utilizado
 
**Last.fm-360K** (Celma, 2010), distribuído pelo Music Technology Group (UPF). Contém interações usuário-artista (contagem de reproduções), com identificador de usuário anonimizado na origem (hash SHA1). Fonte: https://www.upf.edu/web/mtg/lastfm360k
 
## Como reproduzir

Requer Python 3.10 ou superior. Todos os comandos são executados a partir da raiz do repositório.

```bash
python -m venv .venv
source .venv/bin/activate            # no Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_dataset.py   # baixa (569 MB) e extrai os dois .tsv em data/raw/
```

Depois, execute os notebooks na ordem (pelo Jupyter ou pela linha de comando):

```bash
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/01_eda_p1.ipynb
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/02_preparacao_split.ipynb
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/03_hiperparametros_graphsage.ipynb
python -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/04_avaliacao.ipynb
```

Os notebooks localizam a raiz do projeto sozinhos, então funcionam tanto abertos a partir da raiz quanto a partir da pasta `notebooks/`. Os arquivos `.tsv` são encontrados em `data/raw/` ou em qualquer subpasta dela (por exemplo `data/raw/lastfm-dataset-360K/`, que é onde o `.tar.gz` os coloca quando extraído manualmente).

Tempo aproximado em CPU (8 núcleos, sem GPU): notebooks 01 e 02, cerca de 10 minutos cada; notebook 03 (grade de 8 configurações), cerca de 1 h 40 min; notebook 04 (3 sementes), cerca de 1 h 30 min. Os notebooks 03 e 04 guardam cada treino concluído em `results/cache/`, então uma execução interrompida continua de onde parou. São necessários cerca de 8 GB de memória livre.

## Notebooks

- [`01_eda_p1.ipynb`](notebooks/01_eda_p1.ipynb): análise exploratória (carregamento, valores nulos, contagem de usuários e artistas, distribuição de reproduções, artistas mais ouvidos, esparsidade, cauda longa).
- [`02_preparacao_split.ipynb`](notebooks/02_preparacao_split.ipynb): limpeza, filtros de atividade, subamostra de 20 mil usuários, divisão 80/10/10 por usuário, verificações e ajuste do ALS na validação.
- [`03_hiperparametros_graphsage.ipynb`](notebooks/03_hiperparametros_graphsage.ipynb): grade de hiperparâmetros do GraphSAGE na validação (T, dimensão e taxa de aprendizado).
- [`04_avaliacao.ipynb`](notebooks/04_avaliacao.ipynb): rodada final no teste com 3 sementes, tabela principal, bootstrap pareado, figuras, NDCG por quartil de atividade e análise dos pesos aprendidos.

## Código (`src/`)

| Arquivo | Conteúdo |
|---|---|
| `data.py` | leitura dos dados brutos, limpeza, filtros, subamostra e divisão |
| `walks.py` | grafo de treino, caminhadas aleatórias e vizinhança top-T |
| `baselines.py` | popularidade, caminhada aleatória (estilo Pixie) e ALS |
| `model.py` | GraphSAGE com pesos de agregação fixos ou aprendidos |
| `train.py` | treino com perda BPR e parada antecipada |
| `metrics.py` | Precision, Recall, NDCG, cobertura e popularidade @K |
| `experiments.py` | execução da grade e das sementes em processos paralelos, com cache |

## Resultados (teste, média ± desvio-padrão de 3 sementes)

| Modelo | Precision@10 | Recall@10 | NDCG@10 | Cobertura@10 | Popularidade@10 | Treino (s) |
|---|---|---|---|---|---|---|
| Popularidade | 0,0174 | 0,0396 | 0,0341 | 0,002 | 7,80 | 0 |
| Caminhada (Pixie) | 0,0152 ± 0,0001 | 0,0349 ± 0,0003 | 0,0275 ± 0,0005 | 0,672 ± 0,005 | 6,87 | 2 |
| ALS | 0,0699 ± 0,0003 | 0,1621 ± 0,0006 | 0,1449 ± 0,0004 | 0,169 ± 0,000 | 6,11 | 20 |
| GraphSAGE (pesos fixos) | 0,0492 ± 0,0011 | 0,1144 ± 0,0028 | 0,0934 ± 0,0024 | 0,561 ± 0,032 | 5,77 ± 0,07 | 1.128 |
| GraphSAGE (pesos aprendidos) | 0,0496 ± 0,0005 | 0,1157 ± 0,0013 | 0,0949 ± 0,0010 | 0,535 ± 0,065 | 5,79 ± 0,16 | 3.607 |

Configuração final: T = 20 vizinhos, dimensão 32, taxa de aprendizado 0,01, até 100 épocas com parada antecipada; ALS com 128 fatores, regularização 0,01 e alpha 1. As tabelas e figuras completas ficam em `results/`, e a discussão está no relatório em `docs/`.
