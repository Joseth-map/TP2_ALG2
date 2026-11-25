# TP2 — Análise Empírica e Otimização para o Problema do Caixeiro Viajante

Trabalho prático da disciplina DCC207 (Algoritmos 2), Departamento de Ciência da Computação, UFMG. Prof. Renato Vimieiro.

**Autor:** Joseph Patrick Alves de Moura (josephmoura@ufmg.br)

## Conteúdo do repositório

```
.
├── aproximativos.py        # leitura TSPLIB, matriz D, Prim, TAT e Christofides
├── branch_bound.py         # Branch-and-Bound com cinco estratégias de warm-start
├── worker.py               # executa uma única tarefa (modo tempo ou memória)
├── harness.py              # orquestra as tarefas, retomada e CSVs
├── features.py             # features geométricas por instância
├── experimentos.json       # lista de instâncias e configuração
├── requirements.txt        # Bibliotecas necessárias
├── analise/
│   └── regressao.py        # complexidade empírica, sensibilidade e resumo
├── instancias/_extraidos/  # arquivos .tsp e .opt.tour da TSPLIB
├── resultados/             # CSVs mesclados (execuções, profiling, features, ambiente)
├── relatorio/              # artigo em LaTeX (SBC) e PDF
└── anexo_prompts/          # histórico de prompts usados com IA generativa
```

## Requisitos

- Python 3.10 ou superior
- Bibliotecas: `numpy`, `scipy`, `pandas`, `networkx`, `matplotlib`, `scikit-learn`, `psutil`

```
pip install -r requirements.txt
```

Para compilar o relatório: uma distribuição LaTeX com o template SBC (`sbc.cls`).

## Instâncias

As 12 instâncias são da TSPLIB (<http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/>): burma14, ulysses16, gr17, gr21, ulysses22, gr24, fri26, bayg29, bays29, att48, berlin52 e st70. Coloque os arquivos `.tsp` e `.opt.tour` em `instancias/_extraidos/`, com os nomes exatos listados em `experimentos.json`.

Os tipos de distância são GEO, ATT, EUC_2D e EXPLICIT. O professor confirmou que qualquer instância cuja função de custo seja uma distância pode ser usada.

## Como reproduzir

Os experimentos de tempo e memória rodam em separado, porque o `tracemalloc` aumenta o tempo de execução em 5 a 6 vezes.

**1. Experimento de tempo** (uma máquina, 1 worker, 3 repetições)

```
python harness.py --config experimentos.json --modo tempo --workers 1 --reps 3
```

**2. Experimento de memória** (1 repetição; pode ser dividido entre máquinas por instância, usando configs separados)

```
python harness.py --config experimentos_B.json --modo memoria --workers 1 --reps 1
```

O harness retoma de onde parou: tarefas concluídas em `resultados/execucoes.csv` são puladas, e falhas (`motivo_NA = erro`) são reexecutadas.

**3. Features geométricas**

```
python features.py
```

**4. Análise**

```
python analise/regressao.py
```

Os resultados vão para `resultados/regressao_resumo.txt` e para as tabelas em `resultados/`.

## Máquinas usadas

| Modo | Máquina | CPU | Núcleos (físicos/lógicos) | RAM |
|---|---|---|---|---|
| tempo | AMD64 Family 23 Model 104 | AMD | 8 / 16 | 31,4 GB |
| memória | Intel64 Family 6 Model 126 | Intel | 2 / 4 | 15,7 GB |
| memória | Intel64 Family 6 Model 60 | Intel | 4 / 8 | 15,9 GB |

Python 3.13.1 (tempo) e 3.15.2 (memória); NumPy 2,3 e 2,5; NetworkX 3,7. Os tempos e as memórias vêm de hardware diferente e não devem ser comparados entre modos. O registro completo está em `resultados/ambiente.csv`.

## Limitações

- Amostra de 12 instâncias; o B&B concluiu apenas as instâncias com N ≤ 21.
- Timeouts (1800 s) são censura à direita, não falhas.
- `tracemalloc` mede apenas alocações Python, não a memória total do processo.
- Features de GEO e ATT são calculadas sobre coordenadas não euclidianas.
- Correlações de Spearman com n = 6 são indícios, não conclusões causais.

## Licença

Trabalho acadêmico, sem licença de reuso definida.
