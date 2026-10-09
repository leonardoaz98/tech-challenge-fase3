# Documentação técnica

Complementa o README com o que um revisor técnico precisa para auditar o projeto: dicionário de dados, registro das decisões analíticas, contrato de cada script e como reproduzir.

---

## 1. Dicionário de dados

### 1.1 Gold (`data/gold/`, star schema da Fase 2)

**`fato_alfabetizacao`** — grão `id_municipio + ano`, 43.008 linhas

| Coluna | Tipo | Descrição |
|---|---|---|
| `id_municipio` | texto (7) | Código IBGE do município |
| `ano` | inteiro | 2023 e 2024 medidos; 2025 a 2030 só meta |
| `taxa_realizada` | float | % de alunos alfabetizados (≥ 743 pontos). Nulo em 2025+ |
| `media_portugues` | float | Média de proficiência em português. Nulo em 2025+ |
| `percentual_participacao` | float | % de alunos que fizeram a avaliação. Nulo em 2025+ |
| `meta` | float | Meta pactuada para o ano |
| `rede` | inteiro | 3 = municipal |
| `gap_meta` | float | `taxa_realizada − meta` |
| `meta_atingida` | 0/1 | 1 se `taxa_realizada ≥ meta` |
| `nivel_alfabetizacao` | 0–5 | Faixa da taxa (Crítico a Muito alto) |

**Dimensões:** `dim_municipio` (id, UF), `dim_uf` (UF, região), `dim_tempo` (ano, medido/só meta), `dim_nivel` (nível, descrição, faixa).

### 1.2 Base analítica (`data/processed/`, gerada)

Uma linha por município. `base_treino` (2023 → 2024) e `base_previsao` (2024 → 2025) têm as mesmas colunas.

| Coluna | Uso | Descrição |
|---|---|---|
| `taxa_atual` | feature | Taxa do ano de entrada |
| `media_portugues` | feature | Média em português do ano de entrada |
| `participacao` | feature | Participação do ano de entrada |
| `esforco_exigido` | feature | `meta_alvo − taxa_atual` |
| `log_populacao` | feature | `log(1 + população)`, IBGE 2021 |
| `log_pib_per_capita` | feature | `log(PIB × 1000 / população)`, IBGE 2021 |
| `sigla_uf`, `regiao`, `porte_municipio` | feature | Categóricas |
| `meta_alvo` | fora | Colinear com taxa + esforço |
| `_taxa_alvo` | fora | Resultado do ano-alvo (auditoria) |
| `risco` | **alvo** | 1 se `_taxa_alvo < meta_alvo` |
| `nome_municipio`, `microrregiao`, `mesorregiao`, `populacao`, `pib_per_capita` | identificação | Não entram no modelo |

---

## 2. Registro de decisões analíticas

Cada decisão com o motivo e a evidência que a sustenta.

| # | Decisão | Motivo | Evidência |
|---|---|---|---|
| 1 | Unidade = município, não aluno | Não há microdado individual aberto (LGPD); política é por rede | Granularidade mínima da fonte: município × rede |
| 2 | Rede municipal (código 3) | É a rede das metas municipais | Metas existem só para rede municipal |
| 3 | Treino 2023→2024, previsão 2024→2025 | Separação temporal contra vazamento | Simula o gestor no início do ano |
| 4 | Alvo = **não** atingiu | Classe de interesse da política | Recall passa a medir "falhas detectadas" |
| 5 | Remover `meta_alvo` | Colinear: `meta = taxa + esforço` | Coeficientes ilegíveis com as três |
| 6 | Remover `nivel_alfabetizacao` | Faixa da própria taxa | Determinística |
| 7 | Log em população e PIB | Assimetria forte | Metrópoles distorcem a escala linear |
| 8 | Manter participação | É gestão, não ruído | Volatilidade igual entre faixas; direção diferente (+0,3 vs +4,2 p.p.) |
| 9 | Sem rebalanceamento | Classes 47/53 | — |
| 10 | AUC como métrica principal | Mede a fila de prioridade | Independe de limiar |
| 11 | Logística se empatar (≤ 0,005) | Prestação de contas | Regra fixada antes de treinar |
| 12 | Limiar pelo recall ≥ 80% | Perder um município custa mais que um alarme falso | Escolhido fora da amostra do treino; 81% no teste |
| 13 | `LimitarFaixa` no pipeline | 36% dos municípios de 2025 fora da faixa de esforço do treino | Esforço máximo no treino: 7,2 p.p. |
| 14 | RS em 2025 por modelo sem RS | Queda de 2024 causada pelas enchentes | RS −20 p.p. vs +4,8 no resto; MEC confirma |
| 15 | AC avaliado sem efeito de UF | Sem dado de 2024 no treino | `handle_unknown="ignore"` |
| 16 | KMeans em taxa, participação, esforço | Perfis acionáveis de gestão | k = 4 pela silhueta (0,361) |
| 17 | Faixas ancoradas no limiar | Coerência entre alerta e modelo | Validadas fora da amostra: 16% a 84% de falha |

---

## 3. Contrato dos scripts

| Script | Entrada | Saída |
|---|---|---|
| `src/preprocessing/gold.py` | `data/raw/inep/*.csv` | `data/gold/*.parquet` |
| `src/preprocessing/baixar_dados_ibge.py` | APIs do IBGE | `data/raw/ibge/*.csv` (já versionados) |
| `src/preprocessing/base_analitica.py` | Gold + IBGE | `data/processed/base_treino.parquet`, `base_previsao.parquet` |
| `src/visualization/eda.py` | `base_treino` | `images/eda_*.png`, `reports/eda_resumo.json` |
| `src/modeling/treinar.py` | `base_treino` | `models/modelo.joblib`, `reports/resultado_modelo.json`, `images/modelo_precisao_recall.png` |
| `src/evaluation/interpretar.py` | modelo + `base_treino` | `images/interp_*.png`, `reports/interpretabilidade.json` |
| `src/evaluation/aplicacao.py` | modelo + as duas bases | `reports/ranking_2025.csv`, `ranking_2024_oof.csv`, `perfis.csv`, `risco_uf_2025.csv`, `aplicacao.json`, `images/aplicacao_*.png` |
| `src/run_pipeline.py` | — | roda tudo na ordem |

Parâmetros compartilhados (anos, semente, proporção de teste, folds, recall mínimo) ficam em `src/config.py`.

---

## 4. Requisitos do pipeline e onde estão no código

| Requisito do enunciado | Onde |
|---|---|
| Imputação de faltantes numéricos | `treinar.py` → `SimpleImputer(strategy="median")` |
| Transformação numérica | `StandardScaler` + `LimitarFaixa`; log em `base_analitica.py` |
| Transformação categórica / encoding | `OneHotEncoder(handle_unknown="ignore")` |
| Tratamento de data leakage | Separação temporal; `_taxa_alvo` fora; transformações no Pipeline; limiar e busca só no treino |
| Pré-processamento integrado ao modelo | `Pipeline([("prep", ColumnTransformer), ("modelo", ...)])` |
| Treino, validação e teste | 75/25 estratificado + `StratifiedKFold(5)` no treino |
| Otimização | `GridSearchCV` (23 combinações × 5 folds) |
| Redução de overfitting | Regularização na grade; gap CV × teste dentro do desvio |
| Replicabilidade | Semente 42 em tudo; versões fixas; dados versionados; teste de 8 sementes |
| Feature importance | `permutation_importance` + coeficientes |
| SHAP | `shap.LinearExplainer`, resumo e por município |

---

## 5. Reprodutibilidade

```bash
pip install -r requirements.txt
python -m src.run_pipeline
```

- Conferido em ambiente limpo (venv novo, clone novo): os relatórios em `reports/` saem **idênticos byte a byte**. Só os PNG mudam nos metadados de renderização.
- Tempo: ~1min40s com busca de hiperparâmetros.
- Sem internet e sem AWS: dados brutos e Gold estão no repositório.
- Com AWS: `python -m src.run_pipeline --s3` lê a Gold do data lake da Fase 2.

---

## 6. Versionamento

Uma branch por etapa, mergeada na `main` com `--no-ff` para o histórico mostrar a origem de cada mudança:

| Branch | Conteúdo |
|---|---|
| `main` (commit inicial) | Estrutura de pastas, requirements, .gitignore |
| `feat/base-dados` | Gold local da Fase 2, enriquecimento IBGE, base analítica |
| `feat/eda` | Análise exploratória e notebook 01 |
| `feat/modelagem` | Pipeline, GridSearch, limiar, sensibilidade sem RS |
| `feat/interpretabilidade` | Permutação, estabilidade, SHAP |
| `feat/previsao-2025` | Ranking 2025, score fora da amostra, perfis, notebook 02 |
| `docs/readme` | README, documentação técnica, roteiro do vídeo |

Cada branch entrou na `main` por pull request.

Padrão de mensagem: `tipo: descrição` (`feat`, `fix`, `docs`, `chore`), corpo explicando o porquê.
