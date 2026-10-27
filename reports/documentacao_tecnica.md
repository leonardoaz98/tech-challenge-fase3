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

**Dimensões**

| Tabela | Chave | Colunas |
|---|---|---|
| `dim_municipio` | `id_municipio` | `id_municipio`, `sigla_uf` (o nome do município vem do IBGE, na base analítica) |
| `dim_uf` | `sigla_uf` | `sigla_uf`, `regiao` |
| `dim_tempo` | `ano` | `ano`, `tipo` (`medido` ou `so_meta`) |
| `dim_nivel` | `nivel_alfabetizacao` | `nivel_alfabetizacao`, `descricao`, `faixa` |

**`fato_meta_uf`** — grão `sigla_uf + ano`, 224 linhas (27 UFs + Brasil como `BR`, 2023 a 2030), rede pública

| Coluna | Tipo | Descrição |
|---|---|---|
| `sigla_uf` | texto | UF, ou `BR` para o Brasil |
| `ano` | inteiro | 2023 a 2025 medidos; 2026 a 2030 só meta |
| `taxa_realizada` | float | % de alfabetizados da rede pública. 2023 e 2024 da Base dos Dados; 2025 da planilha do INEP, que divulga inteiros |
| `percentual_participacao` | float | % de alunos avaliados |
| `meta` | float | Meta pactuada (publicação de 2024, a mesma dos municípios) |
| `gap_meta` | float | `taxa_realizada − meta` |
| `meta_atingida` | 0/1 | 1 se `taxa_realizada ≥ meta` |

O esquema de todas as tabelas da Gold está em `src/preprocessing/contrato_gold.py`, conferido ao gravar, ao ler e contra este dicionário.

**`fato_escola`** — grão `ano + id_municipio + id_escola + rede`, 79.273 linhas, a partir dos microdados por aluno

| Coluna | Tipo | Descrição |
|---|---|---|
| `id_escola` | texto | Código da escola, **mascarado pelo INEP a cada edição** |
| `rede` | inteiro | 2 = estadual, 3 = municipal, 4 = federal |
| `alunos_registrados` | inteiro | Alunos do 2º ano registrados |
| `alunos_com_resultado` | inteiro | Alunos com proficiência (presentes) |
| `alfabetizados_743` | inteiro | Alunos com proficiência ≥ 743 |
| `alfabetizados_flag` | inteiro | Alunos com `IN_ALFABETIZADO = 1` (igual ao anterior em 100% dos casos) |
| `soma_peso` / `soma_peso_alfabetizados` | float | Soma do peso amostral do INEP dos avaliados / dos alfabetizados |
| `media_proficiencia` / `desvio_proficiencia` | float | Média e desvio da proficiência na escola |
| `participacao` | float | `alunos_com_resultado / alunos_registrados × 100` |
| `taxa_escola` | float | `soma_peso_alfabetizados / soma_peso × 100` (mesma conta do indicador oficial) |

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
| 1 | Unidade = município nesta etapa | A Gold da Fase 2 é municipal e a política é decidida por rede; o microdado por aluno (tabela `alunos`, BigQuery) fica para a etapa seguinte | Gold agregada por município × ano |
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
| 18 | Microdados direto do INEP, não do BigQuery | Dado aberto, sem credencial; quem clona consegue refazer | Mesma fonte da tabela `alunos` da Base dos Dados |
| 19 | Agregar alunos por escola | Sem variável da criança, o modelo por aluno e o modelo nas contagens por escola são idênticos | 79 mil linhas em vez de 3,9 milhões |
| 20 | Taxa da escola com o peso do INEP | É a conta do indicador oficial | Reproduz o oficial com erro médio de 0,03 p.p. |
| 21 | Sem histórico de escola | Código mascarado a cada edição | 2% das escolas de 2024 com o mesmo código em 2023 |
| 22 | Participação da escola fora do modelo por aluno | Só é conhecida no dia da prova | — |
| 23 | Teste e validação separados por município no modelo por aluno | Escolas vizinhas no treino inflariam a AUC | 25% dos municípios no teste; `GroupKFold(5)` no treino |
| 24 | Tetos com o resultado real | Medir quanto ainda cabe melhorar com dados de escola | Teto da escola: AUC 0,736 |
| 25 | Simular o acaso na dispersão entre escolas | Escola pequena oscila por sorte | Metade das escolas críticas observadas seria acaso |
| 26 | Lista por município, não por escola | O código é fictício; a secretaria localiza as próprias escolas | `municipios_com_escolas_criticas.csv` |
| 27 | Busca em grade com AUC ponderada pelo peso dos alunos | Cada linha representa um grupo de alunos; o scorer padrão ignoraria o peso | 12 combinações; validação 0,652, teste 0,663 |
| 28 | Semente fixa antes de cada SHAP | O SHAP sorteia a amostra de fundo e a posição dos pontos | Gráficos idênticos a cada execução |
| 29 | Esquema da Gold num contrato único | Na Fase 2 a Gold gravada divergiu da lida pelas views | Conferido ao gravar, ao ler e contra este documento |
| 30 | Leitura do S3 retirada | Sem acesso ao bucket não havia como testar; caminho documentado e não executado não fica | Tudo roda do snapshot versionado |
| 31 | Duplicata para o pipeline em vez de `drop_duplicates` | Ficar com a primeira linha sem critério esconde erro de fonte | 0 duplicatas hoje nos três CSVs do IBGE |
| 32 | Funil de descartes | Toda linha que sai precisa de contagem e motivo | 216 municípios fora do treino = 52 sem resultado de 2024 + 164 sem meta |
| 33 | Registro acumulado, não sobrescrito | Histórico de cada execução (commit, versões, dados, métricas) | `reports/execucoes.jsonl` |
| 34 | Deriva avisa, não para | Mudança de distribuição é informação para o gestor, não erro de dado | `esforco_exigido` em alerta (PSI 1,26), demais estáveis |
| 35 | Teste automático compara métricas com tolerância | Máquinas diferentes mudam pixels e última casa decimal, não a conclusão | `referencia_metricas.json` |
| 36 | Metas do Brasil e das UFs na Gold | O enunciado lista metas nacionais e estaduais; os CSVs estavam no repositório sem uso | `fato_meta_uf`, 28 linhas por ano (2025 das UFs em inteiros, como o INEP divulga) |

---

## 3. Contrato dos scripts

| Script | Entrada | Saída |
|---|---|---|
| `src/preprocessing/gold.py` | `data/raw/inep/*.csv` e `*.xlsx` | `data/gold/*.parquet` (para se fugir do contrato) |
| `src/preprocessing/contrato_gold.py` | `data/gold/*.parquet` + esta documentação | confere esquema, tipos, chave e dicionário de dados |
| `src/preprocessing/baixar_dados_ibge.py` | APIs do IBGE | `data/raw/ibge/*.csv` (já versionados) |
| `src/preprocessing/base_analitica.py` | Gold + IBGE | `data/processed/base_treino.parquet`, `base_previsao.parquet` |
| `src/visualization/eda.py` | `base_treino` | `images/eda_*.png`, `reports/eda_resumo.json` |
| `src/modeling/treinar.py` | `base_treino` | `models/modelo.joblib`, `reports/resultado_modelo.json`, `images/modelo_precisao_recall.png` |
| `src/evaluation/interpretar.py` | modelo + `base_treino` | `images/interp_*.png`, `reports/interpretabilidade.json` |
| `src/evaluation/aplicacao.py` | modelo + as duas bases | `reports/ranking_2025.csv`, `ranking_2024_oof.csv`, `perfis.csv`, `risco_uf_2025.csv`, `aplicacao.json`, `images/aplicacao_*.png` |
| `src/preprocessing/microdados_inep.py` | zips do INEP em `data/raw/inep/microdados/` (fora do git) | `data/raw/inep/alunos_por_escola.csv`, `reports/perfil_microdados.json` |
| `src/preprocessing/escolas.py` | `alunos_por_escola.csv` + Gold | `data/gold/fato_escola.parquet`, `reports/validacao_microdados.json` (para se não reproduzir o indicador oficial) |
| `src/visualization/desigualdade.py` | `fato_escola` | `images/escolas_*.png`, `reports/escolas_resumo.json`, `reports/municipios_com_escolas_criticas.csv` |
| `src/modeling/modelo_aluno.py` | `fato_escola` + `base_treino` | `models/modelo_aluno.joblib`, `reports/resultado_modelo_aluno.json`, `images/aluno_*.png` (comparação, permutação e SHAP) |
| `src/preprocessing/funil.py` | chamado por `base_analitica.py` e `modelo_aluno.py` | `reports/funil.json` (linhas que entram e saem de cada filtro, com motivo) |
| `src/evaluation/drift.py` | as duas bases | `reports/drift.json` (PSI e % fora da faixa do treino por variável) |
| `src/run_pipeline.py` | — | roda tudo na ordem (a extração dos microdados é à parte); grava `logs/pipeline_<data>.log` e acrescenta uma linha em `reports/execucoes.jsonl` |
| `src/checar_resultados.py` | `reports/*.json` + `reports/referencia_metricas.json` | sai com erro se alguma métrica sair da tolerância |

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
| Otimização | `GridSearchCV` (23 combinações × 5 folds) no municipal; busca em grade ponderada (12 × 5) no modelo por aluno |
| Redução de overfitting | Regularização na grade; gap CV × teste dentro do desvio |
| Replicabilidade | Semente 42 em tudo; versões fixas; dados versionados; teste de 8 sementes; GitHub Actions a cada pull request |
| Monitoramento | `drift.py` (deriva treino × previsão) e `execucoes.jsonl` (histórico de métricas) |
| Feature importance | `permutation_importance` + coeficientes |
| SHAP | `shap.LinearExplainer`, resumo e por município |
| Treino, validação e teste no modelo por aluno | `GroupShuffleSplit` (25% dos municípios no teste) + `GroupKFold(5)` no treino |
| SHAP no modelo por aluno | `shap.LinearExplainer`, resumo numa amostra do teste |

---

## 5. Reprodutibilidade

```bash
pip install -r requirements.txt
python -m src.run_pipeline
```

- Conferido em ambiente limpo (venv novo, clone novo) e em duas execuções seguidas: `reports/` e `images/` saem **idênticos byte a byte** na mesma máquina, inclusive os gráficos SHAP.
- Em outra máquina, os gráficos podem mudar alguns pixels (fontes do sistema) e as métricas, a última casa decimal. Por isso o teste automático compara as métricas principais com tolerância (`reports/referencia_metricas.json`), e não os arquivos byte a byte.
- Teste automático: `.github/workflows/pipeline.yml` roda, a cada pull request e a cada push na `main`, o contrato da Gold, o pipeline completo e `checar_resultados.py`. O log e o registro da execução ficam salvos como artefato do Actions.
- Tempo: ~3 min com as duas buscas de hiperparâmetros (municipal e por aluno).
- A extração dos microdados (`microdados_inep.py`) fica fora do `run_pipeline`: depende de baixar ~40 MB do INEP, e o resultado já está versionado.
- Sem internet e sem AWS: dados brutos e Gold estão no repositório.

---

## 6. Versionamento

Uma branch por etapa, mergeada na `main` com `--no-ff` para o histórico mostrar a origem de cada mudança:

| Branch | Conteúdo |
|---|---|
| `main` (commit inicial) | Estrutura de pastas, requirements, .gitignore, .mailmap |
| `feat/base-dados` | Gold local da Fase 2, enriquecimento IBGE, base analítica |
| `feat/eda` | Análise exploratória e notebook 01 |
| `feat/modelagem` | Pipeline, GridSearch, limiar, sensibilidade sem RS |
| `feat/interpretabilidade` | Permutação, estabilidade, SHAP |
| `feat/previsao-2025` | Ranking 2025, score fora da amostra, perfis, notebook 02 |
| `docs/readme` | README e documentação técnica |
| `feat/microdados-alunos` | Extração dos microdados do INEP, `fato_escola`, validação contra o indicador oficial |
| `feat/desigualdade-escolas` | Decomposição da variância, dispersão entre escolas, escolas críticas escondidas |
| `feat/modelo-aluno` | Modelo por aluno, tetos, notebook 03 |
| `docs/readme-aluno` | README e documentação técnica com o nível de aluno |
| `fix/modelo-aluno-teste-e-busca` | Teste separado por município, busca em grade e SHAP no modelo por aluno; SHAP reprodutível |
| `refactor/download-ibge` | Script de download do IBGE reescrito |
| `fix/contrato-gold` | Contrato de esquema da Gold conferido ao gravar, ao ler e contra a documentação; leitura do S3 retirada |
| `fix/descartes-com-regra` | Duplicata no IBGE para o pipeline; funil de descartes com contagem e motivo |
| `feat/registro-execucao` | Logger padronizado com arquivo; registro acumulado de cada execução |
| `feat/monitoramento-deriva` | PSI por variável entre treino e previsão |
| `ci/pipeline-github-actions` | Pipeline do zero a cada pull request, com comparação das métricas |
| `docs/qualidade-e-monitoramento` | README e documentação técnica com qualidade, rastreabilidade e custo |
| `feat/metas-estaduais` | Metas e resultados do Brasil e das UFs na Gold (`fato_meta_uf`) |

Cada branch entrou na `main` por pull request.

Padrão de mensagem: `tipo: descrição` (`feat`, `fix`, `docs`, `chore`), corpo explicando o porquê.
