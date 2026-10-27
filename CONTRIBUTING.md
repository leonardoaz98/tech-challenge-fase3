# Como trabalhar neste repositório

O fluxo é o mesmo para qualquer pessoa do grupo: nada entra na `main` sem branch, pull request e teste automático.

## 1. Uma branch por mudança

    git checkout main && git pull
    git checkout -b tipo/descricao-curta

| Prefixo | Quando usar | Exemplo |
|---|---|---|
| `feat/` | Análise, modelo ou dado novo | `feat/resultado-2025` |
| `fix/` | Correção de algo que estava errado | `fix/contrato-gold` |
| `docs/` | README, documentação técnica, notebooks | `docs/validacao-2025` |
| `ci/` | Teste automático | `ci/pipeline-github-actions` |
| `refactor/` | Reescrita sem mudar resultado | `refactor/download-ibge` |

## 2. Commits que explicam o porquê

    tipo: o que mudou, em uma linha

    Por que mudou e o que isso muda no resultado. O "o quê" já está no diff.

## 3. Antes de abrir o pull request

    python -m src.run_pipeline              # roda tudo do zero
    python -m src.preprocessing.contrato_gold
    python -m src.checar_resultados         # métricas dentro da tolerância?

- Se uma métrica mudou **de propósito**, grave a referência nova (`python -m src.checar_resultados --gravar`) num commit próprio, explicando o motivo.
- Toda decisão analítica nova entra no registro da [documentação técnica](reports/documentacao_tecnica.md#2-registro-de-decisões-analíticas).
- Gráficos regerados sem mudança de conteúdo não entram no commit: mudam alguns pixels de máquina para máquina.

## 4. Pull request

- Preencha o modelo que o GitHub abre (`.github/pull_request_template.md`).
- O GitHub Actions roda o pipeline do zero numa máquina limpa. Pull request com o teste vermelho não entra.
- Revisão: outra pessoa do grupo lê o diff e a descrição antes do merge. O merge é feito com `--merge`, para o histórico mostrar de qual branch cada mudança veio.
