"""
Monitoramento de deriva: a base de previsao ainda se parece com a de treino?

O modelo aprendeu com 2023 -> 2024 e e aplicado em 2024 -> 2025. Se as
variaveis mudarem muito de distribuicao, o ranking de 2025 merece mais
cautela. Esta etapa mede essa mudanca a cada execucao.

PSI (Population Stability Index), por variavel:
    abaixo de 0,10   estavel
    0,10 a 0,25      atencao
    acima de 0,25    alerta

Numericas: 10 faixas pelos decis do treino, mais a % da previsao fora da
faixa vista no treino. Categoricas: proporcao de cada categoria; categoria
que nao existia no treino conta como fora da faixa.

Nao para o pipeline: deriva e aviso para quem usa o ranking, nao erro de
dado. Os erros de dado param o pipeline em contrato_gold.py.

Saida: reports/drift.json

Execucao:
    python -m src.evaluation.drift
"""
import json

import numpy as np
import pandas as pd

from src.config import PROCESSED, REPORTS
from src.preprocessing.base_analitica import CATEGORICAS, NUMERICAS

LIMITE_ATENCAO = 0.10
LIMITE_ALERTA = 0.25
PISO = 1e-4  # evita log(0) quando uma faixa fica vazia


def psi(esperado: np.ndarray, atual: np.ndarray) -> float:
    e, a = np.clip(esperado, PISO, None), np.clip(atual, PISO, None)
    return float(np.sum((a - e) * np.log(a / e)))


def nivel(valor: float) -> str:
    if valor > LIMITE_ALERTA:
        return "alerta"
    return "atencao" if valor >= LIMITE_ATENCAO else "estavel"


def numerica(treino: pd.Series, atual: pd.Series) -> dict:
    treino, atual = treino.dropna(), atual.dropna()
    cortes = np.unique(np.quantile(treino, np.linspace(0, 1, 11)))
    cortes[0], cortes[-1] = -np.inf, np.inf
    def proporcoes(s):
        return pd.cut(s, cortes).value_counts(normalize=True, sort=False).to_numpy()
    fora = ((atual < treino.min()) | (atual > treino.max())).mean()
    return {"tipo": "numerica", "psi": psi(proporcoes(treino), proporcoes(atual)),
            "pct_fora_da_faixa_do_treino": float(fora * 100),
            "media_treino": float(treino.mean()), "media_previsao": float(atual.mean())}


def categorica(treino: pd.Series, atual: pd.Series) -> dict:
    categorias = sorted(set(treino.dropna()) | set(atual.dropna()))
    def proporcoes(s):
        return s.value_counts(normalize=True).reindex(categorias, fill_value=0).to_numpy()
    novas = sorted(set(atual.dropna()) - set(treino.dropna()))
    return {"tipo": "categorica", "psi": psi(proporcoes(treino), proporcoes(atual)),
            "pct_fora_da_faixa_do_treino": float(atual.isin(novas).mean() * 100),
            "categorias_novas": novas}


def main() -> None:
    treino = pd.read_parquet(PROCESSED / "base_treino.parquet")
    previsao = pd.read_parquet(PROCESSED / "base_previsao.parquet")

    variaveis = {}
    for col in NUMERICAS:
        variaveis[col] = numerica(treino[col], previsao[col])
    for col in CATEGORICAS:
        variaveis[col] = categorica(treino[col], previsao[col])
    for v in variaveis.values():
        v["nivel"] = nivel(v["psi"])
        for k, x in v.items():
            if isinstance(x, float):
                v[k] = round(x, 4)

    em = {n: sorted(c for c, v in variaveis.items() if v["nivel"] == n)
          for n in ["alerta", "atencao"]}
    resultado = {
        "referencia": "base de treino (2023 -> 2024)",
        "comparada": "base de previsao (2024 -> 2025)",
        "limites_psi": {"atencao": LIMITE_ATENCAO, "alerta": LIMITE_ALERTA},
        "variaveis": variaveis,
        "resumo": {"qtd_alerta": len(em["alerta"]), "em_alerta": em["alerta"],
                   "qtd_atencao": len(em["atencao"]), "em_atencao": em["atencao"]},
    }
    (REPORTS / "drift.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"  {'variavel':22} {'PSI':>7}  {'fora do treino':>14}  nivel")
    for col, v in sorted(variaveis.items(), key=lambda kv: -kv[1]["psi"]):
        print(f"  {col:22} {v['psi']:>7.3f}  {v['pct_fora_da_faixa_do_treino']:>13.1f}%  {v['nivel']}")
    print(f"\n  em alerta: {', '.join(em['alerta']) or 'nenhuma'}")
    print(f"  em atencao: {', '.join(em['atencao']) or 'nenhuma'}")


if __name__ == "__main__":
    main()
