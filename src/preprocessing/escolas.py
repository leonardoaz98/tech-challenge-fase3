"""
Escolas: do microdado por aluno para a Gold, com validacao contra o INEP.

Le data/raw/inep/alunos_por_escola.csv (gerado por microdados_inep.py) e
grava data/gold/fato_escola.parquet: uma linha por escola x ano x rede.

TAXA PONDERADA
--------------
O INEP calcula o indicador com o peso de cada aluno (VL_PESO_ALUNO_LP),
que compensa quem faltou. A taxa da escola e, portanto:

    soma do peso dos alfabetizados / soma do peso dos avaliados

VALIDACAO
---------
Somando as escolas da rede municipal de cada municipio, a taxa tem de
reproduzir o indicador oficial que ja esta na Gold. Se nao reproduzir, ha
erro na leitura ou no peso, e o resto do bloco nao deve rodar.

CODIGO DA ESCOLA
----------------
O ID_ESCOLA e mascarado pelo INEP a cada edicao: so 2% das escolas de 2024
tem o mesmo codigo em 2023. Nao ha como acompanhar uma escola entre anos.

Execucao:
    python -m src.preprocessing.escolas
"""
import json
import sys

import numpy as np
import pandas as pd

from src.config import GOLD, RAW_INEP, REDE_MUNICIPAL, REPORTS
from src.preprocessing.contrato_gold import checar
from src.preprocessing.gold import carregar

TOLERANCIA_PP = 0.1          # diferenca aceitavel contra o indicador oficial
MINIMO_MUNICIPIOS_OK = 0.95  # fracao de municipios dentro da tolerancia


def construir() -> pd.DataFrame:
    d = pd.read_csv(RAW_INEP / "alunos_por_escola.csv",
                    dtype={"id_municipio": str, "id_escola": str, "rede": str})
    d["rede"] = d["rede"].astype(int)
    d["participacao"] = (d["alunos_com_resultado"] / d["alunos_registrados"] * 100).round(2)
    d["taxa_escola"] = np.where(
        d["soma_peso"] > 0, d["soma_peso_alfabetizados"] / d["soma_peso"] * 100, np.nan
    ).round(2)
    return d.sort_values(["ano", "id_municipio", "id_escola"]).reset_index(drop=True)


def validar(escolas: pd.DataFrame) -> dict:
    mun = (escolas[escolas.rede == REDE_MUNICIPAL]
           .groupby(["ano", "id_municipio"])[["soma_peso", "soma_peso_alfabetizados"]]
           .sum().reset_index())
    mun["taxa_microdado"] = mun.soma_peso_alfabetizados / mun.soma_peso * 100
    fato = carregar("fato_alfabetizacao")
    j = mun.merge(fato[["id_municipio", "ano", "taxa_realizada"]], on=["id_municipio", "ano"])
    erro = (j.taxa_microdado - j.taxa_realizada).abs()
    return {
        "municipios_comparados": int(len(j)),
        "erro_medio_pp": round(float(erro.mean()), 3),
        "erro_p95_pp": round(float(erro.quantile(0.95)), 3),
        "pct_dentro_de_0_1_pp": round(float((erro < TOLERANCIA_PP).mean() * 100), 1),
    }


def main() -> None:
    escolas = construir()
    v = validar(escolas)
    print(json.dumps(v, indent=2))
    if v["pct_dentro_de_0_1_pp"] < MINIMO_MUNICIPIOS_OK * 100:
        sys.exit("Microdado nao reproduz o indicador oficial. Revise a extracao.")

    perfil = json.loads((REPORTS / "perfil_microdados.json").read_text())
    v["escolas_mesmo_codigo_2023_2024_pct"] = perfil["pct_escolas_2024_que_existiam_em_2023"]
    (REPORTS / "validacao_microdados.json").write_text(json.dumps(v, indent=2))

    checar("fato_escola", escolas).to_parquet(GOLD / "fato_escola.parquet", index=False)
    print(f"\n  fato_escola: {len(escolas)} linhas")
    print(escolas.groupby(["ano", "rede"]).agg(
        escolas=("id_escola", "size"), alunos=("alunos_registrados", "sum")).to_string())


if __name__ == "__main__":
    main()
