"""
Validacao real: a previsao de 2025 contra o resultado de 2025.

O ranking de 2025 (aplicacao.py) usa so o que se sabia no inicio de 2025:
resultado de 2024 e meta de 2025. O resultado de 2025 nunca entra na base
de previsao (base_analitica.py o descarta). Aqui ele e usado pela primeira
vez, so para conferir a previsao.

O QUE E MEDIDO
--------------
- AUC do modelo completo e da probabilidade recomendada (RS sem o choque)
- AUC de uma regra simples: ordenar so pelo esforco exigido (meta - taxa)
- taxa real de falha por faixa de risco e no alerta (precisao e recall)
- Rio Grande do Sul: a hipotese de recuperacao depois das enchentes
- municipios com esforco acima do maximo visto no treino
- calibracao: falha prevista em media x falha real
- robustez: as mesmas contas com o indicador oficial arredondado do INEP
- UFs: quantas bateram a propria meta de 2025 (fato_meta_uf)
- por UF: % em alerta x % que de fato nao bateu a meta, e onde estao os criticos

Saidas: reports/validacao_2025.json, images/validacao_2025.png

Execucao:
    python -m src.evaluation.validacao_2025
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.config import IMAGES, REPORTS
from src.preprocessing.gold import PLANILHA_MUNICIPIOS_2025, carregar, ler_planilha_inep
from src.visualization import estilo
from src.visualization.estilo import plt

ANO = 2025
ORDEM_FAIXAS = ["baixo", "moderado", "alto", "critico"]


def auc(y: pd.Series, score: pd.Series) -> float:
    return round(float(roc_auc_score(y, score)), 4)


def montar() -> tuple[pd.DataFrame, dict]:
    ranking = pd.read_csv(REPORTS / f"ranking_{ANO}.csv", dtype={"id_municipio": str})
    fato = carregar("fato_alfabetizacao")
    real = fato[fato["ano"] == ANO][["id_municipio", "taxa_realizada", "meta"]]
    base = ranking.merge(real, on="id_municipio", how="left")
    # a meta do ranking e a mesma da Gold: confere
    assert np.allclose(base["meta_2025"], base["meta"]), "meta do ranking difere da Gold"
    contagem = {
        "municipios_previstos": int(len(base)),
        "com_resultado_2025": int(base["taxa_realizada"].notna().sum()),
        "sem_resultado_2025": int(base["taxa_realizada"].isna().sum()),
    }
    base = base[base["taxa_realizada"].notna()].copy()
    base["falhou"] = (base["taxa_realizada"] < base["meta_2025"]).astype(int)
    return base, contagem


def resumo_alerta(base: pd.DataFrame) -> dict:
    a = base["alerta"].astype(bool)
    return {
        "municipios_em_alerta": int(a.sum()),
        "precisao": round(float(base.loc[a, "falhou"].mean()), 4),
        "recall": round(float((a & (base["falhou"] == 1)).sum() / base["falhou"].sum()), 4),
    }


def robustez_indicador_oficial(base: pd.DataFrame) -> dict:
    """Mesmas contas com a planilha oficial (taxa e meta em inteiros)."""
    p = ler_planilha_inep(PLANILHA_MUNICIPIOS_2025)
    oficial = pd.DataFrame({
        "id_municipio": p["CO_MUNICIPIO"].astype("Int64").astype(str).str.zfill(7),
        "taxa_oficial": pd.to_numeric(p["PC_ALUNO_ALFABETIZADO_2025"], errors="coerce"),
        "meta_oficial": pd.to_numeric(p["META_FINAL_2025"], errors="coerce"),
    })
    b = base.merge(oficial, on="id_municipio", how="inner").dropna(subset=["taxa_oficial"])
    b["falhou_oficial"] = (b["taxa_oficial"] < b["meta_oficial"]).astype(int)
    return {
        "municipios": int(len(b)),
        "taxa_falha_real": round(float(b["falhou_oficial"].mean()), 4),
        "concordancia_com_microdado": round(float((b["falhou_oficial"] == b["falhou"]).mean()), 4),
        "auc_modelo": auc(b["falhou_oficial"], b["prob_risco"]),
        "auc_regra_simples_esforco": auc(b["falhou_oficial"], b["esforco_exigido"]),
    }


def resumo_ufs() -> dict:
    uf = carregar("fato_meta_uf")
    u = uf[(uf["ano"] == ANO) & (uf["sigla_uf"] != "BR")]
    br = uf[(uf["ano"] == ANO) & (uf["sigla_uf"] == "BR")].iloc[0]
    com_meta = u[u["meta_atingida"].notna()]
    return {
        "brasil": {"taxa": float(br["taxa_realizada"]), "meta": float(br["meta"]),
                   "atingiu": bool(br["meta_atingida"] == 1)},
        "ufs_com_meta": int(len(com_meta)),
        "ufs_que_atingiram": int((com_meta["meta_atingida"] == 1).sum()),
        "ufs_abaixo_da_meta": sorted(com_meta.loc[com_meta["meta_atingida"] == 0, "sigla_uf"]),
    }


def grafico(por_faixa: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(por_faixa))
    ax.bar(x - 0.2, por_faixa["falha_prevista"] * 100, 0.4, label="prevista (média do modelo)",
           color=estilo.TINTA_2)
    ax.bar(x + 0.2, por_faixa["falha_real"] * 100, 0.4, label="real em 2025", color=estilo.COR_RISCO)
    for i, (p, r) in enumerate(zip(por_faixa["falha_prevista"], por_faixa["falha_real"])):
        ax.text(i - 0.2, p * 100 + 1, f"{p * 100:.0f}%", ha="center", fontsize=8)
        ax.text(i + 0.2, r * 100 + 1, f"{r * 100:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x, [f"{f}\n({n})" for f, n in zip(por_faixa.index, por_faixa["municipios"])])
    ax.set_ylim(0, 100)
    ax.set_ylabel("% que não atingiu a meta de 2025")
    ax.set_title("Previsão de 2025 x resultado real, por faixa de risco")
    ax.legend(frameon=False, fontsize=9)
    estilo.salvar(fig, IMAGES / "validacao_2025.png")


def main() -> None:
    base, contagem = montar()
    por_faixa = (base.groupby("faixa")
                 .agg(municipios=("falhou", "size"), falha_prevista=("prob_risco", "mean"),
                      falha_real=("falhou", "mean"))
                 .reindex(ORDEM_FAIXAS))
    rs = base[base["sigla_uf"] == "RS"]
    acima = base.groupby("esforco_acima_do_treino")["falhou"].agg(["size", "mean"])
    por_uf = (base.assign(alerta=base["alerta"].astype(bool))
              .groupby("sigla_uf")
              .agg(municipios=("falhou", "size"), pct_alerta=("alerta", "mean"),
                   falha_real=("falhou", "mean"))
              .sort_values(["pct_alerta", "municipios"], ascending=False))
    criticos = (base[base["faixa"] == "critico"].groupby("sigla_uf")
                .agg(criticos=("falhou", "size"), falha_real=("falhou", "mean"))
                .sort_values("criticos", ascending=False).head(5))

    resultado = {
        **contagem,
        "taxa_falha_real": round(float(base["falhou"].mean()), 4),
        "taxa_falha_prevista_media": round(float(base["prob_risco"].mean()), 4),
        "auc_modelo": auc(base["falhou"], base["prob_risco"]),
        "auc_probabilidade_recomendada": auc(base["falhou"], base["prob_recomendada"]),
        "auc_regra_simples_esforco": auc(base["falhou"], base["esforco_exigido"]),
        "alerta": resumo_alerta(base),
        "por_faixa": por_faixa.round(4).reset_index().to_dict("records"),
        "rio_grande_do_sul": {
            "municipios": int(len(rs)),
            "falha_real": round(float(rs["falhou"].mean()), 4),
            "prob_media_modelo_completo": round(float(rs["prob_risco"].mean()), 4),
            "prob_media_sem_choque": round(float(rs["prob_risco_sem_rs"].mean()), 4),
        },
        "esforco_acima_do_treino": {
            "sim": {"municipios": int(acima.loc[True, "size"]), "falha_real": round(float(acima.loc[True, "mean"]), 4)},
            "nao": {"municipios": int(acima.loc[False, "size"]), "falha_real": round(float(acima.loc[False, "mean"]), 4)},
        },
        "por_uf": por_uf.round(4).reset_index().to_dict("records"),
        "criticos_por_uf_top5": criticos.round(4).reset_index().to_dict("records"),
        "robustez_indicador_oficial": robustez_indicador_oficial(base),
        "ufs": resumo_ufs(),
    }
    (REPORTS / "validacao_2025.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    grafico(por_faixa)

    print(f"  municipios previstos {contagem['municipios_previstos']}, "
          f"com resultado em 2025 {contagem['com_resultado_2025']}")
    print(f"  falha real {resultado['taxa_falha_real']:.1%} | prevista em media "
          f"{resultado['taxa_falha_prevista_media']:.1%}")
    print(f"  AUC modelo {resultado['auc_modelo']} | recomendada "
          f"{resultado['auc_probabilidade_recomendada']} | regra simples (esforco) "
          f"{resultado['auc_regra_simples_esforco']}")
    print("\n" + por_faixa.round(3).to_string())
    print(f"\n  RS: falha real {resultado['rio_grande_do_sul']['falha_real']:.1%}")
    print(f"  robustez (indicador oficial): {resultado['robustez_indicador_oficial']}")
    print(f"  UFs: {resultado['ufs']}")


if __name__ == "__main__":
    main()
