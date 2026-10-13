"""
Analise exploratoria da base de treino (2023 -> 2024).

Cada bloco responde uma pergunta que muda alguma decisao de modelagem.
Os numeros vao para reports/eda_resumo.json e os graficos para images/.

Execucao:
    python -m src.visualization.eda
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import ALVO, IMAGES, PROCESSED, REPORTS, SEMENTE
from src.preprocessing.base_analitica import CATEGORICAS, NUMERICAS
from src.preprocessing.gold import carregar
from src.visualization import estilo
from src.visualization.estilo import plt

ROTULOS = {
    "taxa_atual": "Taxa de alfabetização (%)",
    "media_portugues": "Média em português (pontos)",
    "participacao": "Participação na avaliação (%)",
    "esforco_exigido": "Esforço exigido (meta − taxa, p.p.)",
    "log_populacao": "log(população)",
    "log_pib_per_capita": "log(PIB por habitante)",
}


def auc_univariada(base: pd.DataFrame) -> pd.DataFrame:
    """Poder de separacao de cada variavel sozinha (0,5 = nenhum).

    Categoricas: cada categoria vira a taxa de risco do grupo, calculada
    fora da propria particao (out-of-fold), para nao vazar o alvo.
    """
    y = base[ALVO].values
    linhas = []
    for c in NUMERICAS:
        auc = roc_auc_score(y, base[c])
        linhas.append({"variavel": c, "auc": max(auc, 1 - auc),
                       "direcao": "maior -> mais risco" if auc > 0.5 else "maior -> menos risco"})
    cv = StratifiedKFold(5, shuffle=True, random_state=SEMENTE)
    for c in CATEGORICAS:
        oof = np.zeros(len(base))
        for tr, te in cv.split(base, y):
            medias = base.iloc[tr].groupby(c)[ALVO].mean()
            oof[te] = base.iloc[te][c].map(medias).fillna(y[tr].mean())
        linhas.append({"variavel": c, "auc": roc_auc_score(y, oof), "direcao": "categorica"})
    return pd.DataFrame(linhas).sort_values("auc", ascending=False)


def grafico_distribuicoes(base: pd.DataFrame) -> None:
    fig, eixos = plt.subplots(2, 3, figsize=(13, 7))
    for ax, c in zip(eixos.flat, NUMERICAS):
        faixa = np.linspace(base[c].quantile(0.005), base[c].quantile(0.995), 40)
        for valor, cor, nome in [(0, estilo.COR_ATINGIU, "Atingiu"),
                                 (1, estilo.COR_RISCO, "Não atingiu")]:
            ax.hist(base.loc[base[ALVO] == valor, c], bins=faixa, density=True,
                    histtype="step", linewidth=2, color=cor, label=nome)
        ax.set_title(ROTULOS[c], fontsize=10)
        ax.set_yticks([])
    eixos.flat[0].legend(loc="upper left")
    fig.suptitle("Distribuição de cada variável por resultado em 2024",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    estilo.salvar(fig, IMAGES / "eda_distribuicoes.png")


def grafico_correlacoes(base: pd.DataFrame) -> pd.DataFrame:
    corr = base[NUMERICAS + [ALVO]].corr()
    fig, ax = plt.subplots(figsize=(7.5, 6))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("div", estilo.DIVERGENTE)
    im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    nomes = [c.replace("_", " ") for c in corr.columns]
    ax.set_xticks(range(len(nomes)), nomes, rotation=40, ha="right")
    ax.set_yticks(range(len(nomes)), nomes)
    ax.grid(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color=estilo.TINTA)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlação entre variáveis numéricas e o alvo")
    estilo.salvar(fig, IMAGES / "eda_correlacoes.png")
    return corr


def grafico_risco_uf(base: pd.DataFrame) -> pd.DataFrame:
    uf = (base.groupby("sigla_uf")
          .agg(municipios=(ALVO, "size"), risco=(ALVO, "mean"))
          .sort_values("risco"))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(uf.index, uf["risco"] * 100, color=estilo.COR_RISCO, width=0.7)
    media = base[ALVO].mean() * 100
    ax.axhline(media, color=estilo.TINTA_2, linewidth=1, linestyle="--")
    ax.text(0, media + 2, f"Brasil: {media:.0f}%", color=estilo.TINTA_2, fontsize=9)
    for nome in [uf.index[0], uf.index[-1]]:
        v = uf.loc[nome, "risco"] * 100
        ax.text(nome, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9)
    ax.set_ylabel("% de municípios que não atingiram a meta")
    ax.set_ylim(0, 100)
    ax.set_title("Não atingimento da meta de 2024 por UF")
    estilo.salvar(fig, IMAGES / "eda_risco_por_uf.png")
    return uf


def grafico_participacao(base: pd.DataFrame) -> pd.DataFrame:
    faixas = pd.cut(base["participacao"], [0, 85, 90, 95, 101],
                    labels=["< 85%", "85–90%", "90–95%", "≥ 95%"], right=False)
    tab = base.groupby(faixas, observed=True).agg(
        municipios=(ALVO, "size"), risco=(ALVO, "mean"),
        variacao=("variacao", "mean"), volatilidade=("variacao", "std"))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(tab.index.astype(str), tab["risco"] * 100, color=estilo.COR_RISCO, width=0.6)
    for i, v in enumerate(tab["risco"] * 100):
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9)
    ax.set_ylim(0, 75)
    ax.set_xlabel("Participação na avaliação de 2023")
    ax.set_ylabel("% que não atingiu a meta de 2024")
    ax.set_title("Mais participação, menos risco")
    estilo.salvar(fig, IMAGES / "eda_participacao.png")
    return tab


def grafico_teto(base: pd.DataFrame) -> dict:
    """Ruido (variacao ano a ano) x sinal (esforco exigido)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(-60, 60, 61)
    ax.hist(base["variacao"].clip(-60, 60), bins=bins, color=estilo.AZUL,
            alpha=0.85, label="Variação real 2023→2024")
    esf = base["esforco_exigido"].median()
    ax.axvline(esf, color=estilo.COR_RISCO, linewidth=2)
    ax.text(esf + 2, ax.get_ylim()[1] * 0.9,
            f"esforço mediano exigido: {esf:.1f} p.p.", color=estilo.TINTA, fontsize=9)
    ax.set_xlabel("Variação da taxa de alfabetização (p.p.)")
    ax.set_ylabel("Municípios")
    ax.set_title("O indicador oscila muito mais do que a meta pede")
    estilo.salvar(fig, IMAGES / "eda_teto_previsibilidade.png")
    negativo = base[base["esforco_exigido"] <= 0]
    return {"desvio_variacao": round(base["variacao"].std(), 1),
            "esforco_mediano": round(esf, 1),
            "pct_esforco_negativo": round(len(negativo) / len(base) * 100, 1),
            "pct_falharam_com_esforco_negativo": round(negativo[ALVO].mean() * 100, 1)}


def riqueza_por_uf(base: pd.DataFrame) -> dict:
    uf = base.groupby("sigla_uf").agg(pib=("log_pib_per_capita", "median"),
                                      risco=(ALVO, "mean"))
    dentro = base.groupby("sigla_uf").apply(
        lambda g: g["log_pib_per_capita"].corr(g[ALVO]), include_groups=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(np.exp(uf["pib"]) / 1000, uf["risco"] * 100, s=60,
               color=estilo.AZUL, edgecolor=estilo.SUPERFICIE, linewidth=2)
    for nome, r in uf.iterrows():
        ax.annotate(nome, (np.exp(r.pib) / 1000, r.risco * 100), fontsize=8,
                    xytext=(4, 3), textcoords="offset points", color=estilo.TINTA_2)
    ax.set_xlabel("PIB por habitante mediano da UF (R$ mil, 2021)")
    ax.set_ylabel("% que não atingiu a meta")
    ax.set_title("Estado mais rico não é estado com menos risco")
    estilo.salvar(fig, IMAGES / "eda_riqueza_vs_risco.png")
    return {"corr_entre_ufs": round(uf["pib"].corr(uf["risco"]), 3),
            "corr_media_dentro_uf": round(dentro.mean(), 3)}


def anomalia_rs(base: pd.DataFrame) -> dict:
    """RS em 2024: a avaliacao ocorreu no ano das enchentes."""
    rs = base[base.sigla_uf == "RS"]
    outros = base[base.sigla_uf != "RS"]
    return {"variacao_media_rs": round(rs["variacao"].mean(), 1),
            "variacao_media_demais": round(outros["variacao"].mean(), 1),
            "risco_rs": round(rs[ALVO].mean() * 100, 1),
            "taxa_2023_rs": round(rs["taxa_atual"].mean(), 1),
            "taxa_2023_demais": round(outros["taxa_atual"].mean(), 1)}


def main() -> None:
    base = pd.read_parquet(PROCESSED / "base_treino.parquet")
    base["variacao"] = base["_taxa_alvo"] - base["taxa_atual"]
    fato = carregar("fato_alfabetizacao")

    auc = auc_univariada(base)
    grafico_distribuicoes(base)
    corr = grafico_correlacoes(base)
    uf = grafico_risco_uf(base)
    part = grafico_participacao(base)
    teto = grafico_teto(base)
    riqueza = riqueza_por_uf(base)
    rs = anomalia_rs(base)

    resumo = {
        "municipios": len(base),
        "pct_risco": round(base[ALVO].mean() * 100, 1),
        "ufs": int(base.sigla_uf.nunique()),
        "ufs_ausentes_no_treino": sorted(
            set(carregar("dim_uf").sigla_uf) - set(base.sigla_uf)),
        "municipios_sem_meta_2024": int(
            fato[(fato.ano == 2023) & fato.taxa_realizada.notna()].shape[0] - len(base)),
        "auc_univariada": auc.round(3).to_dict("records"),
        "pares_correlacao_alta": [
            (a, b, round(corr.loc[a, b], 3)) for i, a in enumerate(NUMERICAS)
            for b in NUMERICAS[i + 1:] if abs(corr.loc[a, b]) > 0.8],
        "risco_por_uf": {"menor": [uf.index[0], round(uf.risco.iat[0] * 100, 1)],
                         "maior": [uf.index[-1], round(uf.risco.iat[-1] * 100, 1)]},
        "participacao": part.round(2).reset_index(names="faixa")
                            .astype({"faixa": str}).to_dict("records"),
        "teto_previsibilidade": teto,
        "riqueza": riqueza,
        "rio_grande_do_sul": rs,
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "eda_resumo.json").write_text(
        json.dumps(resumo, indent=2, ensure_ascii=False, default=float))

    print(auc.round(3).to_string(index=False))
    for k in ["pares_correlacao_alta", "risco_por_uf", "teto_previsibilidade",
              "riqueza", "rio_grande_do_sul", "ufs_ausentes_no_treino",
              "municipios_sem_meta_2024"]:
        print(f"\n{k}: {resumo[k]}")
    print("\n", part.round(2))


if __name__ == "__main__":
    main()
