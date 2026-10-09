"""
Aplicacao estrategica: do modelo para a decisao do gestor.

PRODUTOS
--------
1. ranking_2025.csv     risco de cada municipio NAO atingir a meta de 2025
                        (features de 2024, modelo re-treinado com toda a
                        base 2023->2024). E o produto acionavel.
2. ranking_2024_oof.csv risco 2024 calculado FORA DA AMOSTRA
                        (cross_val_predict): cada municipio recebe um score
                        de um modelo que nao o viu no treino. Serve para
                        validar o ranking contra o que de fato aconteceu.
3. perfis.csv           grupos de municipios parecidos (KMeans), com a taxa
                        real de falha de cada grupo em 2024.
4. risco_uf_2025.csv    onde a acao precisa ser estadual.

FAIXAS
------
Ancoradas no limiar escolhido no treino (detecta 80% de quem falha):
    baixo     < 0,25
    moderado  0,25 a limiar
    alto      limiar a 0,70
    critico   >= 0,70
Alto + critico = municipios em alerta.

RIO GRANDE DO SUL
-----------------
A queda do RS em 2024 foi causada pelas enchentes de abril e maio. O
modelo aprende "RS = risco" a partir desse choque. Para nao repassar o
choque para 2025, o ranking traz uma segunda coluna com o score de um
modelo treinado SEM o RS (a UF do RS vira desconhecida e o municipio e
avaliado so pelas demais variaveis). Para municipios gauchos, essa e a
coluna recomendada.

Execucao:
    python -m src.evaluation.aplicacao
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from src.config import (ALVO, IMAGES, MODELS, PARTICOES_CV, PROCESSED,
                        REPORTS, SEMENTE)
from src.preprocessing.base_analitica import FEATURES
from src.visualization import estilo
from src.visualization.estilo import plt

VARIAVEIS_PERFIL = ["taxa_atual", "participacao", "esforco_exigido"]
ORDEM_FAIXAS = ["baixo", "moderado", "alto", "critico"]


def faixa(prob: pd.Series, limiar: float) -> pd.Series:
    return pd.cut(prob, [-0.01, 0.25, limiar, 0.70, 1.01],
                  labels=ORDEM_FAIXAS, right=False).astype(str)


def ajustar(modelo, X, y):
    m = clone(modelo)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(X, y)
    return m


def score_fora_da_amostra(modelo, base) -> pd.Series:
    cv = StratifiedKFold(PARTICOES_CV, shuffle=True, random_state=SEMENTE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        oof = cross_val_predict(clone(modelo), base[FEATURES], base[ALVO],
                                cv=cv, method="predict_proba")[:, 1]
    return pd.Series(oof, index=base.index)


def perfis(treino: pd.DataFrame, previsao: pd.DataFrame):
    """KMeans nas variaveis de gestao/desempenho. Nao ve o alvo."""
    escala = StandardScaler().fit(treino[VARIAVEIS_PERFIL])
    Z = escala.transform(treino[VARIAVEIS_PERFIL])
    silhuetas = {k: silhouette_score(Z, KMeans(k, n_init=10, random_state=SEMENTE)
                                     .fit_predict(Z), sample_size=3000,
                                     random_state=SEMENTE)
                 for k in range(3, 7)}
    k = max(silhuetas, key=silhuetas.get)
    km = KMeans(k, n_init=10, random_state=SEMENTE).fit(Z)
    treino = treino.assign(perfil=km.labels_)
    previsao = previsao.assign(
        perfil=km.predict(escala.transform(previsao[VARIAVEIS_PERFIL])))

    resumo = treino.groupby("perfil").agg(
        municipios=(ALVO, "size"), taxa_atual=("taxa_atual", "mean"),
        participacao=("participacao", "mean"),
        esforco_exigido=("esforco_exigido", "median"),
        pct_falhou_2024=(ALVO, "mean"))
    resumo["nome"] = nomear_perfis(resumo)
    return treino, previsao, resumo, silhuetas


def nomear_perfis(r: pd.DataFrame) -> pd.Series:
    """Nome descritivo a partir dos centroides (relativo a media geral)."""
    media_taxa, media_part = r.taxa_atual.mean(), r.participacao.mean()
    nomes = {}
    for p, linha in r.iterrows():
        nivel = "alto desempenho" if linha.taxa_atual > media_taxa + 8 else \
                "baixo desempenho" if linha.taxa_atual < media_taxa - 8 else \
                "desempenho médio"
        gestao = "alta participação" if linha.participacao > media_part + 1.5 else \
                 "baixa participação" if linha.participacao < media_part - 1.5 else \
                 "participação média"
        nomes[p] = f"{nivel}, {gestao}"
    return pd.Series(nomes)


def grafico_risco_uf(uf: pd.DataFrame) -> None:
    uf = uf.sort_values("pct_alerta")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(uf.index, uf["pct_alerta"], color=estilo.COR_RISCO, width=0.7)
    if "RS" in uf.index:
        # fantasma: o que o modelo completo diria para o RS
        ax.bar(["RS"], [uf.loc["RS", "pct_alerta_modelo_completo"]], width=0.7,
               color="none", edgecolor=estilo.TINTA_2, linestyle="--", linewidth=1.2)
        ax.annotate("RS com o choque das enchentes de 2024\n(tracejado) e sem o choque (barra)",
                    ("RS", uf.loc["RS", "pct_alerta_modelo_completo"]),
                    xytext=(-20, 22), textcoords="offset points", fontsize=8.5,
                    ha="right", arrowprops={"arrowstyle": "-", "color": estilo.TINTA_2})
    ax.set_ylabel("% de municípios em alerta (alto + crítico)")
    ax.set_ylim(0, 112)
    ax.set_title("Previsão 2025: municípios em alerta por UF")
    estilo.salvar(fig, IMAGES / "aplicacao_alerta_uf_2025.png")


def grafico_perfis(resumo: pd.DataFrame) -> None:
    r = resumo.sort_values("pct_falhou_2024")
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.barh([f"{n}\n({m} municípios)" for n, m in zip(r.nome, r.municipios)],
            r.pct_falhou_2024 * 100, color=estilo.COR_RISCO, height=0.6)
    for i, v in enumerate(r.pct_falhou_2024 * 100):
        ax.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% que não atingiu a meta de 2024")
    ax.set_title("Perfis municipais (KMeans) e taxa real de falha")
    estilo.salvar(fig, IMAGES / "aplicacao_perfis.png")


def main() -> None:
    treino = pd.read_parquet(PROCESSED / "base_treino.parquet")
    previsao = pd.read_parquet(PROCESSED / "base_previsao.parquet")
    resultado = json.loads((REPORTS / "resultado_modelo.json").read_text())
    limiar = resultado["teste_limiar_escolhido"]["limiar"]
    modelo_base = joblib.load(MODELS / "modelo.joblib")

    # 1. Score 2024 fora da amostra (validacao do ranking)
    treino["prob_risco"] = score_fora_da_amostra(modelo_base, treino)
    treino["faixa"] = faixa(treino["prob_risco"], limiar)
    validacao = treino.groupby("faixa")[ALVO].agg(["size", "mean"]).reindex(ORDEM_FAIXAS)

    # 2. Modelos finais: toda a base, e toda a base sem RS
    final = ajustar(modelo_base, treino[FEATURES], treino[ALVO])
    sem_rs = treino[treino.sigla_uf != "RS"]
    final_sem_rs = ajustar(modelo_base, sem_rs[FEATURES], sem_rs[ALVO])
    joblib.dump(final, MODELS / "modelo_final.joblib")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        previsao["prob_risco"] = final.predict_proba(previsao[FEATURES])[:, 1]
        previsao["prob_risco_sem_rs"] = final_sem_rs.predict_proba(previsao[FEATURES])[:, 1]
    # Score recomendado: sem o choque para o RS, completo para o resto
    previsao["prob_recomendada"] = np.where(previsao.sigla_uf == "RS",
                                            previsao["prob_risco_sem_rs"],
                                            previsao["prob_risco"])
    previsao["faixa"] = faixa(previsao["prob_recomendada"], limiar)
    previsao["alerta"] = previsao["faixa"].isin(["alto", "critico"])
    limite = treino["esforco_exigido"].max()
    previsao["esforco_acima_do_treino"] = previsao["esforco_exigido"] > limite
    previsao["observacao"] = np.select(
        [previsao.sigla_uf == "RS", previsao.sigla_uf == "AC"],
        ["score sem o choque das enchentes de 2024",
         "UF sem historico no treino: avaliado so pelas demais variaveis"], "")

    # 3. Perfis
    treino, previsao, resumo_perfis, silhuetas = perfis(treino, previsao)
    previsao["perfil_nome"] = previsao["perfil"].map(resumo_perfis["nome"])

    # 4. UF
    uf = previsao.groupby("sigla_uf").agg(
        municipios=("prob_recomendada", "size"),
        risco_medio=("prob_recomendada", "mean"),
        pct_alerta=("alerta", "mean"))
    uf["pct_alerta"] *= 100
    # Para comparacao: o que o modelo completo diria (difere so no RS)
    uf["pct_alerta_modelo_completo"] = previsao.assign(
        a=faixa(previsao.prob_risco, limiar).isin(["alto", "critico"])
    ).groupby("sigla_uf")["a"].mean() * 100
    grafico_risco_uf(uf)
    grafico_perfis(resumo_perfis)

    # Saidas
    colunas = ["nome_municipio", "sigla_uf", "regiao", "taxa_atual", "meta_alvo",
               "esforco_exigido", "participacao", "prob_risco", "prob_risco_sem_rs",
               "prob_recomendada", "faixa", "alerta", "perfil_nome",
               "esforco_acima_do_treino", "observacao"]
    ranking = (previsao[colunas].sort_values("prob_recomendada", ascending=False)
               .round(3).rename(columns={"taxa_atual": "taxa_2024",
                                         "meta_alvo": "meta_2025"}))
    ranking.to_csv(REPORTS / "ranking_2025.csv", index_label="id_municipio")
    (treino[["nome_municipio", "sigla_uf", "taxa_atual", "meta_alvo",
             "prob_risco", "faixa", ALVO]]
     .sort_values("prob_risco", ascending=False).round(3)
     .rename(columns={"taxa_atual": "taxa_2023", "meta_alvo": "meta_2024",
                      ALVO: "nao_atingiu_2024"})
     .to_csv(REPORTS / "ranking_2024_oof.csv", index_label="id_municipio"))
    resumo_perfis.round(2).to_csv(REPORTS / "perfis.csv", index_label="perfil")
    uf.round(3).sort_values("pct_alerta", ascending=False).to_csv(
        REPORTS / "risco_uf_2025.csv")

    dist = previsao["faixa"].value_counts().reindex(ORDEM_FAIXAS)
    saida = {
        "limiar": limiar,
        "auc_oof_2024": roc_auc_score(treino[ALVO], treino["prob_risco"]),
        "validacao_faixas_2024": validacao.round(3).reset_index().to_dict("records"),
        "alerta_2024_recall": treino.loc[treino[ALVO] == 1, "faixa"]
                                    .isin(["alto", "critico"]).mean(),
        "previsao_2025": {
            "municipios": len(previsao),
            "faixas": dist.to_dict(),
            "em_alerta": int(previsao["alerta"].sum()),
            "pct_alerta": previsao["alerta"].mean(),
            "rs_alerta_modelo_completo_pct": uf.loc["RS", "pct_alerta_modelo_completo"],
            "rs_alerta_sem_choque_pct": uf.loc["RS", "pct_alerta"],
            "esforco_max_treino": limite,
            "pct_esforco_acima_do_treino": previsao["esforco_acima_do_treino"].mean(),
        },
        "silhuetas": silhuetas,
        "perfis": resumo_perfis.round(3).reset_index().to_dict("records"),
        "top_uf_alerta": uf.sort_values("pct_alerta", ascending=False)
                          .head(5).round(1).reset_index().to_dict("records"),
        "menos_alerta": uf.sort_values("pct_alerta").head(5).round(1)
                          .reset_index().to_dict("records"),
    }
    (REPORTS / "aplicacao.json").write_text(
        json.dumps(saida, indent=2, ensure_ascii=False, default=float))

    print("validacao faixas 2024 (oof):\n", validacao.round(3))
    print(f"\nAUC oof 2024: {saida['auc_oof_2024']:.3f}  "
          f"recall alerta: {saida['alerta_2024_recall']:.3f}")
    print("\n2025:", json.dumps(saida["previsao_2025"], default=float))
    print("\nsilhuetas:", {k: round(v, 3) for k, v in silhuetas.items()})
    print("\nperfis:\n", resumo_perfis.round(2).to_string())
    print("\nUF (mais alerta):\n", uf.sort_values("pct_alerta", ascending=False)
          .round(1).head(8).to_string())
    print("\ntop 10 ranking 2025:\n", ranking.head(10)[
        ["nome_municipio", "sigla_uf", "taxa_2024", "meta_2025",
         "prob_recomendada", "perfil_nome"]].to_string())


if __name__ == "__main__":
    main()
