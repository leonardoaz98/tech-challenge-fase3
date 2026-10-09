"""
Interpretabilidade: quais variaveis pesam e por que um municipio
recebeu o score que recebeu.

Tres visoes, cada uma responde uma pergunta:

  coeficientes    direcao e forca (regressao logistica, variaveis padronizadas)
  permutacao      quanto a AUC cai ao embaralhar cada variavel original
  SHAP            contribuicao de cada variavel em cada previsao

Mais um teste de estabilidade: o ranking da permutacao sobrevive a
troca de semente? Com variaveis correlacionadas (taxa x media de
portugues: r = 0,93) a importancia pode migrar entre elas.

Execucao:
    python -m src.evaluation.interpretar
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from src.config import (ALVO, IMAGES, MODELS, PROCESSED, PROPORCAO_TESTE,
                        REPORTS, SEMENTE)
from src.preprocessing.base_analitica import CATEGORICAS, FEATURES
from src.visualization import estilo
from src.visualization.estilo import plt

SEMENTES_ESTABILIDADE = range(8)


def nome_legivel(coluna: str) -> str:
    return (coluna.replace("num__", "").replace("cat__", "")
            .replace("sigla_uf_", "UF ").replace("regiao_", "região ")
            .replace("porte_municipio_", "porte "))


def variavel_original(coluna: str) -> str:
    coluna = coluna.replace("num__", "").replace("cat__", "")
    for c in CATEGORICAS:
        if coluna.startswith(c + "_"):
            return c
    return coluna


def separar(base):
    return train_test_split(base[FEATURES], base[ALVO], test_size=PROPORCAO_TESTE,
                            stratify=base[ALVO], random_state=SEMENTE)


def coeficientes(modelo) -> pd.DataFrame:
    nomes = modelo.named_steps["prep"].get_feature_names_out()
    coef = modelo.named_steps["modelo"].coef_[0]
    df = pd.DataFrame({"coluna": nomes, "coef": coef})
    df["variavel"] = df["coluna"].map(nome_legivel)
    df["razao_chances"] = np.exp(df["coef"])
    return df.sort_values("coef")


def grafico_coeficientes(df: pd.DataFrame) -> None:
    sel = pd.concat([df.head(8), df.tail(8)]).drop_duplicates()
    fig, ax = plt.subplots(figsize=(8, 6))
    cores = [estilo.COR_RISCO if c > 0 else estilo.COR_ATINGIU for c in sel["coef"]]
    ax.barh(sel["variavel"], sel["coef"], color=cores, height=0.65)
    ax.axvline(0, color=estilo.TINTA_2, linewidth=1)
    ax.set_xlabel("Coeficiente (log-chance de NÃO atingir a meta)")
    ax.set_title("Coeficientes: laranja aumenta o risco, azul reduz")
    estilo.salvar(fig, IMAGES / "interp_coeficientes.png")


def permutacao(modelo, X, y, semente=SEMENTE) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = permutation_importance(modelo, X, y, scoring="roc_auc", n_repeats=20,
                                   random_state=semente, n_jobs=-1)
    return pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)


def grafico_permutacao(imp: pd.Series) -> None:
    imp = imp.sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh([nome_legivel(i).replace("_", " ") for i in imp.index], imp.values,
            color=estilo.AZUL, height=0.65)
    for i, v in enumerate(imp.values):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=8.5)
    ax.set_xlabel("Queda da AUC no teste ao embaralhar a variável")
    ax.set_title("Importância por permutação")
    estilo.salvar(fig, IMAGES / "interp_permutacao.png")


def estabilidade(modelo, base) -> pd.DataFrame:
    """Re-treina com 8 divisoes diferentes e registra a posicao de cada variavel."""
    posicoes = {}
    for s in SEMENTES_ESTABILIDADE:
        Xtr, Xte, ytr, yte = train_test_split(
            base[FEATURES], base[ALVO], test_size=PROPORCAO_TESTE,
            stratify=base[ALVO], random_state=s)
        m = clone(modelo)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(Xtr, ytr)
        imp = permutacao(m, Xte, yte, semente=s)
        posicoes[s] = imp.rank(ascending=False).astype(int)
    df = pd.DataFrame(posicoes)
    return pd.DataFrame({"posicao_media": df.mean(axis=1), "melhor": df.min(axis=1),
                         "pior": df.max(axis=1)}).sort_values("posicao_media")


def shap_valores(modelo, Xtr, Xte):
    prep, final = modelo.named_steps["prep"], modelo.named_steps["modelo"]
    nomes = [nome_legivel(n).replace("_", " ") for n in prep.get_feature_names_out()]
    Ztr = pd.DataFrame(prep.transform(Xtr), columns=nomes)
    Zte = pd.DataFrame(prep.transform(Xte), columns=nomes, index=Xte.index)
    explicador = shap.LinearExplainer(final, Ztr)
    return explicador(Zte), nomes


def grafico_shap_resumo(sv) -> None:
    plt.figure()
    shap.plots.beeswarm(sv, max_display=14, show=False,
                        color=plt.get_cmap("coolwarm"),
                        color_bar_label="Valor da variável (baixo → alto)")
    fig = plt.gcf()
    fig.set_size_inches(9, 6.5)
    ax = plt.gca()
    ax.set_title("SHAP: impacto de cada variável no risco (teste)", loc="left",
                 fontweight="bold")
    ax.set_xlabel("Valor SHAP (log-chance de NÃO atingir a meta)")
    estilo.salvar(fig, IMAGES / "interp_shap_resumo.png")


def shap_agrupado(sv, nomes) -> pd.Series:
    """Media do |SHAP| somando as colunas one-hot de cada variavel original."""
    abs_sv = pd.DataFrame(np.abs(sv.values), columns=nomes)
    mapa = {}
    for n in nomes:
        for c in CATEGORICAS:
            prefixo = {"sigla_uf": "UF ", "regiao": "região ",
                       "porte_municipio": "porte "}[c]
            if n.startswith(prefixo):
                mapa[n] = c
        mapa.setdefault(n, n.replace(" ", "_"))
    return abs_sv.T.groupby(mapa).sum().T.mean().sort_values(ascending=False)


FORMATO_VALOR = {
    "taxa_atual": lambda v: f"{v:.0f}%", "participacao": lambda v: f"{v:.0f}%",
    "media_portugues": lambda v: f"{v:.0f} pts",
    "esforco_exigido": lambda v: f"{v:+.1f} p.p.",
    "log_populacao": lambda v: f"{np.expm1(v):,.0f} hab".replace(",", "."),
    "log_pib_per_capita": lambda v: f"R$ {np.exp(v) / 1000:.0f} mil/hab",
}


def grafico_shap_municipio(sv, nomes, base, id_municipio) -> dict:
    """Contribuicao de cada variavel original para o score de um municipio,
    com o valor real da variavel no rotulo (nao o valor padronizado)."""
    idx = np.where(base.index == id_municipio)[0][0]
    linha = base.iloc[idx]
    contrib = pd.Series(sv.values[idx], index=nomes)
    grupos = {}
    for n, v in contrib.items():
        orig = next((c for c, p in [("sigla_uf", "UF "), ("regiao", "região "),
                                    ("porte_municipio", "porte ")] if n.startswith(p)),
                    n.replace(" ", "_"))
        grupos[orig] = grupos.get(orig, 0) + v
    grupos = pd.Series(grupos).reindex(FEATURES)
    grupos = grupos.reindex(grupos.abs().sort_values().index)

    nome_curto = {"sigla_uf": "UF", "log_populacao": "população",
                  "log_pib_per_capita": "PIB por habitante", "porte_municipio": "porte",
                  "media_portugues": "média em português", "taxa_atual": "taxa atual",
                  "esforco_exigido": "esforço exigido", "participacao": "participação",
                  "regiao": "região"}
    rotulos = [f"{nome_curto[c]} = "
               f"{FORMATO_VALOR.get(c, str)(linha[c])}" for c in grupos.index]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    cores = [estilo.COR_RISCO if v > 0 else estilo.COR_ATINGIU for v in grupos]
    ax.barh(rotulos, grupos.values, color=cores, height=0.65)
    ax.axvline(0, color=estilo.TINTA_2, linewidth=1)
    ax.set_xlim(min(grupos.min() * 1.8, -0.6), grupos.max() * 1.15)
    for i, v in enumerate(grupos.values):
        ax.text(v + (0.05 if v >= 0 else -0.05), i, f"{v:+.2f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8.5)
    prob = 1 / (1 + np.exp(-(sv.base_values[idx] + contrib.sum())))
    ax.set_xlabel("Contribuição SHAP (laranja empurra para o risco, azul para longe)")
    ax.set_title(f"Por que {linha['nome_municipio']} ({linha['sigla_uf']}) "
                 f"recebeu risco de {prob:.0%}")
    estilo.salvar(fig, IMAGES / "interp_shap_municipio.png")
    return {"id_municipio": id_municipio, "nome": linha["nome_municipio"],
            "uf": linha["sigla_uf"],
            "contribuicoes": grupos.round(3).sort_values(ascending=False).to_dict()}


def main() -> None:
    base = pd.read_parquet(PROCESSED / "base_treino.parquet")
    modelo = joblib.load(MODELS / "modelo.joblib")
    Xtr, Xte, ytr, yte = separar(base)

    coef = coeficientes(modelo)
    grafico_coeficientes(coef)

    imp = permutacao(modelo, Xte, yte)
    grafico_permutacao(imp)

    estab = estabilidade(modelo, base)

    sv, nomes = shap_valores(modelo, Xtr, Xte)
    grafico_shap_resumo(sv)
    agrupado = shap_agrupado(sv, nomes)

    # Exemplo para o video: o municipio de maior risco previsto no teste
    # que estava acima da media nacional em 2023 (caso contraintuitivo).
    prob = pd.Series(modelo.predict_proba(Xte)[:, 1], index=Xte.index)
    teste = base.loc[Xte.index].assign(prob=prob)
    exemplo = (teste[teste.taxa_atual > base.taxa_atual.mean()]
               .sort_values("prob", ascending=False).index[0])
    exemplo_info = grafico_shap_municipio(sv, nomes, teste, exemplo)
    exemplo_info["prob_risco"] = round(float(prob[exemplo]), 3)

    saida = {
        "coeficientes": coef[["variavel", "coef", "razao_chances"]].round(3)
                            .to_dict("records"),
        "permutacao": imp.round(4).to_dict(),
        "estabilidade": estab.round(2).reset_index(names="variavel").to_dict("records"),
        "shap_medio_agrupado": agrupado.round(4).to_dict(),
        "exemplo_shap": exemplo_info,
    }
    (REPORTS / "interpretabilidade.json").write_text(
        json.dumps(saida, indent=2, ensure_ascii=False, default=float))

    print("permutacao:\n", imp.round(4).to_string())
    print("\nestabilidade:\n", estab.to_string())
    print("\nshap agrupado:\n", agrupado.round(4).to_string())
    print("\ncoef extremos:\n", pd.concat([coef.head(5), coef.tail(5)])
          [["variavel", "coef"]].round(3).to_string(index=False))
    print("\nexemplo:", exemplo_info)


if __name__ == "__main__":
    main()
