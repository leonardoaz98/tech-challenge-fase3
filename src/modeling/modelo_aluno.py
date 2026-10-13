"""
Modelo por aluno: qual a chance de uma crianca do 2o ano ser alfabetizada?

UNIDADE E ALVO
--------------
Cada aluno avaliado em 2024 e uma observacao; o alvo e alfabetizado = 1
quando a proficiencia e >= 743 (o indicador do INEP bate 100% com esse
corte). Cada aluno entra com o peso amostral do INEP.

POR QUE TREINAR NAS CONTAGENS DA ESCOLA
---------------------------------------
Os microdados nao trazem nada da crianca antes da prova (sexo, cor, renda,
idade). Todas as variaveis disponiveis sao da escola ou do municipio, entao
todos os alunos de uma escola tem as mesmas variaveis. Nesse caso, uma
linha por aluno ou duas linhas por escola (alfabetizados e nao
alfabetizados, com o peso somado de cada grupo) produzem exatamente a mesma
verossimilhanca, os mesmos coeficientes e a mesma AUC. A segunda forma usa
80 mil linhas em vez de 2 milhoes.

VARIAVEIS (todas conhecidas antes da prova de 2024)
---------------------------------------------------
Municipio, de 2023: taxa, media em portugues, participacao, esforco exigido
para a meta de 2024, populacao, PIB por habitante, UF, regiao, porte.
Escola, de 2024: rede (estadual ou municipal) e numero de alunos do 2o ano
registrados. A participacao da escola na prova fica de fora: so e conhecida
no dia da prova. O codigo da escola muda a cada edicao (2% de coincidencia
entre 2023 e 2024), entao nao existe historico da escola.

PROTOCOLO (o mesmo rigor do modelo municipal)
---------------------------------------------
1. Teste: 25% dos MUNICIPIOS ficam fora de tudo (GroupShuffleSplit). Todas
   as escolas de um municipio caem do mesmo lado; sem isso, o modelo veria
   escolas vizinhas no treino e a AUC sairia otimista.
2. Busca em grade no treino, com GroupKFold por municipio e AUC ponderada
   pelo peso dos alunos (os scorers padrao do scikit-learn ignorariam o
   peso, e cada linha aqui representa um grupo de alunos).
3. Escolha: maior AUC de validacao; se a logistica ficar a ate 0,005 do
   melhor, ela vence, por ser legivel. Regra fixada antes de rodar.
4. Avaliacao unica no teste.
5. Tetos, no mesmo teste: a AUC de quem soubesse a taxa real do municipio
   e a taxa real da escola em 2024. Impossiveis na pratica; o segundo e o
   maximo que qualquer modelo com variaveis de escola alcancaria.

Execucao:
    python -m src.modeling.modelo_aluno
"""
import json
import warnings
from itertools import product

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (GOLD, IMAGES, MODELS, PARTICOES_CV, PROCESSED,
                        PROPORCAO_TESTE, REPORTS, SEMENTE)
from src.modeling.transformadores import LimitarFaixa
from src.visualization import estilo
from src.visualization.estilo import plt

ANO = 2024
NUM_MUNICIPIO = ["taxa_atual", "media_portugues", "participacao", "esforco_exigido",
                 "log_populacao", "log_pib_per_capita"]
CAT_MUNICIPIO = ["sigla_uf", "regiao", "porte_municipio"]
NUM_ESCOLA = ["log_alunos_escola"]
CAT_ESCOLA = ["rede_escola"]
REDES = {2: "estadual", 3: "municipal"}
TOLERANCIA_INTERPRETAVEL = 0.005


def montar_base() -> pd.DataFrame:
    """Duas linhas por escola (alfabetizados e nao), com o peso do grupo."""
    e = pd.read_parquet(GOLD / "fato_escola.parquet")
    e = e[(e.ano == ANO) & e.rede.isin(REDES) & (e.soma_peso > 0)].copy()
    # a participacao da escola so e conhecida no dia da prova: fica fora
    e = e.rename(columns={"participacao": "participacao_escola"})
    mun = pd.read_parquet(PROCESSED / "base_treino.parquet")
    mun = mun[NUM_MUNICIPIO + CAT_MUNICIPIO + ["nome_municipio"]]
    total_alunos = e.alunos_com_resultado.sum()
    e = e.merge(mun, left_on="id_municipio", right_index=True, how="inner")
    e["rede_escola"] = e.rede.map(REDES)
    e["log_alunos_escola"] = np.log1p(e.alunos_registrados)
    e["taxa_escola_real"] = e.soma_peso_alfabetizados / e.soma_peso

    sim = e.assign(alfabetizado=1, peso=e.soma_peso_alfabetizados)
    nao = e.assign(alfabetizado=0, peso=e.soma_peso - e.soma_peso_alfabetizados)
    base = pd.concat([sim, nao], ignore_index=True)
    base = base[base.peso > 0].reset_index(drop=True)
    base.attrs["cobertura_alunos"] = float(e.alunos_com_resultado.sum() / total_alunos)
    base.attrs["escolas"] = int(len(e))
    base.attrs["municipios"] = int(e.id_municipio.nunique())
    return base


def pipeline(modelo, com_escola: bool) -> Pipeline:
    num = NUM_MUNICIPIO + (NUM_ESCOLA if com_escola else [])
    cat = CAT_MUNICIPIO + (CAT_ESCOLA if com_escola else [])
    prep = ColumnTransformer([
        ("num", Pipeline([("imputar", SimpleImputer(strategy="median")),
                          ("limitar", LimitarFaixa()),
                          ("escalonar", StandardScaler())]), num),
        ("cat", Pipeline([("imputar", SimpleImputer(strategy="most_frequent")),
                          ("codificar", OneHotEncoder(handle_unknown="ignore",
                                                      sparse_output=False))]), cat),
    ])
    return Pipeline([("prep", prep), ("modelo", modelo)])


def candidatos() -> dict:
    """Modelo, se usa variaveis de escola e a grade. Todo hiperparametro e um freio de overfitting."""
    return {
        "logistica_so_municipio": (LogisticRegression(max_iter=3000), False,
                                   {"C": [0.01, 0.1, 1, 10]}),
        "logistica_municipio_e_escola": (LogisticRegression(max_iter=3000), True,
                                         {"C": [0.01, 0.1, 1, 10]}),
        "boosting_municipio_e_escola": (
            HistGradientBoostingClassifier(max_iter=300, l2_regularization=1.0,
                                           random_state=SEMENTE), True,
            {"learning_rate": [0.05, 0.1], "max_leaf_nodes": [15, 31]}),
    }


def ajustar(pipe, base, idx):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(base.iloc[idx], base.alfabetizado.values[idx],
                 modelo__sample_weight=base.peso.values[idx])
    return pipe


def auc(base, idx, prob) -> float:
    return roc_auc_score(base.alfabetizado.values[idx], prob, sample_weight=base.peso.values[idx])


def busca_em_grade(base, treino) -> pd.DataFrame:
    """GroupKFold por municipio dentro do treino, AUC ponderada pelo peso dos alunos."""
    cv = GroupKFold(n_splits=PARTICOES_CV)
    sub = base.iloc[treino]
    folds = [(treino[a], treino[b]) for a, b in
             cv.split(sub, sub.alfabetizado, sub.id_municipio)]
    linhas = []
    for nome, (modelo, com_escola, grade) in candidatos().items():
        for valores in product(*grade.values()):
            params = dict(zip(grade, valores))
            aucs = []
            for tr, va in folds:
                pipe = ajustar(pipeline(clone(modelo).set_params(**params), com_escola), base, tr)
                aucs.append(auc(base, va, pipe.predict_proba(base.iloc[va])[:, 1]))
            linhas.append({"modelo": nome, "params": params,
                           "auc_cv": float(np.mean(aucs)), "desvio": float(np.std(aucs))})
    return pd.DataFrame(linhas)


def escolher(busca: pd.DataFrame) -> tuple[str, dict]:
    melhores = busca.loc[busca.groupby("modelo").auc_cv.idxmax()].set_index("modelo")
    escolhido = melhores.auc_cv.idxmax()
    if melhores.auc_cv.max() - melhores.loc["logistica_municipio_e_escola", "auc_cv"] \
            <= TOLERANCIA_INTERPRETAVEL:
        escolhido = "logistica_municipio_e_escola"
    return escolhido, melhores.loc[escolhido, "params"]


def avaliar_no_teste(base, treino, teste, busca) -> pd.DataFrame:
    melhores = (busca.loc[busca.groupby("modelo").auc_cv.idxmax()]
                .set_index("modelo").loc[list(candidatos())].reset_index())
    linhas = [{"modelo": "baseline (taxa nacional)", "auc_cv": 0.5, "auc_teste": 0.5, "tipo": "piso"}]
    for _, r in melhores.iterrows():
        modelo, com_escola, _ = candidatos()[r.modelo]
        pipe = ajustar(pipeline(clone(modelo).set_params(**r.params), com_escola), base, treino)
        linhas.append({"modelo": r.modelo, "params": r.params, "auc_cv": r.auc_cv,
                       "auc_teste": auc(base, teste, pipe.predict_proba(base.iloc[teste])[:, 1]),
                       "tipo": "modelo"})
    t = base.iloc[teste]
    taxa_mun = (t.assign(wy=t.peso * t.alfabetizado).groupby("id_municipio").wy.transform("sum")
                / t.groupby("id_municipio").peso.transform("sum"))
    linhas.append({"modelo": "teto: taxa real do municipio", "auc_teste": auc(base, teste, taxa_mun),
                   "tipo": "teto"})
    linhas.append({"modelo": "teto: taxa real da escola",
                   "auc_teste": auc(base, teste, t.taxa_escola_real), "tipo": "teto"})
    return pd.DataFrame(linhas)


ROTULOS = {
    "baseline (taxa nacional)": "Sem modelo",
    "logistica_so_municipio": "Logística: só município",
    "logistica_municipio_e_escola": "Logística: município + escola",
    "boosting_municipio_e_escola": "Boosting: município + escola",
    "teto: taxa real do municipio": "Teto: saber a taxa real do município",
    "teto: taxa real da escola": "Teto: saber a taxa real da escola",
}


def grafico_auc(tab: pd.DataFrame) -> None:
    t = tab.iloc[::-1]
    cores = {"piso": estilo.TINTA_2, "modelo": estilo.AZUL, "teto": estilo.LARANJA}
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.barh([ROTULOS[m] for m in t.modelo], t.auc_teste, color=[cores[x] for x in t.tipo], height=0.6)
    for i, v in enumerate(t.auc_teste):
        ax.text(v + 0.003, i, f"{v:.3f}".replace(".", ","), va="center", fontsize=9)
    ax.set_xlim(0.45, 0.78)
    ax.set_xlabel("AUC por aluno no teste (25% dos municípios, nunca vistos)")
    ax.set_title("Prever a criança: modelos (azul) e o máximo possível (laranja)")
    estilo.salvar(fig, IMAGES / "aluno_auc_comparacao.png")


def importancia(modelo, base, teste) -> pd.Series:
    cols = NUM_MUNICIPIO + CAT_MUNICIPIO + NUM_ESCOLA + CAT_ESCOLA
    t = base.iloc[teste]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = permutation_importance(modelo, t[cols], t.alfabetizado, sample_weight=t.peso,
                                   scoring="roc_auc", n_repeats=10, random_state=SEMENTE, n_jobs=-1)
    return pd.Series(r.importances_mean, index=cols).sort_values(ascending=False)


def grafico_importancia(imp: pd.Series) -> None:
    imp = imp.sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh([i.replace("_", " ") for i in imp.index], imp.values, color=estilo.AZUL, height=0.6)
    ax.set_xlabel("Queda da AUC no teste ao embaralhar a variável")
    ax.set_title("Modelo por aluno: importância por permutação")
    estilo.salvar(fig, IMAGES / "aluno_importancia.png")


def legivel(coluna: str) -> str:
    return (coluna.replace("num__", "").replace("cat__", "").replace("sigla_uf_", "UF ")
            .replace("regiao_", "região ").replace("porte_municipio_", "porte ")
            .replace("rede_escola_", "rede ").replace("_", " "))


def analisar_shap(modelo, base, treino, teste) -> pd.Series:
    """SHAP do modelo logistico, numa amostra do teste; agrega por variavel original."""
    prep, final = modelo.named_steps["prep"], modelo.named_steps["modelo"]
    nomes = [legivel(n) for n in prep.get_feature_names_out()]
    fundo = base.iloc[treino].sample(n=2000, random_state=SEMENTE)
    amostra = base.iloc[teste].sample(n=3000, random_state=SEMENTE)
    Zf = pd.DataFrame(prep.transform(fundo), columns=nomes)
    Za = pd.DataFrame(prep.transform(amostra), columns=nomes)
    np.random.seed(SEMENTE)
    sv = shap.LinearExplainer(final, Zf)(Za)

    np.random.seed(SEMENTE)
    plt.figure()
    shap.plots.beeswarm(sv, max_display=12, show=False, color=plt.get_cmap("coolwarm"),
                        color_bar_label="Valor da variável (baixo → alto)")
    fig = plt.gcf()
    fig.set_size_inches(9, 6)
    plt.gca().set_title("SHAP do modelo por aluno (amostra do teste)", loc="left", fontweight="bold")
    plt.gca().set_xlabel("Valor SHAP (log-chance de ser alfabetizado)")
    estilo.salvar(fig, IMAGES / "aluno_shap_resumo.png")

    grupos = {}
    prefixos = {"UF ": "sigla_uf", "região ": "regiao", "porte ": "porte_municipio", "rede ": "rede_escola"}
    for i, n in enumerate(nomes):
        orig = next((v for p, v in prefixos.items() if n.startswith(p)), n.replace(" ", "_"))
        grupos.setdefault(orig, []).append(i)
    absv = np.abs(sv.values)
    return pd.Series({k: absv[:, idx].sum(axis=1).mean() for k, idx in grupos.items()}
                     ).sort_values(ascending=False)


def main() -> None:
    base = montar_base()
    divisor = GroupShuffleSplit(n_splits=1, test_size=PROPORCAO_TESTE, random_state=SEMENTE)
    treino, teste = next(divisor.split(base, base.alfabetizado, base.id_municipio))
    print(f"  escolas {base.attrs['escolas']}  municipios {base.attrs['municipios']}  "
          f"cobertura de alunos {base.attrs['cobertura_alunos']:.1%}")
    print(f"  municipios no treino {base.iloc[treino].id_municipio.nunique()}  "
          f"no teste {base.iloc[teste].id_municipio.nunique()}")

    busca = busca_em_grade(base, treino)
    print("\n" + busca.round(4).to_string(index=False))
    escolhido, params = escolher(busca)
    print(f"\n  escolhido: {escolhido} {params}")

    tab = avaliar_no_teste(base, treino, teste, busca)
    print("\n" + tab.drop(columns="params", errors="ignore").round(4).to_string(index=False))
    grafico_auc(tab)

    modelo, com_escola, _ = candidatos()[escolhido]
    final = ajustar(pipeline(clone(modelo).set_params(**params), com_escola), base, treino)
    MODELS.mkdir(exist_ok=True)
    joblib.dump(final, MODELS / "modelo_aluno.joblib")
    imp = importancia(final, base, teste)
    grafico_importancia(imp)
    shap_medio = analisar_shap(final, base, treino, teste) \
        if escolhido.startswith("logistica") else pd.Series(dtype=float)

    t = tab.set_index("modelo")
    melhor_teste, teto = t.loc[escolhido, "auc_teste"], t.loc["teto: taxa real da escola", "auc_teste"]
    resultado = {
        "ano": ANO, "escolas": base.attrs["escolas"], "municipios": base.attrs["municipios"],
        "alunos_avaliados_cobertos_pct": round(base.attrs["cobertura_alunos"] * 100, 1),
        "municipios_treino": int(base.iloc[treino].id_municipio.nunique()),
        "municipios_teste": int(base.iloc[teste].id_municipio.nunique()),
        "busca_em_grade": busca.round(4).to_dict("records"),
        "combinacoes_testadas": int(len(busca)),
        "comparacao_teste": tab.round(4).to_dict("records"),
        "modelo_escolhido": escolhido, "parametros": params,
        "auc_cv": round(float(t.loc[escolhido, "auc_cv"]), 4),
        "auc_teste": round(float(melhor_teste), 4),
        "gap_cv_teste": round(float(t.loc[escolhido, "auc_cv"] - melhor_teste), 4),
        "ganho_das_variaveis_de_escola_teste": round(float(
            t.loc["logistica_municipio_e_escola", "auc_teste"]
            - t.loc["logistica_so_municipio", "auc_teste"]), 4),
        "pct_do_caminho_ate_o_teto": round(float((melhor_teste - 0.5) / (teto - 0.5) * 100), 1),
        "importancia_permutacao": imp.round(4).to_dict(),
        "shap_medio_por_variavel": shap_medio.round(4).to_dict(),
    }
    (REPORTS / "resultado_modelo_aluno.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False, default=float))
    print(json.dumps({k: v for k, v in resultado.items()
                      if k not in ("busca_em_grade", "comparacao_teste")},
                     indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
