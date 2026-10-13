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
85 mil linhas em vez de 2 milhoes.

VARIAVEIS (todas conhecidas antes da prova de 2024)
---------------------------------------------------
Municipio, de 2023: taxa, media em portugues, participacao, esforco exigido
para a meta de 2024, populacao, PIB por habitante, UF, regiao, porte.
Escola, de 2024: rede (estadual ou municipal) e numero de alunos do 2o ano
registrados. A participacao da escola na prova fica de fora: so e conhecida
no dia da prova.

O codigo da escola muda a cada edicao (2% de coincidencia entre 2023 e
2024), entao nao existe historico da escola.

VALIDACAO
---------
GroupKFold por municipio: todas as escolas de um municipio caem na mesma
particao. Sem isso, o modelo veria escolas vizinhas no treino e a AUC sairia
otimista.

TETOS
-----
Para saber quanto ainda cabe melhorar, a AUC e comparada com dois "oraculos"
que usam o resultado real de 2024 (portanto impossiveis na pratica):
saber a taxa real do municipio e saber a taxa real da escola. O segundo e o
maximo que qualquer modelo com variaveis de escola alcancaria.

Execucao:
    python -m src.modeling.modelo_aluno
"""
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import GOLD, IMAGES, MODELS, PARTICOES_CV, PROCESSED, REPORTS, SEMENTE
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
    return {
        "logistica_so_municipio": (LogisticRegression(max_iter=3000), False),
        "logistica_municipio_e_escola": (LogisticRegression(max_iter=3000), True),
        "boosting_municipio_e_escola": (HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, random_state=SEMENTE), True),
    }


def validar(base: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    y, w, grupos = base.alfabetizado.values, base.peso.values, base.id_municipio.values
    cv = GroupKFold(n_splits=PARTICOES_CV)
    oof = {nome: np.zeros(len(base)) for nome in candidatos()}
    for tr, te in cv.split(base, y, grupos):
        for nome, (modelo, com_escola) in candidatos().items():
            pipe = pipeline(modelo, com_escola)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(base.iloc[tr], y[tr], modelo__sample_weight=w[tr])
            oof[nome][te] = pipe.predict_proba(base.iloc[te])[:, 1]

    linhas = [{"modelo": "baseline (taxa nacional)", "auc": 0.5, "tipo": "piso"}]
    for nome, p in oof.items():
        linhas.append({"modelo": nome, "auc": roc_auc_score(y, p, sample_weight=w), "tipo": "modelo"})
    # oraculos: usam o resultado real de 2024, impossiveis na pratica
    taxa_mun_real = (base.assign(wy=w * y).groupby("id_municipio").wy.transform("sum")
                     / base.groupby("id_municipio").peso.transform("sum"))
    linhas.append({"modelo": "teto: taxa real do municipio",
                   "auc": roc_auc_score(y, taxa_mun_real, sample_weight=w), "tipo": "teto"})
    linhas.append({"modelo": "teto: taxa real da escola",
                   "auc": roc_auc_score(y, base.taxa_escola_real, sample_weight=w), "tipo": "teto"})
    return pd.DataFrame(linhas), oof


def grafico_auc(tab: pd.DataFrame) -> None:
    rotulos = {
        "baseline (taxa nacional)": "Sem modelo",
        "logistica_so_municipio": "Logística: só município",
        "logistica_municipio_e_escola": "Logística: município + escola",
        "boosting_municipio_e_escola": "Boosting: município + escola",
        "teto: taxa real do municipio": "Teto: saber a taxa real do município",
        "teto: taxa real da escola": "Teto: saber a taxa real da escola",
    }
    t = tab.iloc[::-1]
    cores = {"piso": estilo.TINTA_2, "modelo": estilo.AZUL, "teto": estilo.LARANJA}
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.barh([rotulos[m] for m in t.modelo], t.auc, color=[cores[x] for x in t.tipo], height=0.6)
    for i, v in enumerate(t.auc):
        ax.text(v + 0.003, i, f"{v:.3f}".replace(".", ","), va="center", fontsize=9)
    ax.set_xlim(0.45, 0.75)
    ax.set_xlabel("AUC por aluno (validação agrupada por município)")
    ax.set_title("Prever a criança: modelos (azul) e o máximo possível (laranja)")
    estilo.salvar(fig, IMAGES / "aluno_auc_comparacao.png")


def importancia(modelo, base) -> pd.Series:
    cols = NUM_MUNICIPIO + CAT_MUNICIPIO + NUM_ESCOLA + CAT_ESCOLA
    amostra = base.sample(n=min(20000, len(base)), random_state=SEMENTE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = permutation_importance(modelo, amostra[cols], amostra.alfabetizado,
                                   sample_weight=amostra.peso, scoring="roc_auc",
                                   n_repeats=10, random_state=SEMENTE, n_jobs=-1)
    return pd.Series(r.importances_mean, index=cols).sort_values(ascending=False)


def grafico_importancia(imp: pd.Series) -> None:
    imp = imp.sort_values()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh([i.replace("_", " ") for i in imp.index], imp.values, color=estilo.AZUL, height=0.6)
    ax.set_xlabel("Queda da AUC ao embaralhar a variável")
    ax.set_title("Modelo por aluno: o que pesa")
    estilo.salvar(fig, IMAGES / "aluno_importancia.png")


def main() -> None:
    base = montar_base()
    print(f"  escolas {base.attrs['escolas']}  municipios {base.attrs['municipios']}  "
          f"cobertura de alunos {base.attrs['cobertura_alunos']:.1%}")
    tab, oof = validar(base)
    print(tab.round(4).to_string(index=False))
    grafico_auc(tab)

    # Mesma regra do modelo municipal, fixada antes de rodar: se a logistica
    # ficar a ate 0,005 de AUC do melhor, ela vence, por ser legivel.
    modelos = tab[tab.tipo == "modelo"].set_index("modelo").auc
    escolhido = modelos.idxmax()
    if modelos.max() - modelos["logistica_municipio_e_escola"] <= TOLERANCIA_INTERPRETAVEL:
        escolhido = "logistica_municipio_e_escola"
    modelo, com_escola = candidatos()[escolhido]
    final = pipeline(modelo, com_escola)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final.fit(base, base.alfabetizado, modelo__sample_weight=base.peso)
    MODELS.mkdir(exist_ok=True)
    joblib.dump(final, MODELS / "modelo_aluno.joblib")
    imp = importancia(final, base) if com_escola else pd.Series(dtype=float)
    if len(imp):
        grafico_importancia(imp)

    t = tab.set_index("modelo").auc
    ganho_escola = t["logistica_municipio_e_escola"] - t["logistica_so_municipio"]
    melhor = t[escolhido]
    teto = t["teto: taxa real da escola"]
    resultado = {
        "ano": ANO, "escolas": base.attrs["escolas"], "municipios": base.attrs["municipios"],
        "alunos_avaliados_cobertos_pct": round(base.attrs["cobertura_alunos"] * 100, 1),
        "comparacao": tab.round(4).to_dict("records"),
        "modelo_escolhido": escolhido,
        "ganho_das_variaveis_de_escola": round(float(ganho_escola), 4),
        "pct_do_caminho_ate_o_teto": round(float((melhor - 0.5) / (teto - 0.5) * 100), 1),
        "importancia": imp.round(4).to_dict(),
    }
    (REPORTS / "resultado_modelo_aluno.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False, default=float))
    print(json.dumps({k: v for k, v in resultado.items() if k != "comparacao"},
                     indent=2, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
