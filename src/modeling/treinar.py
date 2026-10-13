"""
Treino, otimizacao e validacao do modelo.

PIPELINE
--------
Imputacao, padronizacao e one-hot ficam DENTRO do Pipeline do
scikit-learn. Em cada particao da validacao cruzada esses parametros
(mediana, media, desvio, categorias) sao aprendidos so com o treino
daquela particao. Fazer isso antes do split vazaria informacao do teste.

PROTOCOLO
---------
1. Split 75/25 estratificado. O teste so e tocado no fim.
2. GridSearchCV (5 folds, AUC) em cada modelo, so no treino.
3. Escolha do modelo: maior AUC de validacao. Regra definida antes de
   rodar: se a regressao logistica ficar a menos de 0,005 do melhor,
   ela vence, porque o gestor precisa entender o porque do alerta.
4. Limiar de decisao: escolhido com previsoes fora da amostra do
   TREINO (cross_val_predict), como o maior corte que ainda detecta
   RECALL_MINIMO dos municipios que falham. O teste nao participa.
5. Avaliacao unica no teste.
6. Sensibilidade: mesmo protocolo sem o Rio Grande do Sul, cuja queda
   em 2024 foi causada pelas enchentes, nao por gestao.

Execucao:
    python -m src.modeling.treinar
    python -m src.modeling.treinar --sem-busca     (rapido)
"""
import argparse
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_predict, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.modeling.transformadores import LimitarFaixa
from src.config import (ALVO, IMAGES, MODELS, PARTICOES_CV, PROCESSED,
                        PROPORCAO_TESTE, RECALL_MINIMO, REPORTS, SEMENTE)
from src.preprocessing.base_analitica import CATEGORICAS, FEATURES, NUMERICAS
from src.visualization import estilo
from src.visualization.estilo import plt

TOLERANCIA_INTERPRETAVEL = 0.005


def preprocessador() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("imputar", SimpleImputer(strategy="median")),
                          # nao extrapola alem do que o treino viu
                          ("limitar", LimitarFaixa()),
                          ("escalonar", StandardScaler())]), NUMERICAS),
        ("cat", Pipeline([("imputar", SimpleImputer(strategy="most_frequent")),
                          # UF nunca vista no treino (ex.: AC em 2025) vira
                          # vetor de zeros em vez de quebrar a previsao
                          ("codificar", OneHotEncoder(handle_unknown="ignore",
                                                      sparse_output=False))]),
         CATEGORICAS),
    ])


def candidatos() -> dict:
    """Modelo + grade. Todo hiperparametro da grade e um freio de overfitting."""
    return {
        "baseline": (DummyClassifier(strategy="prior"), {}),
        "regressao_logistica": (
            LogisticRegression(max_iter=5000),
            {"modelo__C": [0.01, 0.1, 1, 10, 100]}),       # C menor = mais regularizacao
        "random_forest": (
            RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEMENTE),
            {"modelo__min_samples_leaf": [1, 5, 15],
             "modelo__max_depth": [None, 12]}),
        "gradient_boosting": (
            HistGradientBoostingClassifier(max_iter=300, random_state=SEMENTE),
            {"modelo__learning_rate": [0.03, 0.06, 0.12],
             "modelo__max_leaf_nodes": [15, 31],
             "modelo__l2_regularization": [0.0, 1.0]}),
    }


def comparar(X, y, buscar: bool):
    cv = StratifiedKFold(PARTICOES_CV, shuffle=True, random_state=SEMENTE)
    linhas, ajustados = [], {}
    for nome, (modelo, grade) in candidatos().items():
        pipe = Pipeline([("prep", preprocessador()), ("modelo", modelo)])
        grade = grade if buscar else {}
        # grade vazia = um unico candidato (valores padrao)
        busca = GridSearchCV(pipe, grade, scoring="roc_auc", cv=cv, n_jobs=-1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            busca.fit(X, y)
        r = busca.cv_results_
        i = busca.best_index_
        linhas.append({
            "modelo": nome, "auc_cv": r["mean_test_score"][i],
            "desvio": r["std_test_score"][i], "combinacoes": len(r["params"]),
            "pior_auc": float(np.min(r["mean_test_score"])),
            "parametros": {k.replace("modelo__", ""): v
                           for k, v in busca.best_params_.items()},
        })
        ajustados[nome] = busca.best_estimator_
    tabela = pd.DataFrame(linhas).sort_values("auc_cv", ascending=False)
    return tabela.reset_index(drop=True), ajustados


def escolher(tabela: pd.DataFrame) -> str:
    melhor = tabela.iloc[0]
    rl = tabela.set_index("modelo").loc["regressao_logistica"]
    if melhor.auc_cv - rl.auc_cv <= TOLERANCIA_INTERPRETAVEL:
        return "regressao_logistica"
    return melhor.modelo


def escolher_limiar(modelo, X, y) -> tuple[float, np.ndarray]:
    """Maior limiar com recall >= RECALL_MINIMO, em previsoes fora da amostra."""
    cv = StratifiedKFold(PARTICOES_CV, shuffle=True, random_state=SEMENTE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        oof = cross_val_predict(modelo, X, y, cv=cv, method="predict_proba")[:, 1]
    precisao, recall, limiares = precision_recall_curve(y, oof)
    validos = np.where(recall[:-1] >= RECALL_MINIMO)[0]
    return float(limiares[validos[-1]]), oof


def metricas(y, prob, limiar) -> dict:
    prev = (prob >= limiar).astype(int)
    return {"limiar": round(limiar, 3),
            "auc": roc_auc_score(y, prob),
            "recall_risco": recall_score(y, prev),
            "precisao_risco": precision_score(y, prev),
            "f1_risco": f1_score(y, prev),
            "acuracia": accuracy_score(y, prev),
            "sinalizados_pct": prev.mean(),
            "matriz_confusao": confusion_matrix(y, prev).tolist()}


def grafico_limiar(y, prob, limiar) -> None:
    precisao, recall, limiares = precision_recall_curve(y, prob)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(recall, precisao, color=estilo.AZUL, linewidth=2)
    for lim, rot in [(0.5, "limiar 0,5"), (limiar, f"limiar escolhido {limiar:.2f}".replace(".", ","))]:
        k = np.argmin(np.abs(limiares - lim))
        ax.scatter(recall[k], precisao[k], s=70, zorder=3,
                   color=estilo.COR_RISCO if lim == limiar else estilo.TINTA_2,
                   edgecolor=estilo.SUPERFICIE, linewidth=2)
        ax.annotate(f"{rot}\nrecall {recall[k]:.0%} · precisão {precisao[k]:.0%}",
                    (recall[k], precisao[k]), xytext=(-150, -45),
                    textcoords="offset points", fontsize=9,
                    arrowprops={"arrowstyle": "-", "color": estilo.TINTA_2})
    ax.axhline(y.mean(), color=estilo.TINTA_2, linestyle="--", linewidth=1)
    ax.text(0.02, y.mean() + 0.015, f"sem modelo: {y.mean():.0%}", fontsize=8,
            color=estilo.TINTA_2)
    ax.set_xlabel("Recall: % dos municípios que falham e são detectados")
    ax.set_ylabel("Precisão: % dos alertas que estavam certos")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0.3, 1.02)
    ax.set_title("Curva precisão × recall no teste (classe: não atingir a meta)")
    estilo.salvar(fig, IMAGES / "modelo_precisao_recall.png")


def sensibilidade_sem_rs(base, parametros_modelo, nome_modelo) -> dict:
    """Mesmo protocolo, sem o RS: quanto do resultado vem do choque de 2024."""
    sem = base[base["sigla_uf"] != "RS"]
    X, y = sem[FEATURES], sem[ALVO]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=PROPORCAO_TESTE,
                                          stratify=y, random_state=SEMENTE)
    modelo, _ = candidatos()[nome_modelo]
    pipe = Pipeline([("prep", preprocessador()), ("modelo", modelo)])
    pipe.set_params(**{f"modelo__{k}": v for k, v in parametros_modelo.items()})
    cv = StratifiedKFold(PARTICOES_CV, shuffle=True, random_state=SEMENTE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        oof = cross_val_predict(pipe, Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
        pipe.fit(Xtr, ytr)
    return {"municipios": len(sem), "pct_risco": float(y.mean()),
            "auc_cv": roc_auc_score(ytr, oof),
            "auc_teste": roc_auc_score(yte, pipe.predict_proba(Xte)[:, 1])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sem-busca", action="store_true")
    args = parser.parse_args()

    base = pd.read_parquet(PROCESSED / "base_treino.parquet")
    X, y = base[FEATURES], base[ALVO]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=PROPORCAO_TESTE,
                                          stratify=y, random_state=SEMENTE)
    print(f"  treino {len(Xtr)}  teste {len(Xte)}  risco {y.mean():.1%}")

    tabela, ajustados = comparar(Xtr, ytr, buscar=not args.sem_busca)
    print("\n" + tabela.drop(columns="parametros").round(3).to_string(index=False))

    nome = escolher(tabela)
    modelo = ajustados[nome]
    params = tabela.set_index("modelo").loc[nome, "parametros"]
    print(f"\n  escolhido: {nome} {params}")

    limiar, oof = escolher_limiar(modelo, Xtr, ytr)
    prob_teste = modelo.predict_proba(Xte)[:, 1]
    teste_padrao = metricas(yte, prob_teste, 0.5)
    teste_limiar = metricas(yte, prob_teste, limiar)
    grafico_limiar(yte.values, prob_teste, limiar)

    sens = sensibilidade_sem_rs(base, params, nome)

    MODELS.mkdir(exist_ok=True)
    joblib.dump(modelo, MODELS / "modelo.joblib")
    resultado = {
        "modelo_escolhido": nome, "parametros": params,
        "features": FEATURES, "semente": SEMENTE,
        "treino": len(Xtr), "teste": len(Xte), "pct_risco": float(y.mean()),
        "comparacao_cv": tabela.to_dict("records"),
        "auc_cv_oof": roc_auc_score(ytr, oof),
        "recall_minimo_alvo": RECALL_MINIMO,
        "teste_limiar_0_5": teste_padrao,
        "teste_limiar_escolhido": teste_limiar,
        "gap_cv_teste": float(tabela.set_index("modelo").loc[nome, "auc_cv"]
                              - teste_padrao["auc"]),
        "sensibilidade_sem_rs": sens,
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "resultado_modelo.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False, default=float))

    for rotulo, m in [("limiar 0,5", teste_padrao), ("limiar escolhido", teste_limiar)]:
        print(f"\n  teste ({rotulo}): " + "  ".join(
            f"{k} {v:.3f}" for k, v in m.items() if isinstance(v, float)))
        print(f"    matriz {m['matriz_confusao']}")
    print(f"\n  gap cv x teste: {resultado['gap_cv_teste']:.3f}")
    print(f"  sem RS: {sens}")


if __name__ == "__main__":
    main()
