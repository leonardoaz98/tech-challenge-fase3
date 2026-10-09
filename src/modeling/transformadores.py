"""Transformadores proprios usados dentro do Pipeline."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class LimitarFaixa(BaseEstimator, TransformerMixin):
    """Corta cada variavel na faixa vista no treino (minimo e maximo).

    POR QUE
    -------
    A regressao logistica extrapola em linha reta. Em 2025, 37% dos
    municipios tem esforco exigido acima do maior valor visto no treino
    (7,2 p.p.), porque as metas de 2025 seguem a trajetoria original e
    nao foram recalculadas apos o resultado de 2024. Sem o corte, o
    modelo daria probabilidades extremas com base em um territorio que
    ele nunca observou.

    No treino o corte nao muda nada (os limites vem do proprio treino);
    ele so age em dados novos fora da faixa.
    """

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.minimo_ = np.nanmin(X, axis=0)
        self.maximo_ = np.nanmax(X, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.minimo_, self.maximo_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)
