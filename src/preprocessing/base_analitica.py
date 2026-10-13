"""
Base analitica: uma linha por municipio, pronta para o modelo.

DESENHO TEMPORAL
----------------
A mesma funcao monta duas bases com as MESMAS colunas:

    treino     features de 2023  ->  atingiu a meta de 2024?   (alvo conhecido)
    previsao   features de 2024  ->  vai atingir a meta de 2025? (alvo desconhecido)

Os nomes das colunas nao carregam o ano (taxa_atual, e nao taxa_2023).
Assim o modelo treinado no primeiro par e aplicado no segundo sem
nenhuma traducao, que e exatamente o uso real: o gestor no inicio de
2025 conhece o resultado de 2024 e a meta de 2025.

ALVO
----
`risco` = 1 quando o municipio NAO atingiu a meta. A classe positiva e
o evento que a politica publica quer detectar. Com isso recall,
precisao e limiar passam a falar diretamente de "municipios que vao
falhar", sem inversao mental.

VARIAVEIS FORA DO MODELO (decididas aqui, nao na EDA)
------------------------------------------------------
- meta_alvo: e taxa_atual + esforco_exigido. As tres juntas sao
  perfeitamente colineares; o coeficiente de cada uma deixa de ter
  leitura. Mantemos taxa_atual e esforco_exigido.
- nivel_alfabetizacao: faixa da propria taxa (0-5). Informacao repetida.
- _taxa_alvo: resultado do ano-alvo. E de onde o alvo sai; entrar no
  modelo seria vazamento direto. Fica so para auditoria.

ENRIQUECIMENTO IBGE
-------------------
Logica de populacao, porte e PIB por habitante adaptada do trabalho da
Fernanda (github.com/fernanda161082/tech-challenge-fase3).

Execucao:
    python -m src.preprocessing.base_analitica
"""
import numpy as np
import pandas as pd

from src.config import (ALVO, ANO_ALVO_PREVISAO, ANO_ALVO_TREINO,
                        ANO_ENTRADA_PREVISAO, ANO_ENTRADA_TREINO,
                        PROCESSED, RAW_IBGE)
from src.preprocessing.gold import carregar

FAIXAS_PORTE = [
    (0, 20_000, "pequeno_I"), (20_000, 50_000, "pequeno_II"),
    (50_000, 100_000, "medio"), (100_000, 900_000, "grande"),
    (900_000, np.inf, "metropole"),
]

NUMERICAS = ["taxa_atual", "media_portugues", "participacao",
             "esforco_exigido", "log_populacao", "log_pib_per_capita"]
CATEGORICAS = ["sigla_uf", "regiao", "porte_municipio"]
FEATURES = NUMERICAS + CATEGORICAS


def _ibge(nome: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_IBGE / f"ibge_{nome}.csv", dtype={"id_municipio": str})
    df["id_municipio"] = df["id_municipio"].str.zfill(7)
    return df.drop_duplicates("id_municipio").set_index("id_municipio")


def contexto_ibge() -> pd.DataFrame:
    """Nome, territorio, populacao, porte e PIB por habitante (IBGE)."""
    loc = _ibge("localidades")[["nome_municipio", "microrregiao", "mesorregiao",
                                "sigla_uf_ibge"]]
    pop = _ibge("populacao")[["populacao"]]
    pib = _ibge("pib")
    assert pib["pib_unidade"].str.contains("Mil", case=False).all(), \
        "PIB deveria vir em mil reais"

    ctx = loc.join(pop).join(pib[["pib"]])
    ctx["pib_per_capita"] = ctx["pib"] * 1000 / ctx["populacao"]
    ctx["log_populacao"] = np.log1p(ctx["populacao"])
    ctx["log_pib_per_capita"] = np.log(ctx["pib_per_capita"])
    ctx["porte_municipio"] = pd.cut(
        ctx["populacao"], bins=[f[0] for f in FAIXAS_PORTE] + [np.inf],
        right=False, labels=[f[2] for f in FAIXAS_PORTE]).astype(str)
    return ctx.drop(columns=["pib"])


def montar(ano_entrada: int, ano_alvo: int) -> pd.DataFrame:
    fato = carregar("fato_alfabetizacao").set_index("id_municipio")
    mun = carregar("dim_municipio").merge(carregar("dim_uf"), on="sigla_uf")

    entrada = fato[fato["ano"] == ano_entrada][
        ["taxa_realizada", "media_portugues", "percentual_participacao"]]
    entrada.columns = ["taxa_atual", "media_portugues", "participacao"]

    alvo = fato[fato["ano"] == ano_alvo][["meta", "taxa_realizada"]]
    alvo.columns = ["meta_alvo", "_taxa_alvo"]

    base = (entrada.join(alvo, how="inner")
            .join(mun.set_index("id_municipio"))
            .join(contexto_ibge()))

    base = base[base["meta_alvo"].notna() & base["taxa_atual"].notna()].copy()
    base["esforco_exigido"] = (base["meta_alvo"] - base["taxa_atual"]).round(2)

    if base["_taxa_alvo"].notna().any():
        base = base[base["_taxa_alvo"].notna()]
        base[ALVO] = (base["_taxa_alvo"] < base["meta_alvo"]).astype(int)

    # Conferencia cruzada: UF pelo codigo IBGE x UF informada pelo IBGE
    divergentes = (base["sigla_uf"] != base["sigla_uf_ibge"]).sum()
    assert divergentes == 0, f"{divergentes} municipios com UF divergente"
    base = base.drop(columns=["sigla_uf_ibge"])

    base["ano_entrada"], base["ano_alvo"] = ano_entrada, ano_alvo
    return base.sort_index()


def resumo(base: pd.DataFrame, nome: str) -> None:
    print(f"\n  [{nome}] {base.ano_entrada.iat[0]} -> {base.ano_alvo.iat[0]}")
    print(f"    municipios: {len(base)}   UFs: {base.sigla_uf.nunique()}")
    nulos = base[FEATURES].isna().sum()
    print(f"    nulos nas features: {int(nulos.sum())}")
    if ALVO in base:
        print(f"    nao atingiram (risco=1): {base[ALVO].mean():.1%}")


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    treino = montar(ANO_ENTRADA_TREINO, ANO_ALVO_TREINO)
    previsao = montar(ANO_ENTRADA_PREVISAO, ANO_ALVO_PREVISAO)
    treino.to_parquet(PROCESSED / "base_treino.parquet")
    previsao.to_parquet(PROCESSED / "base_previsao.parquet")
    resumo(treino, "treino")
    resumo(previsao, "previsao")


if __name__ == "__main__":
    main()
