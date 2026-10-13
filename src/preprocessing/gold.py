"""
Camada Gold da Fase 2: reconstrucao local e leitura.

ORIGEM
------
Na Fase 2 (github.com/leonardoaz98/aws-tech-challenge2-fiap) a Gold e um
star schema materializado no S3 e consultado via Athena. Este modulo
reproduz o mesmo modelo dimensional a partir dos CSVs publicos do INEP
(Base dos Dados), para que o projeto rode em qualquer maquina, sem
credencial AWS.

    fato_alfabetizacao  id_municipio + ano  (2023 e 2024 medidos; 2025-2030 so meta)
    dim_municipio       id_municipio
    dim_uf              sigla_uf
    dim_tempo           ano
    dim_nivel           nivel_alfabetizacao (0 a 5)

LEITURA
-------
`carregar(tabela)` le do snapshot local (padrao) ou do S3 quando
GOLD_FONTE=s3. O resto do projeto nao sabe de onde o dado veio.

Execucao (reconstroi o snapshot local):
    python -m src.preprocessing.gold
"""
import sys

import numpy as np
import pandas as pd

from src.config import GOLD, GOLD_FONTE, RAW_INEP, REDE_MUNICIPAL, S3_GOLD

ANOS_META = range(2024, 2031)
ANO_ULTIMO_RESULTADO = 2024

REGIOES = {
    "AC": "Norte", "AP": "Norte", "AM": "Norte", "PA": "Norte",
    "RO": "Norte", "RR": "Norte", "TO": "Norte",
    "AL": "Nordeste", "BA": "Nordeste", "CE": "Nordeste", "MA": "Nordeste",
    "PB": "Nordeste", "PE": "Nordeste", "PI": "Nordeste", "RN": "Nordeste",
    "SE": "Nordeste",
    "DF": "Centro-Oeste", "GO": "Centro-Oeste", "MT": "Centro-Oeste",
    "MS": "Centro-Oeste",
    "ES": "Sudeste", "MG": "Sudeste", "RJ": "Sudeste", "SP": "Sudeste",
    "PR": "Sul", "RS": "Sul", "SC": "Sul",
}

# Prefixo de 2 digitos do codigo IBGE -> UF
CODIGO_UF = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP",
    "17": "TO", "21": "MA", "22": "PI", "23": "CE", "24": "RN", "25": "PB",
    "26": "PE", "27": "AL", "28": "SE", "29": "BA", "31": "MG", "32": "ES",
    "33": "RJ", "35": "SP", "41": "PR", "42": "SC", "43": "RS", "50": "MS",
    "51": "MT", "52": "GO", "53": "DF",
}

# Mesmas faixas da Gold da Fase 2
FAIXAS_NIVEL = [
    (0, "Critico", 0, 40), (1, "Muito baixo", 40, 50), (2, "Baixo", 50, 60),
    (3, "Medio", 60, 70), (4, "Alto", 70, 80), (5, "Muito alto", 80, 101),
]

TABELAS = ["fato_alfabetizacao", "dim_municipio", "dim_uf", "dim_tempo", "dim_nivel"]


# ---------------------------------------------------------------------
# Leitura
# ---------------------------------------------------------------------
def carregar(tabela: str) -> pd.DataFrame:
    """Le uma tabela da Gold (snapshot local ou S3)."""
    if GOLD_FONTE == "s3":
        import awswrangler as wr  # dependencia opcional
        return wr.s3.read_parquet(path=f"{S3_GOLD}/{tabela}/", dataset=True)

    caminho = GOLD / f"{tabela}.parquet"
    if not caminho.exists():
        sys.exit(f"Gold nao encontrada em {caminho}. "
                 "Rode: python -m src.preprocessing.gold")
    return pd.read_parquet(caminho)


# ---------------------------------------------------------------------
# Reconstrucao
# ---------------------------------------------------------------------
def _ler_inep(nome: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_INEP / f"br_inep_avaliacao_alfabetizacao_{nome}.csv",
                     dtype={"id_municipio": str})
    if "id_municipio" in df.columns:
        df["id_municipio"] = df["id_municipio"].str.strip().str.zfill(7)
    return df


def classificar_nivel(taxa: pd.Series) -> pd.Series:
    cortes = [f[2] for f in FAIXAS_NIVEL] + [101]
    return pd.cut(taxa, bins=cortes, right=False,
                  labels=[f[0] for f in FAIXAS_NIVEL]).astype("Int64")


def construir() -> dict:
    resultado = _ler_inep("municipio")
    resultado = resultado[resultado["rede"] == REDE_MUNICIPAL]
    resultado = resultado[["id_municipio", "ano", "taxa_alfabetizacao", "media_portugues"]]

    metas = _ler_inep("meta_alfabetizacao_municipio")

    # Participacao e publicada junto com as metas, uma linha por ano.
    participacao = metas[["id_municipio", "ano", "percentual_participacao"]]

    # Metas: usa a publicacao mais recente. Conferido: as publicacoes de
    # 2023 e 2024 sao identicas para todos os anos-meta (0 divergencias).
    ultima = metas[metas["ano"] == metas["ano"].max()]
    metas_longo = ultima.melt(
        id_vars="id_municipio",
        value_vars=[f"meta_alfabetizacao_{a}" for a in ANOS_META],
        var_name="ano", value_name="meta",
    )
    metas_longo["ano"] = metas_longo["ano"].str[-4:].astype(int)

    # Fato: anos medidos (resultado + meta) e anos futuros (so meta)
    medidos = (resultado
               .merge(participacao, on=["id_municipio", "ano"], how="left")
               .merge(metas_longo, on=["id_municipio", "ano"], how="left"))
    futuros = metas_longo[metas_longo["ano"] > ANO_ULTIMO_RESULTADO]
    fato = pd.concat([medidos, futuros], ignore_index=True)
    fato = fato.rename(columns={"taxa_alfabetizacao": "taxa_realizada"})
    fato["rede"] = REDE_MUNICIPAL
    fato["gap_meta"] = (fato["taxa_realizada"] - fato["meta"]).round(2)
    fato["meta_atingida"] = np.where(
        fato["taxa_realizada"].isna() | fato["meta"].isna(), pd.NA,
        (fato["taxa_realizada"] >= fato["meta"]).astype(int),
    )
    fato["meta_atingida"] = fato["meta_atingida"].astype("Int64")
    fato["nivel_alfabetizacao"] = classificar_nivel(fato["taxa_realizada"])
    fato = fato.sort_values(["id_municipio", "ano"]).reset_index(drop=True)

    ids = pd.Series(sorted(fato["id_municipio"].unique()), name="id_municipio")
    dim_municipio = pd.DataFrame({"id_municipio": ids,
                                  "sigla_uf": ids.str[:2].map(CODIGO_UF)})
    ufs = sorted(dim_municipio["sigla_uf"].unique())
    dim_uf = pd.DataFrame({"sigla_uf": ufs, "regiao": [REGIOES[u] for u in ufs]})
    dim_tempo = pd.DataFrame({"ano": sorted(fato["ano"].unique())})
    dim_tempo["tipo"] = np.where(dim_tempo["ano"] <= ANO_ULTIMO_RESULTADO,
                                 "medido", "so_meta")
    dim_nivel = pd.DataFrame([{"nivel_alfabetizacao": n, "descricao": d,
                               "faixa": f"{a} a {min(b, 100)}%"}
                              for n, d, a, b in FAIXAS_NIVEL])

    return {"fato_alfabetizacao": fato, "dim_municipio": dim_municipio,
            "dim_uf": dim_uf, "dim_tempo": dim_tempo, "dim_nivel": dim_nivel}


def validar(gold: dict) -> None:
    fato = gold["fato_alfabetizacao"]
    assert not fato.duplicated(["id_municipio", "ano"]).any(), "fato com duplicata"
    assert fato["taxa_realizada"].dropna().between(0, 100).all(), "taxa fora de 0-100"
    assert fato["id_municipio"].isin(gold["dim_municipio"]["id_municipio"]).all()
    assert gold["dim_municipio"]["sigla_uf"].notna().all(), "municipio sem UF"


def main() -> None:
    gold = construir()
    validar(gold)
    GOLD.mkdir(parents=True, exist_ok=True)
    for nome, df in gold.items():
        df.to_parquet(GOLD / f"{nome}.parquet", index=False)
        print(f"  {nome:22} {len(df):>7} linhas")
    print("\n  linhas do fato por ano:")
    print(gold["fato_alfabetizacao"].groupby("ano").size().to_string())


if __name__ == "__main__":
    main()
