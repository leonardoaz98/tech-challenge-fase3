"""
Camada Gold da Fase 2: reconstrucao local e leitura.

ORIGEM
------
Na Fase 2 (github.com/leonardoaz98/aws-tech-challenge2-fiap) a Gold e um
star schema no S3, consultado via Athena. Este modulo reconstroi a Gold a
partir dos CSVs publicos do INEP (Base dos Dados), para que o projeto
rode em qualquer maquina, sem credencial AWS.

ESQUEMA
-------
Segue o esquema que as views do Athena e o dashboard da Fase 2 leem
(taxa_realizada, meta, gap_meta, meta_atingida), e nao os nomes que o
build_gold.py da Fase 2 grava (taxa_alfabetizacao). Na Fase 2 essa
diferenca quebrava as views. Aqui o esquema esta escrito em
contrato_gold.py e e conferido ao gravar e ao ler.

    fato_alfabetizacao  id_municipio + ano  (2023 e 2024 medidos; 2025-2030 so meta)
    dim_municipio       id_municipio
    dim_uf              sigla_uf
    dim_tempo           ano
    dim_nivel           nivel_alfabetizacao (0 a 5)
    fato_meta_uf        sigla_uf + ano      (Brasil e UFs, rede publica: resultado e meta)

LEITURA
-------
`carregar(tabela)` le o snapshot local e confere o contrato antes de
entregar a tabela. A leitura direta do S3 foi retirada: sem acesso ao
bucket nao havia como testa-la, e caminho documentado que ninguem rodou
nao fica no projeto.

Execucao (reconstroi o snapshot local):
    python -m src.preprocessing.gold
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import GOLD, RAW_INEP, REDE_MUNICIPAL
from src.preprocessing.contrato_gold import ContratoQuebrado, checar

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

TABELAS = ["fato_alfabetizacao", "dim_municipio", "dim_uf", "dim_tempo", "dim_nivel",
           "fato_meta_uf"]

# Metas pactuadas: publicacao de 2024, a mesma usada para os municipios.
# A publicacao de 2025 do Brasil arredonda as metas para inteiros.
ANO_PUBLICACAO_METAS = 2024

# Planilha do INEP com o resultado de 2025 do Brasil e das UFs, como publicada:
# https://download.inep.gov.br/avaliacao_da_alfabetizacao/resultados/resultados_e_metas_ufs_2025_v2.xlsx
PLANILHA_UF_2025 = RAW_INEP / "resultados_e_metas_ufs_2025.xlsx"


# ---------------------------------------------------------------------
# Leitura
# ---------------------------------------------------------------------
def carregar(tabela: str) -> pd.DataFrame:
    """Le uma tabela da Gold e confere o contrato antes de devolver."""
    caminho = GOLD / f"{tabela}.parquet"
    if not caminho.exists():
        sys.exit(f"Gold nao encontrada em {caminho}. "
                 "Rode: python -m src.preprocessing.gold")
    try:
        return checar(tabela, pd.read_parquet(caminho))
    except ContratoQuebrado as erro:
        sys.exit(str(erro))


# ---------------------------------------------------------------------
# Reconstrucao
# ---------------------------------------------------------------------
def _ler_inep(nome: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_INEP / f"br_inep_avaliacao_alfabetizacao_{nome}.csv",
                     dtype={"id_municipio": str})
    if "id_municipio" in df.columns:
        df["id_municipio"] = df["id_municipio"].str.strip().str.zfill(7)
    return df


def ler_planilha_inep(caminho: Path) -> pd.DataFrame:
    """Planilha de divulgacao do INEP: 1a linha com rotulos, 2a com os nomes
    tecnicos das colunas, notas de rodape no fim."""
    df = pd.read_excel(caminho, sheet_name=0, header=1)
    return df[pd.to_numeric(df["ANO"], errors="coerce").notna()].reset_index(drop=True)


def construir_meta_uf() -> pd.DataFrame:
    """Resultado e meta do Brasil e de cada UF (rede publica), 2023 a 2030."""
    uf = _ler_inep("meta_alfabetizacao_uf")
    br = _ler_inep("meta_alfabetizacao_brasil").assign(sigla_uf="BR")
    pub = pd.concat([uf, br], ignore_index=True)

    # Resultado medido: cada linha de publicacao traz a taxa do proprio ano
    medidos = pub[["sigla_uf", "ano", "taxa_alfabetizacao", "percentual_participacao"]]

    # 2025 das UFs ainda nao esta na Base dos Dados: vem da planilha do INEP
    # (inteiros, como o INEP divulga). O Brasil 2025 ja esta no CSV.
    p = ler_planilha_inep(PLANILHA_UF_2025)
    novo = pd.DataFrame({
        "sigla_uf": p["SIGLA_UF"].fillna("BR").astype(str).str.replace("*", "", regex=False).str.strip(),
        "ano": 2025,
        "taxa_alfabetizacao": pd.to_numeric(p["PC_ALUNO_ALFABETIZADO_2025"], errors="coerce").astype(float),
        "percentual_participacao": pd.to_numeric(p["PC_AVALIADOS_LP"], errors="coerce").astype(float),
    })
    ja_tem = set(medidos.loc[medidos["ano"] == 2025, "sigla_uf"])
    medidos = pd.concat([medidos, novo[~novo["sigla_uf"].isin(ja_tem)]], ignore_index=True)

    metas = pub[pub["ano"] == ANO_PUBLICACAO_METAS].melt(
        id_vars="sigla_uf", value_vars=[f"meta_alfabetizacao_{a}" for a in ANOS_META],
        var_name="ano", value_name="meta")
    metas["ano"] = metas["ano"].str[-4:].astype(int)

    fato = (medidos.merge(metas, on=["sigla_uf", "ano"], how="outer")
            .rename(columns={"taxa_alfabetizacao": "taxa_realizada"}))
    fato["ano"] = fato["ano"].astype(int)
    for c in ["taxa_realizada", "percentual_participacao", "meta"]:
        fato[c] = fato[c].astype(float)
    fato["gap_meta"] = (fato["taxa_realizada"] - fato["meta"]).round(2)
    fato["meta_atingida"] = np.where(
        fato["taxa_realizada"].isna() | fato["meta"].isna(), pd.NA,
        (fato["taxa_realizada"] >= fato["meta"]).astype(int))
    fato["meta_atingida"] = fato["meta_atingida"].astype("Int64")
    colunas = ["sigla_uf", "ano", "taxa_realizada", "percentual_participacao",
               "meta", "gap_meta", "meta_atingida"]
    return fato[colunas].sort_values(["sigla_uf", "ano"]).reset_index(drop=True)


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
            "dim_uf": dim_uf, "dim_tempo": dim_tempo, "dim_nivel": dim_nivel,
            "fato_meta_uf": construir_meta_uf()}


def validar(gold: dict) -> None:
    for nome, df in gold.items():
        checar(nome, df)
    fato = gold["fato_alfabetizacao"]
    assert fato["taxa_realizada"].dropna().between(0, 100).all(), "taxa fora de 0-100"
    assert fato["id_municipio"].isin(gold["dim_municipio"]["id_municipio"]).all()
    assert gold["dim_municipio"]["sigla_uf"].notna().all(), "municipio sem UF"
    uf = gold["fato_meta_uf"]
    assert set(uf["sigla_uf"]) == set(REGIOES) | {"BR"}, "fato_meta_uf sem todas as UFs e o Brasil"
    assert uf["taxa_realizada"].dropna().between(0, 100).all(), "taxa de UF fora de 0-100"


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
