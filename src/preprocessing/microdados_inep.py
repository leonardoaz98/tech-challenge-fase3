"""
Microdados por aluno da Avaliacao da Alfabetizacao (INEP).

O INEP publica um zip por ano com o arquivo TS_ALUNO.csv: uma linha por
aluno do 2o ano, com escola, municipio, rede, presenca, proficiencia e o
indicador de alfabetizado. E a mesma fonte da tabela `alunos` da Base dos
Dados, sem passar pelo BigQuery.

O QUE ESTE SCRIPT ENTREGA
-------------------------
1. data/raw/inep/alunos_por_escola.csv
   Agregado por ano x municipio x escola x rede. Como nenhuma variavel
   descreve a crianca individualmente (so escola, municipio e rede), um
   modelo por aluno com essas variaveis e matematicamente identico a um
   modelo treinado nas contagens por escola. O agregado tem ~90 mil linhas
   em vez de ~4 milhoes e cabe no repositorio.

2. reports/perfil_microdados.json
   Conferencias da fonte: linhas por ano, escolas, alunos sem resultado,
   valores de cada codigo e, principalmente, quantas escolas aparecem nos
   dois anos com o mesmo codigo (o ID_ESCOLA e mascarado pelo INEP).

Os zips ficam em data/raw/inep/microdados/ e nao sao versionados.

Execucao:
    python -m src.preprocessing.microdados_inep
"""
import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import RAW_INEP, REPORTS

ANOS = [2023, 2024]
URL = "https://download.inep.gov.br/dados_abertos/microdados_avaliacao_da_alfabetizacao_{ano}.zip"
PASTA_ZIP = RAW_INEP / "microdados"
# Pasta onde o navegador salvou os zips (o site do INEP as vezes recusa
# download por script). Padrao: ~/Downloads. No WSL, apontar para a pasta
# do Windows: PASTA_DOWNLOADS=/mnt/c/Users/<usuario>/Downloads
DOWNLOADS = Path(os.getenv("PASTA_DOWNLOADS", str(Path.home() / "Downloads")))
CORTE = 743  # ponto de corte da escala Saeb para "alfabetizado"

COLUNAS = {
    "NU_ANO_AVALIACAO": "ano", "CO_MUNICIPIO": "id_municipio",
    "ID_ESCOLA": "id_escola", "ID_ALUNO": "id_aluno",
    "TP_DEPENDENCIA": "rede", "IN_PRESENCA_LP": "presenca",
    "IN_PREENCHIMENTO_LP": "preenchimento", "VL_PESO_ALUNO_LP": "peso",
    "VL_PROFICIENCIA_LP": "proficiencia", "IN_ALFABETIZADO": "alfabetizado",
}


def obter_zip(ano: int) -> Path:
    """Usa o zip ja baixado (pasta do projeto ou Downloads); senao baixa."""
    nome = f"microdados_avaliacao_da_alfabetizacao_{ano}.zip"
    PASTA_ZIP.mkdir(parents=True, exist_ok=True)
    destino = PASTA_ZIP / nome
    if destino.exists() and zipfile.is_zipfile(destino):
        return destino
    baixado = DOWNLOADS / nome
    if baixado.exists() and zipfile.is_zipfile(baixado):
        print(f"  [{ano}] usando {baixado}")
        destino.write_bytes(baixado.read_bytes())
        return destino
    print(f"  [{ano}] baixando do INEP (pode levar alguns minutos)...")
    try:
        urllib.request.urlretrieve(URL.format(ano=ano), destino)
    except Exception as erro:
        destino.unlink(missing_ok=True)
        sys.exit(f"\n  Nao consegui baixar {ano}: {erro}\n"
                 f"  Baixe pelo navegador e salve em Downloads:\n  {URL.format(ano=ano)}")
    return destino


def ler_alunos(caminho: Path, ano: int) -> pd.DataFrame:
    with zipfile.ZipFile(caminho) as z:
        alvo = [n for n in z.namelist() if n.upper().endswith("TS_ALUNO.CSV")]
        if not alvo:
            sys.exit(f"  TS_ALUNO.csv nao encontrado no zip de {ano}")
        with z.open(alvo[0]) as f:
            df = pd.read_csv(f, sep=";", encoding="latin1", dtype=str,
                             usecols=lambda c: c in COLUNAS)
    df = df.rename(columns=COLUNAS)
    for c in ["peso", "proficiencia"]:
        df[c] = pd.to_numeric(df[c].str.replace(",", ".", regex=False), errors="coerce")
    for c in ["id_municipio", "id_escola", "id_aluno", "rede", "presenca",
              "preenchimento", "alfabetizado"]:
        df[c] = df[c].str.strip()
    df["ano"] = ano
    print(f"  [{ano}] {len(df):,} alunos".replace(",", "."))
    return df


def agregar(df: pd.DataFrame) -> pd.DataFrame:
    tem = df["proficiencia"].notna()
    alf = df["proficiencia"] >= CORTE
    base = df.assign(
        com_resultado=tem.astype(int),
        alf_743=alf.astype(int),
        alf_flag=(df["alfabetizado"] == "1").astype(int),
        peso_alf=np.where(alf, df["peso"], 0.0),
        prof2=df["proficiencia"] ** 2,
    )
    g = base.groupby(["ano", "id_municipio", "id_escola", "rede"], dropna=False)
    out = g.agg(
        alunos_registrados=("id_aluno", "size"),
        alunos_com_resultado=("com_resultado", "sum"),
        alfabetizados_743=("alf_743", "sum"),
        alfabetizados_flag=("alf_flag", "sum"),
        soma_peso=("peso", "sum"),
        soma_peso_alfabetizados=("peso_alf", "sum"),
        media_proficiencia=("proficiencia", "mean"),
        desvio_proficiencia=("proficiencia", "std"),
    ).reset_index()
    return out.round({"soma_peso": 3, "soma_peso_alfabetizados": 3,
                      "media_proficiencia": 2, "desvio_proficiencia": 2})


def perfil(dfs: dict) -> dict:
    p = {}
    for ano, df in dfs.items():
        tem = df["proficiencia"].notna()
        p[str(ano)] = {
            "alunos": len(df),
            "escolas": int(df[["id_municipio", "id_escola"]].drop_duplicates().shape[0]),
            "municipios": int(df["id_municipio"].nunique()),
            "sem_proficiencia": int((~tem).sum()),
            "valores_rede": df["rede"].value_counts().to_dict(),
            "valores_presenca": df["presenca"].value_counts(dropna=False).to_dict(),
            "valores_alfabetizado": df["alfabetizado"].value_counts(dropna=False).to_dict(),
            "concordancia_flag_x_743": float(
                ((df.loc[tem, "alfabetizado"] == "1") == (df.loc[tem, "proficiencia"] >= CORTE)).mean()),
            "taxa_alfabetizados_ponderada": float(
                (df.loc[tem, "peso"] * (df.loc[tem, "proficiencia"] >= CORTE)).sum() / df.loc[tem, "peso"].sum()),
        }
    if len(dfs) == 2:
        a, b = (set(map(tuple, d[["id_municipio", "id_escola"]].drop_duplicates().values))
                for d in dfs.values())
        ia, ib = (set(d["id_aluno"]) for d in dfs.values())
        p["escolas_nos_dois_anos"] = len(a & b)
        p["pct_escolas_2024_que_existiam_em_2023"] = round(len(a & b) / len(b) * 100, 1)
        p["id_aluno_repetido_entre_anos"] = len(ia & ib)
    return p


def main() -> None:
    dfs = {ano: ler_alunos(obter_zip(ano), ano) for ano in ANOS}
    agregado = pd.concat([agregar(d) for d in dfs.values()], ignore_index=True)
    destino = RAW_INEP / "alunos_por_escola.csv"
    agregado.to_csv(destino, index=False)
    prf = perfil(dfs)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "perfil_microdados.json").write_text(
        json.dumps(prf, indent=2, ensure_ascii=False, default=str))
    linhas = f"{len(agregado):,}".replace(",", ".")
    print(f"\n  {destino.name}: {linhas} linhas, {destino.stat().st_size / 1e6:.1f} MB")
    print("\n" + json.dumps(prf, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
