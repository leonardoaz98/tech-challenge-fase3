"""
Contrato da Gold: o que cada tabela tem de ter para o resto do projeto rodar.

POR QUE EXISTE
--------------
Na Fase 2 a Gold gravada pelo build_gold.py divergiu do esquema que as
views e o dashboard liam, e a sequencia do README quebrava. Aqui o
esquema e escrito uma vez, neste arquivo, e conferido em tres pontos:

    1. ao gravar a Gold (gold.py e escolas.py)
    2. ao ler a Gold (gold.carregar), antes de qualquer consumidor usar
    3. contra a documentacao tecnica: toda coluna do contrato tem de
       estar no dicionario de dados

Coluna faltando, coluna a mais, tipo errado ou chave duplicada param a
execucao com a lista do que divergiu.

Execucao (confere os parquets de data/gold e a documentacao):
    python -m src.preprocessing.contrato_gold
"""
import sys

import pandas as pd
from pandas.api import types

from src.config import GOLD, REPORTS

TIPOS = {
    "texto": types.is_string_dtype,
    "inteiro": types.is_integer_dtype,
    "decimal": types.is_float_dtype,
}

CONTRATO = {
    "fato_alfabetizacao": {
        "chave": ["id_municipio", "ano"],
        "colunas": {
            "id_municipio": "texto", "ano": "inteiro",
            "taxa_realizada": "decimal", "media_portugues": "decimal",
            "percentual_participacao": "decimal", "meta": "decimal",
            "rede": "inteiro", "gap_meta": "decimal",
            "meta_atingida": "inteiro", "nivel_alfabetizacao": "inteiro",
        },
    },
    "dim_municipio": {
        "chave": ["id_municipio"],
        "colunas": {"id_municipio": "texto", "sigla_uf": "texto"},
    },
    "dim_uf": {
        "chave": ["sigla_uf"],
        "colunas": {"sigla_uf": "texto", "regiao": "texto"},
    },
    "dim_tempo": {
        "chave": ["ano"],
        "colunas": {"ano": "inteiro", "tipo": "texto"},
    },
    "dim_nivel": {
        "chave": ["nivel_alfabetizacao"],
        "colunas": {"nivel_alfabetizacao": "inteiro", "descricao": "texto",
                    "faixa": "texto"},
    },
    "fato_meta_uf": {
        "chave": ["sigla_uf", "ano"],
        "colunas": {
            "sigla_uf": "texto", "ano": "inteiro", "taxa_realizada": "decimal",
            "percentual_participacao": "decimal", "meta": "decimal",
            "gap_meta": "decimal", "meta_atingida": "inteiro",
        },
    },
    "fato_escola": {
        "chave": ["ano", "id_municipio", "id_escola", "rede"],
        "colunas": {
            "ano": "inteiro", "id_municipio": "texto", "id_escola": "texto",
            "rede": "inteiro", "alunos_registrados": "inteiro",
            "alunos_com_resultado": "inteiro", "alfabetizados_743": "inteiro",
            "alfabetizados_flag": "inteiro", "soma_peso": "decimal",
            "soma_peso_alfabetizados": "decimal", "media_proficiencia": "decimal",
            "desvio_proficiencia": "decimal", "participacao": "decimal",
            "taxa_escola": "decimal",
        },
    },
}


class ContratoQuebrado(Exception):
    pass


def divergencias(tabela: str, df: pd.DataFrame) -> list[str]:
    """Lista o que a tabela tem de diferente do contrato (vazia = conforme)."""
    c = CONTRATO[tabela]
    esperadas, presentes = list(c["colunas"]), list(df.columns)
    erros = [f"falta a coluna {col}" for col in esperadas if col not in presentes]
    erros += [f"coluna fora do contrato: {col}" for col in presentes if col not in esperadas]
    for col, tipo in c["colunas"].items():
        if col in df.columns and not TIPOS[tipo](df[col]):
            erros.append(f"{col} deveria ser {tipo}, veio {df[col].dtype}")
    if all(k in df.columns for k in c["chave"]):
        dup = int(df.duplicated(c["chave"]).sum())
        if dup:
            erros.append(f"{dup} linhas duplicadas na chave {c['chave']}")
        nulos = int(df[c["chave"]].isna().any(axis=1).sum())
        if nulos:
            erros.append(f"{nulos} linhas com chave nula")
    return erros


def checar(tabela: str, df: pd.DataFrame) -> pd.DataFrame:
    """Para a execucao se a tabela nao seguir o contrato; devolve o df intacto."""
    erros = divergencias(tabela, df)
    if erros:
        raise ContratoQuebrado(f"{tabela} fora do contrato:\n  - " + "\n  - ".join(erros))
    return df


def divergencias_documentacao() -> list[str]:
    doc = (REPORTS / "documentacao_tecnica.md").read_text(encoding="utf-8")
    return [f"{t}.{col} nao esta na documentacao tecnica"
            for t, c in CONTRATO.items() for col in c["colunas"]
            if f"`{col}`" not in doc]


def main() -> None:
    erros = []
    for tabela in CONTRATO:
        caminho = GOLD / f"{tabela}.parquet"
        if not caminho.exists():
            erros.append(f"{caminho.name} nao existe")
            continue
        df = pd.read_parquet(caminho)
        problemas = divergencias(tabela, df)
        erros += [f"{tabela}: {p}" for p in problemas]
        print(f"  {tabela:20} {len(df):>7} linhas  {'OK' if not problemas else 'FORA DO CONTRATO'}")
    doc = divergencias_documentacao()
    erros += doc
    print(f"  {'documentacao':20} {'':>7}        {'OK' if not doc else 'DESATUALIZADA'}")
    if erros:
        sys.exit("\n".join(["", "Contrato quebrado:"] + [f"  - {e}" for e in erros]))
    print("\n  Gold e documentacao conferem com o contrato.")


if __name__ == "__main__":
    main()
