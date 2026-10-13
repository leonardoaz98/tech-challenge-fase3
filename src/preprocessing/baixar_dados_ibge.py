"""
Download das fontes do IBGE usadas no enriquecimento da base.

Grava em data/raw/ibge/ tres CSVs, sem transformacao alem de achatar o
JSON das APIs. Os arquivos ja estao versionados; este script so e
necessario para atualizar os dados.

    ibge_localidades.csv  API de Localidades     nome, micro/mesorregiao, regioes imediata e intermediaria
    ibge_populacao.csv    SIDRA, tabela 6579     populacao residente estimada
    ibge_pib.csv          SIDRA, tabela 5938     PIB a precos correntes (mil reais)

Execucao:
    python -m src.preprocessing.baixar_dados_ibge
    python -m src.preprocessing.baixar_dados_ibge --ano 2021 --forcar
"""
import argparse
import json
import sys
import urllib.request

import pandas as pd

from src.config import RAW_IBGE

URL_LOCALIDADES = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"
URL_SIDRA = "https://apisidra.ibge.gov.br/values/t/{tabela}/n6/all/v/{variavel}/p/{ano}"
POPULACAO = {"tabela": 6579, "variavel": 9324}   # populacao residente estimada
PIB = {"tabela": 5938, "variavel": 37}           # PIB a precos correntes


def baixar(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


def _nome(no: dict | None, *chaves: str):
    """Desce no JSON aninhado; devolve None se algum nivel faltar."""
    for chave in chaves:
        if not isinstance(no, dict):
            return None
        no = no.get(chave)
    return no


def localidades(dados: list) -> pd.DataFrame:
    linhas = []
    for m in dados:
        # municipios criados depois da ultima divisao regional vem sem
        # microrregiao; a UF sai entao da regiao intermediaria
        uf = (_nome(m, "microrregiao", "mesorregiao", "UF", "sigla")
              or _nome(m, "regiao-imediata", "regiao-intermediaria", "UF", "sigla"))
        linhas.append({
            "id_municipio": str(m["id"]),
            "nome_municipio": m["nome"],
            "microrregiao": _nome(m, "microrregiao", "nome"),
            "mesorregiao": _nome(m, "microrregiao", "mesorregiao", "nome"),
            "sigla_uf_ibge": uf,
            "regiao_imediata": _nome(m, "regiao-imediata", "nome"),
            "regiao_intermediaria": _nome(m, "regiao-imediata", "regiao-intermediaria", "nome"),
        })
    return pd.DataFrame(linhas)


def sidra(dados: list) -> pd.DataFrame:
    """A primeira linha do SIDRA traz os rotulos de cada campo; o resto, os valores."""
    rotulos, linhas = dados[0], pd.DataFrame(dados[1:])
    campo_codigo = next(k for k, v in rotulos.items() if v.startswith("Município (Código)"))
    return pd.DataFrame({
        "id_municipio": linhas[campo_codigo].astype(str),
        # '-', '...' e 'X' significam valor indisponivel ou sigiloso
        "valor": pd.to_numeric(linhas["V"], errors="coerce").astype(float),
        "unidade": linhas["MN"],
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ano", type=int, default=2021, help="ano de populacao e PIB")
    parser.add_argument("--forcar", action="store_true", help="baixa mesmo se o arquivo existir")
    args = parser.parse_args()
    RAW_IBGE.mkdir(parents=True, exist_ok=True)

    tarefas = {
        "ibge_localidades.csv": lambda: localidades(baixar(URL_LOCALIDADES)),
        "ibge_populacao.csv": lambda: (
            sidra(baixar(URL_SIDRA.format(ano=args.ano, **POPULACAO)))
            .rename(columns={"valor": "populacao"})[["id_municipio", "populacao"]]),
        "ibge_pib.csv": lambda: (
            sidra(baixar(URL_SIDRA.format(ano=args.ano, **PIB)))
            .rename(columns={"valor": "pib", "unidade": "pib_unidade"})
            .assign(pib_variavel="Produto Interno Bruto a preços correntes")
            [["id_municipio", "pib", "pib_variavel", "pib_unidade"]]),
    }
    for nome, tarefa in tarefas.items():
        destino = RAW_IBGE / nome
        if destino.exists() and not args.forcar:
            print(f"  {nome}: ja existe (use --forcar para baixar de novo)")
            continue
        try:
            df = tarefa()
        except Exception as erro:
            sys.exit(f"  {nome}: falhou ({erro})")
        df.to_csv(destino, index=False)
        print(f"  {nome}: {len(df)} municipios")

    pib = pd.read_csv(RAW_IBGE / "ibge_pib.csv")
    if not pib["pib_unidade"].str.contains("Mil", case=False).all():
        sys.exit("  PIB deveria vir em mil reais: confira a unidade antes de usar")


if __name__ == "__main__":
    main()
