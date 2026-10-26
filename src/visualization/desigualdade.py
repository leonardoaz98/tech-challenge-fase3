"""
Onde esta a desigualdade na alfabetizacao: estado, municipio, escola ou aluno?

Usa as escolas de 2024 (fato_escola). Cada escola entra com o peso dos seus
alunos, entao todas as contas sao no nivel do aluno.

1. DECOMPOSICAO DA VARIANCIA
   A variancia da proficiencia entre todos os alunos se divide, sem sobra,
   em quatro partes:
       entre UFs + entre municipios da mesma UF
       + entre escolas do mesmo municipio + dentro da escola
   Usa a media e o desvio de cada escola, que vem do microdado. Parte da
   diferenca entre escolas pequenas e ruido de amostra (desvio^2 / n); ela
   e descontada na versao corrigida. A mesma conta e repetida com o
   indicador alfabetizado (0/1) como conferencia.

2. DISPERSAO DENTRO DO MUNICIPIO
   Distancia entre a escola do percentil 90 e a do percentil 10, em
   municipios com 3 ou mais escolas com pelo menos 10 alunos avaliados.

3. ESCOLAS CRITICAS ESCONDIDAS
   Escolas municipais com menos de 40% de alfabetizados em municipios que
   bateram a meta de 2024. A media do municipio esconde essas escolas.

O ACASO
   Escola pequena oscila muito por sorte: com 20 alunos, a taxa varia
   cerca de 11 p.p. sem nenhuma diferenca real. Para separar o real do
   acaso, os itens 2 e 3 sao recalculados em 20 simulacoes onde toda escola
   tem a taxa do seu municipio e so o sorteio dos alunos varia. A diferenca
   entre o observado e o simulado e o que e real.

4. REDE E PORTE DA ESCOLA
   Taxa por rede (estadual x municipal) e por tamanho da turma avaliada.

Execucao:
    python -m src.visualization.desigualdade
"""
import json

import numpy as np
import pandas as pd

from src.config import IMAGES, RAW_IBGE, REDE_MUNICIPAL, REPORTS
from src.preprocessing.gold import CODIGO_UF, carregar
from src.visualization import estilo
from src.visualization.estilo import plt

ANO = 2024
MIN_ALUNOS = 10       # escola com menos avaliados fica fora das contas por escola
CORTE_CRITICO = 40    # % de alfabetizados abaixo do qual a escola e critica
REDES = {2: "Estadual", 3: "Municipal", 4: "Federal"}


def carregar_escolas() -> pd.DataFrame:
    e = carregar("fato_escola")
    e = e[(e.ano == ANO) & (e.soma_peso > 0)].copy()
    e["sigla_uf"] = e.id_municipio.str[:2].map(CODIGO_UF)
    e["p"] = e.soma_peso_alfabetizados / e.soma_peso
    return e


def decompor(e: pd.DataFrame) -> dict:
    w, p = e.soma_peso, e.p
    W = w.sum()
    p_geral = (w * p).sum() / W

    def media_grupo(col):
        g = e.assign(wp=w * p).groupby(col)
        return g.wp.transform("sum") / g.soma_peso.transform("sum")

    p_uf = media_grupo("sigla_uf")
    p_mun = media_grupo("id_municipio")
    partes = {
        "entre_ufs": (w * (p_uf - p_geral) ** 2).sum() / W,
        "entre_municipios": (w * (p_mun - p_uf) ** 2).sum() / W,
        "entre_escolas": (w * (p - p_mun) ** 2).sum() / W,
        "dentro_da_escola": (w * p * (1 - p)).sum() / W,
    }
    total = p_geral * (1 - p_geral)
    pct = {k: round(v / total * 100, 1) for k, v in partes.items()}

    # ruido de amostra da diferenca entre escolas (por escola: p(1-p)/n)
    n = e.alunos_com_resultado.clip(lower=1)
    ruido = (w * p * (1 - p) / n).sum() / W
    corr = dict(partes)
    corr["entre_escolas"] = max(partes["entre_escolas"] - ruido, 0)
    corr["dentro_da_escola"] = partes["dentro_da_escola"] + ruido
    pct_corr = {k: round(v / total * 100, 1) for k, v in corr.items()}
    return {"taxa_geral": round(p_geral * 100, 1), "pct": pct, "pct_corrigido": pct_corr,
            "soma_conferida": round(sum(partes.values()) / total * 100, 1)}


def decompor_proficiencia(e: pd.DataFrame) -> dict:
    e = e[(e.alunos_com_resultado >= 2) & e.desvio_proficiencia.notna()]
    n, m, sd = e.alunos_com_resultado, e.media_proficiencia, e.desvio_proficiencia
    N = n.sum()
    geral = (n * m).sum() / N

    def media_grupo(col):
        g = e.assign(nm=n * m).groupby(col)
        return g.nm.transform("sum") / g.alunos_com_resultado.transform("sum")

    m_uf, m_mun = media_grupo("sigla_uf"), media_grupo("id_municipio")
    partes = {
        "entre_ufs": (n * (m_uf - geral) ** 2).sum(),
        "entre_municipios": (n * (m_mun - m_uf) ** 2).sum(),
        "entre_escolas": (n * (m - m_mun) ** 2).sum(),
        "dentro_da_escola": ((n - 1) * sd ** 2).sum(),
    }
    total = sum(partes.values())
    ruido = (sd ** 2).sum()
    corr = dict(partes)
    corr["entre_escolas"] = max(partes["entre_escolas"] - ruido, 0)
    corr["dentro_da_escola"] = partes["dentro_da_escola"] + ruido
    return {"media_geral": round(geral, 1),
            "pct": {k: round(v / total * 100, 1) for k, v in partes.items()},
            "pct_corrigido": {k: round(v / total * 100, 1) for k, v in corr.items()}}


def grafico_decomposicao(d: dict) -> None:
    nomes = {"entre_ufs": "Entre estados", "entre_municipios": "Entre municípios\nda mesma UF",
             "entre_escolas": "Entre escolas\ndo mesmo município", "dentro_da_escola": "Entre alunos\nda mesma escola"}
    v = d["pct_corrigido"]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.barh(list(nomes.values())[::-1], [v[k] for k in nomes][::-1],
            color=[estilo.LARANJA] + [estilo.AZUL] * 3, height=0.6)
    for i, k in enumerate(list(nomes)[::-1]):
        ax.text(v[k] + 1, i, f"{v[k]:.0f}%", va="center", fontsize=10)
    ax.set_xlim(0, 100)
    ax.set_xlabel("% da variação da proficiência entre alunos (2024, ruído amostral descontado)")
    ax.set_title("Onde está a diferença entre as crianças")
    estilo.salvar(fig, IMAGES / "escolas_decomposicao.png")


def dispersao(e: pd.DataFrame) -> pd.DataFrame:
    ok = e[e.alunos_com_resultado >= MIN_ALUNOS]
    g = ok.groupby("id_municipio").p
    disp = pd.DataFrame({"escolas": g.size(), "p10": g.quantile(0.1) * 100,
                         "p90": g.quantile(0.9) * 100})
    disp = disp[disp.escolas >= 3]
    disp["amplitude"] = disp.p90 - disp.p10
    return disp


def grafico_dispersao(disp: pd.DataFrame, acaso: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 90, 46)
    ax.hist(disp.amplitude, bins=bins, color=estilo.AZUL, density=True, label="Observado")
    ax.hist(acaso, bins=bins, histtype="step", linewidth=2, color=estilo.TINTA_2,
            density=True, label="Só o acaso (simulado)")
    ax.legend(loc="upper right")
    ax.set_yticks([])
    ax.set_xlabel("Diferença entre a escola do percentil 90 e a do percentil 10 (p.p.)")
    ax.set_title("Escolas do mesmo município: diferença real além do acaso")
    estilo.salvar(fig, IMAGES / "escolas_dispersao_municipio.png")


def municipios_que_bateram() -> set:
    fato = carregar("fato_alfabetizacao")
    return set(fato[(fato.ano == ANO) & (fato.meta_atingida == 1)].id_municipio)


def escolas_criticas(e: pd.DataFrame, bateu: set) -> pd.DataFrame:
    mun = e[(e.rede == REDE_MUNICIPAL) & (e.alunos_com_resultado >= MIN_ALUNOS)]
    return mun[mun.id_municipio.isin(bateu) & (mun.p * 100 < CORTE_CRITICO)]


def criticas_escondidas(e: pd.DataFrame) -> dict:
    bateu = municipios_que_bateram()
    crit = escolas_criticas(e, bateu)
    return {
        "municipios_que_bateram_a_meta": len(bateu),
        "escolas_criticas_nesses_municipios": int(len(crit)),
        "municipios_com_ao_menos_uma": int(crit.id_municipio.nunique()),
        "pct_municipios_que_bateram_com_escola_critica":
            round(crit.id_municipio.nunique() / max(len(bateu), 1) * 100, 1),
        "alunos_nessas_escolas": int(crit.alunos_registrados.sum()),
    }


def rede_e_porte(e: pd.DataFrame) -> dict:
    def taxa(g):
        return round(g.soma_peso_alfabetizados.sum() / g.soma_peso.sum() * 100, 1)
    por_rede = {REDES[r]: taxa(g) for r, g in e.groupby("rede") if r in (2, 3)}
    faixas = pd.cut(e.alunos_registrados, [0, 10, 25, 50, 100, 1000],
                    labels=["até 10", "11–25", "26–50", "51–100", "mais de 100"])
    por_porte = {str(k): taxa(g) for k, g in e.groupby(faixas, observed=True)}
    alunos_porte = {str(k): int(g.alunos_registrados.sum())
                    for k, g in e.groupby(faixas, observed=True)}
    return {"taxa_por_rede": por_rede, "taxa_por_porte": por_porte, "alunos_por_porte": alunos_porte}


def grafico_porte(rp: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    k, v = list(rp["taxa_por_porte"]), list(rp["taxa_por_porte"].values())
    ax.bar(k, v, color=estilo.AZUL, width=0.6)
    for i, x in enumerate(v):
        ax.text(i, x + 1, f"{x:.0f}%", ha="center", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Alunos do 2º ano registrados na escola")
    ax.set_ylabel("% alfabetizados (2024)")
    ax.set_title("Taxa de alfabetização por porte da escola")
    estilo.salvar(fig, IMAGES / "escolas_porte.png")


def simular_acaso(e: pd.DataFrame, repeticoes: int = 20) -> dict:
    """Recalcula dispersao e escolas criticas num mundo sem diferenca real entre escolas."""
    rng = np.random.default_rng(42)
    g = e.assign(wp=e.soma_peso * e.p).groupby("id_municipio")
    p_mun = (g.wp.transform("sum") / g.soma_peso.transform("sum")).clip(0, 1)
    n = e.alunos_com_resultado.astype(int)
    amplitudes, criticas = [], []
    for _ in range(repeticoes):
        sim = e.copy()
        sim["p"] = np.where(n > 0, rng.binomial(n, p_mun) / np.maximum(n, 1), 0)
        amplitudes.append(dispersao(sim).amplitude)
        criticas.append(criticas_escondidas(sim)["escolas_criticas_nesses_municipios"])
    return {"amplitudes": pd.concat(amplitudes), "criticas_media": float(np.mean(criticas))}


def main() -> None:
    e = carregar_escolas()
    dec = decompor_proficiencia(e)
    dec_bin = decompor(e)
    grafico_decomposicao(dec)
    disp = dispersao(e)
    acaso = simular_acaso(e)
    grafico_dispersao(disp, acaso["amplitudes"])
    crit = criticas_escondidas(e)
    crit["esperadas_so_por_acaso"] = round(acaso["criticas_media"])
    crit["acima_do_acaso"] = crit["escolas_criticas_nesses_municipios"] - crit["esperadas_so_por_acaso"]
    rp = rede_e_porte(e)
    grafico_porte(rp)

    resumo = {
        "ano": ANO, "escolas": int(len(e)), "alunos": int(e.alunos_registrados.sum()),
        "decomposicao_proficiencia": dec,
        "decomposicao_alfabetizado_0_1": dec_bin,
        "dispersao_municipio": {
            "municipios_avaliados": int(len(disp)),
            "amplitude_mediana_pp": round(float(disp.amplitude.median()), 1),
            "amplitude_mediana_so_acaso_pp": round(float(acaso["amplitudes"].median()), 1),
            "amplitude_real_estimada_pp": round(float(np.sqrt(max(
                disp.amplitude.median() ** 2 - acaso["amplitudes"].median() ** 2, 0))), 1),
            "pct_com_amplitude_acima_de_30pp": round(float((disp.amplitude > 30).mean() * 100), 1),
        },
        "escolas_criticas_escondidas": crit,
        "rede_e_porte": rp,
    }
    # Produto para a secretaria: o codigo da escola e ficticio, mas a rede
    # municipal conhece as proprias escolas e pode localizar as criticas.
    tab = escolas_criticas(e, municipios_que_bateram())
    nomes = pd.read_csv(RAW_IBGE / "ibge_localidades.csv", dtype={"id_municipio": str})
    nomes = nomes.set_index(nomes.id_municipio.str.zfill(7)).nome_municipio
    lista = (tab.groupby("id_municipio")
             .agg(escolas_criticas=("id_escola", "size"),
                  alunos_nessas_escolas=("alunos_registrados", "sum"),
                  menor_taxa_escola=("p", lambda x: round(x.min() * 100, 1)))
             .assign(nome_municipio=lambda d: d.index.map(nomes),
                     sigla_uf=lambda d: d.index.str[:2].map(CODIGO_UF))
             .sort_values(["escolas_criticas", "alunos_nessas_escolas"], ascending=False))
    lista[["nome_municipio", "sigla_uf", "escolas_criticas", "alunos_nessas_escolas",
           "menor_taxa_escola"]].to_csv(REPORTS / "municipios_com_escolas_criticas.csv")

    (REPORTS / "escolas_resumo.json").write_text(json.dumps(resumo, indent=2, ensure_ascii=False))
    print(json.dumps(resumo, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
