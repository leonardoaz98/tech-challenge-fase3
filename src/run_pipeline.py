"""
Roda o pipeline completo, da Gold ao ranking de 2025, e registra a execucao.

    python -m src.run_pipeline             # tudo, com busca de hiperparametros (~3 min)
    python -m src.run_pipeline --rapido    # sem busca de hiperparametros

Rodar a partir da raiz do repositorio.

O QUE FICA GRAVADO
------------------
- logs/pipeline_<data>.log   tudo o que as etapas imprimiram, com horario
                             (fora do git: e o detalhe de cada rodada)
- reports/execucoes.jsonl    uma linha por execucao, acumulando: data,
                             commit, versoes, hash dos dados de entrada,
                             tempo por etapa, metricas principais e custo.
                             Nada e sobrescrito: e o historico do projeto.
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from src.logger import get_logger

RAIZ = Path(__file__).resolve().parents[1]
REGISTRO = RAIZ / "reports" / "execucoes.jsonl"
LOGS = RAIZ / "logs"

ETAPAS = [
    ("Gold local (star schema da Fase 2)", "src.preprocessing.gold", []),
    ("Base analitica (treino e previsao)", "src.preprocessing.base_analitica", []),
    ("Analise exploratoria", "src.visualization.eda", []),
    ("Treino e validacao", "src.modeling.treinar", []),
    ("Interpretabilidade", "src.evaluation.interpretar", []),
    ("Aplicacao: ranking 2025 e perfis", "src.evaluation.aplicacao", []),
    ("Validacao real: previsao 2025 x resultado 2025", "src.evaluation.validacao_2025", []),
    ("Monitoramento de deriva (treino x previsao)", "src.evaluation.drift", []),
    ("Escolas: microdado por aluno na Gold", "src.preprocessing.escolas", []),
    ("Desigualdade entre escolas", "src.visualization.desigualdade", []),
    ("Modelo por aluno", "src.modeling.modelo_aluno", []),
]

# Metricas acompanhadas a cada execucao: nome -> (arquivo em reports/, caminho no JSON)
METRICAS = {
    "auc_teste_municipal": ("resultado_modelo.json", ["teste_limiar_escolhido", "auc"]),
    "recall_teste_municipal": ("resultado_modelo.json", ["teste_limiar_escolhido", "recall_risco"]),
    "municipios_criticos_2025": ("aplicacao.json", ["previsao_2025", "faixas", "critico"]),
    "pct_esforco_acima_do_treino": ("aplicacao.json", ["previsao_2025", "pct_esforco_acima_do_treino"]),
    "auc_teste_aluno": ("resultado_modelo_aluno.json", ["auc_teste"]),
    "erro_medio_microdado_pp": ("validacao_microdados.json", ["erro_medio_pp"]),
    "variaveis_em_alerta_de_deriva": ("drift.json", ["resumo", "qtd_alerta"]),
    "taxa_falha_real_2025": ("validacao_2025.json", ["taxa_falha_real"]),
    "auc_real_2025_modelo": ("validacao_2025.json", ["auc_modelo"]),
    "auc_real_2025_regra_simples": ("validacao_2025.json", ["auc_regra_simples_esforco"]),
}

BIBLIOTECAS = ["pandas", "numpy", "pyarrow", "scikit-learn", "matplotlib", "shap"]


def ler_metrica(arquivo: str, chaves: list[str]):
    try:
        valor = json.loads((RAIZ / "reports" / arquivo).read_text(encoding="utf-8"))
        for chave in chaves:
            valor = valor[chave]
        return round(valor, 4) if isinstance(valor, float) else valor
    except (FileNotFoundError, KeyError, TypeError):
        return None


def git(*args: str) -> str | None:
    try:
        r = subprocess.run(["git", *args], cwd=RAIZ, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None
    except FileNotFoundError:
        return None


def hash_entradas() -> dict:
    """sha256 (12 primeiros caracteres) de cada dado bruto que alimenta o pipeline."""
    arquivos = sorted(a for a in (RAIZ / "data" / "raw").glob("*/*")
                      if a.suffix in {".csv", ".xlsx"})
    return {str(a.relative_to(RAIZ)): hashlib.sha256(a.read_bytes()).hexdigest()[:12]
            for a in arquivos}


def tamanho_mb(pasta: Path) -> float:
    return round(sum(a.stat().st_size for a in pasta.rglob("*") if a.is_file()) / 2**20, 1)


def rodar_etapa(nome: str, modulo: str, extra: list, env: dict, log) -> bool:
    origem = modulo.rsplit(".", 1)[-1]
    proc = subprocess.Popen([sys.executable, "-m", modulo, *extra], env=env, cwd=RAIZ,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8")
    for linha in proc.stdout:
        if linha.strip():
            log.info(f"[{origem}] {linha.rstrip()}")
    return proc.wait() == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rapido", action="store_true")
    args = parser.parse_args()

    inicio = datetime.now()
    log = get_logger("pipeline", LOGS / f"pipeline_{inicio:%Y%m%d_%H%M%S}.log")
    env = dict(os.environ, PYTHONWARNINGS="ignore", PYTHONUNBUFFERED="1")
    etapas = list(ETAPAS)
    if args.rapido:
        etapas = [(n, m, ["--sem-busca"] if m.endswith("treinar") else a)
                  for n, m, a in etapas]

    registro = {
        "inicio": inicio.isoformat(timespec="seconds"),
        "modo": "rapido" if args.rapido else "completo",
        "commit": git("rev-parse", "--short", "HEAD"),
        "alteracoes_nao_commitadas": bool(git("status", "--porcelain", "--untracked-files=no")),
        "python": platform.python_version(),
        "bibliotecas": {b: version(b) for b in BIBLIOTECAS},
        "entradas_sha256": hash_entradas(),
        "etapas": [],
    }

    status = "ok"
    t0 = time.time()
    for i, (nome, modulo, extra) in enumerate(etapas, 1):
        log.info(f"[{i}/{len(etapas)}] {nome}")
        t = time.time()
        ok = rodar_etapa(nome, modulo, extra, env, log)
        registro["etapas"].append({"etapa": modulo, "segundos": round(time.time() - t, 1),
                                   "status": "ok" if ok else "falhou"})
        if not ok:
            status = f"falhou em {modulo}"
            log.error(f"Falhou em: {nome}")
            break

    total = round(time.time() - t0, 1)
    registro.update({
        "status": status,
        "metricas": {k: ler_metrica(*v) for k, v in METRICAS.items()} if status == "ok" else {},
        "recursos": {
            "segundos_total": total,
            "dados_mb": tamanho_mb(RAIZ / "data"),
            "onde_rodou": "local",
            "custo_estimado_brl": 0.0,
        },
    })
    with REGISTRO.open("a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")

    if status != "ok":
        log.error(f"Execucao registrada em {REGISTRO.relative_to(RAIZ)} com status: {status}")
        return 1
    log.info(f"Pipeline concluido em {total:.0f}s. Resultados em reports/ e images/. "
             f"Execucao registrada em {REGISTRO.relative_to(RAIZ)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
