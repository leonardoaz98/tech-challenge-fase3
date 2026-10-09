"""
Roda o pipeline completo, da Gold ao ranking de 2025.

    python -m src.run_pipeline             # tudo, com busca de hiperparametros (~2 min)
    python -m src.run_pipeline --rapido    # sem busca de hiperparametros
    python -m src.run_pipeline --s3        # le a Gold do S3 da Fase 2 (exige AWS)

Rodar a partir da raiz do repositorio.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

ETAPAS = [
    ("Gold local (star schema da Fase 2)", "src.preprocessing.gold", []),
    ("Base analitica (treino e previsao)", "src.preprocessing.base_analitica", []),
    ("Analise exploratoria", "src.visualization.eda", []),
    ("Treino e validacao", "src.modeling.treinar", []),
    ("Interpretabilidade", "src.evaluation.interpretar", []),
    ("Aplicacao: ranking 2025 e perfis", "src.evaluation.aplicacao", []),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rapido", action="store_true")
    parser.add_argument("--s3", action="store_true")
    args = parser.parse_args()

    env = dict(os.environ, PYTHONWARNINGS="ignore")
    etapas = list(ETAPAS)
    if args.s3:
        env["GOLD_FONTE"] = "s3"
        etapas = etapas[1:]  # a Gold ja existe no S3
    if args.rapido:
        etapas = [(n, m, ["--sem-busca"] if m.endswith("treinar") else a)
                  for n, m, a in etapas]

    inicio = time.time()
    for i, (nome, modulo, extra) in enumerate(etapas, 1):
        print(f"\n[{i}/{len(etapas)}] {nome}")
        r = subprocess.run([sys.executable, "-m", modulo, *extra], env=env, cwd=RAIZ)
        if r.returncode != 0:
            print(f"\nFalhou em: {nome}")
            return r.returncode
    print(f"\nPipeline concluido em {time.time() - inicio:.0f}s. "
          "Resultados em reports/ e images/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
