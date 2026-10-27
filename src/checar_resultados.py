"""
Confere se os resultados da execucao batem com a referencia versionada.

POR QUE EXISTE
--------------
"O README roda" so vale se alguem rodar. Este script e o ultimo passo do
GitHub Actions: depois de o pipeline rodar do zero numa maquina limpa,
compara as metricas principais com reports/referencia_metricas.json.
Cada metrica tem uma tolerancia, porque maquinas diferentes podem mudar
a ultima casa decimal sem mudar a conclusao.

    python -m src.checar_resultados            # compara (sai com erro se divergir)
    python -m src.checar_resultados --gravar   # grava a referencia a partir de reports/

Gravar uma referencia nova e uma decisao: deve vir num commit proprio,
explicando por que o resultado mudou.
"""
import argparse
import json
import sys

from src.config import REPORTS
from src.run_pipeline import METRICAS, ler_metrica

REFERENCIA = REPORTS / "referencia_metricas.json"
TOLERANCIAS = {
    "auc_teste_municipal": 0.01,
    "recall_teste_municipal": 0.02,
    "municipios_criticos_2025": 15,
    "pct_esforco_acima_do_treino": 0.01,
    "auc_teste_aluno": 0.01,
    "erro_medio_microdado_pp": 0.01,
    "variaveis_em_alerta_de_deriva": 0,
    "taxa_falha_real_2025": 0.001,
    "auc_real_2025_modelo": 0.01,
    "auc_real_2025_regra_simples": 0.001,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gravar", action="store_true")
    args = parser.parse_args()
    atuais = {nome: ler_metrica(*origem) for nome, origem in METRICAS.items()}

    if args.gravar:
        faltando = [n for n, v in atuais.items() if v is None]
        if faltando:
            sys.exit(f"Rode o pipeline antes: sem valor para {', '.join(faltando)}")
        ref = {n: {"valor": v, "tolerancia": TOLERANCIAS[n]} for n, v in atuais.items()}
        REFERENCIA.write_text(json.dumps(ref, indent=2) + "\n", encoding="utf-8")
        print(f"  referencia gravada em {REFERENCIA.name}")
        return 0

    ref = json.loads(REFERENCIA.read_text(encoding="utf-8"))
    divergentes = 0
    print(f"  {'metrica':32} {'referencia':>11} {'atual':>9} {'tolerancia':>11}")
    for nome, r in ref.items():
        atual = atuais.get(nome)
        ok = atual is not None and abs(atual - r["valor"]) <= r["tolerancia"]
        divergentes += not ok
        print(f"  {nome:32} {r['valor']:>11} {str(atual):>9} {r['tolerancia']:>11}  "
              f"{'OK' if ok else 'DIVERGE'}")
    if divergentes:
        print(f"\n  {divergentes} metrica(s) fora da tolerancia.")
        return 1
    print("\n  Resultados conferem com a referencia.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
