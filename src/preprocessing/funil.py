"""
Funil de dados: quantas linhas entram e saem de cada etapa, e por que.

POR QUE EXISTE
--------------
Na Fase 2 as metas eram deduplicadas sem criterio documentado, e linhas
sumiam sem ninguem saber quantas nem por que. Aqui todo filtro que
descarta linhas registra a contagem e o motivo em reports/funil.json.

Cada etapa que filtra chama `Funil.passo` depois do filtro; `registrar`
grava a secao no arquivo (uma secao por base: treino, previsao, aluno).
"""
import json

from src.config import REPORTS

ARQUIVO = REPORTS / "funil.json"


class Funil:
    def __init__(self, unidade: str):
        self.unidade = unidade
        self.passos: list[dict] = []

    def passo(self, etapa: str, restam: int, motivo: str = "") -> None:
        anterior = self.passos[-1]["restam"] if self.passos else None
        self.passos.append({
            "etapa": etapa,
            "restam": int(restam),
            "saem": None if anterior is None else int(anterior - restam),
            "motivo": motivo,
        })

    def como_dict(self) -> dict:
        return {"unidade": self.unidade, "passos": self.passos}


def registrar(secao: str, funil: Funil) -> None:
    dados = json.loads(ARQUIVO.read_text(encoding="utf-8")) if ARQUIVO.exists() else {}
    dados[secao] = funil.como_dict()
    dados = dict(sorted(dados.items()))
    ARQUIVO.write_text(json.dumps(dados, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for p in funil.passos:
        saem = "" if p["saem"] is None else f"  (-{p['saem']}: {p['motivo']})" if p["saem"] else ""
        print(f"    {p['etapa']:45} {p['restam']:>9}{saem}")
