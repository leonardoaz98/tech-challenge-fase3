"""
Logger padronizado do projeto.

Mesmo formato da Fase 2 (data | nivel | origem | mensagem), com uma
diferenca: aceita um arquivo, para o log da execucao ficar gravado em
logs/ e nao so no terminal.
"""
import logging
import sys
from pathlib import Path

FORMATO = "%(asctime)s | %(levelname)-7s | %(name)-14s | %(message)s"
DATA_FORMATO = "%Y-%m-%d %H:%M:%S"


def get_logger(nome: str, arquivo: Path | None = None,
               nivel: int = logging.INFO) -> logging.Logger:
    """Devolve um logger configurado, sem duplicar handlers."""
    logger = logging.getLogger(nome)
    if logger.handlers:
        return logger
    logger.setLevel(nivel)
    formato = logging.Formatter(FORMATO, datefmt=DATA_FORMATO)
    saidas = [logging.StreamHandler(sys.stdout)]
    if arquivo is not None:
        arquivo.parent.mkdir(parents=True, exist_ok=True)
        saidas.append(logging.FileHandler(arquivo, encoding="utf-8"))
    for saida in saidas:
        saida.setFormatter(formato)
        logger.addHandler(saida)
    logger.propagate = False
    return logger
