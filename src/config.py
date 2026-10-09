"""
Configuracao central do projeto.

Caminhos, anos de referencia e parametros compartilhados por todas as
etapas. Mudar um ano ou uma semente aqui muda o pipeline inteiro.
"""
import os
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

# --- Pastas ---
RAW_INEP = RAIZ / "data" / "raw" / "inep"
RAW_IBGE = RAIZ / "data" / "raw" / "ibge"
GOLD = RAIZ / "data" / "gold"
PROCESSED = RAIZ / "data" / "processed"
REPORTS = RAIZ / "reports"
IMAGES = RAIZ / "images"
MODELS = RAIZ / "models"

# --- Origem da Gold ---
# "local": snapshot versionado em data/gold (padrao, roda em qualquer maquina)
# "s3":    le direto do data lake da Fase 2 (exige credencial AWS)
GOLD_FONTE = os.getenv("GOLD_FONTE", "local")
S3_GOLD = f"s3://{os.getenv('S3_BUCKET', 'tc2-fiap-datalake')}/gold"

# --- Desenho temporal ---
# Treino:    features de 2023 -> resultado de 2024 (alvo conhecido)
# Previsao:  features de 2024 -> meta de 2025 (alvo desconhecido)
ANO_ENTRADA_TREINO = 2023
ANO_ALVO_TREINO = 2024
ANO_ENTRADA_PREVISAO = 2024
ANO_ALVO_PREVISAO = 2025

REDE_MUNICIPAL = 3

# --- Modelagem ---
ALVO = "risco"            # 1 = NAO atingiu a meta (classe de interesse)
SEMENTE = 42
PROPORCAO_TESTE = 0.25
PARTICOES_CV = 5
RECALL_MINIMO = 0.80      # limiar escolhido para detectar >= 80% dos que falham
