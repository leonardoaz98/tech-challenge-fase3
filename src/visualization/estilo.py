"""Estilo unico dos graficos: superficie, tinta e paleta."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
GRADE = "#e4e3df"

# Paleta categorica em ordem fixa
AZUL, LARANJA, AQUA, AMARELO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MAGENTA, VERDE, VIOLETA, VERMELHO = "#e87ba4", "#008300", "#4a3aa7", "#e34948"
CATEGORICA = [AZUL, LARANJA, AQUA, AMARELO, MAGENTA, VERDE, VIOLETA, VERMELHO]

# Classes do alvo: cor segue a entidade em todos os graficos
COR_ATINGIU = AZUL
COR_RISCO = LARANJA

# Divergente azul <-> vermelho, meio cinza
DIVERGENTE = ["#2a78d6", "#86b6ef", "#f0efec", "#f0a1a0", "#e34948"]

plt.rcParams.update({
    "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE,
    "savefig.facecolor": SUPERFICIE, "figure.dpi": 110, "savefig.dpi": 150,
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelcolor": TINTA_2,
    "text.color": TINTA, "xtick.color": TINTA_2, "ytick.color": TINTA_2,
    "axes.edgecolor": GRADE, "axes.grid": True, "grid.color": GRADE,
    "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})


def salvar(fig, caminho) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, bbox_inches="tight")
    plt.close(fig)
