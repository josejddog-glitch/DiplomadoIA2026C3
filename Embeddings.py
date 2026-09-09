"""Explorador interactivo de embeddings para cualquier lista de textos.

Escribe tantos textos como quieras, uno por línea, y termina con ``salir``.
Después puedes elegir PCA o t-SNE para visualizar la cercanía semántica.
"""

from __future__ import annotations

import textwrap
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from embeddings_provider import build_embeddings


MODEL_NAME = "openai/text-embedding-3-small"
EXIT_COMMANDS = {"salir", "exit", "quit", "fin"}


def leer_textos() -> list[str]:
    """Lee una lista libre de textos desde la terminal."""
    print("Explorador general de embeddings")
    print("Ingresa cualquier palabra, frase o párrafo, uno por línea.")
    print("Escribe 'salir' cuando hayas terminado.\n")

    textos: list[str] = []
    while True:
        texto = input(f"Texto {len(textos) + 1}: ").strip()
        if texto.lower() in EXIT_COMMANDS:
            break
        if texto:
            textos.append(texto)
        else:
            print("El texto no puede estar vacío.")
    return textos


def reducir_embeddings(embeddings: np.ndarray, metodo: str) -> np.ndarray:
    """Reduce embeddings a dos dimensiones con PCA o t-SNE."""
    if len(embeddings) < 2:
        raise ValueError("Se necesitan al menos dos textos para visualizar.")

    metodo = metodo.lower()
    if metodo == "pca":
        return PCA(n_components=2).fit_transform(embeddings)

    if len(embeddings) < 4:
        raise ValueError("t-SNE necesita al menos cuatro textos; usa PCA.")

    perplexity = min(30.0, max(2.0, float(len(embeddings) - 1) / 3))
    try:
        tsne = TSNE(
            n_components=2, perplexity=perplexity, learning_rate="auto",
            max_iter=1000, init="pca", random_state=42,
        )
    except TypeError:  # Compatibilidad con scikit-learn anterior.
        tsne = TSNE(
            n_components=2, perplexity=perplexity, learning_rate="auto",
            n_iter=1000, init="pca", random_state=42,
        )
    return tsne.fit_transform(embeddings)


def _etiqueta(texto: str, limite: int = 42) -> str:
    """Acorta y parte etiquetas largas para que no tapen el gráfico."""
    texto = " ".join(texto.split())
    if len(texto) > limite:
        texto = texto[: limite - 1].rstrip() + "…"
    return "\n".join(textwrap.wrap(texto, width=22))


def graficar_embeddings(
    textos: Sequence[str], coords: np.ndarray, metodo: str
) -> None:
    """Genera una visualización limpia, etiquetada y apta para presentaciones."""
    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(13, 8), dpi=120)
    fig.patch.set_facecolor("#F5F7FB")
    ax.set_facecolor("#FFFFFF")

    colores = plt.get_cmap("viridis")(np.linspace(0.12, 0.88, len(textos)))
    ax.scatter(
        coords[:, 0], coords[:, 1], s=360, c="#172033", alpha=0.10,
        linewidths=0, zorder=1,
    )
    ax.scatter(
        coords[:, 0], coords[:, 1], s=210, c=colores,
        edgecolors="white", linewidths=2, alpha=0.96, zorder=2,
    )

    for indice, (texto, (x, y), color) in enumerate(zip(textos, coords, colores), 1):
        ax.annotate(
            f"{indice}. {_etiqueta(texto)}", xy=(x, y), xytext=(9, 10),
            textcoords="offset points", fontsize=9, color="#24324A",
            bbox={
                "boxstyle": "round,pad=0.42,rounding_size=0.18",
                "facecolor": "white", "edgecolor": color,
                "linewidth": 1.1, "alpha": 0.94,
            }, zorder=3,
        )

    ax.set_title(
        "Mapa semántico de tus textos", loc="left", fontsize=21,
        fontweight="bold", color="#172033", pad=24,
    )
    ax.text(
        0, 1.02, f"Cada punto representa un texto · reducción: {metodo}",
        transform=ax.transAxes, fontsize=11, color="#64748B", va="bottom",
    )
    ax.set_xlabel("Dimensión semántica 1", color="#52627A", labelpad=10)
    ax.set_ylabel("Dimensión semántica 2", color="#52627A", labelpad=10)
    ax.grid(True, color="#DCE3EE", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CBD5E1")
    ax.tick_params(colors="#64748B")

    leyenda = Line2D(
        [0], [0], marker="o", color="w", label=f"{len(textos)} textos ingresados",
        markerfacecolor="#4F46E5", markersize=10,
    )
    ax.legend(
        handles=[leyenda], loc="upper right", frameon=True,
        facecolor="white", edgecolor="#E2E8F0",
    )
    fig.text(
        0.01, 0.015,
        "Puntos cercanos indican mayor similitud semántica aproximada.",
        fontsize=9, color="#64748B",
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.12)
    plt.show()


def main() -> None:
    textos = leer_textos()
    if len(textos) < 2:
        print("\nIngresa al menos dos textos para generar el mapa.")
        return

    print(f"\nGenerando embeddings con: {MODEL_NAME}")
    model = build_embeddings(MODEL_NAME)
    embeddings = np.asarray(model.embed_documents(textos), dtype=np.float32)
    print(f"Embeddings generados: {len(textos)} textos × {embeddings.shape[1]} dimensiones.")

    print("\nMétodo de visualización:")
    print("1. PCA  - rápido y estable (recomendado)")
    print("2. t-SNE - explora agrupaciones locales")
    opcion = input("Selecciona [1/2, Enter=PCA]: ").strip()
    metodo = "t-SNE" if opcion == "2" else "PCA"

    try:
        coords = reducir_embeddings(embeddings, metodo)
    except ValueError as error:
        print(f"\n{error}")
        return

    print(f"Mostrando visualización con {metodo}.")
    graficar_embeddings(textos, coords, metodo)


if __name__ == "__main__":
    main()
