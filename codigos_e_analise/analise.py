"""
Projeto 2 - SO - Custo de gerar imagens com IA local
Script de ANALISE E GRAFICOS - serve para os 5 integrantes

Le o CSV do benchmark.py, calcula a media das 3 repeticoes de cada
pedido e salva dois graficos: tempo por imagem e pico de memoria.

Cada integrante muda APENAS a linha INTEGRANTE, igual a do benchmark.

Como rodar:
  python analise.py
"""

import matplotlib.pyplot as plt
import pandas as pd

# ======================= MUDE SO ESTA LINHA =======================
INTEGRANTE = 5          # o mesmo numero que voce usou no benchmark.py
# ==================================================================

EIXOS = {
    1: "Quantidade de imagens de uma vez",
    2: "Tamanho da imagem (pixels de lado)",
    3: "Imagem pedida",
    4: "Pedido (imagem + tamanho)",
    5: "Pedido (imagem + tamanho + quantidade)",
}

METRICAS = [
    ("tempo_por_imagem_s", "Tempo por imagem", "segundos"),
    ("pico_vram_mb", "Pico de memoria da GPU", "MB"),
]


def main():
    arquivo = f"resultados_integrante{INTEGRANTE}.csv"
    df = pd.read_csv(arquivo)

    # mantem a ordem em que os pedidos foram medidos
    ordem = list(dict.fromkeys(df["rotulo"]))

    for coluna, titulo, unidade in METRICAS:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")
        medias = df.groupby("rotulo")[coluna].mean().reindex(ordem).round(2)
        desvios = df.groupby("rotulo")[coluna].std().reindex(ordem).round(2)

        print(f"\n{titulo} ({unidade}) por pedido:")
        print(medias.to_string())

        serie = medias.dropna()
        if serie.empty:
            print("Sem dados - grafico nao gerado.")
            continue

        rotulos = [str(r) for r in serie.index]
        numerico = all(
            isinstance(r, (int, float)) and not isinstance(r, bool)
            for r in serie.index
        )

        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        if numerico:
            ax.errorbar(
                serie.index, serie.values,
                yerr=desvios.reindex(serie.index).values,
                marker="o", capsize=4, linewidth=2,
            )
            ax.set_xticks(serie.index)
        else:
            ax.bar(
                rotulos, serie.values,
                yerr=desvios.reindex(serie.index).values,
                capsize=4, color="#4C72B0",
            )

        ax.set_title(f"{titulo} - Integrante {INTEGRANTE}")
        ax.set_xlabel(EIXOS[INTEGRANTE])
        ax.set_ylabel(unidade)
        ax.grid(True, axis="y", alpha=0.3)

        for x, y in zip(rotulos if not numerico else serie.index, serie.values):
            ax.annotate(f"{y:g}", (x, y), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9)

        fig.tight_layout()
        nome = f"grafico_int{INTEGRANTE}_{coluna}.png"
        fig.savefig(nome, dpi=150)
        plt.close(fig)
        print(f"Grafico salvo: {nome}")

    falhas = df[df["pico_vram_mb"].isna()]["rotulo"].unique()
    if len(falhas) > 0:
        print(f"\nPedidos que estouraram a memoria da GPU: {list(falhas)}")


if __name__ == "__main__":
    main()
