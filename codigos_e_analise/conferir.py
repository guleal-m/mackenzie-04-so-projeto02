"""
Projeto 2 - SO
Script de CONFERENCIA dos dois testes novos

Teste 1 - ordem invertida: compara resultados_integranteN.csv com
          resultados_integranteN_invertido.csv e diz se a variacao do
          tempo veio do pedido ou do aquecimento da placa.

Teste 2 - memoria reservada: mostra, em todos os CSVs encontrados,
          quanto o alocador do PyTorch realmente reservou na placa e
          onde isso encosta nos 6144 MB.

Como rodar:
  python conferir.py
"""

import glob
import os

import pandas as pd

LIMITE_PLACA = 6144.0     # MB da RTX 3060
INTEGRANTE = 3            # qual frente comparar nas duas ordens


def media_por_pedido(arquivo):
    df = pd.read_csv(arquivo)
    for c in ["tempo_por_imagem_s", "pico_vram_mb", "pico_reservada_mb", "pico_temp_c"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    ordem = list(dict.fromkeys(df["rotulo"]))
    g = df.groupby("rotulo", sort=False).mean(numeric_only=True).reindex(ordem)
    return g


def teste_ordem():
    a = f"resultados_integrante{INTEGRANTE}.csv"
    b = f"resultados_integrante{INTEGRANTE}_invertido.csv"
    if not (os.path.exists(a) and os.path.exists(b)):
        print(f"Teste 1: faltam arquivos ({a} e/ou {b}). Pulando.\n")
        return

    ga, gb = media_por_pedido(a), media_por_pedido(b)
    print("=" * 62)
    print(f"TESTE 1 - frente {INTEGRANTE} nas duas ordens")
    print("=" * 62)
    print(f"{'pedido':<12}{'normal':>10}{'invertido':>12}{'diferenca':>12}")
    for r in ga.index:
        if r not in gb.index:
            continue
        ta, tb = ga.loc[r, "tempo_por_imagem_s"], gb.loc[r, "tempo_por_imagem_s"]
        print(f"{str(r):<12}{ta:>9.2f}s{tb:>11.2f}s{(tb - ta):>+11.2f}s")

    # na ordem normal o ultimo roda com a placa quente; na invertida, o primeiro
    primeiro, ultimo = ga.index[0], ga.index[-1]
    sobe_normal = ga.loc[ultimo, "tempo_por_imagem_s"] - ga.loc[primeiro, "tempo_por_imagem_s"]
    sobe_invert = gb.loc[primeiro, "tempo_por_imagem_s"] - gb.loc[ultimo, "tempo_por_imagem_s"]

    print(f"\nNa ordem normal, '{ultimo}' custou {sobe_normal:+.2f}s a mais que '{primeiro}'.")
    print(f"Na ordem invertida, '{primeiro}' custou {sobe_invert:+.2f}s a mais que '{ultimo}'.")
    print()
    if sobe_normal > 0 and sobe_invert > 0:
        print("LEITURA: nas duas ordens quem rodou POR ULTIMO ficou mais lento.")
        print("O efeito segue a posicao na sequencia, nao o pedido.")
        print("=> a variacao do tempo e aquecimento da placa, confirmado.")
    elif sobe_normal > 0 and sobe_invert < 0:
        print("LEITURA: o mesmo pedido ficou lento nas duas ordens.")
        print("=> a variacao vem do proprio pedido, nao do aquecimento.")
    else:
        print("LEITURA: o padrao nao ficou claro. Vale repetir com a placa fria.")
    print()


def teste_reservada():
    arquivos = sorted(glob.glob("resultados_integrante*.csv"))
    linhas = []
    for f in arquivos:
        df = pd.read_csv(f)
        if "pico_reservada_mb" not in df.columns:
            continue
        for c in ["pico_vram_mb", "pico_reservada_mb", "tempo_total_s", "resolucao", "lote"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["mp"] = (df.resolucao ** 2) * df.lote / 1e6
        g = df.groupby(["resolucao", "lote"]).agg(
            mp=("mp", "first"), aloc=("pico_vram_mb", "mean"),
            reserv=("pico_reservada_mb", "mean"), t=("tempo_total_s", "mean")).reset_index()
        linhas.append(g)

    if not linhas:
        print("Teste 2: nenhum CSV com a coluna pico_reservada_mb ainda. Pulando.")
        print("Rode o benchmark_v2.py nos cinco perfis primeiro.\n")
        return

    g = pd.concat(linhas).drop_duplicates(["resolucao", "lote"]).sort_values("reserv")
    g["s_por_MP"] = g.t / g.mp
    g["extra_%"] = (g.reserv / g.aloc - 1) * 100
    g["ocupacao_%"] = g.reserv / LIMITE_PLACA * 100

    print("=" * 76)
    print("TESTE 2 - memoria alocada x reservada, e onde o tempo dispara")
    print("=" * 76)
    print(f"{'pedido':<12}{'alocada':>10}{'reservada':>11}{'extra':>8}"
          f"{'da placa':>10}{'s/MP':>9}")
    for _, r in g.iterrows():
        print(f"{int(r.resolucao)}x{int(r.lote):<8}{r.aloc:>9.0f}{r.reserv:>11.0f}"
              f"{r['extra_%']:>7.0f}%{r['ocupacao_%']:>9.0f}%{r.s_por_MP:>9.2f}")

    # O criterio e a capacidade da placa, nao um limite de tempo: o que
    # define "nao cabe" e a memoria RESERVADA passar dos 6144 MB.
    # (Classificar pelo s/MP erra no 256x1, que e lento por ser pequeno
    #  demais - o custo fixo de cada chamada domina -, nao por falta de
    #  memoria.)
    cabe = g[g.reserv < LIMITE_PLACA]
    estoura = g[g.reserv >= LIMITE_PLACA]

    if len(cabe) and len(estoura):
        print(f"\nCabem na placa ({len(cabe)} pedidos, reservada < {LIMITE_PLACA:.0f} MB):")
        print(f"  custo de {cabe.s_por_MP.min():.2f} a {cabe.s_por_MP.max():.2f} s/MP, "
              f"mediana {cabe.s_por_MP.median():.2f}")
        print(f"Estouram ({len(estoura)} pedidos, reservada >= {LIMITE_PLACA:.0f} MB):")
        print(f"  custo de {estoura.s_por_MP.min():.2f} a {estoura.s_por_MP.max():.2f} s/MP")

        acertos = (cabe.s_por_MP.max() < estoura.s_por_MP.min())
        print()
        if acertos:
            print("A memoria RESERVADA separa os dois grupos sem nenhuma excecao:")
            print(f"  todo pedido que cabe custa menos de {estoura.s_por_MP.min():.2f} s/MP,")
            print(f"  todo pedido que estoura custa mais de {cabe.s_por_MP.max():.2f} s/MP.")
        else:
            print("Os dois grupos se sobrepoem - vale olhar caso a caso.")

        print(f"\nO joelho esta entre {cabe.reserv.max():.0f} MB "
              f"({cabe.reserv.max()/LIMITE_PLACA*100:.0f}% da placa) e "
              f"{estoura.reserv.min():.0f} MB "
              f"({estoura.reserv.min()/LIMITE_PLACA*100:.0f}%).")
        print(f"Quando nao cabe, o mesmo trabalho custa ate "
              f"{estoura.s_por_MP.max()/cabe.s_por_MP.median():.0f}x mais.")

        # o ponto que mais engana: alocada baixa, reservada estourada
        engana = estoura[estoura.aloc < LIMITE_PLACA]
        if len(engana):
            r = engana.iloc[0]
            print(f"\nAtencao ao {int(r.resolucao)}x{int(r.lote)}: alocada "
                  f"{r.aloc:.0f} MB ({r.aloc/LIMITE_PLACA*100:.0f}% da placa, "
                  f"parece seguro),")
            print(f"mas reservada {r.reserv:.0f} MB "
                  f"({r.reserv/LIMITE_PLACA*100:.0f}%). E a reservada que conta.")
    print()


if __name__ == "__main__":
    teste_ordem()
    teste_reservada()
