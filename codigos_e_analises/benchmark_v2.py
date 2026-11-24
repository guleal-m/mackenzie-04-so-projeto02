"""
Projeto 2 - SO - Custo de gerar imagens com IA local
Script de MEDICAO - serve para os 5 integrantes

Cada integrante muda APENAS a linha INTEGRANTE abaixo e roda o script.
Tudo o mais (modelo, semente, passos, repeticoes) fica igual para o
grupo inteiro, senao as medidas nao podem ser comparadas.

A ideia do experimento e uma escada: o integrante 1 deixa tudo travado e
muda so a quantidade de imagens; cada integrante seguinte solta mais uma
trava; o integrante 5 muda as tres coisas ao mesmo tempo.

Como rodar:
  python benchmark.py
"""

import csv
import os
import threading
import time

import psutil
import torch
from diffusers import AutoPipelineForText2Image

# ====================== MUDE SO ESTAS DUAS ========================
INTEGRANTE = 3         # 1, 2, 3, 4 ou 5
ORDEM_INVERTIDA = True # True roda os pedidos de tras para frente
# ==================================================================
#
# Para que serve ORDEM_INVERTIDA:
# a placa esquenta ao longo da sessao e vai perdendo velocidade, o que
# soma um erro ao tempo que cresce com a ordem da execucao. Rodando a
# mesma frente nas duas ordens, o que for efeito do pedido se repete e
# o que for aquecimento troca de lado. Vale sobretudo para a frente 3,
# onde os quatro pedidos deveriam custar o mesmo.

# ---- combinados com o grupo: NAO mudar ----
MODELO = "stabilityai/sd-turbo"
SEMENTE = 42
PASSOS = 4
REPETICOES = 3

P1 = "a slice of chocolate cake on a white plate"
P2 = "a lighthouse on a rocky cliff at sunset, dramatic clouds"
P3 = "an old stone bridge over a river"
P4 = "a close up of a sunflower against a blue sky"
# -------------------------------------------

# Cada caso e um "pedido": qual imagem, que tamanho e quantas de uma vez.
# "rotulo" e o nome que aparece no eixo do grafico.
PERFIS = {
    1: {
        "nome": "Integrante 1 - mesma imagem, mesmo tamanho, quantidade varia",
        "eixo": "Quantidade de imagens de uma vez",
        "casos": [
            {"rotulo": 1, "prompt": P1, "resolucao": 512, "lote": 1},
            {"rotulo": 2, "prompt": P1, "resolucao": 512, "lote": 2},
            {"rotulo": 4, "prompt": P1, "resolucao": 512, "lote": 4},
            {"rotulo": 8, "prompt": P1, "resolucao": 512, "lote": 8},
        ],
    },
    2: {
        "nome": "Integrante 2 - mesma imagem, quantidade fixa, tamanho varia",
        "eixo": "Tamanho da imagem (pixels de lado)",
        "casos": [
            {"rotulo": 256, "prompt": P1, "resolucao": 256, "lote": 1},
            {"rotulo": 384, "prompt": P1, "resolucao": 384, "lote": 1},
            {"rotulo": 512, "prompt": P1, "resolucao": 512, "lote": 1},
            {"rotulo": 768, "prompt": P1, "resolucao": 768, "lote": 1},
        ],
    },
    3: {
        "nome": "Integrante 3 - imagens diferentes, mesmo tamanho, quantidade fixa",
        "eixo": "Imagem pedida",
        "casos": [
            {"rotulo": "bolo", "prompt": P1, "resolucao": 512, "lote": 1},
            {"rotulo": "farol", "prompt": P2, "resolucao": 512, "lote": 1},
            {"rotulo": "ponte", "prompt": P3, "resolucao": 512, "lote": 1},
            {"rotulo": "girassol", "prompt": P4, "resolucao": 512, "lote": 1},
        ],
    },
    4: {
        "nome": "Integrante 4 - imagens e tamanhos diferentes, quantidade fixa",
        "eixo": "Pedido (imagem + tamanho)",
        "casos": [
            {"rotulo": "bolo 256", "prompt": P1, "resolucao": 256, "lote": 1},
            {"rotulo": "farol 384", "prompt": P2, "resolucao": 384, "lote": 1},
            {"rotulo": "ponte 512", "prompt": P3, "resolucao": 512, "lote": 1},
            {"rotulo": "girassol 768", "prompt": P4, "resolucao": 768, "lote": 1},
        ],
    },
    5: {
        "nome": "Integrante 5 - imagem, tamanho e quantidade, tudo diferente",
        "eixo": "Pedido (imagem + tamanho + quantidade)",
        "casos": [
            {"rotulo": "bolo 256 x1", "prompt": P1, "resolucao": 256, "lote": 1},
            {"rotulo": "farol 384 x2", "prompt": P2, "resolucao": 384, "lote": 2},
            {"rotulo": "ponte 512 x4", "prompt": P3, "resolucao": 512, "lote": 4},
            {"rotulo": "girassol 768 x8", "prompt": P4, "resolucao": 768, "lote": 8},
        ],
    },
}

SUFIXO = f"integrante{INTEGRANTE}" + ("_invertido" if ORDEM_INVERTIDA else "")
ARQUIVO_SAIDA = f"resultados_{SUFIXO}.csv"
PASTA_IMAGENS = f"imagens_{SUFIXO}"

processo = psutil.Process()

# Energia e temperatura vem da biblioteca pynvml (pacote nvidia-ml-py).
# Se nao estiver instalada, o script continua e deixa esses campos vazios.
try:
    import pynvml

    pynvml.nvmlInit()
    HANDLE_GPU = pynvml.nvmlDeviceGetHandleByIndex(0)
    TEM_NVML = True
except Exception:
    HANDLE_GPU = None
    TEM_NVML = False


class MonitorGPU(threading.Thread):
    """Le energia e temperatura da placa a cada 0,2 s enquanto a geracao roda."""

    def __init__(self):
        super().__init__(daemon=True)
        self.rodando = True
        self.pico_watts = 0.0
        self.pico_temp = 0.0

    def run(self):
        while self.rodando:
            try:
                watts = pynvml.nvmlDeviceGetPowerUsage(HANDLE_GPU) / 1000.0
                temp = pynvml.nvmlDeviceGetTemperature(
                    HANDLE_GPU, pynvml.NVML_TEMPERATURE_GPU
                )
                self.pico_watts = max(self.pico_watts, watts)
                self.pico_temp = max(self.pico_temp, float(temp))
            except Exception:
                pass
            time.sleep(0.2)

    def parar(self):
        self.rodando = False
        self.join(timeout=1)


def nome_de_arquivo(rotulo):
    """Transforma o rotulo do pedido num nome de arquivo seguro."""
    limpo = []
    for c in str(rotulo):
        limpo.append(c if (c.isalnum() or c in "-_") else "_")
    return "".join(limpo).strip("_")


def medir(pipe, caso, salvar_em=None):
    """Gera um pedido e devolve um dicionario com as metricas da rodada.

    Se salvar_em for passado, grava as imagens geradas nessa pasta.
    A gravacao acontece DEPOIS do cronometro parar, entao nao atrapalha
    a medicao de tempo.
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ram_antes = processo.memory_info().rss

    monitor = None
    if TEM_NVML:
        monitor = MonitorGPU()
        monitor.start()

    gerador = torch.Generator("cuda").manual_seed(SEMENTE)

    inicio = time.perf_counter()
    saida = pipe(
        prompt=[caso["prompt"]] * caso["lote"],
        num_inference_steps=PASSOS,
        guidance_scale=0.0,
        width=caso["resolucao"],
        height=caso["resolucao"],
        generator=gerador,
    )
    torch.cuda.synchronize()   # espera a GPU terminar de verdade
    tempo_total = time.perf_counter() - inicio

    # cronometro ja parou: salvar daqui para baixo nao afeta a medicao
    if salvar_em is not None:
        base = nome_de_arquivo(caso["rotulo"])
        for i, img in enumerate(saida.images, start=1):
            img.save(os.path.join(salvar_em, f"{base}_img{i}.png"))

    pico_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
    # reservada: o que o alocador do PyTorch pediu a placa, incluindo o
    # cache que ele guarda para as proximas alocacoes. E sempre >= a
    # alocada, e e ela que de fato ocupa espaco na placa.
    pico_reservada = torch.cuda.max_memory_reserved() / (1024 ** 2)
    pico_ram = max(processo.memory_info().rss, ram_antes) / (1024 ** 2)

    pico_watts = ""
    pico_temp = ""
    if monitor is not None:
        monitor.parar()
        pico_watts = round(monitor.pico_watts, 1)
        pico_temp = round(monitor.pico_temp, 1)

    return {
        "tempo_total_s": round(tempo_total, 3),
        "tempo_por_imagem_s": round(tempo_total / caso["lote"], 3),
        "pico_vram_mb": round(pico_vram, 1),
        "pico_reservada_mb": round(pico_reservada, 1),
        "folga_ate_o_limite_mb": round(6144 - pico_reservada, 1),
        "pico_ram_mb": round(pico_ram, 1),
        "pico_energia_w": pico_watts,
        "pico_temp_c": pico_temp,
    }


COLUNAS = [
    "rotulo", "rodada", "prompt", "resolucao", "lote",
    "tempo_total_s", "tempo_por_imagem_s",
    "pico_vram_mb", "pico_reservada_mb", "folga_ate_o_limite_mb",
    "pico_ram_mb", "pico_energia_w", "pico_temp_c",
]


def main():
    if INTEGRANTE not in PERFIS:
        print("INTEGRANTE precisa ser 1, 2, 3, 4 ou 5.")
        return

    if not torch.cuda.is_available():
        print("Placa de video NVIDIA nao encontrada. Verifique a instalacao do torch.")
        return

    perfil = PERFIS[INTEGRANTE]

    print(f"Placa: {torch.cuda.get_device_name(0)}")
    print(f"{perfil['nome']}\n")
    if not TEM_NVML:
        print("Aviso: pynvml nao encontrado - energia e temperatura ficarao vazias.")
        print("Para medir isso, rode: pip install nvidia-ml-py\n")

    print("Carregando o modelo (pode demorar alguns minutos na primeira vez)...")
    pipe = AutoPipelineForText2Image.from_pretrained(
        MODELO, torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    os.makedirs(PASTA_IMAGENS, exist_ok=True)

    print("Aquecimento (essa geracao e descartada)...\n")
    medir(pipe, perfil["casos"][0])

    casos = list(perfil["casos"])
    if ORDEM_INVERTIDA:
        casos.reverse()
        print("Ordem INVERTIDA: " + " -> ".join(str(c["rotulo"]) for c in casos) + "\n")

    linhas = []
    for caso in casos:
        for rodada in range(1, REPETICOES + 1):
            base = {
                "rotulo": caso["rotulo"], "rodada": rodada,
                "prompt": caso["prompt"], "resolucao": caso["resolucao"],
                "lote": caso["lote"],
            }
            # guarda as imagens so na primeira rodada de cada pedido:
            # as outras duas sairiam identicas, por causa da semente fixa
            salvar = PASTA_IMAGENS if rodada == 1 else None
            try:
                m = medir(pipe, caso, salvar_em=salvar)
            except torch.cuda.OutOfMemoryError:
                print(f"{str(caso['rotulo']):>14} | rodada {rodada} | "
                      f"ESTOUROU A MEMORIA DA GPU")
                linhas.append({**base, **{c: "erro" for c in COLUNAS[5:]}})
                torch.cuda.empty_cache()
                continue

            print(
                f"{str(caso['rotulo']):>14} | rodada {rodada} | "
                f"{m['tempo_total_s']:7.2f} s | "
                f"{m['tempo_por_imagem_s']:6.2f} s/img | "
                f"{m['pico_vram_mb']:7.1f} aloc | "
                f"{m['pico_reservada_mb']:7.1f} reserv | "
                f"{m['pico_energia_w']} W"
            )
            linhas.append({**base, **m})

    with open(ARQUIVO_SAIDA, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=COLUNAS)
        escritor.writeheader()
        escritor.writerows(linhas)

    perto = [l for l in linhas
             if isinstance(l.get("pico_reservada_mb"), float)
             and l["pico_reservada_mb"] > 6144 * 0.9]
    if perto:
        print("\nAviso: em " + str(len(perto)) + " medicoes a memoria RESERVADA")
        print("passou de 90% dos 6144 MB da placa. E nessa faixa que o driver")
        print("comeca a usar a RAM do sistema e o tempo dispara.")

    print(f"\nPronto. Resultados salvos em {ARQUIVO_SAIDA}")
    print(f"Imagens geradas salvas na pasta {PASTA_IMAGENS}")
    print("Agora rode: python analise.py")


if __name__ == "__main__":
    main()
