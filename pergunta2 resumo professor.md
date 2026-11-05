# Projeto 2 de SO: Pergunta 2 do grupo

## Pergunta do grupo

**Quanto custa gerar mais imagens por vez com uma IA local, medido em pico de memória (MB) e tempo por imagem (s)?**

## Pilar e conceito

- **Pilar do OSTEP:** virtualização de memória.
- **Conceito:** tipo de dado e tamanho em memória; swap e falta de página quando o modelo não cabe na RAM.

## O que vamos fazer

Rodar um modelo pequeno de geração de imagens (sd-turbo, com Python e a biblioteca diffusers) em uma máquina do grupo, sem GPU se não houver.

Os pesos do modelo ocupam uma memória fixa (a *base*). Cada imagem gerada junto reserva memória extra. Queremos medir quanto cada mudança custa e onde o modelo deixa de caber na RAM.

## Hipótese

A memória cresce com o número de imagens simultâneas, o tempo por imagem melhora pouco, e quando passa da RAM o sistema usa swap e o tempo piora muito.

## O que vamos medir e como

- **Pico de memória:** um script lê a memória residente (RSS) do processo a cada 20 ms durante a geração e guarda o maior valor.
- **Tempo por imagem:** tempo total medido com o relógio do sistema, dividido pelo número de imagens.
- **Repetição:** 1 rodada de aquecimento descartada e 5 rodadas por condição, com média e desvio padrão.
- **Controle:** máquina na tomada, outros programas fechados, uma medição por vez e especificações da máquina anotadas.
- **Origem dos dados:** todo dado é medido pelo grupo. Os CSV e os scripts ficam no repositório. A IA será usada apenas na visualização, e isso será declarado no relatório.

## Frentes de investigação

Cada integrante muda uma variável, com o resto fixo.

| # | Frente (variável) | Valores | O que mostra |
|---|---|---|---|
| 1 | **Lote**: imagens geradas ao mesmo tempo | 1, 2, 4, 8 | Quanto a memória e o tempo por imagem mudam, e onde o modelo deixa de caber |
| 2 | **Resolução** | 128, 192, 256, 384 (pixels de lado) | O custo em memória e tempo do tamanho da imagem |
| 3 | **Passos de geração** | 1, 2, 4, 8 | Frente de controle: tempo crescendo e memória quase constante |
| 4 | **Tipo de dado dos pesos** | fp32, bf16 | O que a precisão custa em memória e em velocidade (análogo ao float32 e int8 do exemplo do Kindle) |
| 5 | **Threads de CPU** | 1, 2, 4 e o número de processadores lógicos | O ganho de tempo até saturar os núcleos, com a memória constante |

## Entrega de cada integrante

Uma afirmação, um gráfico e um limite, defendidos na apresentação com o relatório aberto.
