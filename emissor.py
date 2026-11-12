"""Projeto 8 parte 2 — EMISSOR do chat PSK.

Digite uma frase e tecle ENTER: os caracteres viram bits, os bits modulam uma portadora em BPSK
e o áudio é tocado no alto-falante. O outro computador roda o receptor.py e escreve a frase.

Uso:
    python emissor.py                          # chat interativo, 100 bit/s, portadora 4 kHz
    python emissor.py --taxa 200 --fc 3000     # muda taxa / portadora (o receptor tem que usar os mesmos!)
    python emissor.py -v                       # mostra bits de cada caractere e o quadro
    python emissor.py --plot                   # abre a FFT do sinal emitido a cada mensagem
    python emissor.py -m "oi" --sem-som        # só gera audios/psk_tx.wav (teste sem alto-falante)

Comandos no prompt:  /sair   /info (mostra parâmetros)
"""
import argparse
import sys

import numpy as np

from psk import (FS, PN, N_ABERTURA, N_CAUDA, argumentos_comuns, config_de, crc16, divide_mensagem,
                 modula_texto, resumo_banda)
from utils import salva_wav

p = argumentos_comuns(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
p.add_argument("-m", "--mensagem", help="envia só esta mensagem e sai")
p.add_argument("-v", "--verbose", action="store_true", help="mostra os bits de cada caractere")
p.add_argument("--plot", action="store_true", help="abre a figura do sinal emitido após cada envio")
p.add_argument("--sem-som", action="store_true", help="não toca, só salva o WAV")
p.add_argument("--wav", default="audios/psk_tx.wav", help="onde salvar o último áudio emitido")
args = p.parse_args()
sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # acentos/emoji no terminal do Windows
sys.stdin.reconfigure(encoding="utf-8", errors="replace")
cfg = config_de(args)

if not args.plot:
    import matplotlib
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from graficos_psk import figura_emitido


def mostra_quadro(texto, quadros):
    dados = texto.encode("utf-8")
    print(f"  Texto: {texto!r} -> {len(dados)} bytes UTF-8 em {len(quadros)} quadro(s)")
    if args.verbose:
        for ch in texto:
            b = ch.encode("utf-8")
            bits = " ".join(f"{x:08b}" for x in b)
            print(f"    {ch!r:>5} = {b.hex(' ').upper():<11} = {bits}")
    for i, (bloco, q) in enumerate(zip(divide_mensagem(texto), quadros), start=1):
        n_dados = 8 * len(bloco)
        crc = crc16(bytes([len(bloco)]) + bloco)
        print(f"  Quadro {i}: abertura {N_ABERTURA} + preâmbulo {len(PN)} + tamanho 8 + dados {n_dados} "
              f"+ CRC 16 + cauda {N_CAUDA} = {len(q)} bits  (CRC-16 = 0x{crc:04X}, tamanho = {len(bloco)})")
        if args.verbose:
            print(f"    preâmbulo PN: {''.join(map(str, PN))}")
            print(f"    quadro      : {''.join(map(str, q))}")


def envia(texto):
    x, quadros = modula_texto(texto, cfg)
    n_bits = sum(len(q) for q in quadros)
    mostra_quadro(texto, quadros)
    b = resumo_banda(x, cfg)
    print(f"  {n_bits} bits a {cfg.taxa:g} bit/s = {n_bits * cfg.T:.2f} s de sinal | {len(x)} amostras "
          f"({len(x) / FS:.2f} s com silêncio)")
    print(f"  Banda: necessária (lóbulo principal) {b['teorica']:.0f} Hz | medida: 90% da potência em "
          f"{b['90%']:.0f} Hz, 99% em {b['99%']:.0f} Hz")
    salva_wav(args.wav, x)
    fig = figura_emitido(x, cfg, texto, quadros[0])
    fig.savefig("figuras/psk_emitido.png", dpi=110)
    if not args.sem_som:
        import sounddevice as sd
        print("  ▶ Transmitindo...", end="", flush=True)
        # +0,5 s de silêncio: o driver de áudio do Windows costuma descartar o finalzinho do buffer
        sd.play(np.concatenate([x, np.zeros(FS // 2)]), FS, device=args.dispositivo)
        sd.wait()
        print(" ✔ enviado")
    print(f"  (áudio em {args.wav}, gráfico em figuras/psk_emitido.png)")
    if args.plot:
        plt.show()
    plt.close(fig)


print("=" * 72)
print(" EMISSOR PSK —", cfg.descricao())
print("=" * 72)
if args.mensagem is not None:
    envia(args.mensagem)
    sys.exit()

print("Digite a mensagem e tecle ENTER (/sair para encerrar, /info para parâmetros)\n")
while True:
    try:
        texto = input("você> ")
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if texto.strip() == "/sair":
        break
    if texto.strip() == "/info":
        print(" ", cfg.descricao())
        continue
    if texto:
        envia(texto)
