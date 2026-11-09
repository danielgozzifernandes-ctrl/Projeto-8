"""Projeto 8 — Três estações de rádio AM em um mesmo meio físico.

EMISSOR
  - 3 áudios (44100 Hz) são limitados em banda (passa-baixa) e modulados em AM:
        Estação 1: faixa  9–12 kHz -> portadora 10,5 kHz
        Estação 2: faixa 12–15 kHz -> portadora 13,5 kHz
        Estação 3: faixa 15–18 kHz -> portadora 16,5 kHz
    Cada faixa tem 3 kHz => o áudio pode ocupar no máx. 1,5 kHz. Usamos passa-baixa com
    banda passante até 1,2 kHz e rejeição (-80 dB) a partir de 1,5 kHz.
  - Os 3 sinais modulados são somados: sinal presente no "meio físico".
RECEPTOR
  - Um passa-faixa por estação extrai cada sinal modulado da soma.
  - Demodulação síncrona: multiplica pela portadora e aplica passa-baixa.
  - Os 3 áudios recuperados são salvos em audios/demodulado{1,2,3}.wav.

Uso:
    python projeto8.py [--sem-som] [--sem-janela]               # usa audios/audio1..3.wav
    python projeto8.py musica1.mp3 voz.wav outro.flac           # usa os arquivos indicados
Aceita wav, mp3, flac e ogg em qualquer taxa de amostragem (convertidos para 44100 Hz, máx. 5 s).
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from utils import (FS, le_audio, normaliza, senoide, passa_baixa, passa_faixa,
                   plot_fourier, energia_fora_da_banda, salva_wav, toca)

ESTACOES = [  # (faixa mínima, faixa máxima)
    (9000, 12000),
    (12000, 15000),
    (15000, 18000),
]
F_AUDIO = 1200      # banda passante do áudio
TRANSICAO = 300     # 1200 + 300 = 1500 Hz = metade da largura de cada faixa
CORES = ["tab:blue", "tab:orange", "tab:green"]
XLIM = (0, 22050)

tocar = "--sem-som" not in sys.argv
arquivos = [a for a in sys.argv[1:] if not a.startswith("--")]
if not arquivos:
    arquivos = [f"audios/audio{i}.wav" for i in range(1, 4)]
assert len(arquivos) == 3, "Informe exatamente 3 arquivos de áudio"

# ============================================================ EMISSOR
print("=" * 60, "\nEMISSOR\n" + "=" * 60)
audios = []
for i, arq in enumerate(arquivos, start=1):
    x = le_audio(arq)
    print(f"Áudio {i}: {arq} ({len(x)/FS:.1f} s)")
    audios.append(normaliza(x))

# todos os sinais precisam do mesmo número de pontos para serem somados
N = max(len(a) for a in audios)
audios = [np.pad(a, (0, N - len(a))) for a in audios]
t = np.arange(N) / FS

portadoras, filtrados, modulados = [], [], []
for i, (a, (f_lo, f_hi)) in enumerate(zip(audios, ESTACOES), start=1):
    if tocar:
        toca(a, descricao=f"áudio original {i}")
    fc = (f_lo + f_hi) / 2
    c = senoide(fc, t)
    a_f = normaliza(passa_baixa(a, F_AUDIO, TRANSICAO))
    s = a_f * c
    portadoras.append(c); filtrados.append(a_f); modulados.append(s)
    print(f"Estação {i}: portadora {fc/1000:.1f} kHz, faixa {f_lo/1000:.0f}–{f_hi/1000:.0f} kHz | "
          f"energia fora da faixa: {energia_fora_da_banda(s, f_lo, f_hi):.2e}%")

# verificação: nenhuma estação invade a faixa das outras
print("\nVerificação de invasão de bandas (energia de cada estação dentro das faixas das OUTRAS):")
for i, s in enumerate(modulados):
    for j, (f_lo, f_hi) in enumerate(ESTACOES):
        if i != j:
            dentro = 100 - energia_fora_da_banda(s, f_lo, f_hi)
            print(f"  estação {i+1} na faixa {j+1}: {dentro:.2e}%")

# ============================================================ MEIO FÍSICO
meio = sum(modulados)
salva_wav("audios/sinal_meio_fisico.wav", meio)

# ============================================================ RECEPTOR
print("\n" + "=" * 60, "\nRECEPTOR\n" + "=" * 60)
extraidos, demodulados = [], []
for i, ((f_lo, f_hi), c, a_f) in enumerate(zip(ESTACOES, portadoras, filtrados), start=1):
    # passa-faixa: passa [fc-1,2k, fc+1,2k], rejeita fora de [f_lo, f_hi]
    fc = (f_lo + f_hi) / 2
    r = passa_faixa(meio, fc - F_AUDIO, fc + F_AUDIO, TRANSICAO)
    d = normaliza(passa_baixa(r * c, F_AUDIO, TRANSICAO))
    extraidos.append(r); demodulados.append(d)
    salva_wav(f"audios/demodulado{i}.wav", d)
    corr = np.corrcoef(d, a_f)[0, 1]
    print(f"Estação {i}: demodulado salvo em audios/demodulado{i}.wav | "
          f"correlação com o áudio transmitido: {corr:.4f}")

if tocar:
    for i, d in enumerate(demodulados, start=1):
        toca(d, descricao=f"áudio demodulado {i}")

# ============================================================ GRÁFICOS
def faixas(ax):
    for (f_lo, f_hi), cor in zip(ESTACOES, CORES):
        ax.axvspan(f_lo, f_hi, color=cor, alpha=0.12)

# 1) áudios originais
fig1, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
for i in range(3):
    plot_fourier(ax[i], audios[i], f"Áudio original {i+1}", xlim=(0, 8000), cor=CORES[i])
fig1.suptitle("FFT dos áudios originais", fontsize=14); fig1.tight_layout()
fig1.savefig("figuras/p8_1_originais.png", dpi=110)

# 2) três modulados
fig2, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
for i in range(3):
    plot_fourier(ax[i], modulados[i], f"Áudio {i+1} modulado — faixa "
                 f"{ESTACOES[i][0]/1000:.0f}–{ESTACOES[i][1]/1000:.0f} kHz", xlim=XLIM, cor=CORES[i])
    faixas(ax[i])
fig2.suptitle("FFT dos 3 áudios modulados (emissor)", fontsize=14); fig2.tight_layout()
fig2.savefig("figuras/p8_2_modulados.png", dpi=110)

# 3) soma
fig3, ax = plt.subplots(figsize=(12, 4))
plot_fourier(ax, meio, "FFT do sinal resultante da soma (meio físico)", xlim=XLIM, cor="black")
faixas(ax)
fig3.tight_layout(); fig3.savefig("figuras/p8_3_soma.png", dpi=110)

# 4) extraídos no receptor
fig4, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
for i in range(3):
    plot_fourier(ax[i], extraidos[i], f"Estação {i+1} extraída pelo passa-faixa", xlim=XLIM, cor=CORES[i])
    faixas(ax[i])
fig4.suptitle("FFT dos 3 sinais modulados extraídos da soma (receptor)", fontsize=14); fig4.tight_layout()
fig4.savefig("figuras/p8_4_extraidos.png", dpi=110)

# 5) demodulados
fig5, ax = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
for i in range(3):
    plot_fourier(ax[i], demodulados[i], f"Áudio {i+1} demodulado", xlim=(0, 8000), cor=CORES[i])
fig5.suptitle("FFT dos 3 áudios demodulados (receptor)", fontsize=14); fig5.tight_layout()
fig5.savefig("figuras/p8_5_demodulados.png", dpi=110)

print("\nGráficos salvos em figuras/p8_*.png")
if "--sem-janela" not in sys.argv:
    plt.show()
