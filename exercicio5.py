"""Exercício 5 — Modulação e demodulação AM de um áudio real.

Portadora: 16 kHz. Banda permitida: 14 kHz a 18 kHz  =>  a informação deve ocupar no máx. 2 kHz.
Passos: (1) FFT do áudio original, (2) portadora com o mesmo nº de pontos,
(3) passa-baixa no áudio + modulação por multiplicação, (4) FFT do modulado,
(6) demodulação: multiplica pela portadora e aplica passa-baixa, (7) FFT do demodulado.

Uso: python exercicio5.py [arquivo.wav] [--sem-som] [--sem-janela]
"""
import sys
import matplotlib.pyplot as plt
from utils import (FS, le_audio, normaliza, senoide, passa_baixa, plot_fourier,
                   faixa_ocupada, energia_fora_da_banda, salva_wav, toca)

F_C = 16000
BANDA = (14000, 18000)
F_CORTE = 1700        # 1700 Hz + transição de 300 Hz = 2000 Hz  -> cabe em 16 ± 2 kHz

args = [a for a in sys.argv[1:] if not a.startswith("--")]
arquivo = args[0] if args else "audios/audio1.wav"
tocar = "--sem-som" not in sys.argv

fs = FS
audio = normaliza(le_audio(arquivo))
t = __import__("numpy").arange(len(audio)) / fs

# 1) áudio original
print(f"1) Áudio original ocupa aprox. {faixa_ocupada(audio)[0]:.0f}–{faixa_ocupada(audio)[1]:.0f} Hz (limiar -40 dB)")
if tocar: toca(audio, descricao="áudio original")

# 2) portadora com o mesmo nº de pontos
portadora = senoide(F_C, t)

# 3) trata o sinal (passa-baixa) e modula
audio_filtrado = normaliza(passa_baixa(audio, F_CORTE))
modulado = audio_filtrado * portadora

# 4) verifica banda
print(f"4) Energia do modulado fora de {BANDA}: {energia_fora_da_banda(modulado, *BANDA):.2e}%")

# 5) toca o modulado
if tocar: toca(modulado, descricao="sinal modulado (16 kHz)")

# 6) demodula: multiplica pela portadora e filtra
demodulado = normaliza(passa_baixa(modulado * portadora, F_CORTE))

# 8) toca o demodulado
if tocar: toca(demodulado, descricao="sinal demodulado")
salva_wav("audios/ex5_modulado.wav", modulado)
salva_wav("audios/ex5_demodulado.wav", demodulado)

fig, ax = plt.subplots(5, 1, figsize=(12, 14))
plot_fourier(ax[0], audio, "1) Fourier do áudio original")
plot_fourier(ax[1], audio_filtrado, f"3) Áudio após passa-baixa ({F_CORTE} Hz)")
plot_fourier(ax[2], modulado, "4) Fourier do sinal modulado (portadora 16 kHz)", faixa=BANDA, cor="tab:purple")
plot_fourier(ax[3], modulado * portadora, "6) Modulado × portadora (antes do passa-baixa)", cor="tab:red")
plot_fourier(ax[4], demodulado, "7) Fourier do sinal demodulado", cor="tab:green")
fig.suptitle(f"Exercício 5 — AM com áudio ({arquivo})", fontsize=14)
fig.tight_layout()
fig.savefig("figuras/exercicio5.png", dpi=110)
if "--sem-janela" not in sys.argv:
    plt.show()
