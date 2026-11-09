"""Exercício 3 — Modulação AM por multiplicação ponto a ponto.

m(t): 2 kHz, C(t): 15 kHz, 2 s, fs = 44100 Hz.
S(t) = m(t)·C(t) possui componentes apenas em fc - fm = 13 kHz e fc + fm = 17 kHz,
dentro da banda mínima [13 kHz, 17 kHz] (largura 4 kHz).
"""
import sys
import matplotlib.pyplot as plt
from utils import tempo, senoide, plot_tempo, plot_fourier, faixa_ocupada, energia_fora_da_banda

F_M, F_C = 2000, 15000
BANDA_MIN = (F_C - F_M, F_C + F_M)

t = tempo(2.0)
m = senoide(F_M, t)
c = senoide(F_C, t)
s = m * c                    # multiplicação ponto a ponto

fmin, fmax = faixa_ocupada(s)
print(f"Componentes de S(t) entre {fmin:.0f} Hz e {fmax:.0f} Hz")
print(f"Energia fora de {BANDA_MIN}: {energia_fora_da_banda(s, *BANDA_MIN):.2e}%")

fig, ax = plt.subplots(3, 2, figsize=(13, 9))
plot_tempo(ax[0, 0], t, m, "m(t) — informação 2 kHz", janela=(0, 0.002))
plot_fourier(ax[0, 1], m, "Fourier de m(t)")
plot_tempo(ax[1, 0], t, c, "C(t) — portadora 15 kHz", janela=(0, 0.002), cor="tab:orange")
plot_fourier(ax[1, 1], c, "Fourier de C(t)", cor="tab:orange")
plot_tempo(ax[2, 0], t, s, "S(t) = m(t)·C(t)", janela=(0, 0.002), cor="tab:purple")
plot_fourier(ax[2, 1], s, "Fourier de S(t) — picos em 13 kHz e 17 kHz", faixa=BANDA_MIN, cor="tab:purple")
fig.suptitle("Exercício 3 — Modulação AM por multiplicação", fontsize=14)
fig.tight_layout()
fig.savefig("figuras/exercicio3.png", dpi=120)
if "--sem-janela" not in sys.argv:
    plt.show()
