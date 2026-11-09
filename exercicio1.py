"""Exercício 1 — Somar informação (2 kHz) e portadora (15 kHz) NÃO é modulação.

Banda permitida: 12 kHz a 18 kHz. A FFT da soma mostra uma componente em 2 kHz,
fora da banda — logo esse sinal não pode ser transmitido.
"""
import sys
import matplotlib.pyplot as plt
from utils import FS, tempo, senoide, plot_tempo, plot_fourier, energia_fora_da_banda

F_M, F_C = 2000, 15000
BANDA = (12000, 18000)

t = tempo(1.0)
m = senoide(F_M, t)          # informação
c = senoide(F_C, t)          # portadora
soma = m + c

print(f"Energia fora da banda {BANDA}: {energia_fora_da_banda(soma, *BANDA):.1f}%")
print("=> A soma tem uma componente em 2 kHz, fora da banda permitida: NÃO pode ser enviada.")

fig, ax = plt.subplots(3, 2, figsize=(13, 9))
plot_tempo(ax[0, 0], t, m, "m(t) — informação 2 kHz", janela=(0, 0.002))
plot_fourier(ax[0, 1], m, "Fourier de m(t)", faixa=BANDA)
plot_tempo(ax[1, 0], t, c, "C(t) — portadora 15 kHz", janela=(0, 0.002), cor="tab:orange")
plot_fourier(ax[1, 1], c, "Fourier de C(t)", faixa=BANDA, cor="tab:orange")
plot_tempo(ax[2, 0], t, soma, "m(t) + C(t)", janela=(0, 0.002), cor="tab:red")
plot_fourier(ax[2, 1], soma, "Fourier da SOMA — componente em 2 kHz fora da banda!", faixa=BANDA, cor="tab:red")
fig.suptitle("Exercício 1 — Soma de sinais não é modulação", fontsize=14)
fig.tight_layout()
fig.savefig("figuras/exercicio1.png", dpi=120)
if "--sem-janela" not in sys.argv:
    plt.show()
