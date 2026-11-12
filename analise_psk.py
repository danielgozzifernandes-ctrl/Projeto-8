"""Projeto 8 parte 2 — Análise do espectro do PSK: banda necessária, efeito da taxa e da portadora.

Gera em figuras/:
    psk_a_modulacao.png        passo a passo: bits -> a(t) -> portadora -> s(t) -> I/Q no receptor
    psk_b_taxas.png            FFT para 50, 100, 200 e 400 bit/s (portadora fixa)
    psk_c_portadoras.png       FFT para portadoras de 2, 4 e 6 kHz (taxa fixa)
    psk_d_pulsos.png           pulso retangular x cosseno levantado (banda limitada)
    psk_e_emitido_recebido.png FFT do sinal emitido x recebido + constelação

Uso:
    python analise_psk.py                                   # recebido = canal simulado
    python analise_psk.py --rx audios/psk_rx.wav            # recebido = gravação real do receptor
    python analise_psk.py --taxas 100 300 1000 --fcs 2500 5000 --sem-janela
"""
import argparse
import sys

import numpy as np
import matplotlib.pyplot as plt

from psk import (FS, Config, espectro, modula_bits, modula_texto, monta_quadro, resumo_banda,
                 demodula, simula_canal, converte_para_banda_base)
from graficos_psk import plot_fft, plot_constelacao, marca_banda
from utils import le_audio

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--taxas", type=float, nargs="+", default=[50, 100, 200, 400])
p.add_argument("--fcs", type=float, nargs="+", default=[2000, 4000, 6000])
p.add_argument("--tx", default="audios/psk_tx.wav", help="sinal emitido (gerado pelo emissor.py)")
p.add_argument("--rx", help="sinal recebido gravado pelo receptor.py (se omitido, usa canal simulado)")
p.add_argument("--mensagem", default="Olá, Camada Física! PSK a 100 bit/s.")
p.add_argument("--sem-janela", action="store_true")
args = p.parse_args()
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

rng = np.random.default_rng(0)
BITS_ALEATORIOS = rng.integers(0, 2, 600)   # bits "típicos" (metade 0, metade 1) para o espectro


def envelope_teorico(f, cfg, pico):
    """|X(f)| teórico do BPSK retangular: |sinc((f - fc)/taxa)| (o espectro do pulso de duração T)."""
    return pico * np.abs(np.sinc((f - cfg.fc) / cfg.taxa))


def linha_tabela(cfg, x):
    b = resumo_banda(x, cfg)
    return (f"  {cfg.taxa:>6g} | {cfg.fc:>6.0f} | {cfg.pulso:>4} | {1000 * cfg.T:>6.2f} | "
            f"{cfg.fc - cfg.banda / 2:>6.0f}–{cfg.fc + cfg.banda / 2:<6.0f} | {b['teorica']:>8.0f} | "
            f"{b['90%']:>7.0f} | {b['99%']:>7.0f}")


CABECALHO = ("  taxa   |   fc   |pulso| T (ms) | faixa necess. | banda    | 90% pot | 99% pot\n"
             "  (bit/s)|  (Hz)  |     |        |     (Hz)      | necess.  |   (Hz)  |   (Hz)\n  " + "-" * 78)

# ============================================================================ A) passo a passo
cfg = Config()
bits = np.array([0, 1, 1, 0, 1, 0, 0, 1])
s = modula_bits(np.concatenate([[0, 0], bits, [0, 0]]), cfg)
t = np.arange(len(s)) / FS
a = np.repeat(np.concatenate([[0, 0], bits, [0, 0]]), int(cfg.spb))
a = 1 - 2.0 * a[:len(s)]
_, z = converte_para_banda_base(s, cfg)
fig, ax = plt.subplots(5, 1, figsize=(12, 12), sharex=True)
ax[0].step(np.arange(len(bits) + 5) * cfg.T * 1000, np.concatenate([[0, 0], bits, [0, 0, 0]]), where="post")
for k, bit in enumerate(np.concatenate([[0, 0], bits, [0, 0]])):
    ax[0].text((k + 0.5) * cfg.T * 1000, 1.15, str(bit), ha="center", color="tab:red" if bit else "tab:blue")
ax[0].set_ylim(-0.2, 1.4); ax[0].set_title("1) Bits a transmitir (cada bit dura T = 10 ms a 100 bit/s)")
ax[1].plot(t * 1000, a, color="tab:red"); ax[1].set_ylim(-1.4, 1.4)
ax[1].set_title("2) a(t): bit 0 → +1 (fase 0°), bit 1 → −1 (fase 180°)")
ax[2].plot(t * 1000, np.cos(2 * np.pi * cfg.fc * t), color="gray", lw=0.6)
ax[2].set_title(f"3) Portadora cos(2π·{cfg.fc:.0f}·t)")
ax[3].plot(t * 1000, s, lw=0.6); ax[3].plot(t * 1000, 0.8 * a, color="tab:red", lw=1, ls="--")
ax[3].set_title("4) s(t) = a(t)·cos(2πfc·t): a cada bit 1 a portadora aparece invertida (180°)")
ax[4].plot(t * 1000, z.real, label="I (em fase)", color="tab:blue")
ax[4].plot(t * 1000, z.imag, label="Q (quadratura)", color="tab:orange")
ax[4].set_title("5) Receptor: s(t)·2e^{−j2πfc·t} + passa-baixa → recupera a(t) (I) — bit = 1 se I < 0")
ax[4].legend(loc="upper right"); ax[4].set_xlabel("Tempo (ms)")
for x_ in ax:
    x_.grid(alpha=0.3)
    for k in range(len(bits) + 5):
        x_.axvline(k * cfg.T * 1000, color="gray", lw=0.4, ls=":")
fig.suptitle("Modulação e demodulação BPSK passo a passo", fontsize=13); fig.tight_layout()
fig.savefig("figuras/psk_a_modulacao.png", dpi=110)

# ============================================================================ B) variando a taxa
print("=" * 82)
print("B) VARIANDO A TAXA DE TRANSMISSÃO (portadora 4 kHz, bits aleatórios)")
print("=" * 82)
print(CABECALHO)
fig, ax = plt.subplots(len(args.taxas), 1, figsize=(12, 2.6 * len(args.taxas) + 1), sharex=True)
ax = np.atleast_1d(ax)
larg = 5 * max(args.taxas)
for i, taxa in enumerate(args.taxas):
    cfg = Config(taxa=taxa, fc=4000)
    x = modula_bits(BITS_ALEATORIOS[: max(200, int(4 * taxa))], cfg)
    print(linha_tabela(cfg, x))
    f, X = espectro(x)
    ax[i].plot(f, X, lw=0.6, color=f"C{i}", label=f"FFT — {taxa:g} bit/s")
    m = np.abs(f - cfg.fc) < larg
    pico = np.sqrt(np.mean(X[np.abs(f - cfg.fc) < cfg.taxa / 4] ** 2)) * 1.6
    ax[i].plot(f[m], envelope_teorico(f[m], cfg, pico), "k--", lw=0.9, label="teórico |sinc((f−fc)/taxa)|")
    marca_banda(ax[i], cfg)
    ax[i].set_xlim(cfg.fc - larg, cfg.fc + larg)
    ax[i].set_title(f"{taxa:g} bit/s → T = {1000 * cfg.T:g} ms → nulos em fc ± {taxa:g} Hz "
                    f"(lóbulo principal = {cfg.banda:.0f} Hz)")
    ax[i].legend(loc="upper right", fontsize=8); ax[i].grid(alpha=0.3); ax[i].set_ylabel("|X(f)|")
ax[-1].set_xlabel("Frequência (Hz)")
fig.suptitle("Efeito da TAXA: dobrar a taxa = bits com metade da duração = espectro com o dobro da largura",
             fontsize=12)
fig.tight_layout(); fig.savefig("figuras/psk_b_taxas.png", dpi=110)
print("  -> a banda cresce PROPORCIONALMENTE à taxa: o lóbulo principal tem largura 2·taxa,")
print("     porque cada bit é um pulso de duração T = 1/taxa e a FFT de um pulso de duração T é um sinc")
print("     com nulos a cada 1/T Hz. Pulso mais curto no tempo => espectro mais largo na frequência.")

# ============================================================================ C) variando a portadora
print("\n" + "=" * 82)
print("C) VARIANDO A PORTADORA (100 bit/s)")
print("=" * 82)
print(CABECALHO)
fig, ax = plt.subplots(len(args.fcs), 1, figsize=(12, 2.4 * len(args.fcs) + 1), sharex=True)
ax = np.atleast_1d(ax)
for i, fc in enumerate(args.fcs):
    cfg = Config(taxa=100, fc=fc)
    x = modula_bits(BITS_ALEATORIOS[:400], cfg)
    print(linha_tabela(cfg, x))
    plot_fft(ax[i], x, cfg, f"Portadora {fc:.0f} Hz — mesmo formato, deslocado para {fc - 100:.0f}–{fc + 100:.0f} Hz",
             cor=f"C{i}", zoom=(0, 8000))
ax[-1].set_xlabel("Frequência (Hz)")
fig.suptitle("Efeito da PORTADORA: o espectro só é DESLOCADO (a largura depende só da taxa)", fontsize=12)
fig.tight_layout(); fig.savefig("figuras/psk_c_portadoras.png", dpi=110)
print("  -> mudar fc só translada o espectro (propriedade do deslocamento na frequência da")
print("     multiplicação por cos(2πfc·t), igual ao AM). A largura de banda não muda.")

# ============================================================================ D) formato do pulso
print("\n" + "=" * 82)
print("D) FORMATO DO PULSO (100 bit/s, 4 kHz)")
print("=" * 82)
print(CABECALHO)
fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
for i, (pulso, cor) in enumerate([("ret", "tab:blue"), ("rc", "tab:purple")]):
    cfg = Config(pulso=pulso)
    x = modula_bits(BITS_ALEATORIOS[:400], cfg)
    print(linha_tabela(cfg, x))
    nome = "retangular (troca de fase instantânea)" if pulso == "ret" else "cosseno levantado α=0,5 (troca suave)"
    b = resumo_banda(x, cfg)
    plot_fft(ax[i], x, cfg, f"Pulso {nome} — 99% da potência em {b['99%']:.0f} Hz", cor=cor, db=True,
             zoom=(2000, 6000))
fig.suptitle("Trocas de fase abruptas espalham energia longe da portadora; pulsos suaves confinam a banda",
             fontsize=12)
fig.tight_layout(); fig.savefig("figuras/psk_d_pulsos.png", dpi=110)
print("  -> com pulso retangular os lóbulos laterais caem devagar (1/f): 99% da potência só cabe em")
print("     ~20× a taxa. Com cosseno levantado a banda é exatamente (1+α)·taxa (use --pulso rc).")

# ============================================================================ E) emitido x recebido
print("\n" + "=" * 82)
print("E) SINAL EMITIDO x RECEBIDO")
print("=" * 82)
cfg = Config()
try:
    tx = le_audio(args.tx, duracao_max=600)
    origem_tx = args.tx
except Exception:
    tx, _ = modula_texto(args.mensagem, cfg)
    origem_tx = "gerado agora"
if args.rx:
    rx = le_audio(args.rx, duracao_max=600)
    origem_rx = args.rx
else:
    rx = simula_canal(tx, snr_db=10, desvio_ppm=80)
    origem_rx = "canal simulado (atraso, eco, ruído, 60 Hz, 80 ppm de relógio)"
quadros = demodula(rx, cfg)
for q in quadros:
    print(f"  decodificado: {q.texto!r} | CRC {'ok' if q.ok else 'FALHOU'} | correlação {q.correlacao:.3f} | "
          f"fase {q.fase_graus:+.1f}° | desvio {q.desvio_hz:+.2f} Hz | SNR {q.snr_db:.1f} dB")
rx_filtrado, _ = converte_para_banda_base(rx, cfg)
btx, brx = resumo_banda(tx, cfg), resumo_banda(rx_filtrado, cfg)
print(f"  emitido : 90% da potência em {btx['90%']:.0f} Hz, 99% em {btx['99%']:.0f} Hz ({origem_tx})")
print(f"  recebido após o passa-faixa do receptor ({cfg.fc - cfg.banda/2:.0f}–{cfg.fc + cfg.banda/2:.0f} Hz):"
      f" 90% em {brx['90%']:.0f} Hz, 99% em {brx['99%']:.0f} Hz")
print("  (o sinal bruto do microfone tem ruído em todas as frequências, por isso o receptor filtra")
print("   só a banda necessária antes de demodular)")
fig = plt.figure(figsize=(13, 9))
g = fig.add_gridspec(2, 2, width_ratios=[2.2, 1])
plot_fft(fig.add_subplot(g[0, 0]), tx, cfg, f"FFT do sinal EMITIDO ({origem_tx})", db=True)
plot_fft(fig.add_subplot(g[1, 0]), rx, cfg, f"FFT do sinal RECEBIDO ({origem_rx})", cor="gray", db=True)
ax_c = fig.add_subplot(g[:, 1])
q = next((q for q in quadros if len(q.simbolos)), None)
if q is not None:
    plot_constelacao(ax_c, q.simbolos, f"Constelação recebida\nSNR ≈ {q.snr_db:.1f} dB")
fig.suptitle(f"Emitido x recebido — {cfg.descricao()}", fontsize=11)
fig.tight_layout(); fig.savefig("figuras/psk_e_emitido_recebido.png", dpi=110)

print("\nFiguras salvas em figuras/psk_*.png")
if not args.sem_janela:
    plt.show()
