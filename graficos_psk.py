"""Gráficos do Projeto 8 parte 2 (PSK): sinal no tempo, FFT com a banda necessária e constelação."""
import numpy as np
import matplotlib.pyplot as plt

from psk import Config, espectro, resumo_banda, banda_base, N_ABERTURA


def marca_banda(ax, cfg: Config, rotulo=True):
    ax.axvspan(cfg.fc - cfg.banda / 2, cfg.fc + cfg.banda / 2, color="tab:green", alpha=0.15,
               label=f"banda necessária ({cfg.banda:.0f} Hz)" if rotulo else None)
    ax.axvline(cfg.fc, color="tab:red", ls="--", lw=0.8, label=f"portadora {cfg.fc:.0f} Hz" if rotulo else None)


def plot_fft(ax, x, cfg: Config, titulo, cor="tab:blue", db=False, zoom=None):
    f, X = espectro(x, cfg.fs)
    if db:
        Y = 20 * np.log10(X / X.max() + 1e-12)
        ax.plot(f, Y, color=cor, lw=0.6)
        ax.set_ylim(-80, 5)
        ax.set_ylabel("|X(f)| (dB)")
    else:
        ax.plot(f, X, color=cor, lw=0.7)
        ax.set_ylabel("|X(f)|")
    marca_banda(ax, cfg)
    ax.set_xlim(*(zoom or (0, cfg.fs / 2)))
    ax.set_xlabel("Frequência (Hz)")
    ax.set_title(titulo)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)


def plot_tempo_bits(ax, x, cfg: Config, inicio=0.0, n_bits=12, cor="tab:blue", titulo=None, bits=None):
    """Mostra n_bits do sinal a partir de 'inicio' (s), com as fronteiras dos bits."""
    a, b = int(inicio * cfg.fs), int((inicio + n_bits * cfg.T) * cfg.fs)
    t = np.arange(a, min(b, len(x))) / cfg.fs
    ax.plot((t - inicio) * 1000, x[a:a + len(t)], color=cor, lw=0.8)
    for k in range(n_bits + 1):
        ax.axvline(k * cfg.T * 1000, color="gray", lw=0.5, ls=":")
    if bits is not None:
        topo = np.max(np.abs(x[a:a + len(t)])) if len(t) else 1
        for k, bit in enumerate(bits[:n_bits]):
            ax.text((k + 0.5) * cfg.T * 1000, 1.08 * topo, str(bit), ha="center", fontsize=9,
                    color="tab:red" if bit else "tab:blue", fontweight="bold")
    ax.set_xlabel("Tempo (ms)")
    ax.set_ylabel("Amplitude")
    ax.set_title(titulo or f"{n_bits} bits no tempo (bit 0 = fase 0°, bit 1 = fase 180°)")
    ax.grid(alpha=0.3)


def plot_constelacao(ax, simbolos, titulo="Constelação dos bits recebidos"):
    s = np.asarray(simbolos)
    if len(s) == 0:
        ax.set_title(titulo + " (vazia)")
        return
    s = s / np.mean(np.abs(s.real))
    cores = np.where(s.real >= 0, "tab:blue", "tab:red")
    ax.scatter(s.real, s.imag, s=8, c=cores, alpha=0.6)
    ax.scatter([1, -1], [0, 0], marker="x", s=120, c="k", label="ideal: bit 0 (+1) / bit 1 (−1)")
    ax.axhline(0, color="gray", lw=0.5); ax.axvline(0, color="gray", lw=0.5)
    lim = max(1.6, np.max(np.abs(s)) * 1.1)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("I (em fase)"); ax.set_ylabel("Q (quadratura)")
    ax.set_title(titulo)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)


def inicio_do_preambulo_tx(x, cfg: Config):
    """Instante (s) em que começa o preâmbulo no sinal emitido (pula o silêncio inicial)."""
    nz = np.flatnonzero(np.abs(x) > 1e-3)
    return (nz[0] / cfg.fs if len(nz) else 0) + N_ABERTURA * cfg.T


def plot_inversao_de_fase(ax, x, cfg: Config, bits_quadro, inicio_quadro):
    """Zoom de ±4 ciclos da portadora em torno da primeira transição 0 -> 1: a fase inverte 180°."""
    k = next(i for i in range(1, len(bits_quadro)) if bits_quadro[i - 1] == 0 and bits_quadro[i] == 1)
    tb = inicio_quadro + k * cfg.T
    meia = 4 / cfg.fc
    a, b = int((tb - meia) * cfg.fs), int((tb + meia) * cfg.fs)
    t = (np.arange(a, b) / cfg.fs - tb) * 1000
    ax.plot(t, x[a:b], color="tab:blue", lw=1.2, marker=".", ms=3, label="s(t) emitido")
    ref = cfg.amplitude * np.cos(2 * np.pi * cfg.fc * np.arange(a, b) / cfg.fs)
    ax.plot(t, ref, color="gray", lw=0.8, ls="--", label="portadora cos(2πfc·t)")
    ax.axvline(0, color="tab:red", lw=1)
    ax.text(-meia * 500, 1.0, "bit 0: em fase", ha="center", color="tab:blue")
    ax.text(meia * 500, 1.0, "bit 1: invertido (180°)", ha="center", color="tab:red")
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("Tempo em torno da troca de bit (ms)")
    ax.set_title("Zoom na troca 0 → 1: inversão de fase de 180°")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)


def figura_emitido(x, cfg: Config, texto, bits_quadro):
    b = resumo_banda(x, cfg)
    fig = plt.figure(figsize=(13, 11))
    g = fig.add_gridspec(3, 2, width_ratios=[1.6, 1])
    a0 = fig.add_subplot(g[0, 0]); a1 = fig.add_subplot(g[0, 1])
    a2 = fig.add_subplot(g[1, :]); a3 = fig.add_subplot(g[2, :])
    inicio_quadro = inicio_do_preambulo_tx(x, cfg) - N_ABERTURA * cfg.T
    t0 = inicio_quadro + (N_ABERTURA - 2) * cfg.T
    plot_tempo_bits(a0, x, cfg, inicio=t0, n_bits=12, bits=bits_quadro[N_ABERTURA - 2:])
    bb = banda_base(bits_quadro, cfg) * cfg.amplitude
    i0 = int(round(inicio_quadro * cfg.fs))
    a, n = int(t0 * cfg.fs), int(12 * cfg.T * cfg.fs)
    a0.plot((np.arange(a, a + n) / cfg.fs - t0) * 1000, bb[a - i0:a - i0 + n], color="tab:red", lw=1.5,
            label="a(t): +1 (bit 0) / −1 (bit 1)")
    a0.legend(loc="lower right", fontsize=8)
    plot_inversao_de_fase(a1, x, cfg, bits_quadro, inicio_quadro)
    zoom = (cfg.fc - 8 * cfg.taxa, cfg.fc + 8 * cfg.taxa)
    plot_fft(a2, x, cfg, f"FFT do sinal emitido (zoom em fc ± 8·taxa) — 90% da potência em "
             f"{b['90%']:.0f} Hz, 99% em {b['99%']:.0f} Hz", zoom=zoom)
    plot_fft(a3, x, cfg, "FFT do sinal emitido (0 a 22 kHz, escala dB)", db=True)
    fig.suptitle(f"EMISSOR — \"{texto[:60]}\"\n{cfg.descricao()}", fontsize=11)
    fig.tight_layout()
    return fig


def figura_recebido(x, cfg: Config, quadros, extras):
    q = next((q for q in quadros if len(q.simbolos)), None)
    fig = plt.figure(figsize=(13, 10))
    g = fig.add_gridspec(3, 2, width_ratios=[2.2, 1])
    a1 = fig.add_subplot(g[0, :]); a2 = fig.add_subplot(g[1, 0]); a3 = fig.add_subplot(g[2, 0])
    a4 = fig.add_subplot(g[1:, 1])
    t = np.arange(len(x)) / cfg.fs
    a1.plot(t, x, lw=0.4, color="gray", label="recebido (microfone)")
    a1.plot(t, extras["r"], lw=0.4, color="tab:orange", label="após passa-faixa")
    for qq in quadros:
        a1.axvline(qq.inicio, color="tab:green" if qq.ok else "tab:red", lw=1.5)
    a1.set_xlabel("Tempo (s)"); a1.set_title("Sinal recebido no tempo (linha verde = preâmbulo encontrado)")
    leg = a1.legend(loc="upper right", fontsize=8); a1.grid(alpha=0.3)
    for linha in leg.get_lines():
        linha.set_linewidth(2)
    zoom = (cfg.fc - 8 * cfg.taxa, cfg.fc + 8 * cfg.taxa)
    b = resumo_banda(extras["r"], cfg)
    plot_fft(a2, x, cfg, "FFT do sinal recebido (dB) — ruído, zumbido e voz aparecem fora da banda",
             cor="gray", db=True)
    plot_fft(a3, extras["r"], cfg, f"FFT após o passa-faixa (zoom) — 90% em {b['90%']:.0f} Hz",
             cor="tab:orange", zoom=zoom)
    if q is not None:
        plot_constelacao(a4, q.simbolos, f"Constelação (SNR ≈ {q.snr_db:.1f} dB)")
    else:
        a4.set_title("Nenhum quadro encontrado")
    txt = " | ".join(f"\"{qq.texto[:40]}\" {'✓' if qq.ok else '✗'}" for qq in quadros) or "nada decodificado"
    fig.suptitle(f"RECEPTOR — {txt}\n{cfg.descricao()}", fontsize=11)
    fig.tight_layout()
    return fig


def salva_figura_recebido(x, cfg: Config, caminho):
    """Demodula e salva a figura do receptor (roda num processo separado, sem travar a captura de áudio)."""
    from psk import demodula
    quadros, extras = demodula(x, cfg, detalhes=True)
    fig = figura_recebido(x, cfg, quadros, extras)
    fig.savefig(caminho, dpi=110)
    plt.close(fig)


def inicializa_processo_sem_janela():
    import matplotlib
    matplotlib.use("Agg")
