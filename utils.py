"""Funções auxiliares do Projeto 8 - Modulação AM (Camada Física da Computação)."""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

FS = 44100  # taxa de amostragem (Hz)


# ---------------------------------------------------------------- sinais
def tempo(duracao, fs=FS):
    return np.arange(int(duracao * fs)) / fs


def senoide(freq, t, amp=1.0, fase=0.0):
    return amp * np.sin(2 * np.pi * freq * t + fase)


def normaliza(x):
    x = x - np.mean(x)
    pico = np.max(np.abs(x))
    return x / pico if pico > 0 else x


# ---------------------------------------------------------------- Fourier
def fourier(x, fs=FS):
    """Retorna (frequências, |X(f)|) apenas para f >= 0, com amplitude normalizada."""
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / fs)
    return f, 2 * np.abs(X) / n


def plot_fourier(ax, x, titulo, fs=FS, faixa=None, xlim=(0, 22050), cor="tab:blue"):
    f, X = fourier(x, fs)
    ax.plot(f, X, color=cor, lw=0.8)
    if faixa is not None:
        ax.axvspan(faixa[0], faixa[1], color="tab:green", alpha=0.15, label="banda permitida")
        ax.legend(loc="upper right")
    ax.set_xlim(*xlim)
    ax.set_title(titulo)
    ax.set_xlabel("Frequência (Hz)")
    ax.set_ylabel("|X(f)|")
    ax.grid(alpha=0.3)


def plot_tempo(ax, t, x, titulo, janela=None, cor="tab:blue"):
    if janela is not None:
        m = (t >= janela[0]) & (t <= janela[1])
        t, x = t[m], x[m]
    ax.plot(t * 1000, x, color=cor, lw=0.8)
    ax.set_title(titulo)
    ax.set_xlabel("Tempo (ms)")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.3)


# ---------------------------------------------------------------- filtros (FIR, janela de Kaiser)
def _ntaps(transicao, atenuacao_db, fs):
    n, beta = signal.kaiserord(atenuacao_db, transicao / (fs / 2))
    return n | 1, beta  # número ímpar de coeficientes


def passa_baixa(x, f_corte, transicao=300, atenuacao_db=80, fs=FS):
    """Passa-baixa FIR: banda passante até f_corte, rejeição a partir de f_corte + transicao."""
    n, beta = _ntaps(transicao, atenuacao_db, fs)
    h = signal.firwin(n, f_corte + transicao / 2, window=("kaiser", beta), fs=fs)
    return signal.filtfilt(h, 1.0, x)


def passa_faixa(x, f_baixa, f_alta, transicao=300, atenuacao_db=80, fs=FS):
    """Passa-faixa FIR: banda passante [f_baixa, f_alta], rejeição fora de [f_baixa-transicao, f_alta+transicao]."""
    n, beta = _ntaps(transicao, atenuacao_db, fs)
    h = signal.firwin(n, [f_baixa - transicao / 2, f_alta + transicao / 2],
                      window=("kaiser", beta), pass_zero=False, fs=fs)
    return signal.filtfilt(h, 1.0, x)


# ---------------------------------------------------------------- análise de banda
def energia_fora_da_banda(x, f_min, f_max, fs=FS):
    """Fração (%) da energia espectral de x fora do intervalo [f_min, f_max]."""
    f, X = fourier(x, fs)
    e = X ** 2
    fora = e[(f < f_min) | (f > f_max)].sum()
    return 100 * fora / e.sum()


def faixa_ocupada(x, limiar_db=-40, fs=FS):
    """Menor e maior frequência cujo módulo está acima de (pico + limiar_db)."""
    f, X = fourier(x, fs)
    Xdb = 20 * np.log10(X / X.max() + 1e-15)
    acima = f[Xdb > limiar_db]
    return acima.min(), acima.max()


# ---------------------------------------------------------------- áudio
def le_audio(caminho, duracao_max=5.0, fs_alvo=FS):
    """Lê wav/mp3/flac/ogg, converte para mono, reamostra para fs_alvo e corta em duracao_max segundos."""
    import soundfile as sf
    from math import gcd
    x, fs = sf.read(caminho, dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    if fs != fs_alvo:
        g = gcd(int(fs), int(fs_alvo))
        x = signal.resample_poly(x, fs_alvo // g, int(fs) // g)
    return x[: int(duracao_max * fs_alvo)]


def salva_wav(caminho, x, fs=FS):
    wavfile.write(caminho, fs, (0.9 * normaliza(x) * 32767).astype(np.int16))


def toca(x, fs=FS, descricao=""):
    try:
        import sounddevice as sd
    except ImportError:
        print("  (sounddevice não instalado — áudio não reproduzido)")
        return
    if descricao:
        print(f"  ▶ Tocando: {descricao}")
    sd.play(0.9 * normaliza(x), fs)
    sd.wait()
