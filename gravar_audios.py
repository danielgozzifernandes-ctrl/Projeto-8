"""Grava os 3 áudios do projeto (audios/audio1.wav, audio2.wav, audio3.wav) a 44100 Hz.

Uso:
    python gravar_audios.py               # grava 3 áudios de 4 s pelo microfone
    python gravar_audios.py --duracao 5   # muda a duração
    python gravar_audios.py --sintetico   # gera 3 áudios de teste (sem microfone)
"""
import argparse
import numpy as np
from utils import FS, tempo, salva_wav

p = argparse.ArgumentParser()
p.add_argument("--duracao", type=float, default=4.0)
p.add_argument("--sintetico", action="store_true")
args = p.parse_args()

if args.sintetico:
    # Três "estações" de teste: melodias com timbres diferentes
    t = tempo(args.duracao)
    notas = [[262, 330, 392, 523], [440, 392, 349, 330], [196, 247, 294, 247]]
    for i, seq in enumerate(notas, start=1):
        x = np.zeros_like(t)
        dur = args.duracao / len(seq)
        for k, f in enumerate(seq):
            msk = (t >= k * dur) & (t < (k + 1) * dur)
            tt = t[msk] - k * dur
            env = np.exp(-2 * tt / dur)
            x[msk] = env * sum(np.sin(2 * np.pi * f * h * tt) / h ** (i * 0.7) for h in range(1, 8))
        salva_wav(f"audios/audio{i}.wav", x)
        print(f"audios/audio{i}.wav gerado (sintético)")
else:
    import sounddevice as sd
    for i in range(1, 4):
        input(f"\nPressione ENTER para gravar o áudio {i} ({args.duracao:.0f} s)...")
        print("  ● Gravando...")
        x = sd.rec(int(args.duracao * FS), samplerate=FS, channels=1, dtype="float64")
        sd.wait()
        salva_wav(f"audios/audio{i}.wav", x[:, 0])
        print(f"  ✔ Salvo em audios/audio{i}.wav")
