"""Projeto 8 parte 2 — RECEPTOR do chat PSK.

Fica gravando o microfone continuamente. A cada 0,4 s procura o preâmbulo no áudio acumulado
(passa-faixa -> I/Q -> passa-baixa -> correlação com o preâmbulo). Ao encontrar um quadro, lê o
tamanho da mensagem, espera o áudio do quadro inteiro chegar, decide os bits, confere o CRC
e escreve a frase recebida no prompt.

Uso:
    python receptor.py                          # escuta o microfone (100 bit/s, portadora 4 kHz)
    python receptor.py --taxa 200 --fc 3000     # mesmos parâmetros usados no emissor!
    python receptor.py -v                       # mostra nível do sinal, fase, SNR, bits...
    python receptor.py --plot                   # abre a FFT do sinal recebido a cada mensagem
    python receptor.py --arquivo audios/psk_tx.wav --plot    # demodula um arquivo WAV
"""
import argparse
import queue
import sys
from dataclasses import replace

import numpy as np

from psk import N_ABERTURA, PN, Config, argumentos_comuns, config_de, demodula, margem_filtros

BLOCO = 2048                 # amostras por bloco lido do microfone (~46 ms)


class ReceptorContinuo:
    """Recebe o áudio em blocos, procura quadros e entrega cada trecho com quadro(s) completo(s)."""

    def __init__(self, cfg: Config, ao_receber, ao_detectar=None, passo=0.4, max_buffer=120):
        self.cfg = cfg
        self.ao_receber = ao_receber            # função(trecho de áudio) chamada a cada quadro completo
        self.ao_detectar = ao_detectar          # função() chamada quando um quadro começa a chegar
        self.passo = int(passo * cfg.fs)
        self.max_buffer = int(max_buffer * cfg.fs)
        self.margem = margem_filtros(cfg)
        # quanto áudio manter quando nada foi encontrado: o suficiente para um preâmbulo que esteja
        # "cortado" no fim do buffer ser encontrado inteiro na próxima busca
        self.manter = int((N_ABERTURA + len(PN) + 8) * cfg.spb) + self.margem + int(0.5 * cfg.fs)
        self.antes = self.margem + int((N_ABERTURA + 2) * cfg.spb)  # áudio guardado antes de um preâmbulo
        self.buf = np.zeros(0)
        self.novo = 0
        self.espera_ate = 0
        self.recebendo = False

    def alimenta(self, bloco):
        """Acrescenta um bloco de áudio. Retorna True se fez uma busca nesta chamada."""
        self.buf = np.concatenate([self.buf, bloco])
        self.novo += len(bloco)
        if self.novo < self.passo or len(self.buf) < self.espera_ate:
            return False
        self.novo = 0
        self._busca()
        return True

    def _busca(self):
        buf, cfg = self.buf, self.cfg
        quadros = demodula(buf, cfg)
        cortar = max(0, len(buf) - self.manter)      # padrão: descarta áudio antigo sem nada
        self.espera_ate = 0
        for q in quadros:
            ini = int(round(q.inicio * cfg.fs))
            if q.incompleto:                          # quadro ainda chegando: guarda desde o início dele
                if not self.recebendo and self.ao_detectar:
                    self.ao_detectar()
                self.recebendo = True
                cortar = max(0, ini - self.antes)
                self.espera_ate = q.fim + self.margem - cortar
                break
            a = max(0, ini - self.antes)
            b = min(len(buf), q.fim + self.margem + int(0.1 * cfg.fs))
            self.ao_receber(buf[a:b])
            self.recebendo = False
            cortar = q.fim
        if len(buf) - cortar > self.max_buffer:
            cortar = len(buf) - self.manter
        self.buf = buf[cortar:]


def nivel_na_banda_db(bloco, cfg: Config):
    """Densidade de potência dentro da banda da portadora / densidade típica (mediana) fora dela."""
    X = np.abs(np.fft.rfft(bloco * np.hanning(len(bloco)))) ** 2
    f = np.fft.rfftfreq(len(bloco), 1 / cfg.fs)
    banda = np.abs(f - cfg.fc) <= max(cfg.banda / 2, 50)
    fora = (f > 200) & (f < 12000) & (np.abs(f - cfg.fc) > 3 * cfg.banda)
    return 10 * np.log10(max(X[banda].mean() / max(np.median(X[fora]), 1e-20), 1e-3))


def mostra_detalhes(q):
    print(f"    preâmbulo em t = {q.inicio:.3f} s | correlação {q.correlacao:.3f} | fase da portadora "
          f"recebida {q.fase_graus:+.1f}° | desvio de frequência {q.desvio_hz:+.2f} Hz")
    if len(q.bits):
        print(f"    SNR ≈ {q.snr_db:.1f} dB | erros no preâmbulo: {q.erros_preambulo}/31 | {len(q.bits)} bits lidos "
              f"| bytes: {q.dados.hex(' ').upper()[:96]}")
        print(f"    bits (após o preâmbulo): {''.join(map(str, q.bits[31:31 + 72]))}{'...' if len(q.bits) > 103 else ''}")
        s = q.simbolos[31:37] / np.mean(np.abs(q.simbolos.real))
        print("    primeiros símbolos (I+jQ): " + "  ".join(f"{v.real:+.2f}{v.imag:+.2f}j" for v in s))


def main():
    p = argumentos_comuns(argparse.ArgumentParser(description=__doc__,
                                                  formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--arquivo", help="demodula um arquivo de áudio em vez do microfone")
    p.add_argument("-v", "--verbose", action="store_true", help="mostra detalhes da demodulação")
    p.add_argument("--plot", action="store_true", help="abre a figura do sinal recebido após cada mensagem")
    p.add_argument("--wav", default="audios/psk_rx.wav", help="onde salvar o último áudio recebido")
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # acentos/emoji no terminal do Windows
    cfg = config_de(args)
    if not args.arquivo:
        # grava na taxa nativa do microfone (48 kHz em muitos notebooks) e demodula nessa mesma taxa:
        # evita a conversão de taxa do driver, que pode introduzir falhas no áudio
        import sounddevice as sd
        cfg = replace(cfg, fs=int(sd.query_devices(args.dispositivo, "input")["default_samplerate"]))

    if not args.plot:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from graficos_psk import figura_recebido, inicializa_processo_sem_janela, salva_figura_recebido
    from utils import le_audio, salva_wav

    # A figura é gerada num processo separado: o matplotlib demora ~2 s e, se rodasse aqui, o
    # callback do microfone ficaria sem executar e amostras seriam perdidas no meio da mensagem.
    graficos = None
    if not args.plot and not args.arquivo:
        from concurrent.futures import ProcessPoolExecutor
        graficos = ProcessPoolExecutor(max_workers=1, initializer=inicializa_processo_sem_janela)

    def processa(x, origem):
        quadros, extras = demodula(x, cfg, detalhes=True)
        if not args.arquivo:
            salva_wav(args.wav, x, fs=cfg.fs)
        if not quadros:
            print(f"  (nenhum preâmbulo encontrado em {len(x) / cfg.fs:.1f} s de {origem})")
            return
        texto = "".join(q.texto for q in quadros if q.ok)
        if texto:
            print(f"\noutro> {texto}", flush=True)
        for q in quadros:
            if not q.ok:
                print(f"  ✗ quadro com erro: {q.motivo}" + (f" — texto parcial: {q.texto!r}" if q.texto else ""))
            if args.verbose or args.arquivo:
                mostra_detalhes(q)
        if graficos is not None:
            graficos.submit(salva_figura_recebido, x, cfg, "figuras/psk_recebido.png")
            return
        fig = figura_recebido(x, cfg, quadros, extras)
        fig.savefig("figuras/psk_recebido.png", dpi=110)
        if args.plot:
            plt.show()
        plt.close(fig)

    print("=" * 72)
    print(" RECEPTOR PSK —", cfg.descricao())
    print("=" * 72)

    if args.arquivo:
        processa(le_audio(args.arquivo, duracao_max=600), args.arquivo)
        return

    fila = queue.Queue()

    def callback(indata, frames, tempo, status):
        if status.input_overflow:
            print("\n⚠ amostras do microfone perdidas (receptor ocupado demais)", flush=True)
        fila.put(indata[:, 0].copy())

    rx = ReceptorContinuo(cfg, ao_receber=lambda x: processa(x, "microfone"),
                          ao_detectar=lambda: print("\n📡 recebendo...", flush=True))
    nome = sd.query_devices(args.dispositivo, "input")["name"]
    print(f"🎧 Ouvindo {nome} a {cfg.fs} Hz... (Ctrl+C para sair)")
    try:
        with sd.InputStream(samplerate=cfg.fs, channels=1, blocksize=BLOCO, device=args.dispositivo,
                            latency="high", callback=callback):
            while True:
                buscou = rx.alimenta(fila.get())
                if buscou and args.verbose and not rx.recebendo:
                    db = nivel_na_banda_db(rx.buf[-rx.passo:], cfg)
                    barra = "#" * int(np.clip(db, 0, 40))
                    print(f"\r  nível na banda da portadora: {db:5.1f} dB [{barra:<40}] "
                          f"atraso {fila.qsize() * BLOCO / cfg.fs:4.1f} s", end="", flush=True)
    except KeyboardInterrupt:
        print("\nReceptor encerrado.")
    finally:
        if graficos is not None:
            graficos.shutdown(cancel_futures=True)


if __name__ == "__main__":
    main()
