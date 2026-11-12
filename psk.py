"""Projeto 8 parte 2 — Núcleo da modulação digital BPSK (usado por emissor, receptor e análise).

MODULAÇÃO (BPSK)
    bit 0 -> fase   0°  -> a = +1
    bit 1 -> fase 180°  -> a = -1
    s(t) = a(t) · cos(2π·fc·t)        (multiplicar por -1 é o mesmo que somar 180° à fase)
    Cada bit dura T = 1/taxa segundos (taxa = 100 bit/s -> T = 10 ms = 441 amostras a 44100 Hz).

QUADRO (o que é transmitido a cada mensagem)
    | ABERTURA 16 bits "0" | PREÂMBULO 31 bits (sequência PN) | TAM 8 bits | DADOS 8·TAM bits | CRC-16 | CAUDA 8 bits "0" |
    - ABERTURA: portadora pura, dá tempo do alto-falante/microfone "acordarem".
    - PREÂMBULO: sequência pseudoaleatória conhecida pelos dois lados. O receptor procura essa sequência
      (correlação) para descobrir ONDE o quadro começa e QUAL a fase da portadora recebida.
    - TAM: número de bytes da mensagem (texto em UTF-8, máx. 255 bytes por quadro).
    - CRC-16 (CCITT): detecta se algum bit chegou errado.

DEMODULAÇÃO (coerente, como na demodulação síncrona do AM)
    1. passa-faixa em torno de fc (só a banda necessária entra no receptor)
    2. multiplica por cos e por sen da portadora local (I/Q) + passa-baixa  ->  z(t) = I + jQ
       (o receptor não conhece a fase da portadora que chegou, então usa as duas componentes)
    3. correlação de z(t) com o preâmbulo -> início do quadro, fase θ e desvio de frequência
    4. para cada bit: média de z no centro do bit, girada de -θ; bit = 1 se a parte real < 0.
       Um laço de fase (PLL dirigido por decisão) corrige θ a cada bit, compensando pequenas diferenças
       entre os relógios das placas de som dos dois computadores.
"""
from binascii import crc_hqx
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy import signal

FS = 44100

N_ABERTURA = 16
PN = signal.max_len_seq(5)[0].astype(np.uint8)       # sequência-m de 31 bits (LFSR de 5 bits)
N_CAUDA = 8
LIMIAR_CORRELACAO = 0.6                               # correlação normalizada mínima para aceitar um preâmbulo
MAX_ERROS_PREAMBULO = 6                               # mais erros que isso nos 31 bits = falso alarme
MAX_BYTES = 255


@dataclass(frozen=True)
class Config:
    taxa: float = 100.0      # bits por segundo
    fc: float = 4000.0       # frequência da portadora (Hz)
    pulso: str = "ret"       # "ret" (retangular, PSK clássico) ou "rc" (cosseno levantado)
    alfa: float = 0.5        # roll-off do cosseno levantado
    fs: int = FS
    amplitude: float = 0.8

    def __post_init__(self):
        assert self.pulso in ("ret", "rc"), "pulso deve ser 'ret' ou 'rc'"
        assert self.fc - self.banda / 2 > 0 and self.fc + self.banda / 2 < self.fs / 2, \
            "a banda do sinal precisa caber entre 0 e fs/2"

    @property
    def T(self):
        """Duração de um bit (s)."""
        return 1 / self.taxa

    @property
    def spb(self):
        """Amostras por bit (pode ser fracionário, p. ex. 44100/400 = 110,25)."""
        return self.fs / self.taxa

    @property
    def banda(self):
        """Banda necessária em torno de fc (Hz).
        ret: lóbulo principal do espectro sinc², de fc - taxa a fc + taxa  ->  2·taxa
        rc : espectro limitado em fc ± (1+α)·taxa/2                        ->  (1+α)·taxa"""
        return 2 * self.taxa if self.pulso == "ret" else (1 + self.alfa) * self.taxa

    def descricao(self):
        p = "retangular" if self.pulso == "ret" else f"cosseno levantado (α={self.alfa})"
        return (f"BPSK | portadora {self.fc:.0f} Hz | {self.taxa:g} bit/s (T = {1000*self.T:g} ms, "
                f"{self.spb:g} amostras/bit) | pulso {p} | banda necessária {self.banda:.0f} Hz "
                f"({self.fc - self.banda/2:.0f}–{self.fc + self.banda/2:.0f} Hz)")


# ============================================================================ bits <-> bytes
def bytes_para_bits(dados: bytes) -> np.ndarray:
    """MSB primeiro: 'A' = 0x41 -> 0 1 0 0 0 0 0 1."""
    return np.unpackbits(np.frombuffer(dados, dtype=np.uint8))


def bits_para_bytes(bits) -> bytes:
    return np.packbits(np.asarray(bits, dtype=np.uint8)).tobytes()


def crc16(dados: bytes) -> int:
    return crc_hqx(dados, 0xFFFF)  # CRC-16-CCITT (polinômio 0x1021)


def monta_quadro(dados: bytes) -> np.ndarray:
    """Bits de um quadro completo (abertura + preâmbulo + tamanho + dados + CRC + cauda)."""
    assert len(dados) <= MAX_BYTES
    cabecalho_e_dados = bytes([len(dados)]) + dados
    crc = crc16(cabecalho_e_dados).to_bytes(2, "big")
    return np.concatenate([
        np.zeros(N_ABERTURA, np.uint8),
        PN,
        bytes_para_bits(cabecalho_e_dados + crc),
        np.zeros(N_CAUDA, np.uint8),
    ])


def divide_mensagem(texto: str):
    """Texto -> lista de blocos de até 255 bytes (UTF-8), sem quebrar caracteres multibyte."""
    blocos, atual = [], b""
    for ch in texto:
        b = ch.encode("utf-8")
        if len(atual) + len(b) > MAX_BYTES:
            blocos.append(atual); atual = b""
        atual += b
    return blocos + [atual] if atual else blocos


# ============================================================================ modulação
def pulso_rc(cfg: Config, span=6):
    """Cosseno levantado amostrado em ±span bits (resposta ao impulso)."""
    t = np.arange(-span * cfg.spb, span * cfg.spb + 1) / cfg.fs / cfg.T   # tempo em unidades de T
    a = cfg.alfa
    den = 1 - (2 * a * t) ** 2
    sing = np.isclose(den, 0)
    p = np.sinc(t) * np.cos(np.pi * a * t) / np.where(sing, 1, den)
    p[sing] = np.pi / 4 * np.sinc(1 / (2 * a))
    return p


def banda_base(bits, cfg: Config) -> np.ndarray:
    """Sinal a(t) em banda base: +1 para bit 0 e -1 para bit 1, com o formato de pulso escolhido."""
    a = 1.0 - 2.0 * np.asarray(bits, dtype=float)
    n = int(round(len(a) * cfg.spb))
    if cfg.pulso == "ret":
        idx = np.minimum((np.arange(n) / cfg.spb).astype(int), len(a) - 1)
        return a[idx]
    impulsos = np.zeros(n)
    centros = np.round((np.arange(len(a)) + 0.5) * cfg.spb).astype(int)
    impulsos[np.minimum(centros, n - 1)] = a
    return signal.oaconvolve(impulsos, pulso_rc(cfg), mode="same")


def modula_bits(bits, cfg: Config) -> np.ndarray:
    a = banda_base(bits, cfg)
    t = np.arange(len(a)) / cfg.fs
    s = a * np.cos(2 * np.pi * cfg.fc * t)
    rampa = int(0.005 * cfg.fs)                     # 5 ms de fade in/out para não dar "clique"
    janela = np.ones_like(s)
    janela[:rampa] = 0.5 - 0.5 * np.cos(np.pi * np.arange(rampa) / rampa)
    janela[-rampa:] = janela[:rampa][::-1]
    s *= janela
    return cfg.amplitude * s / max(np.max(np.abs(s)), 1e-12)


def modula_texto(texto: str, cfg: Config, silencio=0.15):
    """Texto -> (sinal de áudio, lista de quadros em bits). Mensagens > 255 bytes viram vários quadros."""
    quadros = [monta_quadro(b) for b in divide_mensagem(texto)]
    pausa = np.zeros(int(silencio * cfg.fs))
    partes = [pausa]
    for q in quadros:
        partes += [modula_bits(q, cfg), pausa]
    return np.concatenate(partes), quadros


# ============================================================================ filtros (FIR fase zero)
def _fir(x, h):
    return signal.oaconvolve(x, h, mode="same")  # FIR simétrico + modo 'same' => atraso compensado


def _ntaps(transicao, fs, atenuacao_db=60):
    n, beta = signal.kaiserord(atenuacao_db, transicao / (fs / 2))
    return n | 1, beta


@lru_cache(maxsize=16)
def filtro_passa_faixa(cfg: Config):
    trans = max(0.5 * cfg.taxa, 40)
    n, beta = _ntaps(trans, cfg.fs)
    meia = cfg.banda / 2 + trans / 2
    return signal.firwin(n, [cfg.fc - meia, cfg.fc + meia], pass_zero=False,
                         window=("kaiser", beta), fs=cfg.fs)


@lru_cache(maxsize=16)
def filtro_passa_baixa(cfg: Config):
    trans = max(0.5 * cfg.taxa, 40)
    n, beta = _ntaps(trans, cfg.fs)
    return signal.firwin(n, cfg.banda / 2 + trans / 2, window=("kaiser", beta), fs=cfg.fs)


# ============================================================================ demodulação
@dataclass
class Quadro:
    ok: bool                  # CRC confere
    texto: str
    dados: bytes
    inicio: float             # instante (s) do início do preâmbulo no áudio recebido
    correlacao: float         # pico da correlação normalizada com o preâmbulo (0–1)
    fase_graus: float         # fase da portadora recebida estimada no preâmbulo
    desvio_hz: float          # diferença de frequência entre as portadoras do emissor e do receptor
    snr_db: float             # SNR estimado a partir da dispersão dos símbolos
    erros_preambulo: int      # bits errados no preâmbulo (conhecido)
    bits: np.ndarray = field(repr=False)
    simbolos: np.ndarray = field(repr=False)   # valores complexos de cada bit (constelação)
    motivo: str = ""
    incompleto: bool = False  # o áudio acabou antes do fim do quadro (o receptor espera chegar mais)
    fim: int = 0              # amostra onde o quadro termina


def margem_filtros(cfg: Config):
    """Amostras no fim do áudio que ainda estão "contaminadas" pela borda dos filtros FIR."""
    return (len(filtro_passa_faixa(cfg)) + len(filtro_passa_baixa(cfg))) // 2


def converte_para_banda_base(x, cfg: Config):
    """Passa-faixa -> multiplica por 2·e^{-j2πfct} (= 2cos - j2sen) -> passa-baixa. Retorna (r, z)."""
    r = _fir(x, filtro_passa_faixa(cfg))
    t = np.arange(len(r)) / cfg.fs
    z = _fir(r * 2 * np.exp(-2j * np.pi * cfg.fc * t), filtro_passa_baixa(cfg))
    return r, z


def modelo_preambulo(cfg: Config):
    """Forma de onda (banda base) esperada para o preâmbulo PN."""
    bits = np.concatenate([np.zeros(N_ABERTURA, np.uint8), PN, np.zeros(8, np.uint8)])
    bb = banda_base(bits, cfg)
    i0 = int(round(N_ABERTURA * cfg.spb))
    return bb[i0: i0 + int(round(len(PN) * cfg.spb))]


N_SUBBLOCOS = 4


def correlacao_normalizada(z, p, m=N_SUBBLOCOS):
    """Correlação do sinal recebido com o preâmbulo, normalizada em [0, 1].

    O preâmbulo é dividido em m sub-blocos e somam-se os MÓDULOS das correlações de cada um
    (correlação não coerente). Assim, se a portadora do emissor estiver levemente fora de frequência
    (relógios diferentes), a fase que gira ao longo do preâmbulo não destrói o pico.
    Retorna (ρ, c) com c[τ] = vetor das m correlações complexas (usado para estimar fase e desvio)."""
    L = len(p)
    if len(z) < L:
        return np.zeros(0), np.zeros((0, m), complex)
    n = len(z) - L + 1
    lb = L // m
    c = np.empty((n, m), complex)
    for k in range(m):
        pk = p[k * lb:(k + 1) * lb] if k < m - 1 else p[k * lb:]
        ck = signal.correlate(z, pk, mode="valid", method="fft")
        c[:, k] = ck[k * lb: k * lb + n]
    e = np.concatenate([[0], np.cumsum(np.abs(z) ** 2)])
    ez = e[L:] - e[:-L]
    rho = np.abs(c).sum(axis=1) / (np.linalg.norm(p) * np.sqrt(np.maximum(ez, 1e-20)))
    return rho, c


def _le_simbolos(z, inicio, n_bits, cfg, theta, nu, conhecidos=None, estado=None, limite=None):
    """Lê n_bits a partir da amostra 'inicio' com PLL dirigido por decisão. Retorna (bits, símbolos, θ, ν)."""
    larg = 0.3 if cfg.pulso == "ret" else 0.1          # meia-largura da janela de média (fração do bit)
    kp, ki = 0.15, 0.01                                # ganhos do laço de fase
    bits, simb = [], []
    k0 = 0 if estado is None else estado
    for k in range(n_bits):
        c = inicio + (k0 + k + 0.5) * cfg.spb
        a, b = int(round(c - larg * cfg.spb)), int(round(c + larg * cfg.spb)) + 1
        if b > (len(z) if limite is None else limite):
            return None
        y = np.mean(z[a:b]) * np.exp(-1j * theta)
        if conhecidos is not None:
            d = 1.0 - 2.0 * conhecidos[k]
        else:
            d = 1.0 if y.real >= 0 else -1.0
        erro = np.angle(y * d)
        theta += kp * erro + nu
        nu += ki * erro
        bits.append(0 if y.real >= 0 else 1)
        simb.append(y)
    return np.array(bits, np.uint8), np.array(simb), theta, nu


def decodifica_quadro(z, tau, rho_pico, c_pico, cfg: Config):
    """Decodifica um quadro cujo preâmbulo começa na amostra tau."""
    p = modelo_preambulo(cfg)
    L = len(p)
    lb = L // len(c_pico)
    # desvio de frequência: quanto a fase gira de um sub-bloco do preâmbulo para o seguinte
    dphi = np.angle(np.sum(c_pico[1:] * np.conj(c_pico[:-1])))
    desvio = dphi / (2 * np.pi * lb / cfg.fs)
    # fase da portadora recebida no início do preâmbulo (descontando o giro devido ao desvio)
    t_blocos = (np.arange(len(c_pico)) + 0.5) * lb / cfg.fs
    fase0 = np.angle(np.sum(c_pico * np.exp(-2j * np.pi * desvio * t_blocos)))
    nu = 2 * np.pi * desvio * cfg.T                       # rotação de fase por bit
    theta = fase0 + nu * 0.5                              # fase no centro do 1º bit do preâmbulo

    limite = len(z) - margem_filtros(cfg)

    def falha(motivo, fim, **kw):
        return Quadro(False, "", b"", tau / cfg.fs, rho_pico, np.degrees(fase0), desvio,
                      np.nan, kw.get("erros", -1), np.zeros(0, np.uint8), np.zeros(0, complex), motivo,
                      incompleto=True, fim=fim)

    r = _le_simbolos(z, tau, len(PN), cfg, theta, nu, conhecidos=PN, limite=limite)
    if r is None:
        return falha("áudio terminou no meio do preâmbulo", tau + L), tau + L
    b_pre, s_pre, theta, nu = r
    erros_pre = int(np.sum(b_pre != PN))
    if erros_pre > MAX_ERROS_PREAMBULO:       # não era o preâmbulo (falso alarme da correlação)
        return None, tau

    r = _le_simbolos(z, tau, 8, cfg, theta, nu, estado=len(PN), limite=limite)
    if r is None:
        return falha("áudio terminou antes do tamanho", tau + L, erros=erros_pre), tau + L
    b_tam, s_tam, theta, nu = r
    tam = int(bits_para_bytes(b_tam)[0])

    n_resto = 8 * tam + 16
    r = _le_simbolos(z, tau, n_resto, cfg, theta, nu, estado=len(PN) + 8, limite=limite)
    fim = tau + int((len(PN) + 8 + n_resto + N_CAUDA) * cfg.spb)
    if r is None:
        return falha(f"áudio terminou antes do fim do quadro (tamanho lido = {tam} bytes)", fim,
                     erros=erros_pre), fim
    b_resto, s_resto, theta, nu = r

    bits = np.concatenate([b_pre, b_tam, b_resto])
    simb = np.concatenate([s_pre, s_tam, s_resto])
    dados = bits_para_bytes(b_resto[:8 * tam])
    crc_rx = int.from_bytes(bits_para_bytes(b_resto[8 * tam:]), "big")
    ok = crc_rx == crc16(bytes([tam]) + dados)

    ref = 1.0 - 2.0 * bits                                     # decisões como ±1
    proj = (simb * ref).real
    amp = np.mean(proj)
    ruido = np.mean(np.abs(simb * ref - amp) ** 2)
    snr = 10 * np.log10(amp ** 2 / max(ruido, 1e-20))
    quadro = Quadro(ok, dados.decode("utf-8", errors="replace"), dados, tau / cfg.fs, rho_pico,
                    np.degrees(fase0), desvio, snr, erros_pre, bits, simb,
                    "" if ok else "CRC não confere (bits corrompidos)", fim=fim)
    return quadro, fim


def demodula(x, cfg: Config, detalhes=False):
    """Procura e decodifica todos os quadros presentes no áudio x.
    Retorna lista de Quadro (ou (lista, dict com sinais intermediários) se detalhes=True)."""
    x = np.asarray(x, dtype=float)
    r, z = converte_para_banda_base(x, cfg)
    p = modelo_preambulo(cfg)
    rho, c = correlacao_normalizada(z, p)
    quadros, pos = [], 0
    while pos < len(rho):
        acima = np.flatnonzero(rho[pos:] > LIMIAR_CORRELACAO)
        if len(acima) == 0:
            break
        i = pos + acima[0]
        j = min(i + len(p), len(rho))
        tau = i + int(np.argmax(rho[i:j]))                    # pico local = alinhamento exato
        q, fim = decodifica_quadro(z, tau, float(rho[tau]), c[tau], cfg)
        if q is None:
            pos = tau + max(1, int(cfg.spb / 2))
            continue
        quadros.append(q)
        if q.incompleto:
            break
        pos = max(fim, tau + len(p)) if q.ok else tau + len(p)
    if detalhes:
        return quadros, {"r": r, "z": z, "rho": rho}
    return quadros


# ============================================================================ espectro / banda
def espectro(x, fs=FS):
    """(f, |X(f)|) normalizado como no projeto AM."""
    X = np.fft.rfft(x)
    return np.fft.rfftfreq(len(x), 1 / fs), 2 * np.abs(X) / len(x)


def banda_ocupada(x, fc, fracao, fs=FS):
    """Menor largura B tal que [fc - B/2, fc + B/2] contém 'fracao' da potência do sinal."""
    f, X = espectro(x, fs)
    pot = X ** 2
    total = pot.sum()
    ordem = np.argsort(np.abs(f - fc))
    acumulado = np.cumsum(pot[ordem]) / total
    k = np.searchsorted(acumulado, fracao)
    return 2 * np.abs(f[ordem[min(k, len(f) - 1)]] - fc)


def resumo_banda(x, cfg: Config):
    return {
        "teorica": cfg.banda,
        "90%": banda_ocupada(x, cfg.fc, 0.90, cfg.fs),
        "99%": banda_ocupada(x, cfg.fc, 0.99, cfg.fs),
    }


# ============================================================================ canal simulado
def simula_canal(x, fs=FS, atraso=0.3, atenuacao=0.2, snr_db=15.0, desvio_ppm=60.0, eco=0.3,
                 atraso_eco=0.004, semente=0):
    """Simula alto-falante -> ar -> microfone: atraso, atenuação, eco, diferença de relógio
    entre as placas de som (reamostragem de 'desvio_ppm' partes por milhão) e ruído branco."""
    rng = np.random.default_rng(semente)
    y = np.concatenate([np.zeros(int(atraso * fs)), x, np.zeros(int(0.3 * fs))])
    d = int(atraso_eco * fs)
    y = y + eco * np.concatenate([np.zeros(d), y[:-d]])
    if desvio_ppm:
        n = int(round(len(y) * (1 + desvio_ppm * 1e-6)))
        y = np.interp(np.arange(n) * len(y) / n, np.arange(len(y)), y)
    y = atenuacao * y
    p_sinal = np.mean(y[np.abs(y) > 1e-6] ** 2) if np.any(np.abs(y) > 1e-6) else 1.0
    y += rng.normal(0, np.sqrt(p_sinal / 10 ** (snr_db / 10)), len(y))
    y += 0.02 * atenuacao * np.sin(2 * np.pi * 60 * np.arange(len(y)) / fs)   # zumbido da rede
    return y


# ============================================================================ linha de comando
def argumentos_comuns(parser):
    """Parâmetros que emissor e receptor precisam combinar entre si."""
    parser.add_argument("--taxa", type=float, default=100, help="taxa de transmissão em bit/s (padrão 100)")
    parser.add_argument("--fc", type=float, default=4000, help="frequência da portadora em Hz (2000–6000, padrão 4000)")
    parser.add_argument("--pulso", choices=["ret", "rc"], default="ret",
                        help="formato do pulso: ret = retangular (padrão), rc = cosseno levantado")
    parser.add_argument("--alfa", type=float, default=0.5, help="roll-off do cosseno levantado (padrão 0,5)")
    parser.add_argument("--dispositivo", type=int, default=None,
                        help="índice do dispositivo de áudio (veja: python -m sounddevice)")
    return parser


def config_de(args) -> Config:
    if not 2000 <= args.fc <= 6000:
        print(f"⚠ Atenção: o enunciado pede portadora entre 2 e 6 kHz (fc = {args.fc:g} Hz)")
    return Config(taxa=args.taxa, fc=args.fc, pulso=args.pulso, alfa=args.alfa)
