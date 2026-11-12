# Projeto 8 — Modulação AM (parte 1) e PSK (parte 2)

Camada Física da Computação — Insper

## Estrutura

| Arquivo | Conteúdo |
|---|---|
| `utils.py` | FFT, filtros FIR (passa-baixa / passa-faixa, janela de Kaiser), leitura/escrita de WAV, reprodução |
| `exercicio1.py` | Soma informação + portadora → mostra que **não** é modulação |
| `exercicio3.py` | AM por multiplicação ponto a ponto (2 kHz × 15 kHz) |
| `exercicio5.py` | AM com áudio real: portadora 16 kHz, banda 14–18 kHz, modulação + demodulação |
| `gravar_audios.py` | Grava os 3 áudios do projeto pelo microfone |
| `projeto8.py` | **Projeto parte 1**: 3 estações AM, soma no meio físico, receptor com passa-faixa e demodulação |
| `psk.py` | **Parte 2**: núcleo BPSK — quadro, modulação, demodulação coerente, CRC, cálculo de banda, canal simulado |
| `emissor.py` | **Parte 2**: chat — digita a frase, ENTER, o áudio BPSK sai no alto-falante |
| `receptor.py` | **Parte 2**: chat — ouve o microfone, demodula e escreve a frase |
| `analise_psk.py` | **Parte 2**: FFT emitido × recebido, banda necessária, efeito da taxa, da portadora e do pulso |
| `graficos_psk.py` | Gráficos da parte 2 |
| `figuras/` | Gráficos gerados |
| `audios/` | Áudios originais, modulados e demodulados |

## Como executar

```bash
pip install numpy scipy matplotlib sounddevice soundfile

# Opção A: usar arquivos prontos (wav, mp3, flac, ogg; qualquer taxa — convertidos para 44100 Hz, máx. 5 s)
python projeto8.py musica.mp3 voz.wav efeito.flac

# Opção B: gravar pelo microfone em audios/audio1..3.wav
python gravar_audios.py
# (ou: python gravar_audios.py --sintetico   para gerar áudios de teste)

python exercicio1.py
python exercicio3.py
python exercicio5.py musica.mp3
python projeto8.py                   # toca originais e demodulados, mostra os gráficos
```

Por padrão os áudios são reproduzidos; `--sem-som` desliga o som e `--sem-janela` só salva os gráficos em `figuras/`.

---

## Respostas

### Exercício 1
A FFT da soma `m(t) + C(t)` tem picos em **2 kHz** e **15 kHz**. A componente de 2 kHz está fora
da banda 12–18 kHz (≈ 50 % da energia fora da banda), logo esse sinal não pode ser transmitido.
Somar não desloca o espectro da informação: não é modulação. Ver `figuras/exercicio1.png`.

### Exercício 2
`S(t) = m(t)·C(t)` gera componentes apenas em `fc − fm` e `fc + fm`:

- fc − fm = 15 − 2 = **13 kHz**
- fc + fm = 15 + 2 = **17 kHz**

Banda mínima: **de 13 kHz a 17 kHz**, ou seja, largura de **4 kHz** (= 2·fm).

### Exercício 3
A FFT de `S(t)` mostra apenas dois picos, em 13 kHz e 17 kHz, dentro da banda mínima
(energia fora de [13, 17] kHz ≈ 0). Ver `figuras/exercicio3.png`.

### Exercício 4
A banda permitida 14–16 kHz tem largura de 2 kHz em torno da portadora de 15 kHz, então a
informação só pode ocupar até **fm,max = 1 kHz** (15 ± 1 kHz). Antes de modular é preciso
**filtrar m(t) com um passa-baixa de corte em 1 kHz**, eliminando as componentes de 1 kHz a 2 kHz.
Perde-se parte do conteúdo agudo, mas o sinal modulado fica todo dentro da banda.

### Exercício 5
Implementado em `exercicio5.py`. O áudio passa por um passa-baixa (banda passante 1,7 kHz,
rejeição a partir de 2 kHz), é multiplicado pela portadora de 16 kHz (fica em 14–18 kHz),
depois é multiplicado de novo pela portadora e filtrado por passa-baixa para recuperar o áudio.
Ver `figuras/exercicio5.png` e `audios/ex5_*.wav`.

### Exercício 6 — 64-QAM, portadora 2,4 GHz, 1 símbolo a cada 2 períodos
- Bits por símbolo: log₂(64) = **6 bits**
- Taxa de símbolos: 2,4·10⁹ / 2 = **1,2·10⁹ símbolos/s**
- Taxa de transmissão: 6 × 1,2·10⁹ = **7,2·10⁹ bit/s = 7,2 Gbit/s** (900 MB/s)

### Exercício 7 — 256-QAM, portadora 2,4 GHz, 1 símbolo a cada 10 períodos
- Bits por símbolo: log₂(256) = **8 bits** (1 byte por símbolo)
- Taxa de símbolos: 2,4·10⁹ / 10 = **2,4·10⁸ símbolos/s**
- Taxa de transmissão: 8 × 2,4·10⁸ = **1,92 Gbit/s** = 240 MB/s
- 480 MB = 480·10⁶ × 8 = 3,84·10⁹ bits → t = 3,84·10⁹ / 1,92·10⁹ = **2 s**

(Se considerar 1 MB = 2²⁰ bytes: 480 × 2²⁰ × 8 ≈ 4,03·10⁹ bits → ≈ 2,1 s.)

---

## Projeto 8 — 3 estações AM

| Estação | Faixa | Portadora | Áudio (passa-baixa) |
|---|---|---|---|
| 1 | 9–12 kHz | 10,5 kHz | passa até 1,2 kHz, rejeita ≥ 1,5 kHz |
| 2 | 12–15 kHz | 13,5 kHz | idem |
| 3 | 15–18 kHz | 16,5 kHz | idem |

Cada faixa tem 3 kHz de largura; como o AM por multiplicação ocupa `fc ± fm`, o áudio pode ter
no máximo 1,5 kHz. Por isso cada áudio é filtrado antes da modulação.

**Emissor:** passa-baixa → multiplicação pela portadora → soma das 3 estações (meio físico).
**Receptor:** passa-faixa da estação (passa `fc ± 1,2 kHz`, rejeita fora da faixa) →
multiplicação pela portadora (demodulação síncrona) → passa-baixa → `audios/demodulado{1,2,3}.wav`.

Os filtros são FIR com janela de Kaiser (atenuação 80 dB, transição 300 Hz), aplicados com
`filtfilt` (fase zero). O script imprime a energia de cada estação fora da sua faixa e dentro das
faixas vizinhas (≈ 10⁻³ % ou menos) e a correlação de cada áudio demodulado com o transmitido.

### Gráficos (o que apresentar)
1. FFT dos 3 áudios originais — `figuras/p8_1_originais.png`
2. FFT dos 3 áudios modulados — `figuras/p8_2_modulados.png`
3. FFT da soma dos 3 modulados — `figuras/p8_3_soma.png`
4. FFT dos 3 sinais extraídos no receptor — `figuras/p8_4_extraidos.png`
5. FFT dos 3 áudios demodulados — `figuras/p8_5_demodulados.png`

---

# Parte 2 — Chat por áudio com modulação digital PSK

Dois computadores conversam pelo ar: o **emissor** transforma a frase em bits, modula uma portadora
em **BPSK** a **100 bit/s** e toca o áudio; o **receptor** ouve pelo microfone, demodula e escreve a frase.

## Como usar (2 computadores)

```bash
# Computador B (recebe) — deixe rodando primeiro
python receptor.py

# Computador A (envia)
python emissor.py
você> Olá, Camada Física!          # ENTER -> o áudio é tocado
```
No computador B aparece `outro> Olá, Camada Física!`.

Os dois lados precisam usar os **mesmos parâmetros**:

| Opção | Padrão | O que faz |
|---|---|---|
| `--taxa` | 100 | bits por segundo |
| `--fc` | 4000 | portadora em Hz (o enunciado pede 2–6 kHz) |
| `--pulso` | `ret` | `ret` = PSK clássico (troca de fase instantânea); `rc` = cosseno levantado (banda menor) |
| `--alfa` | 0.5 | roll-off do cosseno levantado |
| `--dispositivo` | padrão do sistema | índice do microfone/alto-falante (`python -m sounddevice` lista) |

Opções úteis na apresentação:

```bash
python emissor.py -v                 # mostra os bits de cada caractere e o quadro inteiro
python emissor.py --plot             # abre a figura do sinal emitido (tempo + FFT) a cada envio
python receptor.py -v                # medidor de nível, fase, desvio de frequência, SNR, bits, símbolos
python receptor.py --plot            # abre a figura do sinal recebido (FFT + constelação) a cada mensagem
python receptor.py --arquivo audios/psk_rx.wav --plot     # reanalisa a última gravação
python emissor.py -m "teste" --sem-som                    # só gera audios/psk_tx.wav
python receptor.py --arquivo audios/psk_tx.wav -v         # demodula o arquivo (teste sem microfone)
python analise_psk.py --rx audios/psk_rx.wav              # gera todas as figuras de análise
python emissor.py --taxa 200 --fc 3000                    # (e o receptor com os mesmos parâmetros)
```

A cada mensagem são salvos `audios/psk_tx.wav` / `figuras/psk_emitido.png` (emissor) e
`audios/psk_rx.wav` / `figuras/psk_recebido.png` (receptor).

## Como funciona

### Modulação BPSK
```
bit 0 -> fase   0°  -> a = +1
bit 1 -> fase 180°  -> a = −1          s(t) = a(t) · cos(2π·fc·t)
```
Multiplicar a portadora por −1 é o mesmo que deslocar sua fase em 180°. Cada bit dura
`T = 1/taxa` (10 ms = 441 amostras a 44,1 kHz e 100 bit/s). Os caracteres são codificados em UTF-8
(acentos e emoji funcionam), e cada byte é enviado com o MSB primeiro.

### Quadro
```
| ABERTURA 16×"0" | PREÂMBULO 31 bits (PN) | TAMANHO 8 bits | DADOS 8·N bits | CRC-16 | CAUDA 8×"0" |
```
- **Abertura**: portadora pura (dá tempo do alto-falante e do microfone estabilizarem).
- **Preâmbulo**: sequência pseudoaleatória (sequência-m de 31 bits, `1111100110100100001010111011000`).
  Sua autocorrelação tem um pico estreito, e é ela que o receptor procura para saber **onde** o quadro começa
  e **qual a fase** da portadora que chegou.
- **Tamanho**: quantos bytes vêm a seguir (mensagens > 255 bytes viram vários quadros).
- **CRC-16-CCITT**: detecta bits errados. Se não conferir, o receptor avisa em vez de mostrar lixo.

### Demodulação (coerente, como a demodulação síncrona do AM)
1. **Passa-faixa** em `fc ± banda/2`: só a banda necessária entra no receptor (ruído, voz e 60 Hz ficam de fora).
2. **Mistura I/Q**: multiplica por `2cos(2πfc·t)` e `−2sen(2πfc·t)` e passa um **passa-baixa**, obtendo
   `z(t) = I + jQ`. Como no AM, a multiplicação leva o sinal para perto de 0 Hz e o termo em `2fc` é filtrado.
   Usa-se I **e** Q porque o receptor não sabe a fase da portadora que chegou: ela depende da distância e do
   atraso entre os computadores.
3. **Sincronismo**: correlação de `z(t)` com o preâmbulo conhecido. O pico dá o instante de início, e o
   ângulo do pico dá a **fase θ** da portadora recebida. A diferença de fase entre partes do preâmbulo dá o
   **desvio de frequência** (os cristais das placas de som dos dois PCs nunca são idênticos, ~10–100 ppm).
4. **Decisão**: para cada bit tira-se a média de `z` no centro do bit, gira-se de `−θ` e decide-se
   **bit = 1 se a parte real < 0**. Um laço de fase (PLL dirigido por decisão) atualiza θ a cada bit,
   acompanhando o desvio de frequência durante a mensagem.
5. **Validação**: preâmbulo com mais de 6 erros = falso alarme; CRC confere → mostra a frase.

O receptor ao vivo grava continuamente na taxa nativa do microfone, procura o preâmbulo a cada 0,4 s e,
ao achar, lê o tamanho e espera o quadro inteiro chegar. A figura é gerada num processo separado, para a
captura nunca perder amostras.

## Banda necessária

O sinal BPSK com pulso retangular é `a(t)·cos(2πfc·t)`, onde `a(t)` é uma sequência de pulsos de duração
`T = 1/taxa`. A transformada de um pulso retangular de duração `T` é um **sinc** com nulos a cada `1/T`:
```
|S(f)| ∝ |sinc((f − fc)·T)|      ->  nulos em fc ± taxa, fc ± 2·taxa, ...
```
A **banda necessária** (lóbulo principal, onde está ~90% da potência) é:
```
B = 2 · taxa = 2 × 100 = 200 Hz        ->   de 3900 Hz a 4100 Hz para fc = 4 kHz
```
Com o pulso **cosseno levantado** (`--pulso rc`), o espectro fica limitado a `B = (1 + α)·taxa = 150 Hz` (α = 0,5).

Medido pela FFT (`python analise_psk.py`):

| taxa (bit/s) | fc (Hz) | pulso | T (ms) | banda necessária | faixa | 90% da potência | 99% da potência |
|---:|---:|---|---:|---:|---|---:|---:|
| 50  | 4000 | ret | 20  | 100 Hz | 3950–4050 | 80 Hz  | 964 Hz |
| 100 | 4000 | ret | 10  | 200 Hz | 3900–4100 | 153 Hz | 1899 Hz |
| 200 | 4000 | ret | 5   | 400 Hz | 3800–4200 | 321 Hz | 4109 Hz |
| 400 | 4000 | ret | 2,5 | 800 Hz | 3600–4400 | 641 Hz | 8615 Hz |
| 100 | 2000 | ret | 10  | 200 Hz | 1900–2100 | 153 Hz | 1949 Hz |
| 100 | 6000 | ret | 10  | 200 Hz | 5900–6100 | 153 Hz | 1892 Hz |
| 100 | 4000 | rc  | 10  | 150 Hz | 3925–4075 | 82 Hz  | 110 Hz |

### Alterando a taxa de transmissão (`figuras/psk_b_taxas.png`)
Dobrar a taxa **dobra a largura do espectro**: os nulos passam de `fc ± 100` para `fc ± 200` Hz, e assim por
diante. Bits mais curtos no tempo ocupam mais banda na frequência (dualidade tempo × frequência da
transformada de Fourier: `Δt · Δf ≈ constante`). Por isso **taxa maior exige banda maior**, e uma banda
limitada impõe uma taxa máxima. O pico também fica mais baixo, porque a mesma potência se espalha por
uma faixa maior.

### Alterando a portadora (`figuras/psk_c_portadoras.png`)
O espectro tem **exatamente a mesma forma e largura**, só **deslocado** para a nova `fc`
(propriedade do deslocamento em frequência: multiplicar por `cos(2πfc·t)` translada o espectro de `a(t)`
para ±fc, como no AM). A portadora escolhe **onde** o sinal fica no espectro; a taxa define **quanto**
espaço ele ocupa. Isso permite vários chats simultâneos em portadoras diferentes (FDM), como as
estações de rádio da parte 1.

### Formato do pulso (`figuras/psk_d_pulsos.png`)
As trocas de fase instantâneas (pulso retangular) criam lóbulos laterais que decaem devagar (1/f):
99% da potência só cabe em ~19× a taxa. Suavizando as trocas (cosseno levantado), toda a energia fica dentro
de `(1+α)·taxa`, que é o que se usa em Wi-Fi, 4G etc.

### Emitido × recebido (`figuras/psk_e_emitido_recebido.png`, `figuras/psk_recebido.png`)
O sinal recebido tem a mesma forma em torno de `fc`, mais ruído em todas as frequências, zumbido de 60 Hz e
voz/ruído ambiente. Por isso o receptor filtra só a banda necessária antes de demodular. A constelação
mostra os dois pontos ±1 do BPSK; quanto mais espalhados, menor o SNR.

## Robustez (testado)
- Canal simulado com atraso, eco, ruído (SNR 0–15 dB), zumbido de 60 Hz e diferença de relógio de até
  ±400 ppm: decodificado em 50, 100, 200, 300 e 400 bit/s, portadoras de 2, 4 e 6 kHz, pulsos `ret` e `rc`.
- Ruído puro não gera mensagens falsas.
- Teste em tempo real alto-falante → "Mixagem estéreo" → receptor (mensagens curtas, longas com 2 quadros,
  acentos e emoji): todas recebidas com CRC ok e SNR ≈ 29 dB.

## Dicas para a apresentação
- **Volume** do emissor alto (sem distorcer) e os notebooks a ~30–50 cm.
- Desligue os **"Aprimoramentos de áudio" / supressão de ruído** do microfone do receptor
  (Configurações → Sistema → Som → microfone → Aprimoramentos de áudio: *Desativado*). Supressores de ruído
  tratam um tom contínuo como "ruído" e o apagam. Neste notebook (Intel Smart Sound) o microfone cancela o
  próprio som do alto-falante, então **um computador não consegue receber o que ele mesmo toca pelo ar**.
  Entre dois computadores isso não acontece. Para testar sozinho, use a "Mixagem estéreo":
  `python receptor.py --dispositivo 13` (veja o índice com `python -m sounddevice`).
- `python receptor.py -v` mostra um medidor do nível na banda da portadora: ao tocar, ele deve subir bem
  (> 15 dB). Se não subir, verifique volume, dispositivo e se a portadora (`--fc`) é a mesma nos dois lados.
- Se o ambiente for barulhento, tente outra portadora (`--fc 5000`) ou uma taxa menor (`--taxa 50`).
