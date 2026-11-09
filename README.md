# Projeto 8 — Modulação AM

Camada Física da Computação — Insper

## Estrutura

| Arquivo | Conteúdo |
|---|---|
| `utils.py` | FFT, filtros FIR (passa-baixa / passa-faixa, janela de Kaiser), leitura/escrita de WAV, reprodução |
| `exercicio1.py` | Soma informação + portadora → mostra que **não** é modulação |
| `exercicio3.py` | AM por multiplicação ponto a ponto (2 kHz × 15 kHz) |
| `exercicio5.py` | AM com áudio real: portadora 16 kHz, banda 14–18 kHz, modulação + demodulação |
| `gravar_audios.py` | Grava os 3 áudios do projeto pelo microfone |
| `projeto8.py` | **Projeto**: 3 estações AM, soma no meio físico, receptor com passa-faixa e demodulação |
| `figuras/` | Gráficos gerados |
| `audios/` | Áudios originais, modulados e demodulados |

## Como executar

```bash
pip install numpy scipy matplotlib sounddevice

# Opção A: usar arquivos prontos (wav, mp3, flac, ogg; qualquer taxa — convertidos para 44100 Hz, máx. 5 s)
python projeto8.py musica.mp3 voz.wav efeito.flac --tocar

# Opção B: gravar pelo microfone em audios/audio1..3.wav
python gravar_audios.py
# (ou: python gravar_audios.py --sintetico   para gerar áudios de teste)

python exercicio1.py
python exercicio3.py
python exercicio5.py musica.mp3 --tocar
python projeto8.py --tocar           # toca originais e demodulados, mostra os gráficos
```

`--tocar` reproduz os áudios; `--sem-janela` só salva os gráficos em `figuras/`.

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
