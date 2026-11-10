# Prova de conceito: circuito com espectro de maré (CUDA-Q)

**Atenção:** o sinal é **sintético** (velocidades angulares reais de 9 constituintes,
amplitudes e fases inventadas, ruído AR(1)). Os números não dizem nada sobre o sensor CCMAR.
Para usar dados reais, troque `sinal()` em `comum.py`.

Rodado em 09/11/2026 em um ambiente na nuvem (CPU de 2 núcleos, sem GPU), CUDA-Q 0.16 no alvo
`qpp-cpu`. As saídas dessa execução estão em [resultados/](resultados/).

| Script | O que faz | Saída |
|---|---|---|
| `e1_espectro.py` | Confere se a saída do circuito tem as frequências previstas | `r1.json` |
| `e2_q.py <mare\|gen> <semente>` | Treina o circuito (codificação de maré ou genérica) | `r2_*.json`, `th_*.npy`, `c_*.npy` |
| `e3_classicos.py` | Harmônica, substituto, MLP, KAN e erro bayesiano AR | `r3.json` |
| `e4_tiros.py` | Ruído de medição (100 a 10.000 medições) como incerteza | `r4.json` |
| `e5_tempo.py` | Tempo CUDA-Q contra PennyLane | `r5.json` |
| `e6_substituto.py` | Substituto clássico do circuito treinado | `r6.json` |
| `corre.sh` | Roda tudo na ordem certa | `log*.txt` |

## Resultado principal (MAE em cm contra a maré sem ruído, média de 3 realizações)

| Modelo | Parâmetros | MAE |
|---|---|---|
| Harmônica, 9 constituintes | 19 | 1,97 |
| Harmônica, 8 constituintes | 17 | 2,27 |
| Harmônica, 4 constituintes | 9 | 2,57 |
| KAN de Chebyshev [8,2,1] | 57 | 3,27 |
| MLP 8-3-1 | 31 | 3,49 |
| Circuito com espectro de maré (4 qubits) | 29 | 4,04 |
| Regressão direta no dicionário | 305 | 6,63 |
| Circuito genérico | 29 | 9,19 |

Outros achados: as combinações previstas explicam a saída do circuito com R² acima de
0,99999999999; um substituto clássico de 305 coeficientes reproduz o circuito treinado
(diferença máxima de 0,00002 cm); o ruído de medição não acompanha o erro (correlação entre
−0,04 e 0,01).

Limites: o circuito treinou 100 iterações de Adam, enquanto MLP e KAN usaram L-BFGS; não há
ajuste de hiperparâmetros nem resíduo de EDP.

## Como rodar

```bash
python -m venv .venv && source .venv/bin/activate
pip install cudaq numpy scipy scikit-learn jax pennylane
bash corre.sh        # as saídas são gravadas nesta pasta
```

`e3_classicos.py` pode rodar a qualquer momento. `e4_tiros.py` precisa de `e2_q.py mare 0`, e
`e6_substituto.py` precisa das três sementes de `mare`. Para usar a GPU, coloque
`cudaq.set_target('nvidia')` no início de `comum.py`; com 4 qubits ela não compensa.
