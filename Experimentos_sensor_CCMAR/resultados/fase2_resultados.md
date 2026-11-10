# Fase 2 – Resultados reais (sensor CCMAR), 10/11/2026

Rodado em um computador local com Linux e GPU RTX 5070. O Chronos-2 zero-shot reproduziu a
fase 1: MAE 4,618 cm, WQL 1,830 cm, cobertura 0,768. Mesmas 39 janelas de teste (06/10 a
26/10/2026). A saída completa do terminal está em
[saida_terminal_2026-11-10.txt](saida_terminal_2026-11-10.txt).

## E1 · Fração da variância do erro do Chronos-2 por banda

| Banda | Nível observado | Erro em 1 h | 6 h | 12 h | 24 h |
|---|---|---|---|---|---|
| Baixa frequência (acima de 30 h) | 73,7% | 5,8% | 40,8% | 67,9% | 87,5% |
| Diurna (22 a 27 h) | 15,5% | 1,7% | 12,1% | 11,4% | 1,6% |
| Semidiurna (11,5 a 13 h) | 4,4% | 3,9% | 11,0% | 0,7% | 1,1% |
| Terdiurna (7,5 a 8,5 h) | 0,3% | 2,5% | 2,0% | 1,7% | 0,3% |
| Quarto-diurna (5,8 a 6,4 h) | 1,3% | 4,8% | 0,3% | 0,4% | 0,7% |
| Demais frequências | 4,8% | 81,3% | 33,8% | 17,9% | 8,8% |

Ruído branco poria cerca de 6,7% da variância acima de 30 h. O Chronos-2 já captura a maré;
o erro restante é de baixa frequência (vento, pressão, descarga).

## E2 a E4 · Correções do erro (MAE em cm; cobertura p10–p90; largura em cm)

| Modelo | MAE | Cobertura | Largura |
|---|---|---|---|
| Chronos-2 sozinho | 4,618 | 0,768 | 13,16 |
| Chronos-2 + conformal (E3) | 4,618 | 0,803 | 14,28 |
| + circuito genérico, 29 parâmetros (E4) | 4,728 | – | – |
| Harmônica + persistência da anomalia (E2) | 4,791 | – | – |
| Bayesiano, inclui a correção harmônica (E3) | 4,798 | 0,849 | 18,53 |
| Bayesiano + conformal (E3) | 4,798 | 0,768 | 14,84 |
| + harmônica, 9 constituintes (E4) | 4,798 | – | – |
| + KAN [8,1], 25 parâmetros (E4) | 4,805 | – | – |
| + harmônica, 4 constituintes (E4) | 4,807 | – | – |
| + 4 constituintes (E2) | 4,814 | 0,714 | 13,16 |
| + 9 constituintes (E2) | 4,817 | 0,722 | 13,16 |
| + circuito com espectro de maré, 29 parâmetros (E4) | 4,817 | – | – |
| + MLP, 31 parâmetros (E4) | 5,153 | – | – |
| + KAN [8,2,1], 57 parâmetros (E4) | 5,547 | – | – |
| Harmônica pura (E2) | 6,466 | – | – |

Nenhuma correção que usa só o tempo supera o Chronos-2. A calibração conformal é a única
melhora (cobertura de 0,768 para 0,803). E2 ajusta a harmônica a cada janela; E4 ajusta ao
erro médio por instante, por isso os valores diferem um pouco.

## E5 · Tempo por avaliação do circuito (3 camadas)

| Qubits | CUDA-Q GPU (nvidia) | CUDA-Q CPU (qpp-cpu) | PennyLane Lightning |
|---|---|---|---|
| 4 | 0,34 ms | 0,38 ms | 4,7 ms |
| 12 | 1,0 ms | 19 ms | 6,0 ms |
| 18 | 1,6 ms | 1.544 ms | 29 ms |
| 22 | 2,8 ms | 36.914 ms | 956 ms |
| 26 | 46 ms | mais de 4 min | 18.465 ms |

## Nova direção

A correção (física, circuito e modelo bayesiano) deve usar vento, pressão e descarga como
entrada, não as frequências de maré. Próximo passo: obter esses dados das 6 estações e repetir
as comparações.
