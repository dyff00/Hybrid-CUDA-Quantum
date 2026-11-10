# Experimentos – sensor CCMAR

Previsão de 24 h do nível d'água do sensor **CCMAR** com modelos fundacionais de séries
temporais, física e circuitos quânticos variacionais em **CUDA-Q**.

Autor: Everson Flores. Colaboração: Miriã Ciex (pipeline original da fase 1).

> **Status (10/11/2026):** resultados preliminares, com um único sensor e cerca de 80 dias
> de dados. A prova de conceito usa sinal sintético e não diz nada sobre o sensor real.

## Resultado em uma frase

O Chronos-2 já captura a maré: em 24 h, 87,5% do erro restante está em períodos acima de
30 h (vento, pressão, descarga), e nenhuma correção que usa só o tempo — clássica ou
quântica — melhorou o MAE de 4,62 cm. O próximo passo é dar à correção as forçantes
meteorológicas e hidrológicas.

## Experimentos

| Exp. | Pergunta | Onde rodou | Resultado |
|---|---|---|---|
| Fase 1 | Qual modelo fundacional prevê melhor? | Local, GPU e CPU | TimesFM-3, Chronos-2 e TiRex-2 empatam perto de 4,6 cm |
| E0 | O Chronos-2 reproduz a fase 1? | Local, GPU RTX 5070 | Sim: MAE 4,618 cm (fase 1: 4,620) |
| E1 | Em que frequências está o erro? | Local, CPU | 87,5% acima de 30 h em 24 h; maré 3,7% |
| E2 | A maré harmônica corrige o erro? | Local, CPU | Não: o MAE sobe para cerca de 4,81 cm |
| E3 | Dá para calibrar a incerteza? | Local, CPU | Conformal: cobertura de 0,768 para 0,803 |
| E4 | Circuito quântico vence KAN e MLP? | Local, CPU | Nenhuma das 7 correções supera o Chronos-2 |
| E5 | A GPU acelera a simulação? | Local, GPU e CPU | 344× sobre o PennyLane com 22 qubits |
| PoC | O circuito alcança as marés compostas? | Nuvem, CPU | Sim, mas a análise harmônica erra menos |

## Estrutura

```
Experimentos_sensor_CCMAR/
├── README.md
├── dados/                          CSV do sensor (ver dados/README.md)
├── fase1_replicacao/               replicação do pipeline da Miriã (4 modelos fundacionais)
│   ├── README.md
│   ├── config.py, comum.py
│   ├── 01_preparar_dados.py … 05_teste_estatistico.py
│   ├── requirements-principal.txt, requirements-moirai.txt
│   └── docs/                       guia de replicação e resultados da fase 1
├── fase2_experimentos/             E0 a E5 sobre o erro do Chronos-2
│   ├── LEIAME.md
│   ├── 00_verificar.py, 01_historico_chronos.py
│   ├── e1_onde_erra.py … e5_gpu.py, rodar_quantico.sh
│   └── requirements-quantico.txt
├── prova_de_conceito_sintetica/    circuito com espectro de maré em sinal sintético
│   ├── LEIAME.md, comum.py, e1_espectro.py … e6_substituto.py, corre.sh
│   └── resultados/                 JSONs, logs e parâmetros treinados (09/11/2026)
└── resultados/
    ├── resultados_fase2.xlsx       todos os números, uma aba por experimento, com gráficos
    ├── fase2_resultados.md         tabelas de E1 a E5
    ├── saida_terminal_2026-11-10.txt
    └── relatorio_fase2.pdf         relatório com métodos, resultados e próximos passos
```

## Como reproduzir

1. Coloque `sensor_1_CCMAR.csv` em `dados/`.
2. Fase 1: siga [fase1_replicacao/docs/fase1-replicacao.md](fase1_replicacao/docs/fase1-replicacao.md).
   Cria os ambientes `.venv-principal` e `.venv-moirai` nesta pasta.
3. Fase 2: siga [fase2_experimentos/LEIAME.md](fase2_experimentos/LEIAME.md). Usa o
   `.venv-principal` e cria o `.venv-quantico`.
4. Prova de conceito: siga [prova_de_conceito_sintetica/LEIAME.md](prova_de_conceito_sintetica/LEIAME.md).
   Não precisa dos dados do sensor.

## Ambiente usado

| Software | Versão | Usado em |
|---|---|---|
| Python | 3.12.3 | Todos |
| chronos-forecasting | 2.3.2 | Fase 1, E0 |
| timesfm / tirex-2 / uni2ts | 3.0.2 / 0.4.0 / 2.0.0 | Fase 1 |
| CUDA-Q | 0.16.0 | E4, E5, PoC |
| PennyLane / Lightning | 0.45.1 / 0.45.0 | E5 |
| JAX (CPU) | 0.12.0.1 | E4, PoC |
| NumPy / SciPy / pandas | 2.5.4 / 1.18.1 / 3.1.1 | E1 a E5 |

Versões fixadas nos arquivos `requirements-*.txt`; as da fase 2 são as instaladas em
10/11/2026. Hardware: computador local com Linux e GPU NVIDIA RTX 5070 (fase 1, E0 a E5);
ambiente na nuvem com CPU de 2 núcleos (prova de conceito).

## Limitações

- Um único sensor, cerca de 80 dias, sem eventos extremos.
- 39 janelas de teste sobrepostas: poder estatístico limitado.
- Os circuitos tiveram menos ajuste que MLP e KAN.
- O modelo bayesiano de E3 incluiu a correção harmônica, que já piora o MAE.

## Próximos passos

1. Obter vento, pressão e descarga das 6 estações, com pelo menos um ano.
2. Chronos-2 com essas forçantes como covariáveis.
3. Correção física com vento e pressão (PINN e KAN).
4. Circuito com forçantes na GPU, contra seu substituto clássico.
5. Modelo bayesiano com forçantes contra a calibração conformal.
