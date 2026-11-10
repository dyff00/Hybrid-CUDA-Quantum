# Fase 1 – Resultados zero-shot (sensor CCMAR, univariado)

Rodado em 09/11/2026 no Linux (RTX 5070). Teste: 39 janelas de 24 h, uma a cada 12 h, de
06/10 a 26/10/2026. Contexto de 480 passos (10 dias), horizonte de 48 passos (24 h).

| Modelo | MAE (cm) | WQL (cm) | WQL norm. | Cobertura p10–p90 |
|---|---|---|---|---|
| TimesFM-3 | 4,53 | 1,842 | 0,0251 | 0,805 |
| Chronos-2 | 4,62 | 1,830 | 0,0250 | 0,768 |
| TiRex-2 | 4,70 | 1,919 | 0,0262 | 0,760 |
| Sazonal 24 h (baseline) | 5,95 | 2,977 | 0,0406 | – |
| Moirai-1.0-R-small | 6,25 | 2,488 | 0,0339 | 0,823 |
| Persistência (baseline) | 7,49 | 3,745 | 0,0511 | – |

Notas de ambiente:

- TiRex-2 rodou em CPU (`CUDA_VISIBLE_DEVICES="" ... --device cpu`): o mlstm_kernels é
  incompatível com o Triton 3.9 do torch 2.15.
- Moirai rodou em CPU: o uni2ts exige torch 2.4, que não suporta a RTX 5070 (sm_120).
- A cobertura dos baselines não tem significado (são previsões pontuais).

## Teste estatístico (`05_teste_estatistico.py`)

Friedman, ANOVA de medidas repetidas, Wilcoxon pareado com correção de Holm e IC 95% por
bootstrap em blocos. Rodado com as 39 janelas e, como robustez, com as 20 sem sobreposição.

Conclusões que valem nos dois cenários:

- TimesFM-3, Chronos-2 e TiRex-2 são estatisticamente equivalentes, tanto no MAE quanto no WQL.
- Os quatro modelos fundacionais são melhores que a persistência.
- No WQL, os três modelos mais novos são melhores que o baseline sazonal, e o TimesFM-3 é
  melhor que o Moirai.
- O Moirai não se distingue do baseline sazonal em nenhuma das métricas.

Inconclusivo (muda sem sobreposição): no MAE, os três primeiros contra o sazonal e contra o
Moirai. Os intervalos de confiança excluem o zero, mas o p corrigido passa de 0,05 com só 20
janelas: falta poder estatístico (período de teste curto).

Pendências: confirmar com a Miriã o período de teste dela, os números zero-shot dela e o
modo (univariado ou com 6 estações).
