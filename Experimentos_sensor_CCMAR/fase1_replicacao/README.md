# Fase 1 – Replicação do experimento da Miriã

Previsão de 24 h do nível do sensor **CCMAR** com quatro modelos fundacionais (Chronos-2,
Moirai-1.0-R-small, TiRex-2, TimesFM-3), com o mesmo pré-processamento e as mesmas
métricas do pipeline de Miriã Ciex.

> **Status (10/11/2026):** os números ainda não foram conferidos com os da Miriã no mesmo
> período de teste.

## Resultados (zero-shot, univariado)

Teste: 39 janelas de 24 h, uma a cada 12 h, de 06/10 a 26/10/2026. Contexto de 10 dias
(480 passos de 30 min), horizonte de 24 h (48 passos).

| Modelo | MAE (cm) | WQL (cm) | Cobertura p10–p90 |
|---|---|---|---|
| TimesFM-3 | 4,53 | 1,842 | 0,805 |
| Chronos-2 | 4,62 | 1,830 | 0,768 |
| TiRex-2 | 4,70 | 1,919 | 0,760 |
| Sazonal 24 h (baseline) | 5,95 | 2,977 | – |
| Moirai-1.0-R-small | 6,25 | 2,488 | 0,823 |
| Persistência (baseline) | 7,49 | 3,745 | – |

TimesFM-3, Chronos-2 e TiRex-2 são estatisticamente equivalentes entre si e melhores que a
persistência. O Moirai não se distingue do baseline sazonal. Detalhes em
[docs/fase1-resultados.md](docs/fase1-resultados.md).

## Arquivos

| Arquivo | O que faz |
|---|---|
| `config.py` | Todos os parâmetros do experimento |
| `comum.py` | Funções compartilhadas (janelas, gravação das previsões) |
| `01_preparar_dados.py` | Grade de 30 min e interpolação de lacunas de até 3 h |
| `02_prever_chronos2.py`, `02_prever_timesfm3.py`, `02_prever_tirex2.py`, `02_prever_moirai1.py` | Previsões zero-shot, uma por modelo |
| `03_avaliar.py` | MAE, WQL, cobertura e dois baselines |
| `04_finetune_chronos2.py` | LoRA com os hiperparâmetros da Miriã |
| `05_teste_estatistico.py` | Friedman, ANOVA, Wilcoxon com Holm e bootstrap em blocos |
| `requirements-principal.txt`, `requirements-moirai.txt` | Dependências dos dois ambientes |

## Como reproduzir

O CSV do sensor fica em `../dados/sensor_1_CCMAR.csv` (pasta compartilhada com a fase 2).
O guia completo, com instalação, está em [docs/fase1-replicacao.md](docs/fase1-replicacao.md).

```bash
cd Experimentos_sensor_CCMAR
uv venv .venv-principal --python 3.12 && source .venv-principal/bin/activate
uv pip install -r fase1_replicacao/requirements-principal.txt
cd fase1_replicacao
python 01_preparar_dados.py
python 02_prever_chronos2.py
python 02_prever_timesfm3.py
python 02_prever_tirex2.py
deactivate

cd ..
uv venv .venv-moirai --python 3.11 && source .venv-moirai/bin/activate
uv pip install -r fase1_replicacao/requirements-moirai.txt
cd fase1_replicacao && python 02_prever_moirai1.py && deactivate

source ../.venv-principal/bin/activate
python 03_avaliar.py
python 05_teste_estatistico.py
```

São dois ambientes porque o `uni2ts` (Moirai) exige versões antigas de PyTorch e NumPy.

## Limitações conhecidas

- **Univariado.** O pipeline da Miriã usa as 6 estações juntas; aqui só há o CCMAR.
- **Período de teste curto.** Com 20 janelas sem sobreposição, parte das comparações perde
  significância.
- **Sem covariáveis TELEMAC** (experimentos 3 e 4 do pipeline original).
- **Ambiente:** TiRex-2 e Moirai rodaram em CPU por incompatibilidades com a RTX 5070.
