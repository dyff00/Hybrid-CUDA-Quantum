# Fase 1 – Guia de replicação (sensor CCMAR)

Objetivo: rodar os mesmos 4 modelos fundacionais, com o mesmo pré-processamento e as mesmas
métricas do pipeline da Miriã, e obter uma tabela comparável com a dela.

Parâmetros (todos em `config.py`, tirados do documento "Processo de previsão de nível"):

| Item | Valor |
|---|---|
| Grade | 30 min; lacunas de até 3 h preenchidas por interpolação linear interna |
| Contexto | 480 passos (10 dias) |
| Horizonte | 48 passos (24 h) |
| Quantis | 0,1 … 0,9 (a mediana p50 é a previsão pontual) |
| Métricas | MAE da p50 (cm), WQL (média da perda quantílica, cm), cobertura p10–p90 |
| Backtest | uma previsão a cada 12 h nos últimos 20 dias (39 janelas) |
| Modelos | Chronos-2, Moirai-1.0-R-small, TiRex-2, TimesFM-3 |

## Passo 0 – Instalar

**Sistema:** Linux ou macOS. No Windows, use o WSL2 (Ubuntu), porque o TiRex-2 só é testado
em Linux e macOS. Reserve cerca de 15 GB (PyTorch e pesos dos modelos). O fine-tuning fica
muito mais rápido com GPU NVIDIA.

```bash
# uv: gerencia o Python e os pacotes de cada ambiente
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version

# código
git clone https://github.com/eversondj/Hybrid-CUDA-Quantum.git
cd Hybrid-CUDA-Quantum/Experimentos_sensor_CCMAR
```

O CSV do sensor fica em `dados/sensor_1_CCMAR.csv` (veja `dados/README.md`).

São dois ambientes porque o Moirai (`uni2ts`) exige versões antigas de PyTorch e NumPy.
Crie os dois na pasta `Experimentos_sensor_CCMAR`:

```bash
uv venv .venv-principal --python 3.12
source .venv-principal/bin/activate
uv pip install -r fase1_replicacao/requirements-principal.txt
deactivate

uv venv .venv-moirai --python 3.11
source .venv-moirai/bin/activate
uv pip install -r fase1_replicacao/requirements-moirai.txt
deactivate
```

Teste os imports:

```bash
source .venv-principal/bin/activate
python -c "import chronos, timesfm3, tirex2, torch; print('ok', torch.__version__, 'GPU:', torch.cuda.is_available())"
deactivate
```

Os pesos são baixados do Hugging Face na primeira execução e ficam em
`~/.cache/huggingface`. Não precisa de conta.

## Passo 1 – Preparar os dados

```bash
source .venv-principal/bin/activate
cd fase1_replicacao
python 01_preparar_dados.py
```

Esperado: 3.814 leituras, 5 pontos faltando, os 5 preenchidos, 0 restantes, e o período de
teste começando em 06/10/2026 com 39 janelas. Gráfico: `resultados/serie_ccmar.png`.

## Passo 2 – Modelos zero-shot

```bash
python 02_prever_chronos2.py
python 02_prever_timesfm3.py
python 02_prever_tirex2.py        # na RTX 5070: python 02_prever_tirex2.py --device cpu
deactivate

source ../.venv-moirai/bin/activate
python 02_prever_moirai1.py --device cpu
deactivate
```

Cada script grava `resultados/previsoes_<modelo>.csv`, com os 9 quantis por janela e passo.

## Passo 3 – Avaliar e testar

```bash
source ../.venv-principal/bin/activate
python 03_avaliar.py
python 05_teste_estatistico.py
python 05_teste_estatistico.py --metrica WQL_cm
```

Arquivos gerados em `resultados/`: `metricas_resumo.csv`, `metricas_por_janela.csv`,
`comparacao.png` e `teste_estatistico_<métrica>.csv`.

## Passo 4 – Fine-tuning (opcional)

`04_finetune_chronos2.py` aplica o LoRA com os hiperparâmetros da Miriã (lr 1e-5, 1.000
passos, lote 32), usando só dados anteriores ao teste:

```bash
python 04_finetune_chronos2.py
python 02_prever_chronos2.py --checkpoint checkpoints/chronos2_lora/finetuned-ckpt --nome chronos2_ft
python 03_avaliar.py
```

## Diferenças conhecidas em relação ao pipeline da Miriã

1. **Univariado vs. multivariado.** Ela passa as 6 estações juntas ao modelo; aqui só há o CCMAR.
2. **Período de teste.** Usamos os últimos 20 dias; falta confirmar as janelas dela.
3. **Sem TELEMAC.** Os experimentos com covariável dependem dos arquivos do S3.
