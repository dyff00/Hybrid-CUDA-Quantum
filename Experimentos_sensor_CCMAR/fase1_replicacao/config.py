"""Configuração única da Fase 1 (replicação do experimento da Miriã).

Todos os scripts leem daqui. Os valores seguem o documento
"Processo de previsão de nível - Lagoa dos Patos" (seções 2.3, 3.2 e 4).
Se a Miriã informar outro valor, mude SÓ aqui.
"""
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
DADOS = RAIZ.parent / "dados"  # pasta compartilhada com a fase 2 (Experimentos_sensor_CCMAR/dados)
RESULTADOS = RAIZ / "resultados"
CHECKPOINTS = RAIZ / "checkpoints"

# Arquivo bruto exportado da API (sensor_1 = CCMAR)
CSV_BRUTO = DADOS / "sensor_1_CCMAR.csv"
CSV_SERIE = DADOS / "serie_30min.csv"  # gerado pelo 01_preparar_dados.py

# --- Pré-processamento (doc §2.3) ---
FREQ = "30min"
LIMITE_INTERPOLACAO = 6  # passos (3 h), só lacunas internas

# --- Previsão (doc §3.2) ---
LOOKBACK_PASSOS = 480  # contexto: 10 dias
HORIZONTE = 48  # 24 h
QUANTIS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# --- Backtest ---
# Uma previsão a cada 12 h (mesma cadência do job_prever, doc §1).
PASSO_ORIGEM = 24
# Período de teste: últimos N dias da série. É o MESMO para todos os modelos,
# zero-shot e fine-tuned. O fine-tuning só pode usar dados anteriores a ele.
DIAS_TESTE = 20

# --- Modelos (doc §3.3) ---
MODELOS_HF = {
    "chronos2": "amazon/chronos-2",
    "moirai1": "Salesforce/moirai-1.0-R-small",
    "tirex2": "NX-AI/TiRex-2",
    "timesfm3": "google/timesfm-3.0-pytorch",
}
MOIRAI_PATCH = 32
MOIRAI_AMOSTRAS = 100
SEMENTE = 42  # SEMENTE_AMOSTRAGEM do doc

# --- Fine-tuning Chronos-2 (doc §4.1) ---
FT_CHRONOS = dict(finetune_mode="lora", learning_rate=1e-5, num_steps=1000, batch_size=32)
