"""Passo 1.4 – Fine-tuning LoRA do Chronos-2 com os hiperparâmetros da Miriã (doc §4.1).

  finetune_mode="lora", learning_rate=1e-5, num_steps=1000, batch_size=32
  validação: últimos 48 passos (24 h) do trecho de treino, nunca vistos no treino.

IMPORTANTE: só usa dados ANTERIORES ao período de teste, para que a comparação
com o zero-shot seja justa (o modelo não pode ter "visto" o teste).

Diferença em relação à Miriã: ela treina com as 6 estações juntas (multivariado);
aqui temos só o CCMAR, então o treino é univariado. Registrar isso no artigo.

Ambiente: .venv-principal (GPU recomendada; em CPU 1000 passos podem levar horas)
Uso:
  python 04_finetune_chronos2.py
  python 02_prever_chronos2.py --checkpoint checkpoints/chronos2_lora/finetuned-ckpt --nome chronos2_ft
"""
import torch
from chronos import Chronos2Pipeline

import comum
import config as C

torch.manual_seed(C.SEMENTE)
serie = comum.carregar_serie()
treino = serie[serie.index < comum.inicio_teste(serie)].dropna()
y = treino.to_numpy(dtype="float32")
print(f"treino: {treino.index[0]} -> {treino.index[-1]}  ({len(y)} pontos)")

entrada_treino = [{"target": y[None, : -C.HORIZONTE]}]  # sem as últimas 24 h
entrada_valid = [{"target": y[None, :]}]  # o modelo valida prevendo as últimas 24 h

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = Chronos2Pipeline.from_pretrained(C.MODELOS_HF["chronos2"], device_map=device)
saida = C.CHECKPOINTS / "chronos2_lora"
pipe.fit(
    inputs=entrada_treino,
    validation_inputs=entrada_valid,
    prediction_length=C.HORIZONTE,
    context_length=C.LOOKBACK_PASSOS,
    output_dir=saida,
    **C.FT_CHRONOS,
)
print(f"checkpoint salvo em {saida / 'finetuned-ckpt'}")
