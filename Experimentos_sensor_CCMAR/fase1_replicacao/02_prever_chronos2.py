"""Passo 1.3 – Chronos-2 zero-shot (ou com um checkpoint de fine-tuning).

Ambiente: .venv-principal

Uso:
  python 02_prever_chronos2.py                       # zero-shot (amazon/chronos-2)
  python 02_prever_chronos2.py --checkpoint checkpoints/chronos2_lora/finetuned-ckpt --nome chronos2_ft
"""
import argparse

import numpy as np
import torch
from chronos import Chronos2Pipeline

import comum
import config as C

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", default=C.MODELOS_HF["chronos2"])
p.add_argument("--nome", default="chronos2")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = p.parse_args()

torch.manual_seed(C.SEMENTE)
print(f"carregando {a.checkpoint} em {a.device} ...")
pipe = Chronos2Pipeline.from_pretrained(a.checkpoint, device_map=a.device)

serie = comum.carregar_serie()
lista = list(comum.janelas(serie))
origens = [o for o, _ in lista]
# Chronos-2 aceita um lote de séries: formato (lote, n_variáveis=1, contexto)
contextos = np.stack([c for _, c in lista])[:, None, :]

with comum.Cronometro():
    quantis, _media = pipe.predict_quantiles(
        contextos, prediction_length=C.HORIZONTE, quantile_levels=C.QUANTIS, batch_size=32
    )
# cada elemento: (n_variáveis=1, HORIZONTE, 9)
registros = [(o, q[0].cpu().numpy()) for o, q in zip(origens, quantis)]
comum.salvar_previsoes(a.nome, registros, serie)
