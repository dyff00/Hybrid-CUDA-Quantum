"""Passo 1.3 – TimesFM-3 zero-shot.

Ambiente: .venv-principal
Atenção: os pesos do TimesFM 3.0 têm licença NÃO comercial (uso acadêmico ok).

Uso:  python 02_prever_timesfm3.py
"""
import argparse

import numpy as np
import torch
from timesfm3 import TimesFM3Forecaster

import comum
import config as C

p = argparse.ArgumentParser()
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = p.parse_args()

torch.manual_seed(C.SEMENTE)
print(f"carregando {C.MODELOS_HF['timesfm3']} em {a.device} ...")
modelo = TimesFM3Forecaster.from_pretrained(C.MODELOS_HF["timesfm3"], device=a.device)

serie = comum.carregar_serie()
lista = list(comum.janelas(serie))

with comum.Cronometro():
    saidas = list(
        modelo.predict_batch(
            [c for _, c in lista], horizon=C.HORIZONTE, return_quantiles=True
        )
    )
# saida.quantiles: (HORIZONTE, 9) = quantis 0.1 ... 0.9
registros = [(o, np.asarray(s.quantiles)) for (o, _), s in zip(lista, saidas)]
comum.salvar_previsoes("timesfm3", registros, serie)
