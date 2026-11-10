"""Passo 1.3 – TiRex-2 zero-shot.

Ambiente: .venv-principal  (pacote pip: tirex-2; import: tirex2)
TiRex-2 não tem fine-tuning na versão aberta (doc §4).

Uso:  python 02_prever_tirex2.py
"""
import argparse
import os
import sys

# Em CPU, esconde a GPU antes de importar o torch: assim o flashrnn não tenta
# achar o CUDA Toolkit (CUDA_HOME), que não é necessário para rodar em CPU.
if "--device" in sys.argv and sys.argv[sys.argv.index("--device") + 1] == "cpu":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
from tirex2 import TimeseriesType, load_model

import comum
import config as C

p = argparse.ArgumentParser()
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = p.parse_args()

torch.manual_seed(C.SEMENTE)
print(f"carregando {C.MODELOS_HF['tirex2']} em {a.device} ...")
modelo = load_model(C.MODELOS_HF["tirex2"], device=a.device)

serie = comum.carregar_serie()
lista = list(comum.janelas(serie))
entradas = [
    TimeseriesType(target=torch.from_numpy(c)[None, :], past_covariates=None, future_covariates=None)
    for _, c in lista
]

with comum.Cronometro():
    saidas = modelo.forecast(entradas, prediction_length=C.HORIZONTE, output_type="numpy")
# cada saída: (n_alvos=1, 9 quantis, HORIZONTE) -> transpõe para (HORIZONTE, 9)
registros = [(o, np.asarray(s)[0].T) for (o, _), s in zip(lista, saidas)]
comum.salvar_previsoes("tirex2", registros, serie)
