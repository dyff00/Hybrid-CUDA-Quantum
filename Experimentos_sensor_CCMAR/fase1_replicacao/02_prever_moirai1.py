"""Passo 1.3 – Moirai-1.0 (R-small) zero-shot.

Ambiente: .venv-moirai  (uni2ts exige Python 3.10/3.11, torch < 2.5, numpy 1.26
— por isso tem ambiente próprio).

Moirai gera AMOSTRAS (100) e os quantis são os quantis empíricos delas (doc §3.3),
por isso a semente é fixa.

Uso:
  python 02_prever_moirai1.py
  python 02_prever_moirai1.py --checkpoint <pasta do fine-tuning> --nome moirai1_ft
"""
import argparse

import numpy as np
import torch
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

import comum
import config as C

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", default=C.MODELOS_HF["moirai1"])
p.add_argument("--nome", default="moirai1")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = p.parse_args()

torch.manual_seed(C.SEMENTE)
np.random.seed(C.SEMENTE)
print(f"carregando {a.checkpoint} em {a.device} ...")
modelo = MoiraiForecast(
    module=MoiraiModule.from_pretrained(a.checkpoint),
    prediction_length=C.HORIZONTE,
    context_length=C.LOOKBACK_PASSOS,
    patch_size=C.MOIRAI_PATCH,
    num_samples=C.MOIRAI_AMOSTRAS,
    target_dim=1,
    feat_dynamic_real_dim=0,
    past_feat_dynamic_real_dim=0,
).to(a.device).eval()

serie = comum.carregar_serie()
lista = list(comum.janelas(serie))
registros = []
with comum.Cronometro(), torch.no_grad():
    for k in range(0, len(lista), 16):  # lotes de 16 janelas
        lote = lista[k : k + 16]
        ctx = torch.tensor(np.stack([c for _, c in lote]), dtype=torch.float32)[:, :, None]
        amostras = modelo(
            past_target=ctx.to(a.device),
            past_observed_target=torch.ones_like(ctx, dtype=torch.bool).to(a.device),
            past_is_pad=torch.zeros(ctx.shape[:2], dtype=torch.bool).to(a.device),
        )  # (lote, amostras, HORIZONTE)
        amostras = amostras.cpu().numpy()
        for (o, _), am in zip(lote, amostras):
            q = np.quantile(am, C.QUANTIS, axis=0).T  # (HORIZONTE, 9)
            registros.append((o, q))
comum.salvar_previsoes(a.nome, registros, serie)
