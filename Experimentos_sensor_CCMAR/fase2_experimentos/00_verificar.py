"""Confere se a série foi entendida. Rode no ambiente principal."""
from comum import *
idx, y = carregar_serie(verboso=True)
n = len(y); te = origens_teste(n)
print(f"Primeira previsão de teste: {idx[te[0]]} (na fase 1: 06/10/2026) | janelas: {len(te)}")
if len(te) != N_TESTE or str(idx[te[0]])[:10] != "2026-10-06":
    print("ATENÇÃO: o período de teste difere da fase 1. Me avise antes de seguir.")
else:
    print("OK: período de teste igual ao da fase 1.")
