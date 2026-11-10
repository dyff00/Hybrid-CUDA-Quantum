"""Funções compartilhadas pelos scripts de previsão.

Só usa pandas e numpy, para funcionar nos dois ambientes (principal e moirai).
"""
import sys
import time

import numpy as np
import pandas as pd

import config as C


def carregar_serie() -> pd.Series:
    if not C.CSV_SERIE.exists():
        sys.exit("Rode antes: python 01_preparar_dados.py")
    df = pd.read_csv(C.CSV_SERIE, parse_dates=["timestamp"])
    return df.set_index("timestamp")["nivel_cm"]


def origens_de_teste(serie: pd.Series) -> list[int]:
    """Índices (posição) do último ponto do contexto de cada janela de teste.

    A previsão feita na origem i usa serie[i-LOOKBACK+1 .. i] e prevê
    serie[i+1 .. i+HORIZONTE]. Todas as janelas ficam dentro do período de teste.
    """
    n = len(serie)
    inicio_teste = n - C.DIAS_TESTE * 48  # 48 passos de 30 min por dia
    primeira = max(inicio_teste - 1, C.LOOKBACK_PASSOS - 1)
    ultima = n - C.HORIZONTE - 1
    return list(range(primeira, ultima + 1, C.PASSO_ORIGEM))


def inicio_teste(serie: pd.Series) -> pd.Timestamp:
    return serie.index[origens_de_teste(serie)[0] + 1]


def janelas(serie: pd.Series):
    """Gera (origem_timestamp, contexto np.float32[LOOKBACK]) para cada janela."""
    valores = serie.to_numpy(dtype=np.float32)
    for i in origens_de_teste(serie):
        ctx = valores[i - C.LOOKBACK_PASSOS + 1 : i + 1]
        if np.isnan(ctx).any():
            print(f"  janela {serie.index[i]} pulada (lacuna no contexto)")
            continue
        yield serie.index[i], ctx


def salvar_previsoes(nome_modelo: str, registros: list[tuple], serie: pd.Series):
    """registros: lista de (origem_ts, matriz quantis [HORIZONTE, 9])."""
    linhas = []
    passo = pd.Timedelta(C.FREQ)
    for origem, q in registros:
        q = np.sort(np.asarray(q, dtype=float), axis=1)  # garante quantis monótonos
        assert q.shape == (C.HORIZONTE, len(C.QUANTIS)), q.shape
        for h in range(C.HORIZONTE):
            linha = {"modelo": nome_modelo, "origem": origem, "passo": h + 1,
                     "timestamp": origem + passo * (h + 1)}
            linha.update({f"q{qq:.1f}": q[h, k] for k, qq in enumerate(C.QUANTIS)})
            linhas.append(linha)
    C.RESULTADOS.mkdir(exist_ok=True)
    saida = C.RESULTADOS / f"previsoes_{nome_modelo}.csv"
    pd.DataFrame(linhas).to_csv(saida, index=False)
    print(f"OK: {len(registros)} janelas -> {saida}")


class Cronometro:
    def __enter__(self):
        self.t = time.time()
        return self

    def __exit__(self, *a):
        print(f"tempo: {time.time() - self.t:.1f} s")
