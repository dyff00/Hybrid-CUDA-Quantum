"""Passo 1.2 – Preparar os dados exatamente como no pipeline da Miriã (doc §2.3).

1. Ordena do mais antigo para o mais recente (a API entrega ao contrário).
2. Reamostra para uma grade regular de 30 min.
3. Preenche lacunas pequenas (até 6 passos = 3 h) por interpolação linear,
   só entre dois valores conhecidos (limit_area="inside").
4. Salva dados/serie_30min.csv e um gráfico da série com o período de teste marcado.

Uso:  python 01_preparar_dados.py
"""
import sys

import pandas as pd

import config as C

if not C.CSV_BRUTO.exists():
    sys.exit(f"Copie o CSV do sensor para {C.CSV_BRUTO}")

bruto = pd.read_csv(C.CSV_BRUTO)
bruto["data_hora"] = pd.to_datetime(bruto["data_hora"], utc=True)
bruto = bruto.sort_values("data_hora").drop_duplicates("data_hora", keep="last")
s = bruto.set_index("data_hora")["valor"].astype(float)

grade = s.resample(C.FREQ).mean()
faltando = int(grade.isna().sum())
serie = grade.interpolate(method="linear", limit=C.LIMITE_INTERPOLACAO, limit_area="inside")
restantes = int(serie.isna().sum())

C.DADOS.mkdir(exist_ok=True)
out = serie.rename("nivel_cm").rename_axis("timestamp").reset_index()
out["timestamp"] = out["timestamp"].dt.tz_localize(None)  # UTC sem fuso, mais simples p/ os modelos
out.to_csv(C.CSV_SERIE, index=False)

print("=== Relatório do pré-processamento ===")
print(f"leituras brutas:            {len(s)}")
print(f"período (UTC):              {s.index.min()}  ->  {s.index.max()}")
print(f"pontos na grade de 30 min:  {len(grade)}")
print(f"pontos faltando na grade:   {faltando}")
print(f"preenchidos (interpolação): {faltando - restantes}")
print(f"ainda faltando:             {restantes}")
print(f"nível: média {serie.mean():.1f} cm, mín {serie.min():.1f}, máx {serie.max():.1f}")
print(f"salvo em: {C.CSV_SERIE}")

# Período de teste (usa a mesma regra dos scripts de previsão)
import comum  # noqa: E402

serie2 = comum.carregar_serie()
origens = comum.origens_de_teste(serie2)
ini = comum.inicio_teste(serie2)
print(f"\nperíodo de teste: a partir de {ini}  ({len(origens)} janelas de 24 h, uma a cada 12 h)")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(serie2.index, serie2.values, lw=0.7, color="#1f5f8b")
    ax.axvspan(ini, serie2.index[-1], color="orange", alpha=0.15, label="período de teste")
    ax.set_ylabel("nível (cm)")
    ax.set_title("Sensor 1 – CCMAR (grade de 30 min)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    C.RESULTADOS.mkdir(exist_ok=True)
    fig.savefig(C.RESULTADOS / "serie_ccmar.png", dpi=150)
    print(f"gráfico: {C.RESULTADOS / 'serie_ccmar.png'}")
except ImportError:
    pass
