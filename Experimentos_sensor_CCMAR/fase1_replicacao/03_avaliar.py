"""Passo 1.5 – Avaliar todos os modelos com a mesma régua (doc §3.4).

Lê resultados/previsoes_*.csv e calcula, por modelo:
  - MAE (cm) da mediana p50
  - WQL (cm): média da perda quantílica (pinball) sobre os 9 quantis e os 48 passos
    — mesma definição de anova_teste_covariaveis._perda_quantilica
  - WQL normalizado (soma das perdas / soma |y|), o formato usado no GIFT-Eval/fev-bench
  - cobertura do intervalo p10–p90 (ideal: perto de 80%)

Inclui dois baselines simples (não fazem parte do experimento da Miriã, servem
de sanidade): persistência (repete o último valor) e sazonal ingênuo (repete as
últimas 24 h).

Saídas:
  resultados/metricas_resumo.csv       uma linha por modelo
  resultados/metricas_por_janela.csv   uma linha por (modelo, janela) – base da ANOVA
  resultados/comparacao.png            exemplo de previsão + MAE por passo

Ambiente: .venv-principal (precisa só de pandas, numpy e matplotlib)
Uso:  python 03_avaliar.py
"""
import numpy as np
import pandas as pd

import comum
import config as C

QCOLS = [f"q{q:.1f}" for q in C.QUANTIS]
Q = np.array(C.QUANTIS)


def baselines(serie: pd.Series) -> pd.DataFrame:
    """Previsões dos baselines no mesmo formato dos modelos (todos os quantis = ponto)."""
    v = serie.to_numpy()
    passo = pd.Timedelta(C.FREQ)
    linhas = []
    for i in comum.origens_de_teste(serie):
        o = serie.index[i]
        prev = {
            "persistencia": np.repeat(v[i], C.HORIZONTE),
            "sazonal_24h": v[i - 48 + 1 : i + 1][: C.HORIZONTE],
        }
        for nome, y in prev.items():
            for h in range(C.HORIZONTE):
                d = {"modelo": nome, "origem": o, "passo": h + 1, "timestamp": o + passo * (h + 1)}
                d.update({c: y[h] for c in QCOLS})
                linhas.append(d)
    return pd.DataFrame(linhas)


def pinball(y, qpred):
    """y: (n,), qpred: (n, 9) -> perda por ponto e quantil (n, 9)."""
    e = y[:, None] - qpred
    return np.maximum(Q * e, (Q - 1) * e)


serie = comum.carregar_serie()
arquivos = sorted(C.RESULTADOS.glob("previsoes_*.csv"))
if not arquivos:
    raise SystemExit("Nenhuma previsão encontrada. Rode os scripts 02_prever_*.py antes.")
prev = pd.concat(
    [pd.read_csv(f, parse_dates=["origem", "timestamp"]) for f in arquivos] + [baselines(serie)],
    ignore_index=True,
)
prev["y"] = prev["timestamp"].map(serie)
prev = prev.dropna(subset=["y"])

# Garante que todos os modelos foram avaliados nas MESMAS janelas
janelas_por_modelo = prev.groupby("modelo")["origem"].nunique()
comuns = set.intersection(*[set(g["origem"]) for _, g in prev.groupby("modelo")])
if janelas_por_modelo.nunique() > 1:
    print("Aviso: modelos com número diferente de janelas; usando só as comuns.")
prev = prev[prev["origem"].isin(comuns)]

qp = prev[QCOLS].to_numpy()
y = prev["y"].to_numpy()
perdas = pinball(y, qp)
prev["erro_abs"] = np.abs(qp[:, 4] - y)  # p50
prev["pinball"] = perdas.mean(axis=1)
prev["pinball_soma"] = perdas.sum(axis=1)
prev["coberto"] = (y >= qp[:, 0]) & (y <= qp[:, 8])
prev["abs_y"] = np.abs(y)
prev["janela"] = prev["origem"]


def agrega(g):
    return pd.Series({
        "MAE_cm": g["erro_abs"].mean(),
        "WQL_cm": g["pinball"].mean(),
        "WQL_norm": g["pinball_soma"].sum() / (len(C.QUANTIS) * g["abs_y"].sum()),
        "cobertura_p10_p90": g["coberto"].mean(),
        "janelas": g["janela"].nunique(),
    })


por_janela = prev.groupby(["modelo", "origem"]).apply(agrega, include_groups=False).reset_index()
resumo = prev.groupby("modelo").apply(agrega, include_groups=False).sort_values("MAE_cm")
por_janela.to_csv(C.RESULTADOS / "metricas_por_janela.csv", index=False)
resumo.to_csv(C.RESULTADOS / "metricas_resumo.csv")

pd.set_option("display.width", 120)
print(f"\nPeríodo de teste: {min(comuns)} -> {max(comuns)}  ({len(comuns)} janelas de 24 h)\n")
print(resumo.round({"MAE_cm": 2, "WQL_cm": 3, "WQL_norm": 4, "cobertura_p10_p90": 3}).to_string())

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modelos = [m for m in resumo.index if m not in ("persistencia", "sazonal_24h")]
    fig, axs = plt.subplots(1, 2, figsize=(14, 4.5))
    # (a) uma janela de exemplo: a última
    o = max(comuns)
    ctx = serie[o - pd.Timedelta(hours=72) : o + pd.Timedelta(hours=24)]
    axs[0].plot(ctx.index, ctx.values, color="black", lw=1, label="observado")
    for m in modelos:
        g = prev[(prev.modelo == m) & (prev.origem == o)]
        (l,) = axs[0].plot(g.timestamp, g["q0.5"], lw=1.2, label=m)
        axs[0].fill_between(g.timestamp, g["q0.1"], g["q0.9"], color=l.get_color(), alpha=0.12)
    axs[0].axvline(o, color="gray", ls=":")
    axs[0].set_title(f"Previsão de 24 h a partir de {o:%d/%m %H:%M} UTC (faixa p10–p90)")
    axs[0].set_ylabel("nível (cm)")
    axs[0].legend(fontsize=8)
    # (b) MAE por passo do horizonte
    for m in resumo.index:
        mae_h = prev[prev.modelo == m].groupby("passo")["erro_abs"].mean()
        axs[1].plot(mae_h.index / 2, mae_h.values, lw=1.2,
                    ls="--" if m in ("persistencia", "sazonal_24h") else "-", label=m)
    axs[1].set_xlabel("horizonte (h)")
    axs[1].set_ylabel("MAE (cm)")
    axs[1].set_title("Erro médio por horizonte")
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.RESULTADOS / "comparacao.png", dpi=150)
    print(f"\ngráfico: {C.RESULTADOS / 'comparacao.png'}")
except ImportError:
    pass
