"""Passo 3b – Os modelos são diferentes de verdade, ou é acaso?

Usa resultados/metricas_por_janela.csv (gerado pelo 03_avaliar.py): o erro de
cada modelo em cada janela de 24 h. Como todos os modelos preveem as MESMAS
janelas, a comparação é PAREADA (janela a janela).

Testes:
  1. Friedman: existe diferença entre os modelos, no conjunto?
     (alternativa não paramétrica à ANOVA de medidas repetidas)
  2. ANOVA de medidas repetidas: o mesmo, no formato do script da Miriã
  3. Wilcoxon pareado, cada par de modelos, com correção de Holm
     (porque fazemos muitas comparações ao mesmo tempo)
  4. Intervalo de confiança de 95% da diferença de MAE, por bootstrap em blocos

Cuidado: as janelas começam a cada 12 h e duram 24 h, então janelas vizinhas
se sobrepõem e não são independentes. Por isso:
  - o bootstrap sorteia blocos de janelas vizinhas, não janelas soltas;
  - repetimos tudo só com janelas SEM sobreposição (uma a cada 24 h) como
    teste de robustez. Se a conclusão muda, ela não é confiável.

Ambiente: .venv-principal
Uso:
  python 05_teste_estatistico.py              # métrica MAE
  python 05_teste_estatistico.py --metrica WQL_cm
"""
import argparse
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

import config as C

p = argparse.ArgumentParser()
p.add_argument("--metrica", default="MAE_cm", choices=["MAE_cm", "WQL_cm"])
p.add_argument("--bootstrap", type=int, default=10000)
p.add_argument("--bloco", type=int, default=4, help="janelas por bloco no bootstrap")
a = p.parse_args()

dados = pd.read_csv(C.RESULTADOS / "metricas_por_janela.csv", parse_dates=["origem"])
tabela = dados.pivot(index="origem", columns="modelo", values=a.metrica).sort_index()
ordem = tabela.mean().sort_values().index.tolist()
tabela = tabela[ordem]
rng = np.random.default_rng(C.SEMENTE)


def holm(pvals):
    """Correção de Holm-Bonferroni (mais poderosa que Bonferroni simples)."""
    p = np.asarray(pvals)
    idx = np.argsort(p)
    m = len(p)
    ajust = np.empty(m)
    acumulado = 0.0
    for k, i in enumerate(idx):
        acumulado = max(acumulado, (m - k) * p[i])
        ajust[i] = min(1.0, acumulado)
    return ajust


def anova_medidas_repetidas(t: pd.DataFrame):
    x = t.to_numpy()
    n, k = x.shape
    media = x.mean()
    ss_mod = n * ((x.mean(axis=0) - media) ** 2).sum()
    ss_jan = k * ((x.mean(axis=1) - media) ** 2).sum()
    ss_tot = ((x - media) ** 2).sum()
    ss_err = ss_tot - ss_mod - ss_jan
    gl_mod, gl_err = k - 1, (k - 1) * (n - 1)
    f = (ss_mod / gl_mod) / (ss_err / gl_err)
    return f, gl_mod, gl_err, stats.f.sf(f, gl_mod, gl_err)


def ic_bootstrap(d: np.ndarray, bloco: int, B: int):
    """IC 95% da média de d, sorteando blocos de janelas consecutivas."""
    n = len(d)
    bloco = max(1, min(bloco, n))
    n_blocos = int(np.ceil(n / bloco))
    inicios = rng.integers(0, n - bloco + 1, size=(B, n_blocos))
    amostras = np.stack([d[i : i + bloco] for i in range(n - bloco + 1)])  # (n-b+1, b)
    medias = amostras[inicios].reshape(B, -1)[:, :n].mean(axis=1)
    return np.percentile(medias, [2.5, 97.5])


def analisar(t: pd.DataFrame, titulo: str, bloco: int):
    print(f"\n{'=' * 78}\n{titulo}  ({len(t)} janelas, métrica {a.metrica})\n{'=' * 78}")
    print("\nMédia por modelo (menor = melhor):")
    print(t.mean().round(3).to_string())

    chi2, p_f = stats.friedmanchisquare(*[t[c] for c in t.columns])
    f, g1, g2, p_a = anova_medidas_repetidas(t)
    print(f"\nFriedman:                    qui² = {chi2:.2f}, p = {p_f:.2g}")
    print(f"ANOVA de medidas repetidas:  F({g1},{g2}) = {f:.2f}, p = {p_a:.2g}")
    print("  p < 0,05 -> há diferença entre pelo menos dois modelos.")

    linhas = []
    for m1, m2 in combinations(t.columns, 2):
        d = (t[m1] - t[m2]).to_numpy()
        _, p_w = stats.wilcoxon(d) if np.any(d != 0) else (0, 1.0)
        lo, hi = ic_bootstrap(d, bloco, a.bootstrap)
        linhas.append(dict(modelo_A=m1, modelo_B=m2, dif_media=d.mean(), ic95_inf=lo, ic95_sup=hi,
                           A_melhor_em=f"{(d < 0).mean():.0%}", p_wilcoxon=p_w))
    pares = pd.DataFrame(linhas)
    pares["p_holm"] = holm(pares["p_wilcoxon"])
    pares["conclusao"] = np.where(
        (pares["p_holm"] < 0.05) & ((pares["ic95_sup"] < 0) | (pares["ic95_inf"] > 0)),
        "A melhor que B", "sem diferença clara")
    print("\nComparação par a par (dif_media = A − B; negativo = A erra menos):")
    print(pares.round({"dif_media": 3, "ic95_inf": 3, "ic95_sup": 3, "p_wilcoxon": 4, "p_holm": 4})
          .to_string(index=False))
    return pares


todas = analisar(tabela, "TODAS AS JANELAS (com sobreposição de 12 h)", a.bloco)
sem_sobrep = tabela.iloc[::2]
robustez = analisar(sem_sobrep, "SÓ JANELAS SEM SOBREPOSIÇÃO (uma a cada 24 h)", max(1, a.bloco // 2))

todas.assign(cenario="todas").pipe(lambda x: pd.concat([x, robustez.assign(cenario="sem_sobreposicao")])) \
    .to_csv(C.RESULTADOS / f"teste_estatistico_{a.metrica}.csv", index=False)
print(f"\nsalvo: {C.RESULTADOS / f'teste_estatistico_{a.metrica}.csv'}")

mudou = (todas["conclusao"].values != robustez["conclusao"].values)
if mudou.any():
    print("\nAtenção: a conclusão muda sem as janelas sobrepostas para:")
    for _, r in todas[mudou].iterrows():
        print(f"  {r.modelo_A} vs {r.modelo_B}")
    print("Nesses pares, trate o resultado como inconclusivo.")
