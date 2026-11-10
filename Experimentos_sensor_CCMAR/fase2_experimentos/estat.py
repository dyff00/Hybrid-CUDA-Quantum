import numpy as np
from scipy.stats import wilcoxon

def _p(d):
    d = d[np.abs(d) > 1e-12]
    return float(wilcoxon(d).pvalue) if len(d) >= 5 else float("nan")

def bootstrap_blocos(d, bloco=4, n=5000, seed=0):
    g = np.random.default_rng(seed); m = len(d); k = int(np.ceil(m / bloco)); r = []
    for _ in range(n):
        ini = g.integers(0, m - bloco + 1, k)
        r.append(np.concatenate([d[i:i + bloco] for i in ini])[:m].mean())
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]

def holm(ps):
    ps = np.array(ps, float); idx = np.argsort(ps); r = np.zeros(len(ps)); ac = 0.0
    for pos, i in enumerate(idx):
        ac = max(ac, min(1.0, ps[i] * (len(ps) - pos))); r[i] = ac
    return r.tolist()

def comparar(base, modelos, chave="mae"):
    """Diferença pareada por janela (modelo menos base): negativo significa modelo melhor."""
    nomes = list(modelos); s = []
    for k in nomes:
        d = modelos[k][chave] - base[chave]
        s.append({"modelo": k, "dif_media": float(d.mean()), "ic95": bootstrap_blocos(d),
                  "p_39": _p(d), "p_20_sem_sobreposicao": _p(d[1::2])})
    h39 = holm([x["p_39"] for x in s]); h20 = holm([x["p_20_sem_sobreposicao"] for x in s])
    for x, a, b in zip(s, h39, h20):
        x["p_holm_39"], x["p_holm_20"] = a, b
    return s
