"""Comparação 2: quanto a maré harmônica corrige do erro do Chronos-2."""
from comum import *
from estat import comparar
h = carregar_historico(); o, Q, Y, y = h["origens"], h["Q"], h["Y"], h["serie"]; n = int(h["n"]); itr, ite, _ = divisao(h)
tv = lambda i: ((o[i][:, None] + np.arange(H)[None]) * DT_H)
def ajusta(nomes, t, e, lam=1e-3):
    A = desenho(t.ravel(), nomes); return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ e.ravel())
Etr = Y[itr] - Q[itr, :, I50]; ttr, tte = tv(itr), tv(ite)
ref = metricas(Y[ite], Q[ite]); mod = {"Chronos-2": ref}; cor = {}
for nome, cs in (("Chronos-2 + 4 constituintes", BASE4), ("Chronos-2 + 9 constituintes", TODAS9)):
    c = ajusta(cs, ttr, Etr); d = (desenho(tte.ravel(), cs) @ c).reshape(tte.shape); cor[nome] = d
    mod[nome] = metricas(Y[ite], Q[ite] + d[:, :, None])
ini = origens_teste(n).min(); tt_all = np.arange(n) * DT_H
c = np.linalg.lstsq(desenho(tt_all[:ini], TODAS9), y[:ini], rcond=None)[0]; ha = desenho(tt_all, TODAS9) @ c
def pontual(p): return {"mae": np.abs(Y[ite] - p).mean(axis=1)}
puro = np.stack([ha[v:v + H] for v in o[ite]]); mod2 = {"Harmônica pura": pontual(puro)}
an = y[:ini] - ha[:ini]; rho = float(an[1:] @ an[:-1] / (an[:-1] @ an[:-1]))
pa = np.stack([ha[v:v + H] + (y[v - 1] - ha[v - 1]) * rho ** (np.arange(H) + 1) for v in o[ite]]); mod2["Harmônica + persistência da anomalia"] = pontual(pa)
s = {"resumo": {k: resumo(v) for k, v in mod.items()} | {k: resumo(v) for k, v in mod2.items()}, "rho_anomalia": rho}
s["comparacao_mae"] = comparar(ref, {k: v for k, v in (mod | mod2).items() if k != "Chronos-2"}, "mae")
s["comparacao_wql"] = comparar(ref, {k: v for k, v in mod.items() if k != "Chronos-2"}, "wql")
salvar("r2_mare.json", s)
ord_ = np.argsort(ref["mae"]); ex = {"melhor": ord_[0], "mediana": ord_[len(ord_) // 2], "pior": ord_[-1]}; L = []
tempos = h["tempos"]
d9 = cor["Chronos-2 + 9 constituintes"]
for nome, j in ex.items():
    v = int(o[ite][j])
    for p in range(H):
        L.append({"exemplo": nome, "tempo": tempos[v + p], "observado": Y[ite][j, p], "chronos_p10": Q[ite][j, p, I10], "chronos_p50": Q[ite][j, p, I50],
                  "chronos_p90": Q[ite][j, p, I90], "com_mare_p50": Q[ite][j, p, I50] + d9[j, p], "harmonica_persistencia": pa[j, p]})
pd.DataFrame(L).to_csv(SAIDA / "exemplos_janelas.csv", index=False)
for k, v in s["resumo"].items(): print(f"{k:40s} " + " ".join(f"{a}={b:.3f}" for a, b in v.items()))
