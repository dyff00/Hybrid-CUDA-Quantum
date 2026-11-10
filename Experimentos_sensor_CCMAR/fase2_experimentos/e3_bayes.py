"""Comparação 3: incerteza do Chronos-2 antes e depois do tratamento bayesiano do erro."""
from scipy.stats import t as student
from comum import *
from estat import comparar
h = carregar_historico(); o, Q, Y = h["origens"], h["Q"], h["Y"]; itr, ite, idx = divisao(h)
FATOR = 2 * 1.2815515655; BIN = 6; NB = H // BIN
def previa(i):
    """Erro já verificado: previsão emitida 12 h antes, média das últimas 3 h antes da origem."""
    v = int(o[i]) - PASSO_TESTE
    return None if v not in idx else float((Y[idx[v], 18:24] - Q[idx[v], 18:24, I50]).mean())
def ok(ids): 
    r = [(i, previa(i)) for i in ids]; r = [(i, e) for i, e in r if e is not None]; return np.array([a for a, _ in r]), np.array([b for _, b in r])
itr2, e_tr = ok(itr); ite2, e_te = ok(ite)
if len(ite2) != len(ite): raise SystemExit("Faltou erro prévio em janelas de teste.")
ordem = np.argsort(o[itr2]); itr2, e_tr = itr2[ordem], e_tr[ordem]
corte = int(len(itr2) * .8); o_cal = o[itr2][corte]
fit = np.where(o[itr2] + H <= o_cal)[0]; cal = np.arange(corte, len(itr2))
tv = lambda ids: (o[ids][:, None] + np.arange(H)[None]) * DT_H
def harm(ids_fit):
    A = desenho(tv(ids_fit).ravel(), TODAS9); e = (Y[ids_fit] - Q[ids_fit, :, I50]).ravel()
    return np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ e)
cH = harm(itr2[fit]); corr = lambda ids: (desenho(tv(ids).ravel(), TODAS9) @ cH).reshape(len(ids), H)
esc = lambda ids: np.maximum((Q[ids, :, I90] - Q[ids, :, I10]) / FATOR, 1e-3)
# Regressão bayesiana conjugada (Normal-Inversa-Gama), por faixa de horizonte
post = []
Rfit = Y[itr2[fit]] - Q[itr2[fit], :, I50] - corr(itr2[fit]); Sfit = esc(itr2[fit])
for b in range(NB):
    p = np.arange(b * BIN, (b + 1) * BIN); yb = (Rfit[:, p] / Sfit[:, p]).ravel()
    xb = np.column_stack([np.ones(yb.size), np.repeat(e_tr[fit] / np.maximum(Sfit[:, p].mean(1), 1e-3), BIN)])
    V0 = np.eye(2) * 25.0; a0, b0 = 2.0, 2.0; Vn = np.linalg.inv(np.linalg.inv(V0) + xb.T @ xb); mn = Vn @ (xb.T @ yb)
    an = a0 + len(yb) / 2; bn = b0 + .5 * (yb @ yb - mn @ np.linalg.inv(Vn) @ mn); post.append((mn, Vn, an, bn))
TAU = np.array(QUANTIS)
def bayes_q(ids, e0):
    S = esc(ids); C = Q[ids, :, I50] + corr(ids); out = np.zeros((len(ids), H, 9))
    for b in range(NB):
        mn, Vn, an, bn = post[b]; p = np.arange(b * BIN, (b + 1) * BIN)
        for k in p:
            x = np.column_stack([np.ones(len(ids)), e0 / np.maximum(S[:, p].mean(1), 1e-3)]); mu = x @ mn
            var = (bn / an) * (1 + np.einsum("ij,jk,ik->i", x, Vn, x))
            z = student.ppf(TAU, 2 * an)[None, :]
            out[:, k, :] = (C[:, k] + S[:, k] * mu)[:, None] + (S[:, k] * np.sqrt(var))[:, None] * z
    return out
def fator_conformal(qc, yc, alvo=.8):
    med = qc[:, :, I50]; fs = []
    for b in range(NB):
        p = np.arange(b * BIN, (b + 1) * BIN); inf = np.maximum(med[:, p] - qc[:, p, I10], 1e-6); sup = np.maximum(qc[:, p, I90] - med[:, p], 1e-6)
        sc = np.maximum((med[:, p] - yc[:, p]) / inf, (yc[:, p] - med[:, p]) / sup).ravel(); m = len(sc)
        fs.append(float(np.quantile(sc, min(1, np.ceil((m + 1) * alvo) / m))))
    return fs
def aplica(q, fs):
    med = q[:, :, I50, None]; out = q.copy()
    for b, f in enumerate(fs):
        p = np.arange(b * BIN, (b + 1) * BIN); out[:, p, :] = med[:, p] + f * (q[:, p, :] - med[:, p])
    return out
idc = itr2[cal]; ec = e_tr[cal]; Yt = Y[ite2]
Qc = Q[idc]; Qb = bayes_q(idc, ec)
fc = fator_conformal(Qc, Y[idc]); fb = fator_conformal(Qb, Y[idc])
Qt = Q[ite2]; Qbt = bayes_q(ite2, e_te)
qs = {"Chronos-2": Qt, "Chronos-2 + conformal": aplica(Qt, fc), "Bayes": Qbt, "Bayes + conformal": aplica(Qbt, fb)}
m = {k: metricas(Yt, v) for k, v in qs.items()}; base = m["Chronos-2"]
s = {"resumo": {k: resumo(v) for k, v in m.items()}, "alvo_cobertura": .8, "janelas_ajuste": len(fit), "janelas_calibracao": len(cal),
     "fatores_conformal": {"chronos": fc, "bayes": fb}, "phi_erro_previo": [float(p[0][1]) for p in post]}
s["comparacao_mae"] = comparar(base, {k: v for k, v in m.items() if k != "Chronos-2"}, "mae")
s["comparacao_wql"] = comparar(base, {k: v for k, v in m.items() if k != "Chronos-2"}, "wql")
s["cobertura_por_horizonte"] = {k: [float(((Yt[:, b*BIN:(b+1)*BIN] >= v[:, b*BIN:(b+1)*BIN, I10]) & (Yt[:, b*BIN:(b+1)*BIN] <= v[:, b*BIN:(b+1)*BIN, I90])).mean()) for b in range(NB)] for k, v in qs.items()}
salvar("r3_bayes.json", s)
for k, v in s["resumo"].items(): print(f"{k:24s} " + " ".join(f"{a}={b:.3f}" for a, b in v.items()))
