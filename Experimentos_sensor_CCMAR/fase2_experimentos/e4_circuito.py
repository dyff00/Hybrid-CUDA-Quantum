"""Comparação 4: circuito de maré contra KAN, MLP e harmônica, no erro real do Chronos-2.
Uso: python e4_circuito.py treinar mare 0 | python e4_circuito.py resumir"""
import os, sys, time, itertools
os.environ.setdefault("OMP_NUM_THREADS", "1")
from comum import *
SEMS_Q, SEMS_C = 3, 5; N, L = 4, 3; IT, LOTE = int(os.environ.get("E4_IT", 300)), 48; LRS = (0.02, 0.05)
h = carregar_historico(); o, Q, Y = h["origens"], h["Q"], h["Y"]; itr, ite, _ = divisao(h)
tv = lambda ids: (o[ids][:, None] + np.arange(H)[None]) * DT_H
ttr = tv(itr).ravel(); etr = (Y[itr] - Q[itr, :, I50]).ravel()
ts = np.unique(ttr); soma = np.bincount(np.searchsorted(ts, ttr), etr); cont = np.bincount(np.searchsorted(ts, ttr))
alvo = soma / cont; corte = int(len(ts) * .8); ORD = {"mare": np.array([W[k] for k in BASE4]), "gen": 2 * np.pi / 24 * np.array([1, 2, 4, 8])}
def ls(F, z, lam=1e-6):
    A = np.c_[np.ones(len(F)), F]; return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ z)
def prev(F, c): return np.c_[np.ones(len(F)), F] @ c
def treinar(cod, seed):
    if (SAIDA / f"q_{cod}_{seed}.npz").exists():
        print(f"{cod} {seed}: já treinado", flush=True); return
    import cudaq
    O = [cudaq.spin.z(j) for j in range(N)]
    @cudaq.kernel
    def qc(ang: list[float], th: list[float], n: int, lay: int):
        q = cudaq.qvector(n); i = 0
        for l in range(lay):
            for j in range(n):
                ry(th[i], q[j]); rz(th[i + 1], q[j]); i += 2
            for j in range(n - 1): x.ctrl(q[j], q[j + 1])
            if l < lay - 1:
                for j in range(n): rz(ang[j], q[j])
    om = ORD[cod]
    def feats(t, th):
        tl = list(map(float, th))
        return np.array([[s.expectation() for s in cudaq.observe(qc, O, [float(a * x) for a in om], tl, N, L)] for x in t])
    melhor = (1e18, None, None); t0 = time.time()
    tre, val = ts[:corte], ts[corte:]; ytre, yval = alvo[:corte], alvo[corte:]
    for lr in LRS:
        g = np.random.default_rng(seed); th = g.uniform(0, 2 * np.pi, 2 * N * L); m = np.zeros_like(th); v = m.copy()
        for it in range(IT):
            ii = g.choice(len(tre), LOTE, replace=False); x, z = tre[ii], ytre[ii]
            F = feats(x, th); c = ls(F, z); r = prev(F, c) - z; G = np.zeros_like(th)
            for k in range(len(th)):
                a = th.copy(); a[k] += np.pi / 2; b = th.copy(); b[k] -= np.pi / 2
                G[k] = 2 * np.mean(r * ((.5 * (feats(x, a) - feats(x, b))) @ c[1:]))
            m = .9 * m + .1 * G; v = .999 * v + .001 * G ** 2
            th = th - lr * (m / (1 - .9 ** (it + 1))) / (np.sqrt(v / (1 - .999 ** (it + 1))) + 1e-8)
            if (it + 1) % 25 == 0 or it == IT - 1:
                sub = np.arange(0, len(tre), 4); c2 = ls(feats(tre[sub], th), ytre[sub]); er = float(np.mean((prev(feats(val[::2], th), c2) - yval[::2]) ** 2))
                if er < melhor[0]: melhor = (er, th.copy(), lr)
    er, th, lr = melhor; c = ls(feats(ts, th), alvo); tt = tv(ite); C = prev(feats(tt.ravel(), th), c).reshape(tt.shape)
    np.savez(SAIDA / f"q_{cod}_{seed}.npz", th=th, c=c, corr=C, val=er, lr=lr, seg=time.time() - t0, om=om)
    print(f"{cod} {seed}: val={er:.3f} lr={lr} {time.time() - t0:.0f} s", flush=True)
def resumir():
    import jax, jax.numpy as jnp
    from scipy.optimize import minimize
    from estat import comparar
    jax.config.update("jax_enable_x64", True); tt = tv(ite); Yt, Qt = Y[ite], Q[ite]; ref = metricas(Yt, Qt); mod = {}; seeds = {}; par = {}
    def reg(nome, d, p, s=None):
        m = metricas(Yt, Qt + d[:, :, None]); seeds.setdefault(nome, []).append(float(m["mae"].mean())); par[nome] = p; return m
    for nome, cs in (("Harmônica, 4 constituintes", BASE4), ("Harmônica, 9 constituintes", TODAS9)):
        A = desenho(ts, cs); c = np.linalg.solve(A.T @ A + 1e-3 * np.eye(A.shape[1]), A.T @ alvo)
        mod[nome] = reg(nome, (desenho(tt.ravel(), cs) @ c).reshape(tt.shape), A.shape[1])
    om = ORD["mare"]; fx = lambda t: np.column_stack([f(w * t) for w in om for f in (np.cos, np.sin)])
    X = fx(ts); Xt = fx(tt.ravel()); mu, sd = alvo.mean(), alvo.std(); z = (alvo - mu) / sd
    def mlp(p, x): return jnp.tanh(x @ p[:24].reshape(8, 3) + p[24:27]) @ p[27:30] + p[30]
    def ch(u, d): return jnp.stack([u, 2 * u ** 2 - 1, 4 * u ** 3 - 3 * u][:d], -1)
    def kan(p, x):
        a = p[:48].reshape(8, 2, 3); hh = jnp.tanh((ch(x, 3)[:, :, None, :] * a[None]).sum((1, 3)) + p[54:56]); return (ch(hh, 3) * p[48:54].reshape(2, 3)[None]).sum((1, 2)) + p[56]
    def kan1(p, x): return (ch(x, 3) * p[:24].reshape(8, 3)[None]).sum((1, 2)) + p[24]
    MODS = [("MLP", mlp, 31, .3), ("KAN [8,2,1]", kan, 57, .1), ("KAN [8,1]", kan1, 25, .1)]
    grade = list(itertools.product((1e-5, 1e-3, 1e-1), (100, 300, 1000, 3000)))
    def fit(fn, X_, z_, p0, l2, mit):
        F = lambda p: jnp.mean((fn(p, X_) - z_) ** 2) + l2 * jnp.mean(p ** 2); vg = jax.jit(jax.value_and_grad(F))
        def f(p): a, b = vg(jnp.asarray(p)); return float(a), np.asarray(b, float)
        return minimize(f, p0, jac=True, method="L-BFGS-B", options={"maxiter": mit, "maxfun": mit * 2}).x
    for nome, fn, p, s0 in MODS:
        melhor_val = 1e18; melhor_m = None; acum = None
        for seed in range(SEMS_C):
            g = np.random.default_rng(seed); p0 = g.normal(0, s0, p); mel = (1e18, None)
            for l2, mit in grade:
                pv = fit(fn, X[:corte], z[:corte], p0, l2, mit); e = float(np.mean((np.asarray(fn(pv, X[corte:])) - z[corte:]) ** 2))
                if e < mel[0]: mel = (e, (l2, mit))
            pf = fit(fn, X, z, p0, *mel[1]); d = (np.asarray(fn(pf, Xt)) * sd + mu).reshape(tt.shape); m = reg(nome, d, p)
            if mel[0] < melhor_val: melhor_val, melhor_m = mel[0], m
        mod[nome] = melhor_m
    extra = {}
    for cod, nome in (("mare", "Circuito com espectro de maré"), ("gen", "Circuito genérico")):
        melhor_val = 1e18; ms = []
        for seed in range(SEMS_Q):
            f = SAIDA / f"q_{cod}_{seed}.npz"
            if not f.exists(): raise SystemExit(f"Falta {f.name}: rode o treino antes.")
            z_ = np.load(f); m = reg(nome, z_["corr"], 29); extra[f"{cod}_{seed}"] = {"val": float(z_["val"]), "lr": float(z_["lr"]), "segundos": float(z_["seg"])}
            if float(z_["val"]) < melhor_val: melhor_val, melhor_m = float(z_["val"]), m
        mod[nome] = melhor_m
    s = {"chronos_mae": float(ref["mae"].mean()), "modelos": {k: {"parametros": par[k], "mae_sementes": v, "mae_media": float(np.mean(v)), "mae_min": float(np.min(v)), "mae_max": float(np.max(v)), "mae_escolhido_por_validacao": float(mod[k]["mae"].mean())} for k, v in seeds.items()}, "circuitos": extra}
    s["comparacao_mae_vs_chronos"] = comparar(ref, mod, "mae")
    s["comparacao_mae_vs_harmonica9"] = comparar(mod["Harmônica, 9 constituintes"], {k: v for k, v in mod.items() if not k.startswith("Harmônica")}, "mae")
    salvar("r4_circuito.json", s); print(f"Chronos-2 sem correção: MAE {s['chronos_mae']:.3f}")
    for k, v in sorted(s["modelos"].items(), key=lambda a: a[1]["mae_media"]): print(f"{k:32s} p={v['parametros']:3d} MAE={v['mae_media']:.3f} ({v['mae_min']:.3f} a {v['mae_max']:.3f})")
if sys.argv[1] == "treinar": treinar(sys.argv[2], int(sys.argv[3]))
else: resumir()
