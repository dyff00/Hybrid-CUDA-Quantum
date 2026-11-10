"""Comparação 1: em que bandas de frequência está o erro do Chronos-2."""
from comum import *
h = carregar_historico(); o, Q, Y, y = h["origens"], h["Q"], h["Y"], h["serie"]
BANDAS = [("Baixa frequência (acima de 30 h)", 30, 1e9), ("Diurna (22 a 27 h)", 22, 27), ("Semidiurna (11,5 a 13 h)", 11.5, 13),
          ("Terdiurna (7,5 a 8,5 h)", 7.5, 8.5), ("Quarto-diurna (5,8 a 6,4 h)", 5.8, 6.4)]
def espectro(x, dt=1.0):
    x = x - x.mean(); w = np.hanning(len(x)); P = np.abs(np.fft.rfft(x * w)) ** 2; f = np.fft.rfftfreq(len(x), dt)
    return f[1:], P[1:]
def fracoes(x):
    f, P = espectro(x); per = 1 / f; tot = P.sum(); s = {}
    for nome, a, b in BANDAS:
        m = (per >= a) & (per < b); s[nome] = {"energia": float(P[m].sum() / tot), "ruido_branco": float(m.mean())}
    m = np.ones_like(per, bool)
    for _, a, b in BANDAS: m &= ~((per >= a) & (per < b))
    s["Demais frequências"] = {"energia": float(P[m].sum() / tot), "ruido_branco": float(m.mean())}
    return s
saida = {"bandas": {}, "constituintes": {}}; linhas = []
nivel = y[::2]
saida["bandas"]["Nível observado"] = fracoes(nivel)
def amps(x, t):
    A = desenho(t, TODAS9); c = np.linalg.lstsq(A, x - x.mean(), rcond=None)[0]
    return {k: float(np.hypot(c[1 + 2 * i], c[2 + 2 * i])) for i, k in enumerate(TODAS9)}
t = (o[0] + np.arange(len(o)) * PASSO_HIST) * DT_H
for horas in (1, 6, 12, 24):
    p = horas * 2 - 1
    e = Y[:, p] - Q[:, p, I50]; tt = (o + p) * DT_H
    saida["bandas"][f"Erro em {horas} h"] = fracoes(e)
    saida["constituintes"][f"Erro em {horas} h"] = amps(e, tt)
    f, P = espectro(e); pd.DataFrame({"periodo_h": 1 / f, "potencia": P}).to_csv(SAIDA / f"espectro_erro_{horas}h.csv", index=False)
    saida.setdefault("desvio_erro_cm", {})[f"{horas} h"] = float(e.std())
t2 = np.arange(len(y)) * DT_H; saida["constituintes"]["Nível observado"] = amps(y, t2)
_, i_te, _ = divisao(h)
saida["mae_por_horizonte_cm"] = {str((p + 1) / 2): float(np.abs(Y[:, p] - Q[:, p, I50]).mean()) for p in range(1, H, 2)}
salvar("r1_onde_erra.json", saida)
print("Fração da variância do erro por banda")
for k, v in saida["bandas"].items():
    print(k, {a[:12]: round(b["energia"], 3) for a, b in v.items()})
