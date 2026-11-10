"""Funções compartilhadas. Ajuste aqui apenas se 00_verificar.py acusar problema."""
import json, os, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

AQUI = Path(__file__).resolve().parent
ARQUIVO_SERIE = AQUI.parent / "dados" / "sensor_1_CCMAR.csv"
COLUNA_TEMPO = None   # None = detectar sozinho; ou informe o nome da coluna
COLUNA_NIVEL = None   # None = detectar sozinho; ou informe o nome da coluna
ESCALA_PARA_CM = None # None = detectar (metros viram cm); ou informe 1 ou 100
SAIDA = AQUI / "saida"; SAIDA.mkdir(exist_ok=True)

CTX, H = 480, 48
QUANTIS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
I10, I50, I90 = 0, 4, 8
PASSO_HIST, PASSO_TESTE, N_TESTE = 2, 24, 39
DT_H = 0.5
FASE1 = {"mae": 4.62, "wql": 1.830, "cobertura": 0.768}

# Velocidades angulares em graus por hora
VEL = {"M2": 28.9841042, "S2": 30.0, "N2": 28.4397295, "K1": 15.0410686, "O1": 13.9430356,
       "M4": 57.9682084, "MS4": 58.9841042, "MK3": 44.0251729, "MSf": 1.0158958}
W = {k: np.deg2rad(v) for k, v in VEL.items()}
BASE4 = ["M2", "S2", "K1", "O1"]
TODAS9 = BASE4 + ["M4", "MS4", "MK3", "MSf", "N2"]


def _coluna_tempo(df):
    if COLUNA_TEMPO:
        return COLUNA_TEMPO
    melhor, nota = None, 0.0
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            continue
        p = pd.to_datetime(s, errors="coerce", dayfirst=_primeiro_dia(s)).notna().mean()
        if p > nota:
            melhor, nota = c, p
    if melhor is None or nota < 0.9:
        raise SystemExit(f"Não achei a coluna de data e hora em {list(df.columns)}. Preencha COLUNA_TEMPO em comum.py.")
    return melhor


def _primeiro_dia(s):
    a = str(s.dropna().iloc[0])
    return a[:4].isdigit() is False


def _para_numero(s):
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def carregar_serie(verboso=False):
    """Devolve (índice de tempo, nível em cm), em grade de 30 min, com lacunas de até 3 h interpoladas."""
    df = pd.read_csv(ARQUIVO_SERIE, sep=None, engine="python")
    ct = _coluna_tempo(df)
    if COLUNA_NIVEL:
        cn = COLUNA_NIVEL
    else:
        cand = [c for c in df.columns if c != ct]
        pref = [c for c in cand if any(p in c.lower() for p in ("nivel", "nível", "level", "altura", "water"))]
        cand2 = pref or cand
        nums = {c: _para_numero(df[c]) for c in cand2}
        nums = {c: s for c, s in nums.items() if s.notna().mean() > 0.9 and s.std() > 0}
        if not nums:
            raise SystemExit(f"Não achei a coluna de nível em {list(df.columns)}. Preencha COLUNA_NIVEL em comum.py.")
        if len(nums) > 1 and not pref:
            print(f"AVISO: várias colunas numéricas {list(nums)}; usando a primeira. Se estiver errado, preencha COLUNA_NIVEL.")
        cn = list(nums)[0]
    t = pd.to_datetime(df[ct], errors="coerce", dayfirst=_primeiro_dia(df[ct]))
    if getattr(t.dt, "tz", None) is not None:
        t = t.dt.tz_localize(None)
    s = pd.Series(_para_numero(df[cn]).values, index=t).dropna()
    s = s[~s.index.isna()].sort_index()
    bruto = len(s)
    s = s.groupby(s.index).mean().resample("30min").mean()
    faltando = int(s.isna().sum())
    s = s.interpolate(limit=6, limit_area="inside")
    restantes = int(s.isna().sum())
    if restantes:
        raise SystemExit(f"Restaram {restantes} lacunas maiores que 3 h. Precisamos decidir como tratá-las.")
    esc = ESCALA_PARA_CM
    if esc is None:
        esc = 100.0 if (s.max() - s.min()) < 10 else 1.0
    s = s * esc
    if verboso:
        print(f"Coluna de tempo: {ct} | coluna de nível: {cn} | escala para cm: {esc}")
        print(f"Leituras: {bruto} | faltando: {faltando} | restantes: {restantes} | grade: {len(s)} pontos")
        print(f"Período: {s.index[0]} a {s.index[-1]} | amplitude: {s.min():.1f} a {s.max():.1f} cm")
    return s.index, s.values.astype(float)


def origens_teste(n):
    return np.array(sorted(n - H - PASSO_TESTE * k for k in range(N_TESTE)))


def desenho(t, nomes):
    x = [np.ones_like(t)]
    for k in nomes:
        x += [np.cos(W[k] * t), np.sin(W[k] * t)]
    return np.column_stack(x)


def carregar_historico():
    f = SAIDA / "historico_chronos.npz"
    if not f.exists():
        raise SystemExit("Falta saida/historico_chronos.npz. Rode antes o 01_historico_chronos.py.")
    z = np.load(f, allow_pickle=True)
    return {k: z[k] for k in z.files}


def divisao(hist):
    """Janelas de treino terminam antes da primeira previsão de teste."""
    o, n = hist["origens"], int(hist["n"])
    te = origens_teste(n)
    ini = te.min()
    idx = {int(v): i for i, v in enumerate(o)}
    i_te = np.array([idx[int(v)] for v in te])
    i_tr = np.where(o + H <= ini)[0]
    return i_tr, i_te, idx


def pinball(y, q):
    """Perda quantílica média por janela (mesma definição de WQL da fase 1)."""
    tau = np.array(QUANTIS)[None, None, :]
    e = y[:, :, None] - q
    return np.maximum(tau * e, (tau - 1) * e).mean(axis=(1, 2))


def metricas(y, q):
    mae = np.abs(y - q[:, :, I50]).mean(axis=1)
    wql = pinball(y, q)
    cob = ((y >= q[:, :, I10]) & (y <= q[:, :, I90])).mean(axis=1)
    lar = (q[:, :, I90] - q[:, :, I10]).mean(axis=1)
    return {"mae": mae, "wql": wql, "cobertura": cob, "largura": lar}


def resumo(m):
    return {k: float(np.mean(v)) for k, v in m.items()}


def salvar(nome, obj):
    def conv(x):
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        return str(x)
    (SAIDA / nome).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=conv), encoding="utf-8")
