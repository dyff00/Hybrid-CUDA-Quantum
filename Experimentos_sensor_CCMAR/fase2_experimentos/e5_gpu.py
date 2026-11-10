"""Tempo do mesmo circuito em CUDA-Q (CPU e GPU) e PennyLane. Sem argumentos, coordena tudo."""
import json, os, subprocess, sys, time
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np
QS = [int(v) for v in os.environ.get("E5_QS", "4,12,18,22,26").split(",")]; L = 3
def medir_cudaq(alvo, n):
    import cudaq
    cudaq.set_target(alvo)
    @cudaq.kernel
    def qc(ang: list[float], th: list[float], n: int, lay: int):
        q = cudaq.qvector(n); i = 0
        for l in range(lay):
            for j in range(n):
                ry(th[i], q[j]); rz(th[i + 1], q[j]); i += 2
            for j in range(n - 1): x.ctrl(q[j], q[j + 1])
            if l < lay - 1:
                for j in range(n): rz(ang[j], q[j])
    H = sum((cudaq.spin.z(j) for j in range(1, n)), cudaq.spin.z(0)); th = list(map(float, np.random.default_rng(0).uniform(0, 6.28, 2 * n * L))); a = [.3] * n
    cudaq.observe(qc, H, a, th, n, L); r, t0 = 0, time.perf_counter()
    while True:
        v = cudaq.observe(qc, H, a, th, n, L).expectation(); r += 1
        if time.perf_counter() - t0 > 2 or r >= 300: break
    print(json.dumps({"ms": (time.perf_counter() - t0) / r * 1e3, "valor": v, "rep": r}))
def medir_pl(n):
    import pennylane as qml
    th = np.random.default_rng(0).uniform(0, 6.28, 2 * n * L); d = qml.device("lightning.qubit", wires=n)
    @qml.qnode(d, diff_method=None)
    def f():
        i = 0
        for l in range(L):
            for j in range(n):
                qml.RY(th[i], j); qml.RZ(th[i + 1], j); i += 2
            for j in range(n - 1): qml.CNOT([j, j + 1])
            if l < L - 1:
                for j in range(n): qml.RZ(.3, j)
        return qml.expval(sum(qml.PauliZ(j) for j in range(n)))
    f(); r, t0 = 0, time.perf_counter()
    while True:
        v = float(f()); r += 1
        if time.perf_counter() - t0 > 2 or r >= 300: break
    print(json.dumps({"ms": (time.perf_counter() - t0) / r * 1e3, "valor": v, "rep": r}))
if len(sys.argv) > 1:
    medir_pl(int(sys.argv[2])) if sys.argv[1] == "pl" else medir_cudaq(sys.argv[1], int(sys.argv[2])); sys.exit()
SAIDA = Path(__file__).resolve().parent / "saida"; SAIDA.mkdir(exist_ok=True); env = {k: v for k, v in os.environ.items() if k != "OMP_NUM_THREADS"}
R = {"nucleos": os.cpu_count()}
try: R["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
except Exception: R["gpu"] = None
tab = {}
for nome, cmd in (("qpp-cpu", "qpp-cpu"), ("nvidia", "nvidia"), ("lightning.qubit", "pl")):
    tab[nome] = {}
    for n in QS:
        try:
            p = subprocess.run([sys.executable, __file__, cmd, str(n)], capture_output=True, text=True, timeout=240, env=env)
            ult = [l for l in p.stdout.splitlines() if l.startswith("{")]
            tab[nome][n] = json.loads(ult[-1]) if ult else {"erro": (p.stderr or p.stdout)[-300:]}
        except subprocess.TimeoutExpired:
            tab[nome][n] = {"erro": "passou de 4 min"}
        print(nome, n, tab[nome][n], flush=True)
        if "erro" in tab[nome][n]: break
R["ms_por_avaliacao"] = tab
(SAIDA / "r5_gpu.json").write_text(json.dumps(R, indent=1, ensure_ascii=False), encoding="utf-8")
