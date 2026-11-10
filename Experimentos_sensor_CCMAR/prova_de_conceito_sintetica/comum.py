import warnings; warnings.filterwarnings('ignore')
import numpy as np, itertools, cudaq
# velocidades angulares em graus/hora (constantes harmônicas padrão)
V={'M2':28.9841042,'S2':30.0,'N2':28.4397295,'K1':15.0410686,'O1':13.9430356,'M4':57.9682084,'MS4':58.9841042,'MK3':44.0251729,'MSf':1.0158958}
W={k:np.deg2rad(v) for k,v in V.items()}
BASE=['M2','S2','K1','O1']
AMP={'M2':8,'S2':6,'N2':2,'K1':9,'O1':11,'M4':1.5,'MS4':1.0,'MK3':1.2,'MSf':2.5}  # cm, SINTÉTICO
DT=0.5; NTR=2880; NTE=960
def sinal(seed):
    g=np.random.default_rng(1000+seed); t=np.arange(NTR+NTE)*DT
    fase={k:g.uniform(0,2*np.pi) for k in AMP}
    limpo=sum(AMP[k]*np.cos(W[k]*t+fase[k]) for k in AMP)
    ar=np.zeros_like(t); e=g.normal(0,1,len(t)); phi=0.995
    for i in range(1,len(t)): ar[i]=phi*ar[i-1]+e[i]
    ar=ar/ar.std()*5
    return t,limpo,limpo+ar+g.normal(0,.5,len(t))
@cudaq.kernel
def qc(ang:list[float],th:list[float],n:int,L:int):
    q=cudaq.qvector(n); i=0
    for l in range(L):
        for j in range(n):
            ry(th[i],q[j]); rz(th[i+1],q[j]); i+=2
        for j in range(n-1): x.ctrl(q[j],q[j+1])
        if l<L-1:
            for j in range(n): rz(ang[j],q[j])
def obs(n): return [cudaq.spin.z(j) for j in range(n)]
def feats(t,om,th,n,L):
    O=obs(n); r=[]
    for x in t:
        a=[float(o*x) for o in om]
        res=cudaq.observe(qc,O,a,list(map(float,th)),n,L)
        r.append([s.expectation() for s in res])
    return np.array(r)
def combos(om,kmax):
    s=set()
    for c in itertools.product(range(-kmax,kmax+1),repeat=len(om)):
        v=float(np.dot(c,om))
        if v>1e-9: s.add(round(v,9))
    return np.array(sorted(s))
def desenho(t,fr):
    return np.column_stack([np.ones_like(t)]+[f(w*t) for w in fr for f in (np.cos,np.sin)])
