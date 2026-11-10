from comum import *
import json
om=np.array([W[k] for k in BASE]); n=4; g=np.random.default_rng(7); t=np.sort(g.uniform(0,8766,4000)); out={}
def fit(y,fr):
    A=desenho(t,fr); c,*_=np.linalg.lstsq(A,y,rcond=None); p=A@c; return 1-((y-p)**2).sum()/((y-y.mean())**2).sum(),c
for L,km,nome in [(2,1,'1 camada'),(3,2,'2 camadas')]:
    fr=combos(om,km); d={'frequencias_distintas':len(fr),'ate':{}}; alvo=['M2','M4','MS4','MK3','MSf']
    tab={f'Z{j}':{k:[] for k in alvo} for j in range(n)}; r2min=1
    for s in range(5):
        th=g.uniform(0,2*np.pi,2*n*L); F=feats(t,om,th,n,L)
        for j in range(n):
            r2,c=fit(F[:,j],fr); r2min=min(r2min,r2)
            for k in alvo:
                i=np.argmin(abs(fr-W[k])); tab[f'Z{j}'][k].append(float(np.hypot(c[1+2*i],c[2+2*i])))
    d['R2_min']=r2min; d['amp_media']={o:{k:float(np.mean(v)) for k,v in x.items()} for o,x in tab.items()}; out[nome]=d
print(json.dumps(out,indent=1)); json.dump(out,open('r1.json','w'),indent=1)
