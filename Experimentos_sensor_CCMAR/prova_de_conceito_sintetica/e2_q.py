import sys,json,time
from comum import *
cod,seed=sys.argv[1],int(sys.argv[2])
n,L,IT,B,lr=4,3,100,48,.05
om=np.array([W[k] for k in BASE]) if cod=='mare' else 2*np.pi/24*np.array([1,2,4,8])
t,limpo,y=sinal(seed); tr,te=t[:NTR],t[NTR:]; yt=y[:NTR]; mu=yt.mean(); yt=yt-mu
g=np.random.default_rng(seed); th=g.uniform(0,2*np.pi,2*n*L); m=np.zeros_like(th); v=m.copy(); t0=time.time()
def ls(F,z):
    A=np.c_[np.ones(len(F)),F]; return np.linalg.lstsq(A,z,rcond=1e-6)[0]
def p(F,c): return np.c_[np.ones(len(F)),F]@c
for it in range(IT):
    idx=g.choice(NTR,B,replace=False); x=tr[idx]; z=yt[idx]
    F=feats(x,om,th,n,L); c=ls(F,z); r=p(F,c)-z; G=np.zeros_like(th)
    for k in range(len(th)):
        tp=th.copy(); tp[k]+=np.pi/2; tm=th.copy(); tm[k]-=np.pi/2
        dF=.5*(feats(x,om,tp,n,L)-feats(x,om,tm,n,L)); G[k]=2*np.mean(r*(dF@c[1:]))
    m=.9*m+.1*G; v=.999*v+.001*G**2; a=m/(1-.9**(it+1)); b=v/(1-.999**(it+1)); th=th-lr*a/(np.sqrt(b)+1e-8)
F=feats(tr,om,th,n,L); c=ls(F,yt); Ft=feats(te,om,th,n,L); yp=p(Ft,c)+mu
res={'modelo':'Q-'+cod,'seed':seed,'mae':float(np.mean(abs(yp-limpo[NTR:]))),'parametros':len(th)+5,'tempo_s':time.time()-t0}
print(res); json.dump(res,open(f'r2_{cod}_{seed}.json','w')); np.save(f'th_{cod}_{seed}.npy',th); np.save(f'c_{cod}_{seed}.npy',c)
