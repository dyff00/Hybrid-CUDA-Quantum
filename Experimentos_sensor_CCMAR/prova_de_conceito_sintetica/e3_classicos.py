from comum import *
import json, jax, jax.numpy as jnp
from scipy.optimize import minimize
jax.config.update('jax_enable_x64',True)
R=[]
om=np.array([W[k] for k in BASE]); dic=combos(om,2)
for seed in range(3):
    t,limpo,y=sinal(seed); tr,te=t[:NTR],t[NTR:]; yt=y[:NTR]; gt=limpo[NTR:]
    def add(nome,yp,p): R.append({'modelo':nome,'seed':seed,'mae':float(np.mean(abs(yp-gt))),'parametros':p})
    for nome,ks in [('HA-4',BASE),('HA-8',BASE+['M4','MS4','MK3','MSf']),('HA-9',BASE+['M4','MS4','MK3','MSf','N2'])]:
        fr=[W[k] for k in ks]; c=np.linalg.lstsq(desenho(tr,fr),yt,rcond=None)[0]; add(nome,desenho(te,fr)@c,1+2*len(fr))
    A=desenho(tr,dic); Ate=desenho(te,dic); nv=NTR-480
    best=None
    for lam in 10.0**np.arange(-3,6):
        c=np.linalg.solve(A[:nv].T@A[:nv]+lam*np.eye(A.shape[1]),A[:nv].T@yt[:nv]); e=np.mean(abs(A[nv:]@c-yt[nv:]))
        if best is None or e<best[0]: best=(e,lam)
    lam=best[1]; c=np.linalg.solve(A.T@A+lam*np.eye(A.shape[1]),A.T@yt); add('Substituto clássico',Ate@c,A.shape[1])
    X=np.column_stack([f(w*t) for w in om for f in (np.cos,np.sin)]); Xtr,Xte=X[:NTR],X[NTR:]; mu=yt.mean(); s=yt.std(); z=(yt-mu)/s
    def fit(fn,p0):
        L=lambda p: jnp.mean((fn(p,Xtr)-z)**2)+1e-5*jnp.sum(p**2); vg=jax.jit(jax.value_and_grad(L))
        r=minimize(lambda p:[np.asarray(a,float) for a in vg(p)][0],p0,jac=lambda p:np.asarray(vg(p)[1]),method='L-BFGS-B',options={'maxiter':3000}); return r.x
    g=np.random.default_rng(seed)
    def mlp(p,x):
        W1=p[:24].reshape(8,3); b1=p[24:27]; w2=p[27:30]; b2=p[30]; return jnp.tanh(x@W1+b1)@w2+b2
    pm=fit(mlp,g.normal(0,.3,31)); add('MLP 8-3-1',np.asarray(mlp(pm,Xte))*s+mu,31)
    def cheb(u,d): 
        return jnp.stack([jnp.ones_like(u),u,2*u**2-1,4*u**3-3*u][1:d+1],-1)
    def kan(p,x):
        a=p[:48].reshape(8,2,3); b=p[48:54].reshape(2,3); c0=p[54:56]; h=(cheb(x,3)[:,:,None,:]*a[None]).sum((1,3))+c0
        h=jnp.tanh(h); return (cheb(h,3)*b[None]).sum((1,2))+p[56]
    pk=fit(kan,g.normal(0,.1,57)); add('KAN [8,2,1] Chebyshev',np.asarray(kan(pk,Xte))*s+mu,57)
    # erro bayesiano: regressão bayesiana conjugada sobre HA-9, com AR(1) branqueado
    ks=BASE+['M4','MS4','MK3','MSf','N2']; fr=[W[k] for k in ks]; A=desenho(tr,fr); c=np.linalg.lstsq(A,yt,rcond=None)[0]; e=yt-A@c
    phi=float(e[1:]@e[:-1]/(e[:-1]@e[:-1])); As=A[1:]-phi*A[:-1]; ys=yt[1:]-phi*yt[:-1]; prior=1e-2
    s2=np.var(ys-As@np.linalg.lstsq(As,ys,rcond=None)[0]); P=As.T@As/s2+prior*np.eye(A.shape[1]); S=np.linalg.inv(P); mn=S@(As.T@ys/s2)
    # previsão a 24 h a partir de cada janela (como na fase 1)
    Ate=desenho(te,fr); yte=y[NTR:]; cob=[];cob_iid=[];lar=[];er=[];er_sem=[]
    z80=1.2815515655; se=np.sqrt(np.sum((yt-A@mn)**2)/(NTR-A.shape[1]))
    for o in range(0,NTE-48,24):
        h=np.arange(1,49); prev=Ate[o:o+48]@mn; e0=(y[NTR+o-1]-(desenho(np.array([t[NTR+o-1]]),fr)@mn)[0])
        pm_=prev+e0*phi**h; vr=s2*(1-phi**(2*h))/(1-phi**2)+np.einsum('ij,jk,ik->i',Ate[o:o+48],S,Ate[o:o+48]); sd=np.sqrt(vr); yy=yte[o:o+48]
        cob.append(np.mean(abs(yy-pm_)<=z80*sd)); cob_iid.append(np.mean(abs(yy-prev)<=z80*se)); lar.append(2*z80*sd.mean()); er.append(np.mean(abs(yy-pm_))); er_sem.append(np.mean(abs(yy-prev)))
    R.append({'modelo':'Bayes-AR','seed':seed,'cob80':float(np.mean(cob)),'largura':float(np.mean(lar)),'mae24h':float(np.mean(er)),'mae24h_sem_erro':float(np.mean(er_sem)),'cob80_iid':float(np.mean(cob_iid)),'phi':phi,'janelas':len(cob)})
json.dump(R,open('r3.json','w'),indent=1)
import collections
d=collections.defaultdict(list)
for r in R:
    if 'mae' in r: d[r['modelo']].append(r['mae'])
for k,v in d.items(): print(k,np.round(v,3))
print([r for r in R if r['modelo']=='Bayes-AR'])
