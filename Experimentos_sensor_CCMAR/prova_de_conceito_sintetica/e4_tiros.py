from comum import *
import json
t,limpo,y=sinal(0); n,L=4,3; om=np.array([W[k] for k in BASE]); th=np.load('th_mare_0.npy'); c=np.load('c_mare_0.npy'); mu=y[:NTR].mean()
g=np.random.default_rng(3); idx=np.sort(g.choice(np.arange(NTR,NTR+NTE),80,replace=False)); O=obs(n); out={}
ex=feats(t[idx],om,th,n,L); yex=c[0]+ex@c[1:]+mu; aerr=abs(yex-limpo[idx])
for sh in [100,1000,10000]:
    Y=[]
    for r in range(30):
        F=[]
        for x in t[idx]:
            res=cudaq.observe(qc,O,[float(o*x) for o in om],list(map(float,th)),n,L,shots_count=sh); F.append([s.expectation() for s in res])
        Y.append(c[0]+np.array(F)@c[1:]+mu)
    sd=np.std(Y,0); out[sh]={'desvio_medio_cm':float(sd.mean()),'corr_spearman_desvio_erro':float(__import__('scipy.stats').stats.spearmanr(sd,aerr)[0])}
out['erro_abs_medio_cm']=float(aerr.mean()); print(json.dumps(out,indent=1)); json.dump(out,open('r4.json','w'),indent=1)
