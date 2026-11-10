from comum import *
import json
n,L=4,3; om=np.array([W[k] for k in BASE]); fr=combos(om,2); out=[]
for seed in range(3):
    th=np.load(f'th_mare_{seed}.npy'); c=np.load(f'c_mare_{seed}.npy'); t,_,_=sinal(seed)
    g=np.random.default_rng(seed); ta=g.uniform(0,8766,2500); F=feats(ta,om,th,n,L); ya=c[0]+F@c[1:]
    A=desenho(ta,fr); b=np.linalg.lstsq(A,ya,rcond=None)[0]; te=t[NTR:]; Ft=feats(te,om,th,n,L); yq=c[0]+Ft@c[1:]; ys=desenho(te,fr)@b
    out.append({'seed':seed,'coeficientes':A.shape[1],'dif_max_cm':float(abs(yq-ys).max()),'dif_media_cm':float(abs(yq-ys).mean())})
print(out); json.dump(out,open('r6.json','w'),indent=1)
