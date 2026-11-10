import time,json,warnings; warnings.filterwarnings('ignore')
import numpy as np, cudaq, pennylane as qml
from comum import qc
R=[]
for n,rep in [(4,300),(12,60),(18,12),(22,3)]:
    L=3; th=np.random.default_rng(0).uniform(0,6.28,2*n*L); a=[.3]*n; thl=list(map(float,th))
    O=[cudaq.spin.z(j) for j in range(n)]
    t=time.perf_counter()
    for _ in range(rep): r=cudaq.observe(qc,O,a,thl,n,L)
    tc=(time.perf_counter()-t)/rep; cv=[s.expectation() for s in r]
    def mk(dev):
        d=qml.device(dev,wires=n)
        @qml.qnode(d,diff_method=None)
        def f():
            i=0
            for l in range(L):
                for j in range(n):
                    qml.RY(th[i],j); qml.RZ(th[i+1],j); i+=2
                for j in range(n-1): qml.CNOT([j,j+1])
                if l<L-1:
                    for j in range(n): qml.RZ(a[j],j)
            return [qml.expval(qml.PauliZ(j)) for j in range(n)]
        return f
    row={'qubits':n,'cudaq_qpp_ms':tc*1e3}
    for dev in ['lightning.qubit','default.qubit']:
        f=mk(dev); r2=rep if dev=='lightning.qubit' or n<18 else 2
        t=time.perf_counter()
        for _ in range(r2): v=f()
        row[dev+'_ms']=(time.perf_counter()-t)/r2*1e3; row[dev+'_dif']=float(np.max(abs(np.array(v)-np.array(cv))))
    print(row,flush=True); R.append(row)
json.dump(R,open('r5.json','w'),indent=1)
