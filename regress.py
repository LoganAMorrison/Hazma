import sys, json; sys.path.insert(0,"/tmp/hz-plots")
import numpy as np, warnings; warnings.filterwarnings("ignore")
from multiprocessing import Pool
from sweep import Generic, reference
from hazma.scalar_mediator import HiggsPortal
from hazma.relic_density._thermal_functions import thermal_cross_section
MS=np.linspace(150,700,56); XS=np.geomspace(0.5,300,60)
def row(ms):
    inner=HiggsPortal(mx=200.0, ms=ms, gsxx=1.0, stheta=1e-4)
    return [abs(thermal_cross_section(x,Generic(inner))/reference(x,inner)-1) for x in XS]
if __name__=="__main__":
    with Pool(8) as p: E=np.array(p.map(row,MS))
    json.dump({"ms":MS.tolist(),"x":XS.tolist(),"err":E.tolist()},open(sys.argv[1],"w"))
    w=E.max(axis=1); 
    for ms,e in zip(MS,w):
        if e>1e-6: print(ms,e)
    print("overall",E.max())
