from mosek.fusion import *
import io
import sys
from skypie.util.my_dataclasses import MosekOptimizerType, NormalizationType

class MyStream(io.StringIO):

    def __init__(self,*args, verbose: int = 0, **kwargs):

        self.verbose = verbose
        super().__init__(*args, **kwargs)

    def write(self, s):
        if self.verbose > 0:
            sys.stdout.write(s)
        return super().write(s)

def setSolverSettings(*, M: Model, optimizerType: MosekOptimizerType, optimizerThreads: int, verbose: int, normalize: NormalizationType) -> MyStream:
    M.setSolverParam("optimizer", optimizerType) # Primal simplex seems most accurate for our problem, interior-point is way off
    M.setSolverParam("numThreads", optimizerThreads)
    M.setSolverParam("presolveLindepUse", "off") # Disable linear dependency check in presolve phase, as this is the problem we're seeking to solve
    
    if normalize == NormalizationType.Mosek:
        M.setSolverParam("simScaling", "free")
        #M.setSolverParam("simScalingMethod", "pow2")
    else:
        M.setSolverParam("simScaling", "none")         # We take care of scaling ourselves

    w = MyStream("", verbose=verbose)
    if verbose > 2:
        M.setLogHandler(w)

    return w